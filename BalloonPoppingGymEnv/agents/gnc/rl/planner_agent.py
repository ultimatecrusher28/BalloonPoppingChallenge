"""PlannerAgent: on-board rollout beam planner + lead-pursuit execution, with the whole-burn tour.

Reads only observation + given_parameters. Every ``replan_dt`` (and right after the current target is popped or lost) the
planner (planner.BeamPlanner) rolls out target sequences in the given-only simulator (given_sim.GivenSim) with the same
guidance law executed here; the first target of the best sequence is flown with ``guidance`` -> ``accel_for_thrust_dir``
-> ``CompensatedInnerLoop``. Launch: at ``launch_t`` (a fixed agent constant), attitude chosen by planning from the pad.
At ``tour_deploy["t"]`` s after launch (again at the retry times if no tour locked) the whole-burn tour is solved from the
agent's own information (_tour_classes); a locked tour is flown at full throttle to burnout and re-shot from the estimate
(_shoot_step). Every decision is made on flight time; the agent never reads the clock.
"""

from __future__ import annotations

import os

import numpy as np

from ....envs.balloon_world import check_action
from ...base_agent import BaseAgent
from .. import rocketnav as _rocketnav  # noqa: F401 (register)
from .. import registry
from ..follow.inner_loop import CompensatedInnerLoop, quat_to_matrix
from ..interfaces import LaunchDecision
from ..vehicle.given_limits import GivenLimits
from ..vehicle.rocket_model import RocketModel
from .given_sim import GivenSim, quat_from_nose
from .planner import BeamPlanner, guidance
from .predict import WindProfilePredictor
from .thrust_dir import accel_for_thrust_dir

DECISION_DT = 0.1
# tour_deploy settings (PlannerAgent._tour_classes): decision time (s after launch), program segments,
# CMA-ES (generations, population, initial step rad, rollout step s), LM polish (iterations, difference step rad, step cap
# rad), pass windows (half widths s: kept burn targets, the substitute; coast window length after burnout), the pass test
# (burn_max m; burn_max_near m for the burn targets the solution passes within near_t s of the decision; coast_hit m with
# pass_coast: both coast misses within it), the CMA-ES coast stop and burn weight, rank_cands (nseg, rsub, rmax, ntop), the
# classes ranked (n_cls candidates each), the lock-value calibration (tour value = burn targets + calib_burn + 2
# calib_coast against npop + V - calib_end_in when the chosen leaf ends by burn_time, npop - calib_end_after when it ends
# after), the schedule (the tour's burn target count first; each candidate's seeds in turn; at most seed_budget seeds per
# decision; workers forked processes, each pinned to its own physical core when the topology is readable), and retry_t
# (s after launch: a decision that did not lock is made again at the first plan at or after the next retry time).
TOUR_DEPLOY = dict(
    t=20.0,
    K=8,
    gens=20,
    pop=128,
    sigma0=0.15,
    es_dt=0.02,
    polish=6,
    h=0.04,
    smax=0.15,
    win=1.5,
    win_sub=2.5,
    coast_t=3.5,
    burn_max=0.8,
    coast_hit=1.2,
    coast_stop=0.96,
    w_burn=2.0,
    seeds=3,
    nseg=40,
    rsub=1.0,
    rmax=1.1,
    ntop=20,
    near_t=None,
    burn_max_near=1.4,
    n_cls=10,
    classes=("keep", "append", "drop", "swap"),
    pass_coast=False,
    calib_coast=0.912,
    calib_burn=0.06,
    calib_end_in=0.18,
    calib_end_after=0.30,
    seed_budget=None,
    workers=12,
    retry_t=None,
    mode="classes",
    value="calib",
    order="nburn",
    sched="cand",
    sigma0b=None,
    wall_max=None,
    parallel=True,
    pin_cores=True,
    rank_lazy=True,
    queue=True,
)
# the tour_deploy keys with the only value this agent implements (other values select code that is not shipped)
TOUR_FIXED = dict(
    mode="classes",
    value="calib",
    order="nburn",
    sched="cand",
    sigma0b=None,
    wall_max=None,
    parallel=True,
    pin_cores=True,
    rank_lazy=True,
    queue=True,
)
# guidance variants searched per target: (nav gain, t_go floor s, along-LOS boost m/s^2, aim offset up m, side m)
VARIANTS = (
    (2.0, 0.8, 0.0, 0.0, 0.0),
    (3.0, 0.3, 0.0, 0.0, 0.0),
    (4.0, 0.3, 0.0, 0.0, 0.0),
    (3.0, 0.3, 5.0, 0.0, 0.0),
    (3.0, 0.3, 0.0, 1.5, 0.0),
    (3.0, 0.3, 0.0, -1.5, 0.0),
    (3.0, 0.3, 0.0, 0.0, 1.5),
    (3.0, 0.3, 0.0, 0.0, -1.5),
)
# viable-intercept variants (6th column = acceleration margin), appended when vi_variants=True
VI_VARIANTS = ((3.0, 0.3, 0.0, 0.0, 0.0, 0.9), (3.0, 0.3, 0.0, 0.0, 0.0, 0.75))


def los_rows(los_variants):
    """Lean rows: one planner row per [nav, t_go floor, boost, fill] of los_variants, the regular law plus guidance's
    line-of-sight fill (12th column, los_fill); the other columns are 0."""
    out = []
    for r in los_variants or ():
        nav, tmin, bst, fill = (float(x) for x in r)
        out.append((nav, tmin, bst) + (0.0,) * 8 + (fill,))
    return tuple(out)


def _idle(launch=False, attitude=(90.0, 0.0)):
    return {
        "launch": bool(launch),
        "launch_inclination_heading": np.array(attitude, float),
        "tvc": np.zeros(2),
        "roll": np.zeros(1),
        "throttle": np.ones(1),
    }


def _require(name, value, ok):
    if value != ok or type(value) is not type(ok):
        raise ValueError(f"{name}={value!r}: this agent implements only {name}={ok!r}")


class PlannerAgent(BaseAgent):
    def __init__(
        self,
        given_parameters,
        *,
        sim_burnout_cut,
        sim_terms_t,
        xrows,
        late_search,
        lam_late,
        lam_late_t,
        tour_deploy,
        shoot_reshoot_late,
        critic_path,
        lam_coast,
        launch_t=36.0,
        beam=12,
        branch=10,
        depth=3,
        horizon=6.0,
        plan_dt=0.02,
        replan_dt=0.5,
        lam=0.3,
        radius_margin=0.8,
        stick=0.2,
        launch_els=(90.0, 75.0, 60.0),
        launch_cands=6,
        rocketnav="mekf",
        variants=VARIANTS,
        loop_kwargs=None,
        critic_beta=0.0,
        depth_gamma=1.0,
        predictor=True,
        pred_horizon=20.0,
        pred_sim_wind=True,
        pred_sim_wind_zlo=30.0,
        vi_variants=False,
        vi_wr=1.5,
        reach_kwargs=None,
        los_variants=(),
        shoot_reshoot_dt=0.5,
        shoot_reshoot_iters=2,
        shoot_reshoot_tmin=0.3,
    ):
        super().__init__(given_parameters)
        gp = given_parameters
        _require("sim_burnout_cut", sim_burnout_cut, True)
        _require("predictor", predictor, True)
        _require("pred_sim_wind", pred_sim_wind, True)
        self.model = RocketModel(gp)
        self.tables = GivenLimits(gp, self.model)
        self.dec_dt = DECISION_DT
        lk = dict(
            loop_kwargs or {}
        )  # kp_rate, ff (the inner loop and the planner's simulator: the same law)
        p3a = {
            k: v for k, v in lk.items() if k in ("act_inv", "ff") or k.startswith("ff_")
        }
        self.sim = GivenSim(
            gp,
            k_att=lk.get("k_att", 2.0),
            w_max=lk.get("w_max", 2.0),
            kp_rate=lk.get("kp_rate", 25.0),
            axial=False,
            cm_kin=False,
            grav_model=False,
            **p3a,
        )
        # sim_terms_t (s after launch): at the first step with t - t_launch >= sim_terms_t, before anything else of that
        # step, this GivenSim's axial (gimbal cos thrust loss), cm_kin (CDM reference point terms) and grav_model
        # (Somigliana gravity, buoyancy) are set True
        self.sim_terms_t = float(sim_terms_t)
        self._sim_terms_on = False
        self.loop = CompensatedInnerLoop(
            gp, self.model, tables=self.tables, ff_model=self.sim, **lk
        )
        # the navigation filter models the actuators: it gets the command the env applied (the previous action, from the
        # step after the launch step on)
        self.nav = registry.build("rocketnav", rocketnav, gp, self.model)
        self._sent = None
        self.radius = float(gp["balloon"]["radius"])
        # critic weights shipped under agents/gnc/rl/weights (trained on training seeds only, training/)
        from pathlib import Path

        cp = Path(critic_path)
        if not cp.is_absolute():
            cp = Path(__file__).resolve().parent / "weights" / cp
        z = np.load(cp)
        critic = {k: z[k] for k in z.files if k != "meta"}
        self.planner = BeamPlanner(
            self.sim,
            self.radius,
            beam=beam,
            branch=branch,
            depth=depth,
            horizon=horizon,
            dt=plan_dt,
            dec_dt=self.dec_dt,
            lam=lam,
            radius_margin=radius_margin,
            stick=stick,
            # los_variants (lean rows): [nav, t_go floor, boost, fill] rows with guidance's line-of-sight fill, appended
            # after the regular / vi rows (x rows follow, enable_x)
            variants=tuple(tuple(v) for v in variants)
            + (VI_VARIANTS if vi_variants else ())
            + los_rows(los_variants),
            critic=critic,
            critic_beta=critic_beta,
            deep_var=1,
            deep_radius=3.0,
            depth_gamma=depth_gamma,
            **dict(reach_kwargs or {}),
        )
        self.planner.vi_wr = self.vi_wr = float(vi_wr)
        # lam_coast: time cost per s after burnout in the leaf key (BeamPlanner)
        self.planner.lam_coast = float(lam_coast)
        # late_search {t: s after launch, depth/beam/branch: ints}: from the first in-flight plan at or after t the planner
        # searches with these sizes (a fixed switch on flight time)
        ls = dict(late_search)
        if not (set(ls) <= {"t", "depth", "beam", "branch"} and "t" in ls):
            raise ValueError(
                f"late_search {late_search}: keys t (required), depth, beam, branch"
            )
        self.late_search = ls
        self._late_on = False
        # the locked tour program is flown at full throttle to burn_time (coast_shoot.program_cmd) and re-shot from the
        # estimate every shoot_reshoot_dt s while more than shoot_reshoot_tmin s of the burn is left, with
        # shoot_reshoot_iters Levenberg-Marquardt iterations on the segments not finished yet (_shoot_step)
        self.shoot_reshoot_dt = float(shoot_reshoot_dt)
        self.shoot_reshoot_iters = int(shoot_reshoot_iters)
        self.shoot_reshoot_tmin = float(shoot_reshoot_tmin)
        # shoot_reshoot_late (t_left, dt, tmin, iters): once at most t_left s of the burn is left the re-shoot runs every
        # dt s (the first at once) while more than tmin s is left, with up to iters iterations (the tour re-shoot stops
        # earlier once every miss is within its stop)
        self.shoot_reshoot_late = tuple(float(x) for x in shoot_reshoot_late)
        if len(self.shoot_reshoot_late) != 4:
            raise ValueError(
                f"shoot_reshoot_late {shoot_reshoot_late}: (t_left, dt, tmin, iters)"
            )
        # tour_deploy: _tour_classes' settings (TOUR_DEPLOY keys); at the first plan at or after tour_deploy["t"] s after
        # launch the tour is solved from the agent's own information and locked in planner.shoot_cur only if it passes.
        # tour_rec: that decision (diagnostics); tour_prev: the earlier decisions' records (retry_t)
        unknown = set(tour_deploy) - set(TOUR_DEPLOY)
        if unknown:
            raise ValueError(f"unknown tour_deploy keys {sorted(unknown)}")
        self.tour_deploy = dict(TOUR_DEPLOY, **tour_deploy)
        for k, v in TOUR_FIXED.items():
            _require(f"tour_deploy {k}", self.tour_deploy[k], v)
        if (
            self.tour_deploy["seed_budget"] is None
            or self.tour_deploy["near_t"] is None
        ):
            raise ValueError("tour_deploy seed_budget and near_t must be set")
        self._tour_done = False
        self.tour_rec = None
        self.tour_prev = []
        self._tour_retry = None  # the retry times left (None: not read yet)
        self._shoot_prog = None  # the program the re-shoot clock belongs to (a new shoot_cur restarts it)
        self._shoot_next = None
        self.shoot_stats = dict(n=0)  # re-shoots (diagnostics)
        # lam_late: in-burn time cost of plans made lam_late_t s or more after launch
        self.planner.lam_late, self.planner.lam_late_t = (
            float(lam_late),
            float(lam_late_t),
        )
        # xrows: moving-waypoint sweep legs as extra level-0 planner rows (BeamPlanner.enable_x); the chosen row's (balloon,
        # pass time, exit direction) is flown in feedback here (BeamPlanner.x_flight) until the next plan
        self.xcur = None
        self.planner.enable_x(xrows)
        # every row's late re-point, exit-blend, terminal-velocity and lateral-push columns (rp, zb, zk, la, lt) are 0: the
        # guidance code for them is not shipped
        if np.any(self.planner.V[:, 6:11] != 0.0):
            raise ValueError(
                "planner rows with rp / zb / zk / la / lt columns are not supported by this agent"
            )
        self.launch_t = float(launch_t)
        self.replan_dt = float(replan_dt)
        self.launch_els = tuple(float(e) for e in launch_els)
        self.launch_cands = int(launch_cands)
        self.elev = float(self.model.elevation)
        self.pad = np.array([0.0, 0.0, self.elev])
        self.t_launch = None
        self.next_dec = None
        self.next_plan = -1.0
        self.target = None
        self.var = 0
        self.n_cmd = np.array([0.0, 0.0, 1.0])
        self._last_good = None
        self.log = []
        # row-change counters on the same target (diagnostics): plans, plans that keep the target (same) and their time
        # (same_s), row index changes (change_idx) and of them returns to the row left by the previous index change on
        # that target within 0.3 s (back_idx); change = leg changes (an index change, or a fresh x row; the keep row
        # continues the x row flown now)
        self.row_n = dict(
            plans=0, same=0, same_s=0.0, change=0, change_idx=0, back_idx=0
        )
        self._row_prev = None  # (t of the previous plan with a choice, row before the last index change, its t)
        # predictor: altitude wind-profile balloon tracks (predict.py) for the planner, and the same profile from
        # pred_sim_wind_zlo (m ASL) to the highest balloon observed as GivenSim's wind (held constant outside)
        self.pred = WindProfilePredictor(z_min=self.elev)
        self.pred_horizon = float(pred_horizon)
        self.pred_sim_wind_zlo = float(pred_sim_wind_zlo)
        self._next_obs = -1.0

    # -- helpers -------------------------------------------------------------------
    @staticmethod
    def _balloons(obs):
        bs = np.asarray(obs["balloon_states"], float)
        st = np.asarray(obs["balloon_status"]).reshape(-1)
        return bs[:, :3], bs[:, 3:6], st

    def _state(self, pos, vel, q, w, g, thr, t, t_launch, nb):
        return dict(
            pos=pos[None].astype(float),
            vel=vel[None].astype(float),
            q=q[None].astype(float),
            w=w[None].astype(float),
            g=g[None].astype(float),
            thr=np.array([thr], float),
            t=np.array([t], float),
            popped=np.zeros((1, nb), bool),
            crash=np.zeros(1, bool),
            t_launch=float(t_launch),
        )

    def _observe(self, obs, t):
        """Feed the predictor every DECISION_DT (also before launch: free wind history)."""
        if t + 1e-9 < self._next_obs:
            return
        self._next_obs = t + DECISION_DT
        bp, bv, st = self._balloons(obs)
        self.pred.observe(t, bp, bv, st == 1)

    def _world(self, bp, bv, rel, t):
        """Wind for GivenSim and balloon tracks for the planner, right before a plan() call."""
        self.pred.refit(t, bp, bv, rel)
        tab = (
            self.pred.wind_table(self.pred_sim_wind_zlo)
            if self.pred.U is not None
            else None
        )
        if tab is not None:
            self.sim.wz, self.sim.wu, self.sim.wv = tab
        else:
            self.sim.update_wind(bp, bv, rel)
        self.planner.track = (
            self.pred.table(bp, bv, rel, horizon=self.pred_horizon, dt=0.1),
            t,
            0.1,
        )

    def _launch(self, obs, t):
        bp, bv, st = self._balloons(obs)
        rel = st == 1
        self._world(bp, bv, rel, t)
        cands = [np.array([0.0, 0.0, 1.0])]
        if rel.any():
            d = np.linalg.norm(bp - self.pad, axis=1)
            d[~rel] = np.inf
            for i in np.argsort(d)[: self.launch_cands]:
                if not np.isfinite(d[i]):
                    break
                hz = np.arctan2(bp[i, 1], bp[i, 0])
                for e in self.launch_els[1:]:
                    er = np.radians(e)
                    cands.append(
                        np.array(
                            [
                                np.cos(er) * np.cos(hz),
                                np.cos(er) * np.sin(hz),
                                np.sin(er),
                            ]
                        )
                    )
        best, bkey = cands[0], -np.inf
        for n in cands:
            S0 = self._state(
                self.pad,
                np.zeros(3),
                quat_from_nose(n[None])[0],
                np.zeros(2),
                np.zeros(2),
                1.0,
                t,
                t,
                len(bp),
            )
            f, info = self.planner.plan(S0, bp, bv, t, rel)
            if info is not None and info[0] > bkey:
                best, bkey = n, info[0]
        el = float(np.degrees(np.arcsin(np.clip(best[2], -1, 1))))
        head = float(np.degrees(np.arctan2(best[0], best[1])) % 360.0)
        return el, head, best

    def _plan_step(self, est, bp, bv, st, rel, lost, t, t_rel):
        """One plan from the navigation estimate; sets target / variant and the next plan time."""
        self._world(bp, bv, rel, t)
        pos = np.asarray(est.position, float)
        vel = np.asarray(est.velocity, float)
        q = np.asarray(est.quaternion, float)
        om = np.asarray(est.omega, float)
        g = np.asarray(self.loop.g_act, float)
        thr = float(self.loop.thr_act)
        S0 = self._state(pos, vel, q, om[:2], g, thr, t, self.t_launch, len(bp))
        S0["ff"] = self.loop.ff_state[None].copy()
        cur = self.target if (self.target is not None and not lost) else None
        td = self.tour_deploy
        if (
            not self._tour_done and self.planner.seq_log_at is None
        ):  # tour_deploy: the plan to log for the decision
            self.planner.seq_log_at = float(td["t"])
        xc = self.xcur  # the keep row continues the x row flown now
        self.planner.x_cur = (
            xc if (xc is not None and cur is not None and xc["tgt"] == cur) else None
        )
        if not self._late_on and t_rel >= self.late_search["t"]:
            for k in ("depth", "beam", "branch"):
                if k in self.late_search:
                    setattr(self.planner, k, int(self.late_search[k]))
            self._late_on = True
        f, info = self.planner.plan(S0, bp, bv, t, rel, cur=cur)
        if f is not None:
            if f[0] != self.target:
                self.log.append(("tgt", round(t_rel, 2), int(f[0]), int(f[1]), info))
            self._row_count(cur, int(f[0]), int(f[1]), t)
            self.target, self.var = int(f[0]), int(f[1])
            self.xcur = (
                self.planner.last_x if self.planner.xcode[self.var] >= 0 else None
            )
        elif lost:
            self.target = None
        self.next_plan = t + self.replan_dt
        if not self._tour_done and self.planner.seq_log is not None:
            self._tour_deploy(est, bp, bv, st, rel, t, t_rel)
            self._tour_retry_after()

    def _tour_retry_after(self):
        """retry_t, after a decision: one that did not lock is made again at the first plan at or after the next retry
        time, with the same settings."""
        if self._tour_retry is None:
            t0 = float(self.tour_deploy["t"])
            self._tour_retry = sorted(
                float(x) for x in self.tour_deploy["retry_t"] or [] if float(x) > t0
            )
        if (
            not self._tour_done
            or self.tour_rec is None
            or self.tour_rec.get("locked")
            or not self._tour_retry
        ):
            return
        self.tour_prev.append(self.tour_rec)
        self.tour_deploy = dict(self.tour_deploy, t=float(self._tour_retry.pop(0)))
        self.tour_rec = None
        self._tour_done = False
        self.planner.seq_log = None
        self.planner.seq_log_at = None

    def _tour_deploy(self, est, bp, bv, st, rel, t, t_rel):
        """The tour decision, once at the plan that filled planner.seq_log (_tour_classes)."""
        self._tour_done = True
        return self._tour_classes(self.tour_deploy, est, bp, bv, st, rel, t, t_rel)

    def _tour_classes(self, cfg, est, bp, bv, st, rel, t, t_rel):
        """Whole-burn tour decision, once at the plan that filled planner.seq_log, from the agent's own information only
        (estimate and full live state, balloon predictor, GivenSim, the chosen leaf):
        1. burn list: the chosen leaf's legs ending by burn_time + 0.01 s (per leg its target as a waypoint at the leg's end,
           the leg's other pops in its group);
        2. the classes keep (every burn target + a coast pair), append (+ one more after the last), drop (without the last
           leg, rank_cands keep_only) and swap (the last leg replaced by one substitute), each ranked by the LQ energy
           ratio (only the classes that can give a task, _tour_rank_lazy); the first n_cls of each, merged by the tour's
           burn target count descending, then ratio <= 1 first, the ratio, the class and the rank;
        3. the value: tour value = burn targets + calib_burn + 2 calib_coast, base = npop + V - calib_end_in if the chosen
           leaf ends by burn_time + 0.01 s, else npop - calib_end_after (V = the leaf's critic term / critic_beta);
           candidates whose tour value does not exceed the base are not solved;
        4. the schedule (_tour_tasks): per candidate in that order, seeds 0..seeds-1 of CMA-ES + LM polish from the lq_dirs
           start; a pass: every burn miss within burn_max (burn_max_near for the targets passed within near_t s of the
           decision) and both coast misses within coast_hit; the first passing tour in the order is locked.
        The lock is planner.shoot_cur with the tour, flown by _shoot_step / _tour_reshoot."""
        sl = self.planner.seq_log
        sim, burn, tl = self.sim, float(self.sim.burn_time), float(self.t_launch)
        T = float(t_rel)
        rec = dict(t_rel=round(T, 3), locked=False, mode="classes", value=cfg["value"])
        self.tour_rec = rec
        legs = sl["legs"]
        way_ids, way_t, groups = [], [], []
        for lg in legs:
            if lg[2] > burn + 0.01 + 1e-9:
                continue
            pops = [
                int(b)
                for b in lg[3]
                if int(b) not in way_ids and all(int(b) not in g_ for g_ in groups)
            ]
            if not pops:
                continue
            tgt = int(lg[0]) if int(lg[0]) in pops else pops[0]
            way_ids.append(tgt)
            way_t.append(float(lg[2]))
            groups.append([b for b in pops if b != tgt])
        burn_ids = way_ids + [b for g_ in groups for b in g_]
        nw = len(way_ids)
        cb = float(self.planner.critic_beta)
        vterm = float(sl.get("vterm") or 0.0)
        V = vterm / cb if cb else 0.0
        npop = float(sl["npop"])
        end_in = (not legs) or legs[-1][2] <= burn + 0.01 + 1e-9
        base = (
            npop + V - float(cfg["calib_end_in"])
            if end_in
            else npop - float(cfg["calib_end_after"])
        )
        rec.update(
            burn_ids=burn_ids,
            way_ids=way_ids,
            way_t=[round(x, 3) for x in way_t],
            groups=groups,
            base_npop=int(npop),
            base_vterm=vterm,
            base_V=V,
            leaf_end_in=bool(end_in),
            base_val=round(base, 4),
            base_legs=legs,
        )
        self.log.append(("tour_deploy", round(T, 2), burn_ids))
        pos, vel = np.asarray(est.position, float), np.asarray(est.velocity, float)
        full = dict(
            q=np.asarray(est.quaternion, float),
            w=np.asarray(est.omega, float)[:2],
            g=np.asarray(self.loop.g_act, float),
            thr=float(self.loop.thr_act),
        )
        full["ff"] = self.loop.ff_state.copy()
        bpos = self.planner._bp_all
        ids = np.flatnonzero(np.asarray(rel, bool) & (np.asarray(st) != 2))
        ids = np.array([i for i in ids if int(i) not in burn_ids], int)
        rk = dict(
            nseg=int(cfg["nseg"]),
            rsub=float(cfg["rsub"]),
            rmax=float(cfg["rmax"]),
            ntop=int(cfg["ntop"]),
        )
        L = {}
        nlast = (1 + len(groups[-1])) if nw else 0
        nb_of = dict(
            keep=len(burn_ids),
            append=len(burn_ids) + 1,
            drop=len(burn_ids) - nlast,
            swap=len(burn_ids) - nlast + 1,
        )
        order = list(cfg["classes"])
        if len(ids) >= 2:
            self._tour_rank_lazy(
                cfg,
                rec,
                L,
                nb_of,
                order,
                base,
                sim,
                pos,
                vel,
                T,
                tl,
                bpos,
                ids,
                way_ids,
                way_t,
                nw,
                rk,
            )
        M = [
            (c[0], c[1], c[2], c[3], c[4], cls, rnk)
            for cls in order
            if cls in L
            for rnk, c in enumerate(L[cls][: int(cfg["n_cls"])])
        ]
        M.sort(key=lambda c: (-nb_of[c[5]], c[0] > 1.0, c[0], order.index(c[5]), c[6]))
        rec.update(n_cands={k: len(v) for k, v in L.items()}, n_merged=len(M))
        return self._tour_tasks(
            cfg, rec, M, nb_of, base, way_ids, way_t, groups, pos, vel, full, bpos, T
        )

    @staticmethod
    def _tour_rank_lazy(
        cfg,
        rec,
        L,
        nb_of,
        order,
        base,
        sim,
        pos,
        vel,
        T,
        tl,
        bpos,
        ids,
        way_ids,
        way_t,
        nw,
        rk,
    ):
        """The classes of _tour_classes that can give a task, ranked into L, the others not ranked. A class whose tour value
        (its burn target count + calib_burn + 2 calib_coast) does not exceed the base never gives a task (_tour_tasks'
        calib filter); the tasks are the first ceil(seed_budget / seeds) candidates of the merged order, which runs by burn
        target count descending first, so the classes are ranked group by group (the same count) from the highest and the
        lower groups are not needed once the groups above have that many candidates (each class's first n_cls). keep and
        swap come from one rank_cands call; keep alone (swap not needed) from the keep_only call, the same list.
        tour_rec rank_lazy: gated (by the calib value), ranked, skipped (not reachable in the budget)."""
        from .tour import rank_cands

        avail = (["keep", "swap", "drop"] if nw else ["keep"]) + ["append"]
        tvc = float(cfg["calib_burn"]) + 2.0 * float(cfg["calib_coast"])
        ok = [c for c in avail if c in order and nb_of[c] + tvc > base]
        D = {}

        def rank(c):
            if c in L:
                return
            if c in ("keep", "swap") and nw:
                if "swap" in need or c == "swap":
                    if "D" not in D:
                        D["D"] = rank_cands(
                            sim, pos, vel, T, tl, bpos, ids, way_ids, way_t, **rk
                        )
                    L["keep"] = [x for x in D["D"] if x[1] < 0]
                    L["swap"] = [x for x in D["D"] if x[1] >= 0]
                else:
                    L["keep"] = rank_cands(
                        sim,
                        pos,
                        vel,
                        T,
                        tl,
                        bpos,
                        ids,
                        way_ids,
                        way_t,
                        keep_only=True,
                        **rk,
                    )
            elif c == "keep":
                L["keep"] = rank_cands(
                    sim, pos, vel, T, tl, bpos, ids, [], [], keep_only=True, **rk
                )
            elif c == "drop":
                L["drop"] = rank_cands(
                    sim,
                    pos,
                    vel,
                    T,
                    tl,
                    bpos,
                    ids,
                    way_ids[:-1],
                    way_t[:-1],
                    keep_only=True,
                    **rk,
                )
            else:
                L["append"] = rank_cands(
                    sim, pos, vel, T, tl, bpos, ids, way_ids, way_t, append=True, **rk
                )

        need = set()
        ncand = -(-int(cfg["seed_budget"]) // int(cfg["seeds"]))
        have = 0
        for v in sorted({nb_of[c] for c in ok}, reverse=True):
            if have >= ncand:
                break
            grp = [c for c in ok if nb_of[c] == v]
            need |= set(grp)
            for c in sorted(
                grp, key=lambda c: c != "swap"
            ):  # swap first: keep then comes from the same call
                rank(c)
            have += sum(min(int(cfg["n_cls"]), len(L[c])) for c in grp)
        rec["rank_lazy"] = dict(
            gated=[c for c in avail if c in order and c not in ok],
            ranked=sorted(L),
            skipped=[c for c in ok if c not in L],
        )

    def _tour_tasks(
        self, cfg, rec, M, nb_of, base, way_ids, way_t, groups, pos, vel, full, bpos, T
    ):
        """The schedule of _tour_classes (the team compute cap of RULES section 4). The merged candidates M (in the order of
        _tour_classes, only those whose tour value exceeds the base) as (candidate, seed) tasks: seeds 0..seeds-1 of the
        first, then of the second, ..., the first seed_budget of them. A task is one seed (the candidate's tour and lq_dirs
        start, CMA-ES at sigma0, LM polish, the pass test). The tasks go in schedule order to the first free of workers
        forked processes (tour.fork_queue) with BLAS one thread; once the first passing task in the order is known with
        every task before it done, the running workers are stopped and that task is locked, so the lock is the
        one-at-a-time run's. pin_cores: tour.phys_cores of this process's allowed CPUs (the whole machine when no cgroup or
        taskset limits it), the first workers of them; worker k runs only on the k-th, this process (the candidates' starts,
        met next while the workers run) on the first, so the decision never runs on more than that many physical cores
        (the ranking before runs on this process's one core); fewer cores than workers: that many workers; one core: the
        tasks one at a time here. The pinning changes where a seed runs, not what it computes.
        tour_rec: tried (per started candidate, its seeds in run order up to the lock), n_seeds (tasks up to the lock),
        extra (tasks finished after the lock, run but not used), batches (tasks run here one at a time), workers,
        pin_cpus (None: not pinned), n_phys (physical cores found), queue (the workers' log)."""
        from .tour import (
            blas_threads,
            fork_queue,
            lq_dirs,
            phys_cores,
            tour_es,
            tour_lm,
        )

        sim, burn, tl = self.sim, float(self.sim.burn_time), float(self.t_launch)
        nw = len(way_ids)
        K = int(cfg["K"])
        win, win_s, bmax = (
            float(cfg["win"]),
            float(cfg["win_sub"]),
            float(cfg["burn_max"]),
        )
        ct = float(cfg["coast_t"])
        seeds = int(cfg["seeds"])
        tvc = float(cfg["calib_burn"]) + 2.0 * float(cfg["calib_coast"])
        C = [c for c in M if nb_of[c[5]] + tvc > base]
        tasks = [(i, sd) for i in range(len(C)) for sd in range(seeds)]
        tasks = tasks[: int(cfg["seed_budget"])]
        W = max(1, int(cfg["workers"]))
        pc = phys_cores()
        n_phys = len(pc)
        pin = None
        if pc:
            W = min(W, len(pc))
            pin = pc[:W]
        specs, recs, started = {}, {}, []

        def spec(i):  # candidate i's tour, lq_dirs start and record
            ratio, s_, ts, j1, j2, cls, rnk = C[i]
            if cls == "keep":
                wi, wt, gr, isub = (
                    list(way_ids),
                    list(way_t),
                    [list(x) for x in groups],
                    None,
                )
            elif cls == "append":
                wi, wt, gr, isub = (
                    list(way_ids) + [int(s_)],
                    list(way_t) + [float(ts)],
                    [list(x) for x in groups] + [[]],
                    nw,
                )
            elif cls == "drop":
                wi, wt, gr, isub = (
                    list(way_ids[:-1]),
                    list(way_t[:-1]),
                    [list(x) for x in groups[:-1]],
                    None,
                )
            else:
                wi, wt = list(way_ids[:-1]) + [int(s_)], list(way_t[:-1]) + [float(ts)]
                gr, isub = [list(x) for x in groups[:-1]] + [[]], nw - 1
            bj, bt_, bw_ = [], [], []
            for k_, (b, tb) in enumerate(zip(wi, wt)):
                bj.append(int(b))
                bt_.append(float(tb))
                bw_.append(win_s if k_ == isub else win)
                for b2 in gr[k_]:
                    bj.append(int(b2))
                    bt_.append(float(tb))
                    bw_.append(win_s)
            bt_, bw_ = np.asarray(bt_, float), np.asarray(bw_, float)
            lo_b = tl + np.maximum(T, bt_ - bw_)
            hi_b = tl + np.minimum(burn + 0.01, bt_ + bw_)
            D0, lq_r, lq_tau = lq_dirs(
                sim,
                pos,
                vel,
                T,
                tl,
                bpos,
                wi,
                wt,
                int(j1),
                int(j2),
                K,
                nseg=int(cfg["nseg"]),
            )
            js = np.array(bj + [int(j1), int(j2)], int)
            lo = np.concatenate([lo_b, [tl + burn] * 2])
            hi = np.concatenate([hi_b, [tl + burn + ct] * 2])
            nt = len(bj)
            stop = np.concatenate(
                [np.full(nt, bmax), np.full(2, float(cfg["coast_stop"]))]
            )
            w = np.concatenate([np.full(nt, float(cfg["w_burn"])), np.ones(2)])
            o0 = tour_lm(
                sim,
                full,
                pos,
                vel,
                T,
                tl,
                bpos,
                js,
                lo,
                hi,
                D0,
                iters=0,
                t_end=burn + ct,
                stop=stop,
                w=w,
            )
            specs[i] = dict(
                D0=D0,
                js=js,
                lo=lo,
                hi=hi,
                nt=nt,
                stop=stop,
                w=w,
                j1=int(j1),
                j2=int(j2),
            )
            recs[i] = dict(
                cls=cls,
                rank=rnk,
                cand=[
                    round(ratio, 4),
                    int(s_),
                    None if s_ < 0 else round(float(ts), 3),
                    int(j1),
                    int(j2),
                ],
                js=js.tolist(),
                n_burn=nt,
                lq_ratio=round(lq_r, 4),
                start_miss=[round(float(x), 3) for x in o0["miss"]],
                tries=[],
                passed=False,
            )

        def solve(task):  # one seed of candidate i
            i, sd = task
            S = specs[i]
            sg = float(cfg["sigma0"])
            o = tour_es(
                sim,
                full,
                pos,
                vel,
                T,
                tl,
                bpos,
                S["js"],
                S["lo"],
                S["hi"],
                S["D0"],
                t_end=burn + ct,
                stop=S["stop"],
                w=S["w"],
                gens=int(cfg["gens"]),
                pop=int(cfg["pop"]),
                sigma0=sg,
                dt=float(cfg["es_dt"]),
                seed=sd,
            )
            o = tour_lm(
                sim,
                full,
                pos,
                vel,
                T,
                tl,
                bpos,
                S["js"],
                S["lo"],
                S["hi"],
                o["dirs"],
                iters=int(cfg["polish"]),
                t_end=burn + ct,
                stop=S["stop"],
                w=S["w"],
                h=float(cfg["h"]),
                smax=float(cfg["smax"]),
            )
            nt = S["nt"]
            m = np.asarray(o["miss"], float)
            bm = np.full(nt, bmax)
            bm[np.asarray(o["tpass"], float)[:nt] - tl - T <= float(cfg["near_t"])] = (
                float(cfg["burn_max_near"])
            )
            okb = bool(np.all(m[:nt] <= bm))
            nco = int(np.sum(m[nt:] <= float(cfg["coast_hit"])))
            ok = okb and (nco == 2 or not cfg["pass_coast"])
            return dict(
                o=dict(
                    dirs=np.asarray(o["dirs"], float),
                    miss=np.asarray(o["miss"], float),
                    tpass=np.asarray(o["tpass"], float),
                ),
                tr=dict(
                    seed=sd,
                    sigma0=sg,
                    miss=[round(float(x), 3) for x in m],
                    ok_burn=okb,
                    n_coast=nco,
                    ok=ok,
                ),
            )

        st_ = dict(lock=None, n=0)

        def before(j):
            i = tasks[j][0]
            if i not in specs:
                spec(i)
                started.append(i)

        def take(j, r):  # the per-result step, in the schedule order
            i, sd = tasks[j]
            st_["n"] += 1
            tr_ = recs[i]
            if "tr" not in r:  # a worker failed: counted as a seed that did not pass
                tr_["tries"].append(dict(seed=sd, error=r.get("error"), ok=False))
                return False
            tr_["tries"].append(r["tr"])
            if r["tr"]["ok"]:
                tr_["passed"] = True
                tr_["val"] = nb_of[tr_["cls"]] + tvc
                S = specs[i]
                st_["lock"] = (
                    tr_,
                    r["o"],
                    S["js"],
                    S["lo"],
                    S["hi"],
                    S["stop"],
                    S["w"],
                    S["j1"],
                    S["j2"],
                )
            return st_["lock"] is not None

        extra, batches = [], []
        blas0 = blas_threads(1)
        aff0 = None
        if pin is not None:
            try:
                aff0 = os.sched_getaffinity(0)
                os.sched_setaffinity(0, {pin[0]})
            except OSError:
                aff0 = None
        qrec = None
        try:
            if W > 1 and len(tasks) > 1:
                q = fork_queue(solve, tasks, W, take, cpus=pin, before=before)
                for j, r in sorted(
                    q["done"].items()
                ):  # finished after the lock in the order: run but not used
                    i, sd = tasks[j]
                    extra.append(
                        [C[i][5], C[i][6], sd, None if "tr" not in r else r["tr"]["ok"]]
                    )
                qrec = dict(log=q["log"], started=q["started"], stopped=q["stopped"])
            else:  # one worker (one physical core) or at most one task: the tasks one at a time here, in order
                for j in range(len(tasks)):
                    before(j)
                    r = solve(tasks[j])
                    batches.append(1)
                    if take(j, r):
                        break
        finally:
            if blas0 is not None:
                blas_threads(blas0)
            if aff0 is not None:
                os.sched_setaffinity(0, aff0)
        lock, n_seeds = st_["lock"], st_["n"]
        tried = [
            recs[i] for i in started if recs[i]["tries"]
        ]  # a candidate first met after the lock is not in the record
        rec.update(
            tried=tried,
            n_tried=len(tried),
            timeout=False,
            sched=cfg["sched"],
            order=cfg["order"],
            seed_budget=cfg["seed_budget"],
            workers=W,
            n_tasks=len(tasks),
            n_seeds=n_seeds,
            extra=extra,
            batches=batches,
            n_solvable=len(C),
            pin_cpus=pin,
            n_phys=n_phys,
        )
        if qrec is not None:
            rec["queue"] = qrec
        if lock is not None:
            tr_, o, js, lo, hi, stop, w, j1, j2 = lock
            rec.update(
                locked=True,
                cand=[tr_["cls"]] + tr_["cand"],
                tour_js=js.tolist(),
                lock_val=tr_.get("val"),
                lock_miss=tr_["tries"][-1]["miss"],
            )
            self.planner.shoot_cur = dict(
                t0=T,
                dt=0.01,
                j1=j1,
                j2=j2,
                dirs=np.asarray(o["dirs"], float).tolist(),
                miss=[float(x) for x in o["miss"]],
                tpass=[float(x) for x in o["tpass"]],
                tour=dict(
                    js=js.tolist(),
                    lo=(lo - tl).tolist(),
                    hi=(hi - tl).tolist(),
                    stop=stop.tolist(),
                    w=w.tolist(),
                    h=float(cfg["h"]),
                    coast_t=ct,
                    smax=float(cfg["smax"]),
                ),
            )
        self.log.append(
            (
                "tour_lock" if lock is not None else "tour_nolock",
                round(T, 2),
                rec.get("cand"),
                n_seeds,
                rec.get("lock_val"),
                round(base, 3),
            )
        )

    def _shoot_step(self, est, obs, t, t_rel):
        """The thrust direction of the locked program planner.shoot_cur at t_rel, from burn_time on the velocity direction
        (without thrust the command does not act). Every shoot_reshoot_dt s from the lock (every dt s of
        shoot_reshoot_late over the burn's last t_left s), while more than tmin s of the burn is left, the program is
        re-shot first from the estimate (the full state as _plan_step's S0) on the angles of the segments not finished
        yet (coast_shoot.active: same boundaries, so the first point tried is the program flown so far); the best point
        replaces them (never worse in the model than keeping the program). Balloon tracks: the planner's last track."""
        from .coast_shoot import active, program_cmd

        sc = self.planner.shoot_cur
        burn = float(self.sim.burn_time)
        vel = np.asarray(est.velocity, float)
        if t_rel >= burn - 1e-9:
            return vel / max(float(np.linalg.norm(vel)), 1e-9)
        if (
            sc is not self._shoot_prog
        ):  # a new lock: the re-shoot clock starts at its t0
            self._shoot_prog = sc
            self._shoot_next = self.t_launch + float(sc["t0"]) + self.shoot_reshoot_dt
        lt = self.shoot_reshoot_late
        late = burn - t_rel <= lt[0] + 1e-9
        if (
            late and self._shoot_next > t + lt[1] + 1e-9
        ):  # entering the late schedule: the first late re-shoot at once
            self._shoot_next = t
        tmin = lt[2] if late else self.shoot_reshoot_tmin
        if t + 1e-9 >= self._shoot_next and burn - t_rel > tmin:
            nxt = self._shoot_next + self.shoot_reshoot_dt
            self._shoot_next = t + lt[1] if late else nxt
            D0, tseg = active(sc["dirs"], float(sc["t0"]), t_rel, burn)
            full = dict(
                q=np.asarray(est.quaternion, float),
                w=np.asarray(est.omega, float)[:2],
                g=np.asarray(self.loop.g_act, float),
                thr=float(self.loop.thr_act),
            )
            full["ff"] = self.loop.ff_state.copy()
            bpos = self.planner._bp_all
            o = self._tour_reshoot(
                sc,
                obs,
                full,
                np.asarray(est.position, float),
                vel,
                t_rel,
                np.asarray(D0, float),
                tseg,
                bpos,
                burn,
                iters=int(lt[3]) if late else None,
            )
            if o is not None:
                sc = dict(
                    sc,
                    dirs=o["dirs"],
                    t0=float(tseg),
                    miss=o["miss"],
                    tpass=o["tpass"],
                    n_re=int(sc.get("n_re", 0)) + 1,
                )
                self.planner.shoot_cur = self._shoot_prog = sc
                self.shoot_stats["n"] += 1
                self.log.append(
                    (
                        "shoot_re",
                        round(t_rel, 2),
                        [round(float(m), 3) for m in o["miss"]],
                        int(o["it"]),
                    )
                )
        return program_cmd(
            np.asarray(sc["dirs"], float)[None],
            np.array([t_rel]),
            float(sc["t0"]),
            burn,
        )[0]

    def _tour_reshoot(
        self, sc, obs, full, pos, vel, t_rel, D0, tseg, bpos, burn, iters=None
    ):
        """Tour re-shoot: tour.tour_lm with shoot_reshoot_iters iterations (iters: the late schedule's) on the targets of
        sc["tour"] (js, windows lo / hi in s after launch, stop, w, h, coast_t) not popped yet whose window has not
        closed; the window starts now at the earliest. The misses are reported for every target of the tour (nan: popped
        or closed). None: no target left."""
        from .tour import tour_lm

        tp = sc["tour"]
        js = np.asarray(tp["js"], int)
        lo, hi = np.asarray(tp["lo"], float), np.asarray(tp["hi"], float)
        _, _, stt = self._balloons(obs)
        rem = (stt[js] != 2) & (hi > t_rel + 1e-9)
        if not rem.any():
            return None
        tl = float(self.t_launch)
        o = tour_lm(
            self.sim,
            full,
            pos,
            vel,
            t_rel,
            tl,
            bpos,
            js[rem],
            tl + np.maximum(lo[rem], t_rel),
            tl + hi[rem],
            D0,
            iters=iters if iters is not None else self.shoot_reshoot_iters,
            t_end=burn + float(tp["coast_t"]),
            stop=np.asarray(tp["stop"], float)[rem],
            w=np.asarray(tp["w"], float)[rem],
            h=float(tp["h"]),
            dt=float(sc["dt"]),
            smax=float(tp["smax"]),
            t_prog0=tseg,
        )
        miss, tpass = np.full(len(js), np.nan), np.full(len(js), np.nan)
        miss[rem], tpass[rem] = o["miss"], o["tpass"]
        return dict(o, miss=miss, tpass=tpass)

    def _row_count(self, cur, tg, v, t):
        """Row-change counters (row_n) for the plan choice (tg, v) at time t, before it replaces (target, var); cur = the
        current target (None: none or lost). Diagnostics only."""
        R = self.row_n
        R["plans"] += 1
        pv = self._row_prev
        if cur is None or tg != cur or pv is None:
            self._row_prev = (t, None, None)
            return
        R["same"] += 1
        R["same_s"] += t - pv[0]
        v0, xc = self.var, self.planner.xcode
        if xc[v] == len(self.planner.xr["vars"]):
            leg = xc[v0] < 0  # the keep row continues the x row flown now
        else:
            leg = v != v0 or xc[v] >= 0  # a fresh x row starts a new leg
        R["change"] += int(leg)
        if v != v0:
            R["change_idx"] += 1
            R["back_idx"] += int(
                pv[1] is not None and v == pv[1] and t - pv[2] <= 0.3 + 1e-6
            )
            self._row_prev = (t, v0, t)
        else:
            self._row_prev = (t, pv[1], pv[2])

    def _x_cmd(self, est, pos, vel, t, bp, bv, st):
        """In-flight command of the chosen x row (None: no x row for this target, or the base law now)."""
        xc = self.xcur
        if xc is None or xc["tgt"] != self.target:
            return None
        nose = quat_to_matrix(np.asarray(est.quaternion, float))[:, 2][None]
        w = xc.get("wp", -1)
        live = (
            w >= 0 and st[w] == 1
        )  # the row's sweep balloon still there (x_flight ignores bpw / bvw in pass mode)
        return self.planner.x_flight(
            pos,
            vel,
            nose,
            t,
            self.t_launch,
            xc,
            bp[w] if live else None,
            bv[w] if live else None,
        )

    # -- main -----------------------------------------------------------------------
    def get_action(self, observation):
        action = self._get_action(observation)
        # the env does not apply the launch step's tvc/throttle
        t = float(observation["simulation_time"])
        launched = self.t_launch is not None and t > self.t_launch + 1e-9
        self._sent = action if launched else None
        return action

    def _sim_terms_step(self, t):
        """sim_terms_t (see __init__): the GivenSim terms on at the first step with t - t_launch >= sim_terms_t."""
        if self.t_launch is not None and t - self.t_launch >= self.sim_terms_t:
            self.sim.axial = self.sim.cm_kin = self.sim.grav_model = True
            self._sim_terms_on = True
            self.log.append(("sim_terms", round(t - self.t_launch, 3)))

    def _get_action(self, observation):
        t = float(observation["simulation_time"])
        if not self._sim_terms_on:
            self._sim_terms_step(t)
        self._observe(observation, t)
        if self.t_launch is None:
            if t + 1e-9 < self.launch_t:
                return _idle()
            el, head, n = self._launch(observation, t)
            self.t_launch = t
            self.next_dec = t + self.dec_dt
            self.next_plan = t
            dec = LaunchDecision(
                t_launch=t, inclination_deg=el, heading_deg=head, max_rail_tilt_deg=90.0
            )
            self.nav.on_launch(dec, t)
            self.loop.on_launch(t)
            self.n_cmd = n
            self.log.append(("launch", round(t, 2), round(el, 1), round(head, 1)))
            return _idle(launch=True, attitude=(el, head))

        if self._sent is not None:
            self.nav.set_actuation(self._sent["tvc"], self._sent["throttle"])
        est = self.nav.update(observation, deadline=float("inf"))
        t_rel = t - self.t_launch
        if t + 1e-9 >= self.next_dec:
            self.next_dec += self.dec_dt
            bp, bv, st = self._balloons(observation)
            rel = st == 1
            lost = self.target is None or st[self.target] != 1
            if lost or t + 1e-9 >= self.next_plan:
                self._plan_step(est, bp, bv, st, rel, lost, t, t_rel)
            if self.target is not None and st[self.target] == 1:
                pos = np.asarray(est.position, float)[None]
                vel = np.asarray(est.velocity, float)[None]
                nav, tmin, bst, ou, os_, vi, rp, zb, zk, la, lt, lf = self.planner.V[
                    self.var
                ][:12]
                xn = None
                if self.planner.xcode[self.var] >= 0:
                    xn = self._x_cmd(est, pos, vel, t, bp, bv, st)
                if (
                    xn is not None
                ):  # x row: the sweep of the rollouts (base law where it has no solution)
                    self.n_cmd = xn
                elif (
                    vi > 0 or lf > 0
                ):  # viable-intercept / lean row (as the planner's rollouts; A = a_av)
                    nose = quat_to_matrix(np.asarray(est.quaternion, float))[:, 2][None]
                    a_av = self.sim.accel_avail(np.array([t_rel]), self.loop.thr_act)
                    tp, tv = bp[self.target][None], bv[self.target][None]
                    self.n_cmd = guidance(
                        pos,
                        vel,
                        tp,
                        tv,
                        pos[:, 2] - self.elev,
                        boost=bst,
                        nav=nav,
                        tgo_min=tmin,
                        off_up=ou,
                        off_side=os_,
                        vi=vi,
                        nose=nose,
                        a_av=a_av,
                        vi_wr=self.vi_wr,
                        vi_tau0=self.sim.gtau,
                        los_fill=lf,
                    )[0]
                else:
                    self.n_cmd = guidance(
                        pos,
                        vel,
                        bp[self.target][None],
                        bv[self.target][None],
                        pos[:, 2] - self.elev,
                        boost=bst,
                        nav=nav,
                        tgo_min=tmin,
                        off_up=ou,
                        off_side=os_,
                    )[0]
            else:
                self.n_cmd = np.array([0.0, 0.0, 1.0])
        thr_c = 1.0
        if (
            self.planner.shoot_cur is not None
        ):  # the locked tour program: full throttle to burn_time
            self.n_cmd = self._shoot_step(est, observation, t, t_rel)
        acc = accel_for_thrust_dir(
            self.n_cmd, thr_c, est, self.model, self.tables, t_rel
        )
        action, _, _ = self.loop.solve(acc, est)
        if check_action(action):
            return self._last_good if self._last_good is not None else _idle()
        self._last_good = action
        return action
