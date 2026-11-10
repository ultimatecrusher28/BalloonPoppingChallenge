"""PlannerAgent: on-board rollout beam planner + lead-pursuit execution.

Reads only observation + given_parameters. Every ``replan_dt`` (and right after
the current target is popped or lost) the planner (planner.BeamPlanner) rolls
out target sequences in the given-only simulator (given_sim.GivenSim) with the
same guidance law executed here; the first target of the best sequence is
flown with ``guidance`` -> ``accel_for_thrust_dir`` -> ``CompensatedInnerLoop``
(the path the simulator was checked against). Launch: at ``launch_t`` (a fixed
agent constant, not tuned per seed), attitude chosen by planning from the pad.
"""

from __future__ import annotations

import time

import numpy as np

from ....envs.balloon_world import check_action
from ...base_agent import BaseAgent
from .. import rocketnav as _rocketnav  # noqa: F401 (register)
from .. import registry
from ..follow.inner_loop import CompensatedInnerLoop, quat_to_matrix
from ..interfaces import LaunchDecision
from ..vehicle.given_limits import GivenLimits
from ..vehicle.rocket_model import RocketModel
from .budget import PlanBudget
from .given_sim import GivenSim, quat_from_nose
from .planner import BeamPlanner, guidance, next_after
from .predict import WindProfilePredictor
from .thrust_dir import accel_for_thrust_dir

DECISION_DT = 0.1
# guidance variants searched per target: (nav gain, t_go floor s, along-LOS boost m/s^2)
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
# W3 P3b viable-intercept variants (6th column = acceleration margin), appended when vi_variants=True
VI_VARIANTS = ((3.0, 0.3, 0.0, 0.0, 0.0, 0.9), (3.0, 0.3, 0.0, 0.0, 0.0, 0.75))
# stream M late re-point variants: viable margin 0.9 + re-point window rp s (7th column), one row per rp_variants
RP_ROW = (3.0, 0.3, 0.0, 0.0, 0.0, 0.9)


def _idle(launch=False, attitude=(90.0, 0.0)):
    return {
        "launch": bool(launch),
        "launch_inclination_heading": np.array(attitude, float),
        "tvc": np.zeros(2),
        "roll": np.zeros(1),
        "throttle": np.ones(1),
    }


class PlannerAgent(BaseAgent):
    def __init__(
        self,
        given_parameters,
        *,
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
        boost=0.0,
        boost_t0=1e9,
        launch_els=(90.0, 75.0, 60.0),
        launch_cands=6,
        rocketnav="passthrough",
        wall_budget_s=450.0,
        variants=VARIANTS,
        loop_kwargs=None,
        critic_path=None,
        critic_beta=0.0,
        coast=False,
        ground_vz=False,
        launch_rule=None,
        launch_probe=None,
        deep_var=1,
        deep_radius=3.0,
        depth_gamma=1.0,
        predictor=False,
        pred_horizon=20.0,
        pred_sim_wind=False,
        pred_sim_wind_zlo=None,
        vi_variants=False,
        vi_wr=1.5,
        reach_kwargs=None,
        rp_variants=(),
        oracle_order=None,
        oracle_skip_s=3.0,
        oracle_k=3,
        floor_kwargs=None,
        zv_hint=False,
        zv_pair=0,
        sim_axial=False,
        budget=None,
        **_kwargs,
    ):
        super().__init__(given_parameters)
        gp = given_parameters
        self.model = RocketModel(gp)
        self.tables = GivenLimits(gp, self.model, turn_model="slew")
        lk = dict(
            loop_kwargs or {}
        )  # e.g. use_lead (given actuator lags, sc3), kp_rate
        # W3 P3a: act_inv / ff / ff_* go to the loop AND the planner's simulator (same law, follow/ff_ref.py)
        p3a = {
            k: v for k, v in lk.items() if k in ("act_inv", "ff") or k.startswith("ff_")
        }
        self.sim = GivenSim(
            gp,
            k_att=lk.get("k_att", 2.0),
            w_max=lk.get("w_max", 2.0),
            kp_rate=lk.get("kp_rate", 25.0),
            axial=sim_axial,  # round 11: env axial thrust factor in the planner's rollouts
            **p3a,
        )
        self.loop = CompensatedInnerLoop(
            gp, self.model, tables=self.tables, ff_model=self.sim, **lk
        )
        self.nav = registry.build("rocketnav", rocketnav, gp, self.model)
        # a nav that models the actuators (mekf) gets the command the env applied: the previous action, from the step
        # after the launch step on
        self._nav_act = bool(getattr(self.nav, "needs_actuation", False))
        self._sent = None
        self.radius = float(gp["balloon"]["radius"])
        critic = None
        if critic_path:  # weights shipped under agents/gnc/rl/weights (trained on training seeds only)
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
            lam=lam,
            radius_margin=radius_margin,
            stick=stick,
            boost=boost,
            boost_t0=boost_t0,
            variants=tuple(tuple(v) for v in variants)
            + (VI_VARIANTS if vi_variants else ())
            + tuple(RP_ROW + (float(r),) for r in rp_variants),
            critic=critic,
            critic_beta=critic_beta,
            coast=coast,
            ground_vz=ground_vz,
            deep_var=deep_var,
            deep_radius=deep_radius,
            depth_gamma=depth_gamma,
            **dict(reach_kwargs or {}),
        )
        self.planner.vi_wr = self.vi_wr = float(vi_wr)
        # round 10 zv_hint (False = unchanged): zv rows shape the exit toward the second balloon of the last plan
        self.planner.zv_hint = bool(zv_hint)
        # round 10 zv_pair (0 = unchanged): zv rows expanded per likely next balloon t2 (BeamPlanner.zv_pair)
        self.planner.zv_pair = int(zv_pair)
        # floor_kwargs (stream OP, research/sc4/streams/opening.md; None = v8 unchanged): {el_lo, vz_min, h_min} relaxes
        # guidance's 60 deg near-ground floor to el_lo deg while climbing, in the rollouts and in flight alike
        fk = dict(floor_kwargs or {})
        self.planner.floor = (
            (
                float(fk.get("el_lo", 45.0)),
                float(fk.get("vz_min", 1.0)),
                float(fk.get("h_min", 2.0)),
            )
            if floor_kwargs
            else None
        )
        # oracle_order: balloon id sequence from the offline reach bound (research/sc4/oracle_run.py). Offline
        # diagnostic only, it uses knowledge the agent may not have (#165); None = off (v8 unchanged)
        if oracle_order is not None:
            self.planner.oracle = np.asarray(oracle_order, int)
        self.oracle_skip_s = float(oracle_skip_s)
        self.planner.oracle_k = int(oracle_k)
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
        self.wall_budget = float(wall_budget_s)
        self.wall_used = 0.0
        self.log = []
        # launch_rule (t_min, t_max, key_thr, check_dt): from t_min, plan from the pad every check_dt and
        # launch as soon as the best plan's key >= key_thr (observations only); launch at t_max otherwise.
        # launch_probe (t0, t1, dt): log the pad plan's key over time without changing the launch (analysis).
        self.launch_rule = tuple(float(x) for x in launch_rule) if launch_rule else None
        self.launch_probe = (
            tuple(float(x) for x in launch_probe) if launch_probe else None
        )
        self._next_probe = -1.0
        # predictor: altitude wind-profile balloon tracks (predict.py) for the planner instead of constant
        # velocity; pred_sim_wind: the same profile as GivenSim's wind instead of its 25 m bins. Off = v7.
        # pred_sim_wind_zlo (m ASL; None = the whole grid as before): hand GivenSim only the profile from this altitude
        # to the highest balloon observed, held constant outside (stream wind-estimation, 2026-10-30).
        self.pred = WindProfilePredictor(z_min=self.elev) if predictor else None
        self.pred_horizon = float(pred_horizon)
        self.pred_sim_wind = bool(pred_sim_wind)
        self.pred_sim_wind_zlo = (
            None if pred_sim_wind_zlo is None else float(pred_sim_wind_zlo)
        )
        self._next_obs = -1.0
        # budget (research/sc40/budget/NOTES.md; None = off, the agent unchanged): before each plan pick the most
        # expensive level of a ladder (horizon, long_horizon, depth, beam, branch, replan_dt) whose projected agent time
        # for the rest of the flight fits budget.target_s and whose projected elapsed time fits budget.elapsed_s
        self.budget = None
        if budget is not None:
            pl = self.planner
            self.budget = PlanBudget(
                budget,
                dict(
                    horizon=pl.horizon,
                    long_horizon=pl.long_horizon,
                    depth=pl.depth,
                    beam=pl.beam,
                    branch=pl.branch,
                    replan_dt=self.replan_dt,
                ),
                burn_time=self.sim.burn_time,
                max_time=float(gp["simulation"]["max_time"]),
                dec_dt=DECISION_DT,
            )

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
        if self.pred is None or t + 1e-9 < self._next_obs:
            return
        self._next_obs = t + DECISION_DT
        bp, bv, st = self._balloons(obs)
        self.pred.observe(t, bp, bv, st == 1)

    def _world(self, bp, bv, rel, t):
        """Wind for GivenSim and balloon tracks for the planner, right before a plan() call."""
        if self.pred is None:
            self.sim.update_wind(bp, bv, rel)
            return
        self.pred.refit(t, bp, bv, rel)
        tab = None
        if self.pred_sim_wind and self.pred.U is not None:
            tab = (
                (self.pred.zg, self.pred.U[:, 0], self.pred.U[:, 1])
                if self.pred_sim_wind_zlo is None
                else self.pred.wind_table(self.pred_sim_wind_zlo)
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

    def _pad_key(self, obs, t):
        """Best plan key from the pad for a vertical launch now (the attitude launch() nearly always picks)."""
        bp, bv, st = self._balloons(obs)
        rel = st == 1
        self._world(bp, bv, rel, t)
        S0 = self._state(
            self.pad,
            np.zeros(3),
            quat_from_nose(np.array([[0.0, 0.0, 1.0]]))[0],
            np.zeros(2),
            np.zeros(2),
            1.0,
            t,
            t,
            len(bp),
        )
        f, info = self.planner.plan(S0, bp, bv, t, rel)
        return (
            (-np.inf, 0, 0.0)
            if info is None
            else (float(info[0]), int(info[1]), float(info[2]))
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

    def _plan_oracle(self, S0, bp, bv, t, rel, st, cur, t_rel):
        """planner.plan; with an oracle order, drop popped ids first. The planner may fly to the first oracle_k
        released listed balloons (planner._oracle_mask). The head (first released listed balloon) is skipped
        when no plan has popped anything for oracle_skip_s in a row. Plans with no pop are normal from the pad
        and while balloons are beyond the horizon, so skipping is slow."""
        o = self.planner.oracle
        if o is None:
            return self.planner.plan(S0, bp, bv, t, rel, cur=cur)
        self.planner.oracle = o = o[st[o] != 2]
        f, info = self.planner.plan(S0, bp, bv, t, rel, cur=cur)
        hr = o[rel[o]]
        head = int(hr[0]) if len(hr) else -1
        self.log.append(
            (
                "oplan",
                round(t_rel, 2),
                head,
                None if info is None else int(info[1]),
                None if f is None else int(f[0]),
            )
        )
        if head < 0 or (info is not None and info[1] > 0):
            self._o_fail = None
            return f, info
        if getattr(self, "_o_fail", None) is None or self._o_fail[0] != head:
            self._o_fail = (head, t)
        elif t - self._o_fail[1] >= self.oracle_skip_s:
            self.log.append(("oskip", round(t_rel, 2), head))
            self.planner.oracle = o[o != head]
            self._o_fail = None
            f, info = self.planner.plan(S0, bp, bv, t, rel, cur=cur)
        return f, info

    def _plan_step(self, est, bp, bv, st, rel, lost, t, t_rel):
        """One plan from the navigation estimate; sets target / variant and the next plan time."""
        self._world(bp, bv, rel, t)
        pos = np.asarray(est.position, float)
        vel = np.asarray(est.velocity, float)
        q = np.asarray(est.quaternion, float)
        om = np.asarray(est.omega, float)
        g = np.asarray(getattr(self.loop, "g_act", np.zeros(2)), float)
        thr = float(getattr(self.loop, "thr_act", 1.0))
        S0 = self._state(pos, vel, q, om[:2], g, thr, t, self.t_launch, len(bp))
        if getattr(self.loop, "ff", False):
            S0["ff"] = self.loop.ff_state[None].copy()
        cur = self.target if (self.target is not None and not lost) else None
        f, info = self._plan_oracle(S0, bp, bv, t, rel, st, cur, t_rel)
        if f is not None:
            if f[0] != self.target:
                self.log.append(("tgt", round(t_rel, 2), int(f[0]), int(f[1]), info))
            self.target, self.var = int(f[0]), int(f[1])
        elif lost:
            self.target = None
        self.next_plan = t + self.replan_dt

    def _budget_step(self, est, bp, bv, st, rel, lost, t, t_rel):
        """Budget mode (gnc/rl/budget.py): choose the ladder level, plan when it is due, else fly on (greedy target
        choice when the target is gone and no plan() is affordable)."""
        B = self.budget
        lost_evt = (
            self.target is not None and st[self.target] != 1
        )  # popped or lost (not 'none found')
        pos = np.asarray(est.position, float)
        vel = np.asarray(est.velocity, float)
        i, dt = B.decide(t_rel, t, self.wall_used, pos[2] - self.elev, vel[2])
        if (
            i == len(B.levels) and self.target is None
        ):  # event mode, nothing chosen: retry at the slowest level rate
            dt = B.levels[-1]["replan_dt"]
        if i <= len(B.levels) and self.wall_used < self.wall_budget:
            if lost_evt or t + 1e-9 >= B.last_plan_t + dt:
                B.apply(i, self.planner, self)
                c0 = time.perf_counter()
                self._plan_step(est, bp, bv, st, rel, lost, t, t_rel)
                B.record(i, time.perf_counter() - c0, t_rel, t, lost_evt)
        elif lost:
            self._greedy(pos, vel, bp, bv, rel, t_rel)

    def _greedy(self, pos, vel, bp, bv, rel, t_rel):
        """No plan() affordable: the released balloon with the smallest rough reach time (BeamPlanner._cands' estimate),
        flown with the deep guidance variant; none released -> no target (fly up)."""
        self.budget.n_greedy += 1
        if not rel.any():
            self.target = None
            return
        r = bp - pos[None]
        d = np.linalg.norm(r, axis=1)
        vc = -((r * (bv - vel[None])).sum(1)) / np.maximum(d, 1e-6)
        est = np.where(rel, d / np.maximum(vc, 15.0), np.inf)
        j = int(np.argmin(est))
        var = min(self.planner.deep_var, len(self.planner.V) - 1)
        if j != self.target:
            self.log.append(("gtgt", round(t_rel, 2), j, var))
        self.target, self.var = j, var

    def budget_stats(self):
        """Budget summary for offline recorders (None when the budget is off)."""
        return None if self.budget is None else self.budget.stats()

    # -- main -----------------------------------------------------------------------
    def get_action(self, observation):
        w0 = time.time()
        if self.budget is not None:
            self.budget.tick()  # elapsed clock starts at the first call (= the end of env.reset())
        try:
            action = self._get_action(observation)
            if self._nav_act:  # the env does not apply the launch step's tvc/throttle
                t = float(observation["simulation_time"])
                launched = self.t_launch is not None and t > self.t_launch + 1e-9
                self._sent = action if launched else None
            return action
        finally:
            self.wall_used += time.time() - w0

    def _get_action(self, observation):
        t = float(observation["simulation_time"])
        self._observe(observation, t)
        if self.t_launch is None:
            go = t + 1e-9 >= self.launch_t
            if self.launch_rule is not None:
                t_min, t_max, thr, dtc = self.launch_rule
                go = t + 1e-9 >= t_max
                if not go and t + 1e-9 >= t_min and t + 1e-9 >= self._next_probe:
                    self._next_probe = t + dtc
                    k = self._pad_key(observation, t)
                    self.log.append(
                        ("probe", round(t, 2), round(k[0], 3), k[1], round(k[2], 2))
                    )
                    go = k[0] >= thr
            elif self.launch_probe is not None and not go:
                t0p, t1p, dtp = self.launch_probe
                if t0p <= t + 1e-9 <= t1p and t + 1e-9 >= self._next_probe:
                    self._next_probe = t + dtp
                    k = self._pad_key(observation, t)
                    self.log.append(
                        ("probe", round(t, 2), round(k[0], 3), k[1], round(k[2], 2))
                    )
            if not go:
                return _idle()
            el, head, n = self._launch(observation, t)
            self.t_launch = t
            self.next_dec = t + DECISION_DT
            self.next_plan = t
            dec = LaunchDecision(
                t_launch=t, inclination_deg=el, heading_deg=head, max_rail_tilt_deg=90.0
            )
            self.nav.on_launch(dec, t)
            self.loop.on_launch(t)
            self.n_cmd = n
            self.log.append(("launch", round(t, 2), round(el, 1), round(head, 1)))
            return _idle(launch=True, attitude=(el, head))

        if self._nav_act and self._sent is not None:
            self.nav.set_actuation(self._sent["tvc"], self._sent["throttle"])
        est = self.nav.update(observation, deadline=float("inf"))
        self.last_est = (
            est  # read by offline recorders (research/twnia3/diag_run.py --rec)
        )
        t_rel = t - self.t_launch
        if t + 1e-9 >= self.next_dec:
            self.next_dec += DECISION_DT
            bp, bv, st = self._balloons(observation)
            rel = st == 1
            lost = self.target is None or st[self.target] != 1
            if self.budget is not None:
                self._budget_step(est, bp, bv, st, rel, lost, t, t_rel)
            elif (
                lost or t + 1e-9 >= self.next_plan
            ) and self.wall_used < self.wall_budget:
                self._plan_step(est, bp, bv, st, rel, lost, t, t_rel)
            if self.target is not None and st[self.target] == 1:
                pos = np.asarray(est.position, float)[None]
                vel = np.asarray(est.velocity, float)[None]
                nav, tmin, bst, ou, os_, vi, rp, zb, zk, la, lt = self.planner.V[
                    self.var
                ][:11]
                if (
                    vi > 0 or zk > 0 or la > 0
                ):  # W3 P3b viable-intercept / round 10 zv variant (as the planner's rollouts)
                    nose = quat_to_matrix(np.asarray(est.quaternion, float))[:, 2][None]
                    a_av = self.sim.accel_avail(np.array([t_rel]), self.loop.thr_act)
                    tp, tv = bp[self.target][None], bv[self.target][None]
                    pn = vn = None
                    if (
                        rp > 0 or zk > 0 or la > 0
                    ):  # next balloon as in the rollouts (BeamPlanner._next_pv)
                        nx = next_after(
                            pos,
                            vel,
                            tp,
                            tv,
                            bp[None],
                            bv,
                            rel[None],
                            np.array([self.target]),
                        )[0]
                        hn = (
                            self.planner.hint[self.target]
                            if (
                                zk > 0
                                and self.planner.zv_hint
                                and self.planner.hint is not None
                            )
                            else -1
                        )
                        if hn >= 0 and rel[hn] and st[hn] == 1:
                            nx = hn
                        h2 = (
                            self.planner.last_t2
                            if ((zk > 0 or la > 0) and self.planner.zv_pair)
                            else -1
                        )
                        if (
                            h2 >= 0 and rel[h2] and st[h2] == 1
                        ):  # pair rows: the t2 the plan scored
                            nx = h2
                        pn = bp[nx][None] if nx >= 0 else np.full((1, 3), np.nan)
                        vn = bv[nx][None] if nx >= 0 else np.full((1, 3), np.nan)
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
                        rp=rp,
                        pn=pn,
                        vn=vn,
                        floor=self.planner.floor,
                        zb=zb,
                        zk=zk,
                        la=la,
                        lt=lt,
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
                        floor=self.planner.floor,
                    )[0]
            else:
                self.n_cmd = np.array([0.0, 0.0, 1.0])
        acc = accel_for_thrust_dir(self.n_cmd, 1.0, est, self.model, self.tables, t_rel)
        action, _, _ = self.loop.solve(acc, est)
        if check_action(action):
            return self._last_good if self._last_good is not None else _idle()
        self._last_good = action
        return action
