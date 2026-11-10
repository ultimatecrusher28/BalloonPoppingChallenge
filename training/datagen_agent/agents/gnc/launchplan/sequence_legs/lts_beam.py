"""sequence-legs-lts: the same beam search over LINEAR-TANGENT legs.

A sibling of ``beam.py``, not a patch to it: the two differ in exactly one
thing, the leg primitive, so a paired bench between them attributes the
difference to the primitive and nothing else.

What changes, and why the measurements say it should matter
(research/limits/REPORT.md section 6-7):

- ``beam.py`` serialises the attitude turn: the vehicle drifts at constant
  velocity while it rotates (measured 0.83 s per leg, 22% of tour time), then
  flies a constant-acceleration intercept. A linear-tangent leg rotates thrust
  THROUGH the leg (``u(s) = normalize(A + B s)``), so that share is not paid at
  all.
- A constant-acceleration leg puts its whole lateral demand in one direction:
  the plan sits at 0.56x the measured lateral authority on the median and
  spikes to 2.1x. Spreading the same turn over the arc is what keeps the
  effective turn radius ``R = v^2/a_lat`` small, and the reachability ceiling
  (``research/limits/cluster_reach.py``) says R is what binds the tour.
- With the arrival velocity left free the costate ``B`` is zero and the leg
  degenerates to a constant direction - still no serialised turn, but no
  control over where the pass leaves the vehicle pointing. ``arrival_aim``
  pins the arrival velocity toward the local cluster centroid instead, which
  is the principled version of ``beam.py``'s two-way ``stretch`` branch (that
  crude version already scored +0.30 paired).

Default is ``arrival_aim=None``: free arrival only, so the FIRST measurement
against beam changes the primitive and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import pops as pops_mod
from ...interfaces import (
    InterceptWindow,
    LaunchDecision,
    LaunchPlanner,
    Leg,
    Plan,
    PopExpectation,
)
from ...refs import SampledReference
from ...registry import register
from ...vehicle import lts
from ...vehicle.limits import LimitTables
from ...vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
REF_DT = 0.05
WINDOW_HALF_S = 0.6
TUBE_RADIUS_M = 30.0


@dataclass
class _LtsLeg:
    """One solved linear-tangent leg, in the form ``_sample`` needs."""

    bid: int
    t_start: float
    tau: float
    A: np.ndarray
    B: np.ndarray
    k: float
    u_start: np.ndarray
    u_end: np.ndarray
    # Liftoff is solved at full thrust and tour legs at ``eta``; re-integrating
    # a leg at the other one puts the reference tens of metres off the target
    # it was solved for, so the leg carries the eta it was solved under.
    eta: float = 1.0


@dataclass
class _Node:
    t_rel: float
    pos: np.ndarray
    vel: np.ndarray
    axis: np.ndarray
    targets: tuple = ()
    legs: list = field(default_factory=list)


@register("launchplan", "sequence-legs-lts")
class LtsSequenceLegs(LaunchPlanner):
    def __init__(
        self,
        given_parameters,
        model,
        *,
        launch_time=48.5,
        max_rail_tilt_deg=30.0,
        depth=14,
        eta=0.95,
        beam=12,
        branch=8,
        sweep_top=4,
        arrival_aim=None,
        aim_near=6,
        tau_max=8.0,
        tau_max_first=20.0,
        lat_factor=1.3,
        branch_max_m=0.0,
        **_kwargs,
    ):
        self.model = model
        self.tables = LimitTables()
        self.elevation = float(given_parameters["environment"]["elevation"])
        self.burn_time = float(model.burn_time)
        self.launch_time = float(launch_time)
        self.max_rail_tilt_deg = float(max_rail_tilt_deg)
        self.depth = int(depth)
        self.eta = float(eta)
        self.beam = int(beam)
        self.branch = int(branch)
        self.sweep_top = int(sweep_top)
        self.tau_max = float(tau_max)
        # The climb-out from rest needs far more than a tour leg: measured
        # first_pop_tb is 7.6-10.0 s and the solved free leg runs to 12 s for
        # the higher targets, so one cap for both silently deletes every root.
        self.tau_max_first = float(tau_max_first)
        # Reject a leg whose velocity-turning demand exceeds this multiple of
        # the MEASURED lateral table. beam.py declares the same test but never
        # calls it (``_feasible`` is dead code there), which is why its plans
        # peak at 2.1x the table; set to 0 to reproduce that.
        self.lat_factor = float(lat_factor)
        # Cloud-frame cap on the candidate set, same definition as beam.py's
        # (REPORT section 7.13): both endpoints predicted at the SAME instant so
        # the drift the cloud shares cancels. 0.0 = off = the baseline argsort.
        # beam.py measured the (chord, leg-time) locus t_leg ~ c_rel^-0.20 under
        # the MIN-TIME primitive: forcing a short chord there made the leg
        # longer and the speed collapse (24.8 m chord at 5.33 m/s). The open
        # question this knob answers here is whether a linear-tangent leg with
        # a PINNED arrival velocity sits on a different locus - the rivals are
        # 2.3-2.4x above ours at the same chord.
        self.branch_max_m = float(branch_max_m)
        # None -> free arrival only. A list of speeds (m/s, relative to the
        # balloon) -> also try arriving at each speed aimed at the local
        # cluster centroid.
        self.arrival_aim = (
            None if arrival_aim is None else [float(x) for x in arrival_aim]
        )
        self.aim_near = int(aim_near)
        self.last_sweep_score = 0

    # -- balloon linearisation -----------------------------------------
    def _linearize(self, predictor, bid, t_abs):
        """``(bp, bv)`` for ``bp + bv*s`` over a leg; None when unpredictable.

        LTS models the target as moving at a constant velocity for the few
        seconds of a leg, which is what a rising balloon does.
        """
        out = predictor.predict(bid, t_abs)
        if not isinstance(out, tuple) or out[0] is None:
            return None
        return np.asarray(out[0], float), np.asarray(out[1], float)

    def _centroid(self, predictor, node, exclude, t_abs):
        """Mean position of the nearest unvisited balloons: where the pass
        should leave the vehicle pointing if it wants to stay in the cluster."""
        cand = [b for b in predictor.released_ids() if b not in exclude]
        if not cand:
            return None
        pos = predictor.states[cand, :3]
        near = np.argsort(np.linalg.norm(pos - node.pos, axis=1))[: self.aim_near]
        return pos[near].mean(axis=0)

    # -- leg solves ----------------------------------------------------
    def _solve(self, t_rel, p0, v0, bp, bv, v_rel, *, from_pad=False):
        """One batched-of-one LTS solve; ``v_rel=None`` leaves arrival free."""
        t0 = np.array([t_rel])
        args = (self.model, t0, p0[None], v0[None], bp[None], bv[None])
        # Liftoff is flown at full thrust (T/W 1.21 leaves no margin to give
        # away), which is what beam.py's first leg does too.
        eta = 1.0 if from_pad else self.eta
        tau_max = self.tau_max_first if from_pad else self.tau_max
        if v_rel is None:
            sol = lts.free_leg(*args, eta=eta, tau_max=tau_max)
        else:
            sol = lts.solve_leg(
                *args, np.asarray(v_rel, float)[None], eta=eta, tau_max=tau_max
            )
        if not bool(sol["ok"][0]):
            return None
        return (
            float(sol["tau"][0]),
            np.asarray(sol["A"][0], float),
            np.asarray(sol["B"][0], float),
            float(sol.get("k", [1.0])[0]),
        )

    def _leg(self, predictor, bid, t_now, node, v_rel_fn=None, relin=2, from_pad=False):
        """Solve a leg, re-linearising the balloon at the arrival time.

        The first solve uses the balloon's state NOW, which is wrong by
        ``bv * tau``; re-solving against the state at the arrival it just
        predicted converges in one or two passes.
        """
        lin = self._linearize(predictor, bid, t_now + node.t_rel)
        if lin is None:
            return None
        bp, bv = lin
        sol = None
        for _ in range(relin):
            v_rel = None if v_rel_fn is None else v_rel_fn(bp, bv)
            sol = self._solve(
                node.t_rel, node.pos, node.vel, bp, bv, v_rel, from_pad=from_pad
            )
            if sol is None:
                return None
            tau = sol[0]
            if node.t_rel + tau > self.burn_time:
                return None
            lin = self._linearize(predictor, bid, t_now + node.t_rel + tau)
            if lin is None:
                return None
            bp, bv = lin[0] - lin[1] * tau, lin[1]  # back to the leg's origin
        tau, A, B, k = sol
        # Ground clearance and the reachable-set check, on the solved profile.
        s, p, v, u = lts.profile(
            self.model,
            np.array([node.t_rel]),
            node.pos[None],
            node.vel[None],
            A[None],
            B[None],
            np.array([tau]),
            k=np.array([k]),
            eta=1.0 if from_pad else self.eta,
        )
        if self.lat_factor > 0.0 and not self._lat_ok(
            v[0], u[0], s[0], leg_t0=node.t_rel, k=k, eta=1.0 if from_pad else self.eta
        ):
            return None
        if from_pad:
            # The pad IS the ground, so a clearance test on the whole profile
            # rejects every liftoff. beam.py's equivalent test is that the
            # first leg must not dive.
            if u[0, 0, 2] <= 0.0 or v[0, -1, 2] < -0.05:
                return None
        elif p[0, :, 2].min() < self.elevation + 3.0:
            return None
        return (
            _LtsLeg(
                bid=int(bid),
                t_start=float(node.t_rel),
                tau=float(tau),
                A=A,
                B=B,
                k=float(k),
                eta=1.0 if from_pad else self.eta,
                u_start=u[0, 0].copy(),
                u_end=u[0, -1].copy(),
            ),
            p[0, -1].copy(),
            v[0, -1].copy(),
        )

    def _lat_ok(self, v, u, s, *, leg_t0, k, eta):
        """Does the leg stay inside the measured lateral authority?

        The check is on the component of net acceleration perpendicular to
        the velocity - the part that turns the flight path - because that is
        what the a_lat table measures. Below 8 m/s the notion of a flight
        path is meaningless and the test is skipped.
        """
        a = k * lts.alpha(self.model, leg_t0 + s, eta)[:, None] * u - GRAVITY * _Z
        use = self.tables.lateral_use(v, a)
        return bool(use.size == 0 or (use <= self.lat_factor).all())

    # -- search --------------------------------------------------------
    def _aim_fns(self, predictor, node, exclude, t_abs):
        """The arrival-velocity branches to try for one candidate target."""
        yield None  # free arrival
        if not self.arrival_aim:
            return
        c = self._centroid(predictor, node, exclude, t_abs)
        if c is None:
            return
        for speed in self.arrival_aim:

            def fn(bp, bv, c=c, speed=speed):
                d = c - bp
                n = np.linalg.norm(d)
                return speed * (d / n) if n > 1e-6 else np.zeros(3)

            yield fn

    def _near(self, predictor, cand, pos, t_abs, k):
        """Indices of the ``k`` nearest candidates, optionally capped.

        Off, this is literally the baseline expression. On, candidates are
        predicted at the same instant as ``pos`` so the comparison is in the
        cloud frame; a node with nothing inside the cap simply does not
        expand, since falling back to the nearest would smuggle the long legs
        straight back in.
        """
        if self.branch_max_m <= 0.0:
            p_c = predictor.states[cand, :3]
            return np.argsort(np.linalg.norm(p_c - pos, axis=1))[:k]
        p_t = predictor.paths(np.asarray(cand, int), np.array([float(t_abs)]))[:, 0, :]
        d_rel = np.linalg.norm(p_t - pos, axis=1)
        idx = np.argsort(d_rel)[:k]
        return idx[d_rel[idx] <= self.branch_max_m]

    def _expand(self, predictor, t_now, ids, node, out):
        cand = [b for b in ids if b not in node.targets]
        if not cand:
            return
        near = self._near(predictor, cand, node.pos, t_now + node.t_rel, self.branch)
        for j in near:
            bid = int(cand[int(j)])
            for fn in self._aim_fns(
                predictor, node, set(node.targets) | {bid}, t_now + node.t_rel
            ):
                got = self._leg(predictor, bid, t_now, node, fn)
                if got is None:
                    continue
                leg, p_end, v_end = got
                out.append(
                    _Node(
                        t_rel=node.t_rel + leg.tau,
                        pos=p_end,
                        vel=v_end,
                        axis=leg.u_end,
                        targets=node.targets + (bid,),
                        legs=node.legs + [leg],
                    )
                )

    def _search(self, predictor, t_now, ids, start=None):
        roots = []
        if start is None:
            pad = np.array([0.0, 0.0, self.elevation])
            root = _Node(t_rel=0.0, pos=pad, vel=np.zeros(3), axis=_Z)
            order = ids[
                np.argsort(np.linalg.norm(predictor.states[ids, :3] - pad, axis=1))
            ][: self.branch * 2]
            for bid in order:
                # Leg 0 arrives free: the rail cannot pin an arrival velocity,
                # and the tilt cap is checked on the solved direction.
                got = self._leg(predictor, int(bid), t_now, root, None, from_pad=True)
                if got is None:
                    continue
                leg, p_end, v_end = got
                tilt = np.degrees(np.arccos(np.clip(leg.u_start[2], -1.0, 1.0)))
                if tilt > self.max_rail_tilt_deg:
                    continue
                roots.append(
                    _Node(
                        t_rel=leg.tau,
                        pos=p_end,
                        vel=v_end,
                        axis=leg.u_end,
                        targets=(int(bid),),
                        legs=[leg],
                    )
                )
        else:
            self._expand(predictor, t_now, ids, start, roots)
        frontier = sorted(roots, key=lambda n: n.t_rel)[: self.beam]
        best = list(frontier)
        while frontier:
            nxt = []
            for node in frontier:
                self._expand(predictor, t_now, ids, node, nxt)
            nxt.sort(key=lambda n: (-len(n.targets), n.t_rel))
            frontier = nxt[: self.beam]
            best.extend(frontier)
            if frontier and len(frontier[0].targets) >= self.depth:
                break
        best.sort(key=lambda n: (-len(n.targets), n.t_rel))
        return best[: self.sweep_top]

    # -- ref building and sweep scoring --------------------------------
    def _sample(self, node, start=None):
        """Integrate the solved legs onto the reference grid."""
        if start is None:
            pos0, vel0, t0 = np.array([0.0, 0.0, self.elevation]), np.zeros(3), 0.0
        else:
            pos0, vel0, t0 = start.pos.copy(), start.vel.copy(), start.t_rel
        ts, ps, vs, accs, axes = (
            [t0],
            [pos0.copy()],
            [vel0.copy()],
            [np.zeros(3)],
            [node.legs[0].u_start],
        )
        pos, vel = pos0.copy(), vel0.copy()
        for leg in node.legs:
            n_s = max(2, int(np.ceil(leg.tau / REF_DT)))
            s, p, v, u = lts.profile(
                self.model,
                np.array([leg.t_start]),
                pos[None],
                vel[None],
                leg.A[None],
                leg.B[None],
                np.array([leg.tau]),
                k=np.array([leg.k]),
                eta=leg.eta,
                n=n_s,
            )
            a = (
                leg.k
                * lts.alpha(self.model, leg.t_start + s[0], leg.eta)[:, None]
                * u[0]
                - GRAVITY * _Z
            )
            for i in range(1, n_s + 1):
                ts.append(float(leg.t_start + s[0, i]))
                ps.append(p[0, i].copy())
                vs.append(v[0, i].copy())
                accs.append(a[i].copy())
                axes.append(u[0, i].copy())
            pos, vel = p[0, -1].copy(), v[0, -1].copy()
        return ts, ps, vs, accs, axes

    def _sweep_score(self, ts, ps, predictor, t_now):
        ids = predictor.released_ids()
        if len(ids) == 0:
            return 0
        bp = predictor.paths(ids, t_now + np.asarray(ts))
        counts, _popped, _dmin, _first = pops_mod.score_paths(np.asarray(ps), bp, 1.5)
        return int(counts)

    def _build(self, tours, t_clock, launch, start=None):
        """Pick the best-scoring tour and turn it into a Plan."""
        best, best_score, best_samples = None, -1, None
        for node in tours:
            samples = self._sample(node, start=start)
            score = self._sweep_score(samples[0], samples[1], self._pred, t_clock)
            if score > best_score:
                best, best_score, best_samples = node, score, samples
        if best is None:
            return None
        ts, ps, vs, accs, axes = best_samples
        ref = SampledReference(ts, ps, vs, accs, axes)
        legs, pop_exp = [], []
        for i, leg in enumerate(best.legs):
            t_pop = t_clock + leg.t_start + leg.tau
            # A linear-tangent leg turns WHILE it accelerates, so the attitude
            # turn is not a separate cost: the slew time a cost model would
            # charge is zero by construction.
            legs.append(
                Leg(
                    leg_id=i,
                    window=InterceptWindow(
                        balloon_id=leg.bid,
                        t_open=t_pop - WINDOW_HALF_S,
                        t_close=t_pop + WINDOW_HALF_S,
                        balloon_state_at=_state_closure(self._pred, leg.bid),
                    ),
                    t_start=leg.t_start,
                    t_end=leg.t_start + leg.tau,
                    u_start=leg.u_start,
                    u_end=leg.u_end,
                    slew_time_required=0.0,
                    tube=_const_tube,
                )
            )
            pop_exp.append(PopExpectation(balloon_id=leg.bid, t=t_pop, margin_m=0.75))
        self.last_sweep_score = best_score
        return Plan(launch=launch, ref=ref, pops=pop_exp, legs=legs)

    # -- planning ------------------------------------------------------
    def plan(self, observation, predictor, *, deadline: float) -> Plan | None:
        t_now = float(observation["simulation_time"])
        if t_now < self.launch_time:
            return None
        ids = predictor.released_ids()
        if len(ids) == 0:
            return None
        self._pred = predictor
        tours = self._search(predictor, t_now, ids)
        if not tours:
            return None
        u0 = tours[0].legs[0].u_start
        tilt = min(
            np.degrees(np.arccos(np.clip(u0[2], -1.0, 1.0))), self.max_rail_tilt_deg
        )
        heading = (
            float(np.degrees(np.arctan2(u0[0], u0[1])) % 360.0)
            if np.hypot(u0[0], u0[1]) > 1e-9
            else 0.0
        )
        launch = LaunchDecision(
            t_launch=t_now,
            inclination_deg=90.0 - tilt,
            heading_deg=heading,
            max_rail_tilt_deg=self.max_rail_tilt_deg,
            first_target=tours[0].legs[0].bid,
        )
        return self._build(tours, t_now, launch)

    def replan_from(self, position, velocity, axis, t_rel, launch, predictor):
        ids = predictor.released_ids()
        if len(ids) == 0:
            return None
        self._pred = predictor
        start = _Node(
            t_rel=float(t_rel),
            pos=np.asarray(position, float),
            vel=np.asarray(velocity, float),
            axis=np.asarray(axis, float),
        )
        tours = self._search(predictor, float(launch.t_launch), ids, start=start)
        if not tours:
            return None
        return self._build(tours, float(launch.t_launch), launch, start=start)


def _const_tube(_t: float) -> float:
    return TUBE_RADIUS_M


def _state_closure(predictor, bid):
    def state_at(t: float):
        out = predictor.predict(bid, t)
        return out if isinstance(out, tuple) else (None, None)

    return state_at
