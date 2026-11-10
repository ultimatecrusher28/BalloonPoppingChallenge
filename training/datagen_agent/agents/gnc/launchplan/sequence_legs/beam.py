"""sequence-legs-beam: searched tour over an HONEST timeline.

What the greedy skeleton got wrong, fixed here with measured numbers
(research/limits):

- Leg 0 is the full-thrust fixed-axis climb the follower actually flies
  (the reach-set protocol): thrust accel along the rail axis minus gravity,
  arrival found by sliding tau. No throttled first leg (T/W 1.21).
- Every later leg pays the measured attitude turn time first
  (turn_time(angle), 0.7/1.1/1.4 s at 30/90/150 deg) as drift at constant
  velocity, then a constant-acceleration intercept.
- Feasibility = the thrust ball, |a+g| <= eta*T/m. The planner does NOT cap
  the lateral part by default: the follower deliberately has no such cap
  (see LegResolveFeasible._feasible - a sustained-lateral cap under-states a
  vehicle whose TVC re-points in about a second, and slides every aim late),
  and enforcing one here measured -0.60 pops paired (REPORT section 7.7).
  ``enforce_lat=True`` turns it on for that comparison.
- Beam search (width `beam`, `branch` nearest candidates per node) ranked by
  depth then time; the final pick re-scores the top tours with the swept-pop
  test against ALL balloons (incidental pops count).
- Optional cumulative-heading penalty on that ranking
  (``heading_penalty_s_per_rad``, default 0.0 = off, ranking then bit-identical
  to the measured baseline). Pure min-time ranking systematically prefers
  far-but-downrange targets, because turning the VELOCITY around is slow;
  that is what makes our cloud-frame chord 85.8 m against the rival's 43.3 m
  at a LARGER turn radius (REPORT section 7.9). The penalty charges a tour
  ``w`` seconds per radian of accumulated velocity heading change, so a
  near-straight sequence can win on rank without being faster in wall time.
- Optional cumulative CHORD penalty on that same ranking
  (``chord_penalty_s_per_m``, default 0.0 = off). The heading penalty bought
  the straightest tour measured with speed intact and the chord did not move
  at all (85.8 -> 85.7 m, REPORT section 7.10): the minimum of cumulative
  heading change is "keep flying forward", which is perfectly compatible with
  long chords. Straightness is not shortness. Nothing in the objective got
  better when the chord got shorter, so this term puts the chord itself in,
  charging ``w`` seconds per metre of cumulative CLOUD-FRAME chord (the same
  ``c_rel`` definition metrics.py uses: both endpoints at one instant, so the
  drift the two targets share cancels). Unlike the string planner
  (REPORT section 7.11), which reached 53 m at full speed but paid a 5 s
  climb-out to a geometrically-chosen entry point, a ranking term pays no
  entry tax: the tour still starts where the tour starts. MEASURED AND
  REJECTED: w = 0.02/0.05/0.12 give -0.45/-0.55/-0.35 pops paired
  (REPORT section 7.13). The chord it buys is partly an artefact - ``c_rel``
  falls 20% while ``c_world`` falls 3.6%, because a longer leg lets the cloud
  drift further and the same-instant difference shrinks for free.
- Optional CANDIDATE-SET restriction by distance (``branch_max_m``, default
  0.0 = off). ``branch`` nearest candidates is a rank cut, not a distance cut:
  the field's nearest-neighbour spacing is 31-37 m and there are 31-32
  balloons within +-50 m of our flight altitude, yet a trace of the candidates
  ``_leg`` is offered found more than a third of them at chords >= 90 m
  (REPORT section 7.9). So "8 nearest" already reaches well past the local
  neighbourhood. This FORBIDS the long legs instead of pricing them, which is
  the other half of the chord question: a penalty lets a long leg buy its way
  back in on time, a cap does not. The cap MUST be measured at the same
  instant as ``node.pos`` (see ``_near``); a snapshot-frame cap lengthens the
  chord instead of shortening it. MEASURED AND REJECTED: caps of
  95/70/50/35 m give -0.75/-0.40/-0.45/-0.60 pops paired and drive the chord
  to 24.8 m - shorter than the best rival's 27.9 m - at no entry cost, yet
  ``t_leg`` rises 3.78 -> 5.32 s throughout. Fitted over the sweep,
  ``t_leg`` ~ ``c_rel`` ** -0.20: on this primitive a shorter chord is a
  SLOWER leg, and min t_leg sits at the long-chord end where the baseline
  already is (REPORT section 7.13). Target selection is not the gap.
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
from ...vehicle.limits import LimitTables
from ...vehicle.limits import turn_time as _default_turn_time
from ...vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
REF_DT = 0.05
WINDOW_HALF_S = 0.6
TUBE_RADIUS_M = 30.0


@dataclass
class _Node:
    t_rel: float
    pos: np.ndarray
    vel: np.ndarray
    axis: np.ndarray
    targets: tuple = ()
    legs: list = field(default_factory=list)  # (bid, t_start, tau, a_vec, axis)
    # Cumulative VELOCITY heading change over the tour, radians. The body axis
    # is the wrong quantity here: ``exit_align`` measures only 15.5 deg because
    # the tour already avoids hard turns, which is a selection effect.
    hdg_sum: float = 0.0
    # Cumulative CLOUD-FRAME chord over the tour, metres. Only accumulated
    # when the chord penalty is on; 0.0 otherwise, and the rank never reads it.
    chord_sum: float = 0.0


@register("launchplan", "sequence-legs-beam")
class BeamSequenceLegs(LaunchPlanner):
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
        lat_factor=1.3,
        enforce_lat=False,
        heading_penalty_s_per_rad=0.0,
        chord_penalty_s_per_m=0.0,
        branch_max_m=0.0,
        tables=None,
        turn_time=None,
        recover_intercept_windows=False,
        **_kwargs,
    ):
        self.recover_intercept_windows = bool(recover_intercept_windows)
        self.intercept_stats = {}
        self.model = model
        # Injectable (Discussion #165 limits_mode="given"): default
        # constructs the legacy LimitTables()/uses the measured turn_time
        # function, bit-identical to before this option existed. agent.py's
        # given mode injects a shared GivenLimits + GivenLimits.turn_time
        # instead, so this candidate never constructs a legacy LimitTables
        # in that mode.
        self.tables = tables if tables is not None else LimitTables()
        self.turn_time_fn = turn_time if turn_time is not None else _default_turn_time
        self.elevation = float(given_parameters["environment"]["elevation"])
        self.burn_time = float(model.burn_time)
        self.launch_time = float(launch_time)
        self.max_rail_tilt_deg = float(max_rail_tilt_deg)
        self.depth = int(depth)
        self.eta = float(eta)
        self.beam = int(beam)
        self.branch = int(branch)
        self.sweep_top = int(sweep_top)
        self.lat_factor = float(lat_factor)
        # ``_feasible`` below was written but never called, so until now the
        # module docstring's claim that "planner and follower agree on what is
        # flyable" was not true: the metrics show plans peaking at 2.1x the
        # measured lateral table (REPORT section 7.2). Default False keeps the
        # measured baseline bit-identical; True is the fix, under measurement.
        self.enforce_lat = bool(enforce_lat)
        # Seconds charged per radian of cumulative velocity heading change in
        # the beam ranking. 0.0 keeps the ranking key literally ``t_rel``.
        self.heading_penalty_s_per_rad = float(heading_penalty_s_per_rad)
        # Seconds charged per metre of cumulative cloud-frame chord in the beam
        # ranking. 0.0 keeps the ranking key literally ``t_rel`` AND skips the
        # per-candidate predictor call that measures the chord.
        self.chord_penalty_s_per_m = float(chord_penalty_s_per_m)
        # Hard cap in metres on how far a branch candidate may sit from the
        # node. 0.0 = off, and then ``_near`` is the exact baseline argsort
        # slice. A node with nothing inside the cap simply does not expand;
        # falling back to the nearest candidate would smuggle the long legs
        # back in and make the measurement mean nothing.
        self.branch_max_m = float(branch_max_m)

    # -- physics helpers ----------------------------------------------
    def _thrust_accel(self, t_rel):
        return float(self.model.thrust(min(t_rel, self.burn_time))) / float(
            self.model.mass(min(t_rel, self.burn_time))
        )

    def _feasible(self, a_vec, v0, tau, t_rel, n=8):
        """The follower's own test, applied ALONG the leg.

        Speed changes across a leg, and so does the lateral authority the
        table gives, so a single-point test at the leg's start would pass
        legs that leave the envelope halfway through. The velocity samples
        are exact for this primitive: ``a_vec`` is constant, so
        ``v(s) = v0 + a_vec s``.
        """
        need = a_vec + GRAVITY * _Z
        if np.linalg.norm(need) > self.eta * self._thrust_accel(t_rel):
            return False
        if hasattr(self.tables, "set_context"):
            self.tables.set_context(t_rel, self.elevation)
        s = np.linspace(0.0, tau, n)[:, None]
        use = self.tables.lateral_use(v0 + a_vec * s, np.broadcast_to(a_vec, (n, 3)))
        return bool(use.size == 0 or (use <= self.lat_factor).all())

    # -- leg solves ----------------------------------------------------
    def _first_leg(self, predictor, bid, t_now):
        """Min-time full-thrust climb from rest (the same primitive the
        follower flies), rejected when the rail tilt cap or the ground
        constraint cannot hold it."""
        from ...follow.intercept import min_time_accel

        pad = np.array([0.0, 0.0, self.elevation])
        avail = self._thrust_accel(1.0)  # full thrust for liftoff (T/W 1.21)
        sol = min_time_accel(
            pad,
            np.zeros(3),
            t_now,
            lambda t_abs: predictor.predict(bid, t_abs),
            avail,
            recover_windows=self.recover_intercept_windows,
            stats=self.intercept_stats,
        )
        if sol is None:
            return None
        tau, a_net = sol
        need = a_net + GRAVITY * _Z
        u = need / max(np.linalg.norm(need), 1e-9)
        if np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0))) > self.max_rail_tilt_deg:
            return None  # rail cannot lean that far
        if a_net[2] < -0.05:
            return None  # diving first leg: ground risk
        target = predictor.predict(bid, t_now + tau)
        if not isinstance(target, tuple):
            return None
        return tau, target[0], a_net, u

    def _leg(self, predictor, bid, t_now, node, stretch=1.0):
        """Measured turn time (drift at constant velocity), then the shared
        min-time intercept from the post-turn state. ``stretch`` > 1 arrives
        later than min-time: the pass gets slower and its exit velocity
        turns, which the beam trades against the NEXT leg's turn cost
        (two-leg pass shaping)."""
        from ...follow.intercept import min_time_accel

        pred = predictor.predict(bid, t_now + node.t_rel + 1.0)
        if not isinstance(pred, tuple):
            return None
        rough_dir = pred[0] - node.pos
        rough_u = rough_dir / max(np.linalg.norm(rough_dir), 1e-9)
        if hasattr(self.tables, "set_context"):
            self.tables.set_context(node.t_rel, float(node.pos[2]))
        t_turn = self.turn_time_fn(np.arccos(np.clip(node.axis @ rough_u, -1.0, 1.0)))
        p0 = node.pos + node.vel * t_turn
        t0 = node.t_rel + t_turn
        avail = self.eta * self._thrust_accel(min(t0 + 2.0, self.burn_time))
        sol = min_time_accel(
            p0,
            node.vel,
            t_now + t0,
            lambda t_abs: predictor.predict(bid, t_abs),
            avail,
            t_hi=max(1.0, self.burn_time - t0),
            recover_windows=self.recover_intercept_windows,
            stats=self.intercept_stats,
        )
        if sol is None:
            return None
        tau, a_vec = sol
        if stretch > 1.0:
            tau *= stretch
            target_s = predictor.predict(bid, t_now + t0 + tau)
            if not isinstance(target_s, tuple):
                return None
            a_vec = 2.0 * (target_s[0] - p0 - node.vel * tau) / (tau * tau)
        if t0 + tau > self.burn_time:
            return None
        target = predictor.predict(bid, t_now + t0 + tau)
        if not isinstance(target, tuple):
            return None
        # Ground constraint along the leg.
        ss = np.linspace(0.0, tau, 8)[:, None]
        z = p0[2] + node.vel[2] * ss[:, 0] + 0.5 * a_vec[2] * ss[:, 0] ** 2
        if z.min() < self.elevation + 3.0:
            return None
        if self.enforce_lat and not self._feasible(a_vec, node.vel, tau, t0):
            return None
        return t_turn, tau, target[0], a_vec

    # -- ranking -------------------------------------------------------
    def _rank(self, node):
        """Beam ranking cost of a node, in seconds.

        With the penalty off this returns ``node.t_rel`` itself, so the sorts
        below are the exact baseline sorts (no float arithmetic introduced).
        """
        if self.heading_penalty_s_per_rad == 0.0 and self.chord_penalty_s_per_m == 0.0:
            return node.t_rel
        return (
            node.t_rel
            + self.heading_penalty_s_per_rad * node.hdg_sum
            + self.chord_penalty_s_per_m * node.chord_sum
        )

    def _near(self, predictor, cand, pos, t_abs, k):
        """Indices of the ``k`` candidates nearest ``pos``, optionally capped.

        With the cap OFF this is literally the baseline expression: the
        argsort of the distance from ``pos`` to the predictor's CURRENT
        snapshot positions, sliced to ``k``.

        With the cap ON the snapshot frame will not do. ``pos`` is the
        previous target at ITS pop time, 10-25 s into the future, while
        ``predictor.states`` is the cloud as it stands now; the cloud drifts
        at 7.6 m/s (REPORT section 7.11), so that distance carries 80-190 m of
        drift and a cap measured on it is a cap around a point ~150 m
        downwind. Predicting the candidates at the SAME instant as ``pos``
        cancels the drift the whole cloud shares - the same cancellation
        ``metrics.py`` uses for ``c_rel``, which is the quantity the cap is
        supposed to bound.
        """
        p_c = predictor.states[cand, :3]
        if self.branch_max_m <= 0.0:
            return np.argsort(np.linalg.norm(p_c - pos, axis=1))[:k]
        p_t = predictor.paths(np.asarray(cand, int), np.array([float(t_abs)]))[:, 0, :]
        d_rel = np.linalg.norm(p_t - pos, axis=1)
        idx = np.argsort(d_rel)[:k]
        return idx[d_rel[idx] <= self.branch_max_m]

    def _chord(self, predictor, node, p_b, t_pop_abs):
        """Cloud-frame chord from the node's last target to the new one.

        Both endpoints are evaluated at the SAME instant (the new leg's pop
        time), so the drift the two targets share over the leg cancels - the
        same definition ``metrics.py`` reports as ``c_rel``. The world-frame
        distance would be ~20% longer and would charge the cloud's motion to
        the leg, which is not what we are trying to shorten.

        Returns 0.0 with NO predictor call when the penalty is off, and 0.0
        when the node has no previous target (the pad root, and the
        replanner's in-flight start state, whose chord is undefined).
        """
        if self.chord_penalty_s_per_m == 0.0 or not node.targets:
            return 0.0
        prev = predictor.predict(int(node.targets[-1]), t_pop_abs)
        if not isinstance(prev, tuple):
            return 0.0
        return float(np.linalg.norm(np.asarray(p_b) - np.asarray(prev[0])))

    # -- search --------------------------------------------------------
    def _search(self, predictor, t_now, ids, start=None):
        """Beam over tours. ``start=None`` searches from the pad (rail-capped
        first legs); a ``_Node`` start searches the remaining tour from an
        in-flight state (used by the replanner)."""
        roots = []
        if start is None:
            pad = np.array([0.0, 0.0, self.elevation])
            p_now = predictor.states[ids, :3]
            order = ids[np.argsort(np.linalg.norm(p_now - pad, axis=1))][
                : self.branch * 2
            ]
            for bid in order:
                sol = self._first_leg(predictor, int(bid), t_now)
                if sol is None:
                    continue
                tau, p_b, a_net, u = sol
                roots.append(
                    _Node(
                        t_rel=tau,
                        pos=p_b,
                        vel=a_net * tau,
                        axis=u,
                        targets=(int(bid),),
                        legs=[(int(bid), 0.0, tau, a_net, u)],
                    )
                )
        else:
            cand = [b for b in ids if b not in start.targets]
            near = (
                self._near(
                    predictor, cand, start.pos, t_now + start.t_rel, self.branch * 2
                )
                if cand
                else np.zeros(0, int)
            )
            for j in near:
                bid = int(cand[j])
                for stretch in (1.0, 1.3):
                    sol = self._leg(predictor, bid, t_now, start, stretch)
                    if sol is None:
                        continue
                    t_turn, tau, p_b, a_vec = sol
                    axis = a_vec + GRAVITY * _Z
                    axis = axis / max(np.linalg.norm(axis), 1e-9)
                    vel_end = start.vel + a_vec * tau
                    t_pop = start.t_rel + t_turn + tau
                    roots.append(
                        _Node(
                            t_rel=t_pop,
                            pos=p_b,
                            vel=vel_end,
                            axis=axis,
                            targets=start.targets + (bid,),
                            legs=[(bid, start.t_rel + t_turn, tau, a_vec, axis)],
                            hdg_sum=start.hdg_sum + _hdg_change(start.vel, vel_end),
                            chord_sum=start.chord_sum
                            + self._chord(predictor, start, p_b, t_now + t_pop),
                        )
                    )
        frontier = sorted(roots, key=self._rank)[: self.beam]
        best = list(frontier)

        while frontier:
            nxt = []
            for node in frontier:
                cand = [b for b in ids if b not in node.targets]
                if not cand:
                    continue
                near = self._near(
                    predictor, cand, node.pos, t_now + node.t_rel, self.branch
                )
                for j in near:
                    bid = int(cand[j])
                    for stretch in (1.0, 1.3):
                        sol = self._leg(predictor, bid, t_now, node, stretch)
                        if sol is None:
                            continue
                        t_turn, tau, p_b, a_vec = sol
                        axis = a_vec + GRAVITY * _Z
                        axis = axis / max(np.linalg.norm(axis), 1e-9)
                        vel_end = node.vel + a_vec * tau
                        t_pop = node.t_rel + t_turn + tau
                        nxt.append(
                            _Node(
                                t_rel=t_pop,
                                pos=p_b,
                                vel=vel_end,
                                axis=axis,
                                targets=node.targets + (bid,),
                                legs=node.legs
                                + [(bid, node.t_rel + t_turn, tau, a_vec, axis)],
                                hdg_sum=node.hdg_sum + _hdg_change(node.vel, vel_end),
                                chord_sum=node.chord_sum
                                + self._chord(predictor, node, p_b, t_now + t_pop),
                            )
                        )
            nxt.sort(key=lambda n: (-len(n.targets), self._rank(n)))
            frontier = nxt[: self.beam]
            best.extend(frontier)
            if frontier and len(frontier[0].targets) >= self.depth:
                break
        best.sort(key=lambda n: (-len(n.targets), self._rank(n)))
        return best[: self.sweep_top]

    # -- ref building and sweep scoring --------------------------------
    def _sample(self, node, t_now, start=None):
        if start is None:
            pos0, vel0, t0 = np.array([0.0, 0.0, self.elevation]), np.zeros(3), 0.0
        else:
            pos0, vel0, t0 = start.pos.copy(), start.vel.copy(), start.t_rel
        ts, ps, vs, accs, axes = (
            [t0],
            [pos0.copy()],
            [vel0.copy()],
            [np.zeros(3)],
            [node.legs[0][4]],
        )
        pos = pos0.copy()
        vel = vel0.copy()
        t_prev = t0
        for bid, t_start, tau, a_vec, axis in node.legs:
            if t_start > t_prev + 1e-9:  # turn drift
                for s in np.arange(REF_DT, t_start - t_prev + 1e-9, REF_DT):
                    ts.append(t_prev + s)
                    ps.append(pos + vel * s)
                    vs.append(vel.copy())
                    accs.append(np.zeros(3))
                    axes.append(axis)
                pos = pos + vel * (t_start - t_prev)
            n_s = max(2, int(np.ceil(tau / REF_DT)))
            for s in np.linspace(0.0, tau, n_s + 1)[1:]:
                ts.append(t_start + s)
                ps.append(pos + vel * s + 0.5 * a_vec * s * s)
                vs.append(vel + a_vec * s)
                accs.append(a_vec.copy())
                axes.append(axis)
            pos = ps[-1].copy()
            vel = vs[-1].copy()
            t_prev = t_start + tau
        return ts, ps, vs, accs, axes

    def _sweep_score(self, ts, ps, predictor, t_now):
        ids = predictor.released_ids()
        if len(ids) == 0:
            return 0
        t_abs = t_now + np.asarray(ts)
        bp = predictor.paths(ids, t_abs)
        counts, _popped, _dmin, _first = pops_mod.score_paths(np.asarray(ps), bp, 1.5)
        return int(counts)

    # -- planning ------------------------------------------------------
    def plan(self, observation, predictor, *, deadline: float) -> Plan | None:
        t_now = float(observation["simulation_time"])
        if t_now < self.launch_time:
            return None
        ids = predictor.released_ids()
        if len(ids) == 0:
            return None
        tours = self._search(predictor, t_now, ids)
        if not tours:
            return None

        best, best_score, best_samples = None, -1, None
        for node in tours:
            samples = self._sample(node, t_now)
            score = self._sweep_score(samples[0], samples[1], predictor, t_now)
            if score > best_score:
                best, best_score, best_samples = node, score, samples
        ts, ps, vs, accs, axes = best_samples
        ref = SampledReference(ts, ps, vs, accs, axes)

        legs, pop_exp = [], []
        for i, (bid, t_start, tau, a_vec, axis) in enumerate(best.legs):
            if hasattr(self.tables, "set_context"):
                self.tables.set_context(t_start, self.elevation)
            t_pop = t_now + t_start + tau
            legs.append(
                Leg(
                    leg_id=i,
                    window=InterceptWindow(
                        balloon_id=bid,
                        t_open=t_pop - WINDOW_HALF_S,
                        t_close=t_pop + WINDOW_HALF_S,
                        balloon_state_at=_state_closure(predictor, bid),
                    ),
                    t_start=t_start,
                    t_end=t_start + tau,
                    u_start=axis,
                    u_end=axis,
                    slew_time_required=0.0
                    if i == 0
                    else self.turn_time_fn(
                        np.arccos(np.clip(best.legs[i - 1][4] @ axis, -1, 1))
                    ),
                    tube=_const_tube,
                )
            )
            pop_exp.append(PopExpectation(balloon_id=bid, t=t_pop, margin_m=0.75))

        u0 = best.legs[0][4]
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
            first_target=best.legs[0][0],
        )
        self.last_sweep_score = best_score
        return Plan(launch=launch, ref=ref, pops=pop_exp, legs=legs)

    # -- in-flight replan (used by the hybrid-insertion replanner) ------
    def replan_from(self, position, velocity, axis, t_rel, launch, predictor):
        """Re-search the remaining tour from an in-flight state. Times in the
        returned Plan stay on the since-launch clock; ``launch`` is carried
        over untouched (immutable by contract)."""
        ids = predictor.released_ids()
        if len(ids) == 0:
            return None
        start = _Node(
            t_rel=float(t_rel),
            pos=np.asarray(position, float),
            vel=np.asarray(velocity, float),
            axis=np.asarray(axis, float),
        )
        t_launch = float(launch.t_launch)
        tours = self._search(predictor, t_launch, ids, start=start)
        if not tours:
            return None
        best, best_score, best_samples = None, -1, None
        for node in tours:
            samples = self._sample(node, t_launch, start=start)
            score = self._sweep_score(samples[0], samples[1], predictor, t_launch)
            if score > best_score:
                best, best_score, best_samples = node, score, samples
        ts, ps, vs, accs, axes = best_samples
        ref = SampledReference(ts, ps, vs, accs, axes)
        legs, pop_exp = [], []
        for i, (bid, t_start, tau, a_vec, leg_axis) in enumerate(best.legs):
            t_pop = t_launch + t_start + tau
            legs.append(
                Leg(
                    leg_id=i,
                    window=InterceptWindow(
                        balloon_id=bid,
                        t_open=t_pop - WINDOW_HALF_S,
                        t_close=t_pop + WINDOW_HALF_S,
                        balloon_state_at=_state_closure(predictor, bid),
                    ),
                    t_start=t_start,
                    t_end=t_start + tau,
                    u_start=leg_axis,
                    u_end=leg_axis,
                    slew_time_required=0.0,
                    tube=_const_tube,
                )
            )
            pop_exp.append(PopExpectation(balloon_id=bid, t=t_pop, margin_m=0.75))
        self.last_sweep_score = best_score
        return Plan(launch=launch, ref=ref, pops=pop_exp, legs=legs)


def _hdg_change(v0, v1) -> float:
    """Angle between two velocity vectors, radians; 0.0 if either is null.

    Used as the per-leg increment of a node's cumulative heading change. The
    pad leg starts from rest and therefore contributes nothing.
    """
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(float(np.dot(v0, v1)) / (n0 * n1), -1.0, 1.0)))


def _const_tube(_t: float) -> float:
    return TUBE_RADIUS_M


def _state_closure(predictor, bid):
    def state_at(t: float):
        out = predictor.predict(bid, t)
        return out if isinstance(out, tuple) else (None, None)

    return state_at
