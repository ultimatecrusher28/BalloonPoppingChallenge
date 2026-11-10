"""sequence-legs-greedy: minimal-flight tour planner.

Greedy nearest-in-time tour over constant-acceleration point-mass legs. This
is the skeleton's planner: correct interfaces, honest feasibility bound
(fraction of thrust acceleration), no search. The wave-1 candidate replaces
the greedy choice with a searched backend; the Plan it emits is identical in
type, so everything downstream stays put.

Launch posture: wait late (probe-launch-window: +0.70 paired, adopted), then
launch immediately with the rail leaned along the first leg (probe showed the
first-leg demand is only 10-16 deg, inside the default cap).
"""

from __future__ import annotations

import numpy as np

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
from ...vehicle.limits import SLEW_REST_TO_REST_90_S
from ...vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
REF_DT = 0.05
WINDOW_HALF_S = 0.5
TUBE_RADIUS_M = 30.0  # loose skeleton tube; wave 1 derives it per leg
SHORTLIST = 15  # candidates evaluated with the intercept solve


@register("launchplan", "sequence-legs-greedy")
class GreedySequenceLegs(LaunchPlanner):
    # eta near 1: T/W is only 1.21 at ignition, so any real planning margin
    # must come off the LATERAL budget, not the thrust magnitude (0.85 here
    # would leave 0.3 m/s^2 of climb and no feasible leg at all).
    def __init__(
        self,
        given_parameters,
        model,
        *,
        launch_time=40.0,
        max_rail_tilt_deg=16.0,
        depth=12,
        eta=0.97,
        **_kwargs,
    ):
        self.model = model
        self.elevation = float(given_parameters["environment"]["elevation"])
        self.burn_time = float(model.burn_time)
        self.launch_time = float(launch_time)
        self.max_rail_tilt_deg = float(max_rail_tilt_deg)
        self.depth = int(depth)
        self.eta = float(eta)

    # -- intercept solve ----------------------------------------------
    def _avail(self, t_rel):
        m = self.model.mass(min(t_rel, self.burn_time))
        return self.eta * float(self.model.thrust(min(t_rel, self.burn_time))) / m

    def _solve_leg(self, predictor, bid, t_now, t_rel, pos, vel, tilt_cap=None):
        """Smallest feasible intercept time tau, or None. Grow tau until the
        constant-acceleration leg fits inside the available thrust accel and,
        for the first leg off the rail, inside the rail tilt cap (a slower
        first leg needs a more vertical thrust direction, so the launch
        attitude and the plan stay one thing)."""
        pred = predictor.predict(bid, t_now + 0.5)
        if not isinstance(pred, tuple):
            return None
        tau = max(0.4, np.linalg.norm(pred[0] - pos) / 40.0)
        for _ in range(30):
            if t_rel + tau > self.burn_time:
                return None
            target = predictor.predict(bid, t_now + tau)
            if not isinstance(target, tuple):
                return None
            p_b = target[0]
            a_vec = 2.0 * (p_b - pos - vel * tau) / (tau * tau)
            need_vec = a_vec + GRAVITY * _Z
            ok = np.linalg.norm(need_vec) <= self._avail(t_rel + 0.5 * tau)
            if ok and tilt_cap is not None:
                u = _unit(need_vec, _Z)
                ok = np.degrees(_angle(_Z, u)) <= tilt_cap
            if ok:
                return tau, p_b, a_vec
            tau *= 1.2
        return None

    # -- planning ------------------------------------------------------
    def plan(self, observation, predictor, *, deadline: float) -> Plan | None:
        t_now = float(observation["simulation_time"])
        if t_now < self.launch_time:
            return None
        ids = predictor.released_ids()
        if len(ids) == 0:
            return None

        pad = np.array([0.0, 0.0, self.elevation])
        pos, vel, t_rel = pad.copy(), np.zeros(3), 0.0
        ts, ps, vs, accs, axes = [0.0], [pos.copy()], [vel.copy()], [], []
        legs, pops = [], []
        taken: set[int] = set()
        prev_axis = _Z.copy()

        for leg_id in range(self.depth):
            # Shortlist by current straight-line distance, then take the leg
            # with the smallest feasible intercept time.
            cand = [b for b in ids if b not in taken]
            if not cand:
                break
            order = sorted(
                cand, key=lambda b: np.linalg.norm(self.states_pos(predictor, b) - pos)
            )[:SHORTLIST]
            best = None
            cap = self.max_rail_tilt_deg if leg_id == 0 else None
            for bid in order:
                sol = self._solve_leg(
                    predictor, bid, t_now + t_rel, t_rel, pos, vel, tilt_cap=cap
                )
                if sol and (best is None or sol[0] < best[1][0]):
                    best = (bid, sol)
            if best is None:
                break
            bid, (tau, p_b, a_vec) = best

            axis0 = _unit(a_vec + GRAVITY * _Z, prev_axis)
            n_samp = max(2, int(np.ceil(tau / REF_DT)))
            ss = np.linspace(0.0, tau, n_samp + 1)[1:]
            for s in ss:
                ts.append(t_rel + s)
                ps.append(pos + vel * s + 0.5 * a_vec * s * s)
                vs.append(vel + a_vec * s)
                accs.append(a_vec.copy())
                axes.append(axis0)
            t_pop_abs = t_now + t_rel + tau
            slew = _angle(prev_axis, axis0) / (np.pi / 2) * SLEW_REST_TO_REST_90_S
            legs.append(
                Leg(
                    leg_id=leg_id,
                    window=InterceptWindow(
                        balloon_id=int(bid),
                        t_open=t_pop_abs - WINDOW_HALF_S,
                        t_close=t_pop_abs + WINDOW_HALF_S,
                        balloon_state_at=_state_closure(predictor, int(bid)),
                    ),
                    t_start=t_rel,
                    t_end=t_rel + tau,
                    u_start=axis0,
                    u_end=axis0,
                    slew_time_required=float(slew),
                    tube=_const_tube,
                )
            )
            pops.append(PopExpectation(balloon_id=int(bid), t=t_pop_abs, margin_m=0.75))
            taken.add(bid)
            pos, vel, t_rel = p_b.copy(), vel + a_vec * tau, t_rel + tau
            prev_axis = axis0

        if not legs:
            return None

        # accel/axis arrays are per-sample of the NEXT interval; pad the end.
        accs = [np.zeros(3)] + accs
        axes = [legs[0].u_start] + axes
        ref = SampledReference(ts, ps, vs, accs, axes)

        u0 = legs[0].u_start
        # The first leg was solved under the rail cap, so this never clips.
        tilt = min(np.degrees(_angle(_Z, u0)), self.max_rail_tilt_deg)
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
            first_target=legs[0].window.balloon_id,
        )
        return Plan(launch=launch, ref=ref, pops=pops, legs=legs)

    @staticmethod
    def states_pos(predictor, bid) -> np.ndarray:
        return predictor.states[bid, :3]


def _unit(v, fallback):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else fallback


def _angle(u, v):
    return float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))


def _const_tube(_t: float) -> float:
    return TUBE_RADIUS_M


def _state_closure(predictor, bid):
    def state_at(t: float):
        out = predictor.predict(bid, t)
        return out if isinstance(out, tuple) else (None, None)

    return state_at
