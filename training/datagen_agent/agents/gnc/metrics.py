"""Plan-side diagnostics: the first-order quantities the score decomposes into.

Measured on 214 legs of a previous architecture, a leg's duration is the chord
between consecutive targets *in the balloon cloud's frame* divided by the speed
in that same frame, with no fixed per-target overhead. So

    pops ~= (usable burn time / t_leg) * realization rate + coast pops

and a change is worth flying only if it moves one of those. The absolute-frame
chord is ~20% longer than the cloud-frame one (the cloud drifts during the leg),
which is why the cloud-frame endpoints here are evaluated at the SAME instant: a
simultaneous difference cancels the drift the two targets share. ``c_world`` is
the same chord WITHOUT that cancellation - the distance the vehicle actually
flies - so the pair separates "the targets are far apart" from "the cloud ran
away while we crossed".

``t_leg`` splits in two for the sequence-legs primitive: the attitude turn paid
before the leg, during which the vehicle drifts at constant velocity and makes
no progress beyond coasting, and the intercept itself. ``turn_frac`` is that
dead time as a share of tour time. A primitive that rotates thrust *during* the
leg pays no such share, so this number is what one would buy back.

Three groups say WHY a chord is long rather than just that it is:

- ``turn_angle`` (body axis) vs ``exit_align`` (velocity): the first is the
  dead time paid, the second is the heading error the next leg must undo.
  At a fixed lateral authority the achievable chord is 2R sin(theta/2) with
  R = v^2/a_lat, so a misaligned exit lengthens the NEXT chord quadratically
  in speed - the dominant term in the wave-1 gap.
- ``accel_use`` / ``lat_use``: how much of the thrust ball and of the measured
  lateral-authority table the plan spends. Near 0 means the envelope is idle;
  above 1 means the plan is writing checks the vehicle cannot cash and the
  follower will miss.
- ``nominal_miss``: the plan's own closest approach to each target, computed
  from the reference rather than asserted. ``PopExpectation.margin_m`` is a
  constant the planners fill in, so it is NOT a measurement and is deliberately
  not used here; a nominal miss above the 1.5 m pop radius means the plan
  claims a pop its own geometry does not make.

Everything here is a read-only derivation from a Plan the planner already
built. Nothing feeds back into flight.
"""

from __future__ import annotations

import numpy as np

from .vehicle.limits import LimitTables
from .vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
POP_RADIUS_M = 1.5
_MISS_HALF_WINDOW_S = 0.6
_MISS_DT = 0.01  # the env's own step: a finer grid claims false margin
_USE_DT = 0.1
_COAST_ACCEL = 0.1  # |a| below this is a drift arc, not a thrust arc


def _p50(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def _r(value, digits=3):
    return None if value is None else round(float(value), digits)


def _angle_deg(a, b) -> float | None:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def plan_metrics(plan, burn_time: float, model=None, tables=None) -> dict:
    """First-order quantities of a launch plan; ``{}`` when it has no legs.

    ``Leg`` times are on the since-launch clock, ``PopExpectation.t`` is
    absolute, and the two lists are built index-aligned by the planner.
    ``model`` enables the control-reserve columns (it supplies thrust/mass);
    without it those columns are omitted rather than guessed.
    """
    legs = getattr(plan, "legs", None)
    pops = getattr(plan, "pops", None)
    if not legs or not pops or len(legs) != len(pops):
        return {}

    taus, turns, angles = [], [], []
    for i, leg in enumerate(legs):
        taus.append(float(leg.t_end) - float(leg.t_start))
        if i == 0:
            continue
        turns.append(float(leg.slew_time_required))
        angles.append(_angle_deg(legs[i - 1].u_end, leg.u_start))
    angles = [a for a in angles if a is not None]

    # Cloud-frame chord: consecutive targets at one instant, so the drift they
    # share cancels. A leg with no predictable endpoint is skipped, not zeroed.
    # World-frame chord: each target at its OWN pop time, which is the
    # displacement the vehicle has to cover.
    chords, speeds, chords_world = [], [], []
    for i in range(1, len(legs)):
        t_pop = float(pops[i].t)
        here = legs[i].window.balloon_state_at(t_pop)
        prev = legs[i - 1].window.balloon_state_at(t_pop)
        prev_own = legs[i - 1].window.balloon_state_at(float(pops[i - 1].t))
        t_leg = turns[i - 1] + taus[i]
        if here[0] is not None and prev_own[0] is not None:
            chords_world.append(
                float(np.linalg.norm(np.asarray(here[0]) - np.asarray(prev_own[0])))
            )
        if here[0] is None or prev[0] is None:
            continue
        chord = float(np.linalg.norm(np.asarray(here[0]) - np.asarray(prev[0])))
        chords.append(chord)
        if t_leg > 1e-6:
            speeds.append(chord / t_leg)

    tour_time = float(legs[-1].t_end) - float(legs[0].t_end)
    t_legs = [turns[i - 1] + taus[i] for i in range(1, len(legs))]
    out = {
        "planned_depth": len(legs),
        "first_pop_tb": _r(legs[0].t_end, 2),
        "last_pop_tb": _r(legs[-1].t_end, 2),
        "span_tb": _r(tour_time, 2),
        # Leg 0 is the fixed-axis climb-out, not a tour leg: it pays no turn
        # and its duration is the climb-out cost, so it is excluded from the
        # per-leg medians and counted in first_pop_tb instead.
        "t_leg_p50": _r(_p50(t_legs), 2),
        "t_turn_p50": _r(_p50(turns), 2),
        "tau_p50": _r(_p50(taus[1:]), 2),
        "t_turn_frac": _r(sum(turns) / tour_time, 3) if tour_time > 1e-6 else None,
        "turn_angle_p50": _r(_p50(angles), 1),
        "c_rel_p50": _r(_p50(chords), 1),
        "c_world_p50": _r(_p50(chords_world), 1),
        "v_rel_p50": _r(_p50(speeds), 2),
        "burn_time_left": _r(burn_time - float(legs[-1].t_end), 2),
    }
    out.update(_exit_alignment(plan, legs, pops))
    out.update(_hit_margin(plan, legs, pops))
    out.update(_control_reserve(plan, legs, model, tables))
    return out


def _t_launch(legs, pops) -> float:
    """Absolute time of launch: pops are absolute, leg times are since-launch."""
    return float(pops[0].t) - float(legs[0].t_end)


def _exit_alignment(plan, legs, pops) -> dict:
    """Angle between the velocity leaving a pop and the straight line to the
    next pop point. 0 deg means the pass already points at the next target;
    90 deg means the whole crossing has to be re-flown sideways.

    Distinct from ``turn_angle``, which is the body/thrust axis: thrust can be
    re-aimed in ~1 s but velocity can only be turned at a_lat, so this is the
    one that lengthens the next chord.
    """
    ref = getattr(plan, "ref", None)
    if ref is None:
        return {}
    aligns = []
    for i in range(len(legs) - 1):
        nxt = legs[i + 1].window.balloon_state_at(float(pops[i + 1].t))
        here = legs[i].window.balloon_state_at(float(pops[i].t))
        if nxt[0] is None or here[0] is None:
            continue
        try:
            _p, v, _a, _ax = ref.state(float(legs[i].t_end))
        except Exception:  # a Reference that cannot be sampled: skip, not zero
            return {}
        ang = _angle_deg(
            np.asarray(v, float).reshape(3),
            np.asarray(nxt[0], float) - np.asarray(here[0], float),
        )
        if ang is not None:
            aligns.append(ang)
    return {"exit_align_p50": _r(_p50(aligns), 1)} if aligns else {}


def _hit_margin(plan, legs, pops) -> dict:
    """Closest approach of the planned reference to each target, on the env's
    own 0.01 s grid. Reported as distance: > 1.5 m is a pop the plan claims
    but its own geometry does not make.
    """
    ref = getattr(plan, "ref", None)
    if ref is None:
        return {}
    t0 = _t_launch(legs, pops)
    misses = []
    for leg, pop in zip(legs, pops):
        t_abs = np.arange(
            float(pop.t) - _MISS_HALF_WINDOW_S,
            float(pop.t) + _MISS_HALF_WINDOW_S + 1e-9,
            _MISS_DT,
        )
        try:
            p, _v, _a, _ax = ref.state(t_abs - t0)
        except Exception:
            return {}
        best = None
        for k, ta in enumerate(t_abs):
            b = leg.window.balloon_state_at(float(ta))
            if b[0] is None:
                continue
            d = float(np.linalg.norm(np.asarray(p)[k] - np.asarray(b[0])))
            best = d if best is None else min(best, d)
        if best is not None:
            misses.append(best)
    if not misses:
        return {}
    return {
        "nominal_miss_p50": _r(_p50(misses), 2),
        "nominal_miss_max": _r(max(misses), 2),
        "nominal_miss_over": sum(1 for m in misses if m > POP_RADIUS_M),
    }


def _control_reserve(plan, legs, model, tables) -> dict:
    """How much of the envelope the plan spends, sampled over the thrust arcs.

    ``accel_use`` is |a + g| / thrust-accel: the acceleration ball is centered
    at -g, so this is the fraction of the ball's radius the plan asks for.
    ``lat_use`` is the velocity-turning component against the MEASURED
    a_lat table. Drift arcs (|a| ~ 0) are excluded; including them would
    dilute the median with time the plan is not steering at all.
    """
    ref = getattr(plan, "ref", None)
    if ref is None or model is None:
        return {}
    tables = tables if tables is not None else LimitTables()
    ts = np.arange(float(legs[0].t_start), float(legs[-1].t_end) + 1e-9, _USE_DT)
    if ts.size == 0:
        return {}
    try:
        _p, v, a, _ax = ref.state(ts)
    except Exception:
        return {}
    v = np.asarray(v, float).reshape(-1, 3)
    a = np.asarray(a, float).reshape(-1, 3)
    thrusting = np.linalg.norm(a, axis=1) >= _COAST_ACCEL
    if not thrusting.any():
        return {}
    accel_use = []
    for t, a_k in zip(ts[thrusting], a[thrusting]):
        tb = min(float(t), float(model.burn_time))
        avail = float(model.thrust(tb)) / float(model.mass(tb))
        if avail > 1e-6:
            accel_use.append(float(np.linalg.norm(a_k + GRAVITY * _Z)) / avail)
    lat_use = list(tables.lateral_use(v[thrusting], a[thrusting]))
    out = {}
    if accel_use:
        out["accel_use_p50"] = _r(_p50(accel_use), 3)
    if lat_use:
        out["lat_use_p50"] = _r(_p50(lat_use), 3)
        out["lat_use_max"] = _r(max(lat_use), 3)
    return out
