"""Thrust-then-ballistic intercept: the only primitive that can score after
burnout.

What it rests on (all measured in ``research/limits/coast_reach.py``, written
up as REPORT section 7.14):

- After burnout the vehicle has **no** lateral authority. Held for 6 s of
  coast in the real env, full gimbal moves the path 0.024 m and full roll
  torque 0.26 m, against a 1.5 m pop radius: TVC torque is thrust x arm and
  thrust is zero, and the aero normal force needs an angle of attack that can
  no longer be commanded. The ``a_lat_max`` table in ``vehicle/limits.py`` was
  measured at throttle 1.0 and does NOT carry over to the coast.
- So the coast is ONE deterministic arc, fixed at the moment of burnout. The
  only control over it is the burnout state, which the last seconds of thrust
  set. That makes a post-burnout pop a *targeting* problem, not a guidance
  one: aim at burnout or not at all.
- A point-mass drag+gravity propagator tracks the FLOWN coast to a p50 of
  0.47 m at 1 s, 1.48 m at 2 s and 2.62 m at 3 s (sc1 seeds 0-19,
  ``research/limits/data/coast_reach.json``). The 1.5 m pop radius is
  therefore spent by about 2 s, which is where ``max_coast_s`` sits: a
  measured horizon, not a tuning knob.

The primitive: hold a constant net acceleration for the ``s`` seconds of burn
that remain, then fly ballistic, and arrive at the target at ``T``. Without
drag that inverts in closed form,

    a_net = (q - p - v*T + 0.5*g*(T-s)^2 * z) / (s*(T - s/2))     (T > s)
    a_net = 2*(q - p - v*T) / T^2                                 (T <= s)

which is the same constant-acceleration inversion ``intercept.min_time_accel``
uses, extended by the unpowered tail; ``T <= s`` degenerates to exactly it.
Drag is then folded in by shooting on the same expression, because the
compensated inner loop cancels drag during the burn but nothing cancels it
during the coast.
"""

from __future__ import annotations

import numpy as np

from ..vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
DEFAULT_DT = 0.05


def coast_arc(p0, v0, model, t_end, dt=DEFAULT_DT):
    """Unpowered trace (drag + gravity, RK2) as an (n, 3) array of positions."""
    k = 0.5 * float(model.cd_power_off) * float(model.ref_area)
    mass = float(model.mass(model.burn_time))
    n = max(int(round(t_end / dt)), 0)
    p = np.asarray(p0, float).copy()
    v = np.asarray(v0, float).copy()
    out = np.empty((n + 1, 3))
    out[0] = p

    def acc(p_, v_):
        sp = float(np.linalg.norm(v_))
        a = -GRAVITY * _Z
        if sp > 1e-9:
            a = a - (k * float(model.density(p_[2])) * sp / mass) * v_
        return a

    for i in range(n):
        a1 = acc(p, v)
        vm = v + 0.5 * dt * a1
        a2 = acc(p + 0.5 * dt * v, vm)
        p = p + dt * vm
        v = v + dt * a2
        out[i + 1] = p
    return out


def powered_coast_end(p, v, a_net, s, T, model, dt=DEFAULT_DT):
    """Position at ``T`` under net accel ``a_net`` for ``s`` s, then coasting.

    ``a_net`` is the net acceleration the compensated inner loop delivers
    (it cancels drag and the aero normal force during the burn), so the
    powered phase is a plain double integral; only the tail carries drag.
    """
    p = np.asarray(p, float).copy()
    v = np.asarray(v, float).copy()
    s_eff = float(min(s, T))
    if s_eff > 0.0:
        p = p + v * s_eff + 0.5 * np.asarray(a_net, float) * s_eff * s_eff
        v = v + np.asarray(a_net, float) * s_eff
    if T > s:
        return coast_arc(p, v, model, T - s, dt=dt)[-1]
    return p


def _denominator(T, s):
    return 0.5 * T * T if T <= s else s * (T - 0.5 * s)


def a_net_closed_form(p, v, q, T, s):
    """Drag-free inversion; the screen, and the seed for the shooting solve."""
    den = _denominator(T, s)
    if den <= 1e-9:
        return None
    tail = 0.0 if T <= s else 0.5 * GRAVITY * (T - s) ** 2
    return (
        np.asarray(q, float)
        - np.asarray(p, float)
        - np.asarray(v, float) * T
        + tail * _Z
    ) / den


def solve_a_net(p, v, q, T, s, model, *, iters=4, tol=0.05, dt=DEFAULT_DT):
    """Net accel whose thrust-then-coast path reaches ``q`` at ``T``.

    Returns ``(a_net, miss_m)``; ``miss_m`` is the residual of the final
    iterate, so a caller can refuse a solve that did not converge instead of
    flying a command that claims an intercept it does not make.
    """
    a_net = a_net_closed_form(p, v, q, T, s)
    if a_net is None:
        return None
    den = _denominator(T, s)
    miss = float("inf")
    for _ in range(iters):
        err = np.asarray(q, float) - powered_coast_end(p, v, a_net, s, T, model, dt=dt)
        miss = float(np.linalg.norm(err))
        if miss < tol:
            break
        a_net = a_net + err / den
    return a_net, miss


def thrust_need(a_net):
    """Thrust acceleration a net-acceleration command asks for."""
    return float(np.linalg.norm(np.asarray(a_net, float) + GRAVITY * _Z))


def select_target(
    p,
    v,
    t_now,
    s,
    avail,
    candidates,
    predict,
    model,
    *,
    max_coast_s=2.0,
    allow_powered=True,
    allow_coast=True,
    t_step=0.25,
    t_max_extra=0.0,
    margin=0.95,
    refine=4,
):
    """Cheapest thrust-then-coast intercept among ``candidates``.

    ``s`` is the burn time left, ``avail`` the thrust acceleration available,
    ``predict(bid, t_abs)`` the balloon predictor. Screening is the drag-free
    closed form (cheap, and it only ever over-states reach because drag
    shortens the tail); survivors are re-solved with drag and re-checked, so
    what comes back is feasible under the model actually flown.

    Returns ``(bid, t_arrive, a_net)`` or None. Ranked by coast time first
    (a shorter tail is a more predictable one) and required thrust second.
    """
    if s <= 1e-3 or avail <= 1e-6:
        return None
    t_hi = s + (max_coast_s if allow_coast else 0.0) + float(t_max_extra)
    grid = np.arange(t_step, t_hi + 1e-9, t_step)
    best = None
    for bid in candidates:
        for T in grid:
            T = float(T)
            if T <= s and not allow_powered:
                continue
            if T > s and not allow_coast:
                continue
            target = predict(int(bid), t_now + T)
            if not isinstance(target, tuple):
                continue
            a_net = a_net_closed_form(p, v, target[0], T, s)
            if a_net is None or thrust_need(a_net) > margin * avail:
                continue
            key = (round(max(T - s, 0.0), 2), thrust_need(a_net))
            if best is not None and key >= best[0]:
                continue
            best = (key, int(bid), T, a_net)
    if best is None:
        return None
    _key, bid, T, _a = best
    target = predict(bid, t_now + T)
    if not isinstance(target, tuple):
        return None
    sol = solve_a_net(p, v, target[0], T, s, model, iters=refine)
    if sol is None:
        return None
    a_net, miss = sol
    if miss > 0.5 or thrust_need(a_net) > avail:
        return None
    return bid, t_now + T, a_net
