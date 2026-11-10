"""Shared analytic min-time intercept primitive.

One solve used by BOTH the planner (with a margin eta < 1) and the follower
(at the physical limit): the constant-acceleration inversion

    a(tau) = 2 * (P_b(t + tau) - p - v*tau) / tau^2

demands thrust accel |a + g|. The original solver assumes a feasible far
horizon and bisects to an intercept root of

    |a(tau) + g| = avail

found by bisection. At the root the command uses exactly full (or margined)
thrust; earlier is unreachable, later wastes time. The planner building legs
and the follower re-solving each step therefore agree on what is flyable,
which the wave-1 debugging showed is the whole game: hand-made aim floors and
feasibility slides diverged and slid every arrival late.
"""

from __future__ import annotations

import numpy as np

from ..vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])


def min_time_accel(
    pos,
    vel,
    t_now,
    predict,
    avail,
    *,
    t_lo=0.25,
    t_hi=26.0,
    tol=0.02,
    recover_windows=False,
    stats=None,
):
    """Minimum-time constant-acceleration intercept of a moving target.

    Parameters: rocket ``pos``/``vel``, absolute time ``t_now``,
    ``predict(t_abs) -> (pos, vel) | None`` for the target, and ``avail``
    thrust acceleration (m/s^2).

    Returns ``(tau, a_cmd)`` or None when the target is unreachable within
    ``t_hi``. ``a_cmd`` is the net acceleration command; at the root
    ``|a_cmd + g| <= avail``. Optional ``recover_windows`` scans for a finite
    feasible window after far-horizon rejection; it does not guarantee the
    earliest root or find windows narrower than its finite sampling grid.
    """

    def demand(tau):
        target = predict(t_now + tau)
        if not isinstance(target, tuple):
            return None
        a = 2.0 * (target[0] - pos - vel * tau) / (tau * tau)
        return float(np.linalg.norm(a + GRAVITY * _Z)) - avail, a

    lo = demand(t_lo)
    if lo is None:
        return None
    if lo[0] <= 0.0:  # reachable sooner than t_lo: aim at t_lo
        return t_lo, lo[1]
    hi = demand(t_hi)
    if hi is None or hi[0] > 0.0:
        if not recover_windows:
            return None
        # An infeasible far horizon does not imply an empty feasible set.
        # Bounded scan: narrow unsampled windows may still be missed.
        if stats is not None:
            stats["attempts"] = stats.get("attempts", 0) + 1
        previous = None
        for tau in np.geomspace(t_lo, t_hi, 65):
            trial = demand(tau)
            if trial is None:
                previous = None
                continue
            if trial[0] <= 0.0:
                lower, upper = previous, float(tau)
                if lower is not None:
                    while upper - lower > tol:
                        mid = (lower + upper) / 2
                        sample = demand(mid)
                        if sample is None:
                            break
                        if sample[0] <= 0:
                            upper = mid
                        else:
                            lower = mid
                if stats is not None:
                    stats["hits"] = stats.get("hits", 0) + 1
                return upper, demand(upper)[1]
            previous = float(tau)
        return None
    a, b = t_lo, t_hi
    for _ in range(40):
        mid = 0.5 * (a + b)
        d = demand(mid)
        if d is None:
            return None
        if d[0] > 0.0:
            a = mid
        else:
            b = mid
        if b - a < tol:
            break
    d = demand(b)
    return (b, d[1]) if d is not None else None
