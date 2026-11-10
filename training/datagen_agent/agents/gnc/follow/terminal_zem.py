"""terminal-zem: last-mile zero-effort-miss correction.

Inside the terminal window the constant-acceleration re-solve rotates its
command faster than the gimbal can follow (measured misses of a few meters at
tau < 1 s). ZEM feedback is the standard fix (Zarchan; the ZEM/t_go^2 family):

    ZEM   = P_b(t + t_go) - (p + v t_go - 0.5 g t_go^2 z)   (coast miss)
    a_thr = N * ZEM / t_go^2, N = 3, projected perpendicular to the
            closing direction (P-perp: only the miss component matters),
            norm-capped at the available thrust accel.

t_go is the closest-approach time of the gravity coast, refined from the
relative kinematics rather than taken from a schedule.
"""

from __future__ import annotations

import numpy as np

from ..vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
N_GAIN = 3.0


def terminal_accel(
    position, velocity, t_now, predict, avail, *, t_max=2.0, engage_frac=0.85
):
    """(t_go, net accel command) for the last-mile ZEM law, or None when the
    closing geometry is gone (receding / no target) or the required
    correction exceeds ``engage_frac * avail`` (out of the ZEM linear
    regime: the min-time solve keeps flying instead; switch condition per
    plan is required-accel vs budget, not a hardcoded time)."""
    target = predict(t_now)
    if not isinstance(target, tuple):
        return None
    p_b, v_b = target
    r = np.asarray(p_b, float) - position
    v_rel = np.asarray(v_b, float) - velocity
    closing = -float(r @ v_rel)
    speed_rel2 = float(v_rel @ v_rel)
    if closing <= 1e-6 or speed_rel2 < 1e-9:
        return None
    t_go = float(np.clip(closing / speed_rel2, 0.05, t_max))

    target_go = predict(t_now + t_go)
    if not isinstance(target_go, tuple):
        return None
    coast = position + velocity * t_go - 0.5 * GRAVITY * t_go * t_go * _Z
    zem = np.asarray(target_go[0], float) - coast
    a_thr = N_GAIN * zem / (t_go * t_go)
    # P-perp: only the component perpendicular to the closing direction
    # changes the miss; the parallel part just re-times the pass.
    los = -v_rel / np.sqrt(speed_rel2)
    a_thr = a_thr - (a_thr @ los) * los
    mag = float(np.linalg.norm(a_thr))
    if mag > engage_frac * avail:
        return None  # correction unaffordable: not engaged
    return t_go, a_thr - GRAVITY * _Z
