"""W3 / P3a: quicker thrust redirects. Shared by CompensatedInnerLoop (N = 1) and GivenSim (N rows), so the
planner's rollouts and the executed loop use the identical law.

1. Exact gimbal-actuator inversion (``act_inv``). The official actuator (ActiveRocketPy Actuator setter) is
   y[n] = clamp(y[n-1] + clipdelta(alpha * (u[n] - y[n-1]), rate * dt)), alpha = dt / (tau + dt), with tau,
   rate and range given parameters. The agent knows y exactly by running this model on its own commands, so
   it can command u = y + (y_des - y) / alpha and get y_des (within the rate limit and while |u| <= range).
   The lag disappears for the whole command; nothing is differentiated, so measurement noise is not
   amplified (unlike the derivative lead ``use_lead``, which hurt on sc3). With tau = 0 it is the identity.
   NOTE: the old loop rate-limited its command u and GivenSim modelled "rate limit, then lag" on y, which
   is ~8 deg/s initial slew instead of 60 deg/s under the sc3/sc4 lag; the act_inv path models it exactly.

2. Setpoint feedforward (``ff``): the commanded thrust direction n_cmd (piecewise constant per guidance
   decision) drives a rate- and acceleration-limited reference direction n_ref (near time-optimal slew with
   a sqrt braking law). The attitude feedback acts on the error to n_ref; the rate command gets
   w_ff = R^T w_ref and the gimbal gets d_ff = asin(I a_ref / (T lever)) (flatness feedforward). Lag
   inversion of the feedforward path is covered by (1), which ff always uses.

State per row (N, FF_DIM): n_ref (3), w_ref (3, launch frame). A zero n_ref (fresh state) is initialised to
the current nose with zero rate; burnout (T = 0) resets it.
"""

from __future__ import annotations

import numpy as np

try:  # speed (research/sc40/speed): the ufunc / C functions behind np.clip, np.interp, np.einsum (optimize=False) and
    # ndarray.sum, called directly (same computation, without the Python wrappers' per-call overhead)
    from numpy._core.multiarray import c_einsum as _einsum, interp as _interp
    from numpy._core.umath import clip as _clip
except ImportError:  # pragma: no cover - other numpy layouts: the public functions (identical results)
    _clip, _interp, _einsum = np.clip, np.interp, np.einsum
_add_reduce = np.add.reduce
try:  # round 2: the C function behind np.count_nonzero (a boolean test; no Python wrapper)
    from numpy._core.multiarray import count_nonzero as _nz
except ImportError:  # pragma: no cover
    _nz = np.count_nonzero

FF_DIM = 6


class FFParams:
    """Reference-model parameters (agent constants; loop_kwargs ff_*)."""

    def __init__(
        self,
        ff_wmax=3.0,
        ff_afrac=0.6,
        ff_klin=6.0,
        ff_brake=0.7,
        ff_tau_a=0.04,
        gmax=15.0,
    ):
        self.wmax = float(ff_wmax)  # rad/s, reference slew-rate cap
        self.afrac = float(
            ff_afrac
        )  # fraction of full-gimbal angular acceleration the reference may use
        self.klin = float(ff_klin)  # 1/s, linear slew-rate gain near alignment
        self.brake = float(ff_brake)  # fraction of a_max assumed for braking (sqrt law)
        self.tau_a = float(ff_tau_a)  # s, rate-tracking time constant of the reference
        self.gmax = float(gmax)  # deg
        # speed (research/sc40/speed): the per-step constant of ff_step, the same numpy expression evaluated once
        self.sin_gmax = np.sin(np.radians(self.gmax))


_CA = np.array([1, 2, 0, 2, 0, 1])
_CB = np.array([2, 0, 1, 1, 2, 0])


def _cross(a, b):
    """np.cross for (N, 3) rows with numpy's own per-component expressions (bit-identical, less overhead):
    out_c = a[c+1] * b[c+2] - a[c+2] * b[c+1]; the six products in one multiply, then one difference."""
    p = a[:, _CA] * b[:, _CB]
    return p[:, :3] - p[:, 3:]


def _sum3(p):
    """(p0 + p1) + p2 per (N, 3) row, used in place of p.sum(1) without the reduction's per-call cost.

    Equal to p.sum(1) bit for bit on every row with a finite or infinite result except two cases: (1) the sign of an
    exact zero when all three terms are -0.0 (numpy's add.reduce starts from its +0.0 identity, so it returns +0.0 where
    this returns -0.0; any other zero sum is +0.0 in both); (2) the sign and payload bits of a NaN result (which NaN
    operand propagates depends on the order of the additions). Neither reaches an output at the call sites (ff_step,
    ff_step_off, the pop test): an all -0.0 row needs every product to be a signed zero, i.e. two orthogonal-signed zero
    patterns that the unit / reset vectors there do not produce, and a NaN row stays NaN (and every comparison with it is
    False) either way. Unlike _norm (a sum of squares, never -0.0), this is therefore an equivalence for these inputs,
    not for every input; research/sc40/speed/microtest.py checks both cases on the machine it runs on."""
    return p[:, 0] + p[:, 1] + p[:, 2]


def _norm(x):
    """np.linalg.norm(x, axis=1) for (N, 3) rows (same sequential sum of squares (x0^2 + x1^2) + x2^2, bit-identical)."""
    sq = x * x
    return np.sqrt(sq[:, 0] + sq[:, 1] + sq[:, 2])


def _l3(p):
    """np.add.reduce(p, -1) (= p.sum(-1)) for a last axis of length 3 and any leading shape, as ((p0 + p1) + p2) + 0.0,
    without the reduction's per-output inner-loop calls (speed round 2, research/sc40/speed).

    numpy reduces a short axis sequentially and adds the result to its +0.0 identity, i.e. +0.0 + ((p0 + p1) + p2) (or,
    in the other iteration order, ((+0.0 + p0) + p1) + p2). Both equal ((p0 + p1) + p2) + 0.0 bit for bit: x + 0.0 == x
    for every x except -0.0 (-> +0.0), a sum with the +0.0 start is never -0.0, and an inner partial sum that differs
    only in the sign of a zero gives the same result once a nonzero term or the final + 0.0 is added. NaN operands
    propagate in the same order. research/sc40/speed/microtest.py checks it against add.reduce at the call shapes."""
    return ((p[..., 0] + p[..., 1]) + p[..., 2]) + 0.0


def _l3sq(p):
    """np.add.reduce(p, -1) for a last axis of length 3 whose entries are squares x * x (never -0.0): (p0 + p1) + p2,
    the same value as _l3 (the + 0.0 changes nothing when no term is -0.0); np.linalg.norm(x, axis=-1) is
    np.sqrt(_l3sq(x * x)) (numpy computes it as sqrt(add.reduce(x * x, axis)))."""
    return (p[..., 0] + p[..., 1]) + p[..., 2]


def ff_step(st, n_cmd, Rm, T, i_lat, lever, dt, p, normed=False, tabs=None):
    """One reference-model step.

    st (N, FF_DIM), n_cmd (N, 3) unit commanded thrust direction (launch frame), Rm (N, 3, 3) body->launch,
    T (N,) current thrust [N], i_lat (N,) lateral inertia about the CM, lever scalar [m], dt step [s].
    Returns (new state, n_ref (N, 3), w_ff body (N, 2) rad/s, d_ff body (N, 2) deg).
    Speed (research/sc40/speed): the same per-element expressions as before with fewer numpy calls (the input state is
    only copied when a row is reset, masked zeroing is skipped when no row is off, the new state is one concatenate).
    normed=True: the caller guarantees every row of st is an earlier ff_step output. Its n_ref is n / max(|n|, 1e-9)
    with |n| >= 1 (a unit n_ref plus a step normal to it), so |n_ref| = 1 to rounding or nan, and the reset test
    |n_ref| < 0.5 is False for every row: reset == off exactly, and the norm is not evaluated.
    tabs (round 2): (a_max, 2 brake a_max, max(T lever, 1e-9)) precomputed by the caller from the same expressions
    (GivenSim.step_tables); the caller then guarantees T > 1e-9 on every row (no row is off).
    st may also be a tuple (n_ref (N, 3), w_ref (N, 3)) of contiguous arrays (round 2: the rollout keeps the state split,
    so no op works on (N, 3) column views of an (N, 6) array); the new state is then returned as such a tuple.
    """
    split = isinstance(st, tuple)
    nref, wref = st if split else (st[:, 0:3], st[:, 3:6])
    if tabs is None:
        off = T <= 1e-9
        any_off = bool(off.any())
    else:  # no row off: off is all False
        off, any_off = False, False
    if normed:
        reset, any_reset = off, any_off
    else:
        reset = (_norm(nref) < 0.5) | off
        any_reset = bool(reset.any())
    if any_reset:
        nref, wref = nref.copy(), wref.copy()
        nref[reset] = Rm[:, :, 2][reset]
        wref[reset] = 0.0
    # angle-axis error n_ref -> n_cmd, _cross(nref, n_cmd); round 2: the gather nref[:, _CA] is reused below for
    # _cross(wref, nref), whose six products are the same products nref[_CA] * wref[_CB] (multiplication commutes)
    nA = nref[:, _CA]
    pa = nA * n_cmd[:, _CB]
    ax = pa[:, :3] - pa[:, 3:]
    s = _norm(ax)
    th = np.arctan2(s, _sum3(nref * n_cmd))
    if _nz(s > 1e-9) == len(s):  # round 2: every row has s > 1e-9 (not NaN): where(True, x, 0.0) == x, max(s, 1e-9) == s
        eh = ax / s[:, None]
    else:
        eh = np.where(s[:, None] > 1e-9, ax / np.maximum(s, 1e-9)[:, None], 0.0)
    if tabs is None:
        a_max = p.afrac * T * lever * p.sin_gmax / np.maximum(i_lat, 1e-6)
        ba = 2.0 * p.brake * a_max
        den = np.maximum(T * lever, 1e-9)
    else:
        a_max, ba, den = tabs
    w_des = np.minimum(np.minimum(p.wmax, np.sqrt(ba * th)), p.klin * th)
    alpha = (w_des[:, None] * eh - wref) / p.tau_a
    an = _norm(alpha)
    alpha = alpha * np.minimum(1.0, a_max / np.maximum(an, 1e-9))[:, None]
    if any_off:
        alpha[off] = 0.0
    wref = wref + alpha * dt
    wref = wref - _sum3(wref * nref)[:, None] * nref  # keep the rate normal to n_ref
    pw = nA * wref[:, _CB]  # _cross(wref, nref)_c = w[c+1] n[c+2] - w[c+2] n[c+1] = pw[3 + c] - pw[c]
    nref = nref + (pw[:, 3:] - pw[:, :3]) * dt
    nref = nref / np.maximum(_norm(nref), 1e-9)[:, None]
    # einsum operands C-ordered as before (gathers inside _cross give F-ordered intermediates)
    w_b = _einsum("nji,nj->ni", Rm, np.ascontiguousarray(wref))[:, :2]  # body rates (nose = body z)
    a_b = _einsum("nji,nj->ni", Rm, np.ascontiguousarray(alpha))[:, :2]
    d_ff = np.degrees(
        np.arcsin(
            _clip(
                i_lat[:, None] * a_b / den[:, None], -1.0, 1.0
            )
        )
    )
    if any_off:
        w_b[off] = 0.0
        d_ff[off] = 0.0
    return ((nref, wref) if split else np.concatenate([nref, wref], 1)), nref, w_b, d_ff


def ff_step_off(Rm, dt, split=False):
    """The new state of ff_step (normed or not) when every row is off (T <= 1e-9), from the same expressions: the reset
    sets n_ref = nose and w_ref = 0, alpha is zeroed, so w_ref + alpha dt = +0.0, and only the rate projection and the
    n_ref update remain (speed, research/sc40/speed; the outputs n_ref / w_ff / d_ff then feed only the gimbal, which is
    zero for rows without thrust)."""
    nose = Rm[:, :, 2]
    w1 = np.zeros(nose.shape)
    wref = w1 - _sum3(w1 * nose)[:, None] * nose
    nref = nose + _cross(wref, nose) * dt
    nref = nref / np.maximum(_norm(nref), 1e-9)[:, None]
    return (nref, wref) if split else np.concatenate([nref, wref], 1)  # split: as ff_step with a tuple state


_AI_C = {}


def act_inv_y(g_des, y, gstep, alpha, gmax):
    """Gimbal output reached this step under exact actuator inversion: the rate-limited demand, further
    limited by what an in-range command u in [-gmax, gmax] can reach through the lag.
    Round 2 (research/sc40/speed): the four bounds y - gstep, y + gstep, y + alpha (-gmax - y), y + alpha (gmax - y) as
    two (2, N, 2) stacks (outer broadcasts, contiguous halves), the same per-element operations: y + (-gstep) == y - gstep
    exactly, alpha x == x alpha and x + y == y + x."""
    c = _AI_C.get((gstep, gmax))
    if c is None:
        c = _AI_C[(gstep, gmax)] = (np.array([-gstep, gstep]).reshape(2, 1, 1), np.array([-gmax, gmax]).reshape(2, 1, 1))
    Y = y + c[0]  # y - gstep, y + gstep
    y1 = _clip(g_des, Y[0], Y[1])
    Z = c[1] - y  # -gmax - y, gmax - y
    Z *= alpha
    Z += y
    return _clip(y1, Z[0], Z[1])
