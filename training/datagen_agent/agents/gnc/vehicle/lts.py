"""Saturated linear-tangent legs: the minimum-time way to reach a moving point.

Ported verbatim from branch ``tour-planner-ideal-15``
(``agents/saturated/lts.py``, with its test file) apart from the GRAVITY
import; the numerics are unchanged, so the legacy tests still pin them.

Why it was worth bringing back, in measured terms (research/limits/REPORT.md
section 6-7): the current beam primitive serialises an attitude turn before a
constant-acceleration intercept, which costs 22% of tour time as dead drift
AND spikes lateral demand to 2.1x the measured authority mid-leg while
sitting at 0.56x on the median. A linear-tangent leg rotates thrust THROUGH
the leg, so it pays no serialised turn and spreads the same lateral demand
over the whole arc. Both effects reduce the turn radius R = v^2/a_lat that
the reachability ceiling says is what actually binds the tour.

The vehicle is a point mass whose thrust acceleration has a bounded magnitude,
``alpha(t) = eta * T(t) / m(t)``, pointed anywhere, under uniform gravity:

    r'' = k * alpha(t) * u(t) - g * z_hat,     |u| = 1,  0 <= k <= 1

For the minimum-time problem the costates give ``k = 1`` and the linear tangent
law ``u(s) = normalize(A + B*s)`` for *any* ``alpha(t)``. With the arrival
velocity left free the costate ``B`` is zero and the direction is constant --
which is exactly the thrust-ball test of :mod:`..phase4.tour`, and exactly the
leg that pumps speed, because it arrives at ``2*dp/T - v0``. Pinning the arrival
velocity rotates the direction across the leg; that is what the leaderboard's
best runs fly (direction fits within 2 degrees per leg).

A leg is ``(t0, p0, v0) -> balloon`` whose position is ``bp + bv*s`` over the
leg (a balloon rises at a near-constant velocity for a few seconds), arriving
with velocity ``bv + v_rel``. With

    ZEV(tau) = bv + v_rel - v0 + g*tau*z_hat
    ZEM(tau) = bp + bv*tau - p0 - v0*tau + 0.5*g*tau^2*z_hat
    Iv(tau)  = int_0^tau k*alpha(t0+s) u(s) ds
    Ir(tau)  = int_0^tau (tau-s) k*alpha(t0+s) u(s) ds

the leg is the root of ``Iv = ZEV, Ir = ZEM``: six equations in ``A, B`` (up to
scale) and either ``tau`` (minimum time, ``k = 1``) or ``k`` (fixed time). It is
solved batched with damped Newton on an analytic Jacobian, the integrals on
Gauss-Legendre nodes, cold-started from the unconstrained LQ (ZEM/ZEV) solution
-- whose thrust is itself linear in time, so it is a point of the same family.

Everything here is batched over a leading axis ``N`` and uses numpy only.
"""

from __future__ import annotations

import numpy as np

from .rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
N_NODES = 16
_X, _W = np.polynomial.legendre.leggauss(N_NODES)
GL_X = 0.5 * (_X + 1.0)  # nodes on [0, 1]
GL_W = 0.5 * _W  # weights on [0, 1]

ETA_PLAN = 0.96
ETA_FLY = 0.97
MIN_TAU = 0.05


def alpha(model, t, eta=1.0):
    """Thrust acceleration available at ``t`` seconds since launch (m/s^2)."""
    t = np.asarray(t, dtype=float)
    return (
        float(eta)
        * np.asarray(model.thrust(t, 1.0), dtype=float)
        / np.asarray(model.mass(t), dtype=float)
    )


def zem_zev(tau, p0, v0, bp, bv, v_rel):
    """Zero-effort miss and velocity after ``tau`` seconds, gravity included."""
    tau = np.asarray(tau, dtype=float)[..., None]
    zev = bv + v_rel - v0 + GRAVITY * tau * _Z
    zem = bp + bv * tau - p0 - v0 * tau + 0.5 * GRAVITY * tau * tau * _Z
    return zem, zev


def lq_coeffs(tau, zem, zev):
    """Minimum-energy thrust ``c0 + c1*s`` that meets ZEM/ZEV exactly in ``tau``."""
    tau = np.asarray(tau, dtype=float)[..., None]
    c0 = 6.0 * zem / tau**2 - 2.0 * zev / tau
    c1 = (6.0 * zev * tau - 12.0 * zem) / tau**3
    return c0, c1


def lq_peak(tau, zem, zev):
    """Peak thrust of the LQ leg; ``|c0 + c1*s|`` is convex so the ends bound it."""
    c0, c1 = lq_coeffs(tau, zem, zev)
    tau = np.asarray(tau, dtype=float)[..., None]
    return np.maximum(
        np.linalg.norm(c0, axis=-1), np.linalg.norm(c0 + c1 * tau, axis=-1)
    )


def _nodes(model, t0, tau, eta):
    s = np.asarray(tau, dtype=float)[:, None] * GL_X[None, :]  # (N, n)
    a = alpha(model, np.asarray(t0, dtype=float)[:, None] + s, eta)  # (N, n)
    return s, a


def _integrals(A, B, tau, s, a, k=None):
    """``(Iv, Ir, u, wn)`` on the nodes; ``k`` scales the thrust (default 1)."""
    w = A[:, None, :] + B[:, None, :] * s[..., None]  # (N, n, 3)
    wn = np.maximum(np.linalg.norm(w, axis=-1), 1e-12)
    u = w / wn[..., None]
    scale = 1.0 if k is None else np.asarray(k, dtype=float)[:, None]
    q = (tau[:, None] * GL_W[None, :] * a * scale)[..., None]  # (N, n, 1)
    Iv = (q * u).sum(axis=1)
    Ir = (q * (tau[:, None] - s)[..., None] * u).sum(axis=1)
    return Iv, Ir, u, wn


def _jac_AB(A, B, tau, s, a, u, wn, k=None):
    """d(Iv, Ir)/d(A, B): ``(N, 6, 6)``."""
    scale = 1.0 if k is None else np.asarray(k, dtype=float)[:, None]
    P = (np.eye(3)[None, None] - u[..., :, None] * u[..., None, :]) / wn[
        ..., None, None
    ]
    q = tau[:, None] * GL_W[None, :] * a * scale  # (N, n)
    r = tau[:, None] - s
    J = np.zeros((A.shape[0], 6, 6))
    J[:, 0:3, 0:3] = np.einsum("nk,nkij->nij", q, P)
    J[:, 0:3, 3:6] = np.einsum("nk,nkij->nij", q * s, P)
    J[:, 3:6, 0:3] = np.einsum("nk,nkij->nij", q * r, P)
    J[:, 3:6, 3:6] = np.einsum("nk,nkij->nij", q * r * s, P)
    return J


def _gauge(A, B):
    n = np.maximum(np.linalg.norm(A, axis=-1), 1e-12)[:, None]
    return A / n, B / n


_TABLES = {}


def _table(model, eta):
    """Cumulative thrust integrals ``C1 = int alpha``, ``C2 = int C1`` on a 0.01 s grid."""
    key = (id(model), float(eta))
    tab = _TABLES.get(key)
    if tab is None:
        t = np.arange(0.0, float(model.burn_time) + 10.0, 0.01)
        a = alpha(model, t, eta)
        c1 = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * 0.01)])
        c2 = np.concatenate([[0.0], np.cumsum(0.5 * (c1[1:] + c1[:-1]) * 0.01)])
        tab = (t, c1, c2)
        _TABLES[key] = tab
    return tab


def thrust_integrals(model, t0, tau, eta=ETA_PLAN):
    """``(int_0^tau alpha, int_0^tau (tau - s) alpha)`` from ``t0``, broadcast."""
    t, c1, c2 = _table(model, eta)
    t0 = np.asarray(t0, dtype=float)
    tau = np.asarray(tau, dtype=float)
    c1a = np.interp(t0, t, c1)
    i1 = np.interp(t0 + tau, t, c1) - c1a
    i2 = np.interp(t0 + tau, t, c2) - np.interp(t0, t, c2) - tau * c1a
    return i1, i2


def time_bounds(
    model, t0, p0, v0, bp, bv, v_rel, *, eta=ETA_PLAN, tau_max=8.0, d_tau=0.05
):
    """Grid lower bound on the leg time, the free-arrival time, and the LQ time.

    A leg can take no less than the first ``tau`` at which both the miss and the
    velocity change fit what the thrust can integrate to --
    ``|ZEM| <= int (tau-s) alpha`` and ``|ZEV| <= int alpha`` -- and the first of
    those alone is the free-arrival (constant-direction) time. The LQ time is the
    first ``tau`` whose minimum-energy leg fits the bound everywhere; it is
    admissible, so it bounds the answer from above. ``inf`` where none exists.
    Returns ``(tau_lo, tau_free, tau_lq, A_lq, B_lq)``.
    """
    t0 = np.asarray(t0, dtype=float)
    n = t0.shape[0]
    grid = np.arange(d_tau, tau_max + 0.5 * d_tau, d_tau)
    tau = np.broadcast_to(grid[None, :], (n, grid.size))
    zem, zev = zem_zev(
        tau, p0[:, None], v0[:, None], bp[:, None], bv[:, None], v_rel[:, None]
    )
    i1, i2 = thrust_integrals(model, t0[:, None], tau, eta)
    alive = alpha(model, t0[:, None] + tau, eta) > 0.0
    zem_n = np.linalg.norm(zem, axis=-1)
    fit_r = (zem_n <= i2) & alive
    fit_v = (np.linalg.norm(zev, axis=-1) <= i1) & alive

    def first(mask):
        k = np.argmax(mask, axis=1)
        has = mask[np.arange(n), k]
        return np.where(has, grid[k], np.inf), k, has

    tau_lo, _k, _h = first(fit_r & fit_v)
    tau_free, _k, _h = first(fit_r)
    c0, c1 = lq_coeffs(tau, zem, zev)
    lim = np.minimum(
        alpha(model, t0[:, None], eta), alpha(model, t0[:, None] + tau, eta)
    )
    fit_lq = (
        (np.linalg.norm(c0, axis=-1) <= lim)
        & (np.linalg.norm(c0 + c1 * tau[..., None], axis=-1) <= lim)
        & alive
    )
    tau_lq, k, has = first(fit_lq)
    rows = np.arange(n)
    A = np.where(has[:, None], c0[rows, k], np.nan)
    B = np.where(has[:, None], c1[rows, k], np.nan)
    return tau_lo, tau_free, tau_lq, A, B


def cold_start(
    model, t0, p0, v0, bp, bv, v_rel, *, eta=ETA_PLAN, tau_max=8.0, d_tau=0.05
):
    """LQ minimum feasible time on a grid and its ``(A, B)``; see :func:`time_bounds`."""
    _lo, _free, tau_lq, A, B = time_bounds(
        model, t0, p0, v0, bp, bv, v_rel, eta=eta, tau_max=tau_max, d_tau=d_tau
    )
    return tau_lq, A, B


def free_arrival_time(
    model, t0, p0, v0, bp, bv, *, eta=ETA_PLAN, tau_max=8.0, d_tau=0.05
):
    """Earliest intercept with the arrival velocity free: constant direction.

    ``|ZEM(tau)| <= int_0^tau (tau-s) alpha ds`` -- the thrust ball with the
    exact thrust integral in place of a constant radius. Returns ``(tau, u)``.
    """
    t0 = np.asarray(t0, dtype=float)
    n = t0.shape[0]
    grid = np.arange(d_tau, tau_max + 0.5 * d_tau, d_tau)
    tau = np.broadcast_to(grid[None, :], (n, grid.size))
    zem = (
        bp[:, None]
        + bv[:, None] * tau[..., None]
        - p0[:, None]
        - v0[:, None] * tau[..., None]
        + 0.5 * GRAVITY * tau[..., None] ** 2 * _Z
    )
    _i1, i2 = thrust_integrals(model, t0[:, None], tau, eta)
    ok = (np.linalg.norm(zem, axis=-1) <= i2) & (
        alpha(model, t0[:, None] + tau, eta) > 0.0
    )
    first = np.argmax(ok, axis=1)
    has = ok[np.arange(n), first]
    out = np.where(has, grid[first], np.inf)
    u = zem[np.arange(n), first]
    u = u / np.maximum(np.linalg.norm(u, axis=-1), 1e-12)[:, None]
    return out, u


def free_leg(
    model, t0, p0, v0, bp, bv, *, eta=ETA_PLAN, tau_max=8.0, d_tau=0.05, fixed_tau=None
):
    """Constant-direction leg with the arrival velocity left free.

    The fastest way to reach the balloon at all: ``u`` along the zero-effort
    miss, ``tau`` the first time the thrust integral covers it -- found on the
    grid and then bisected against the exact integral. ``fixed_tau`` instead
    returns the constant-direction leg of that duration (``k = |ZEM| / reach``).
    Returns the same keys as :func:`solve_min_time` plus ``k`` and ``v_arr``.
    """
    t0 = np.asarray(t0, dtype=float)
    p0, v0, bp, bv = (np.asarray(x, dtype=float) for x in (p0, v0, bp, bv))
    n = t0.shape[0]

    def gap(tau):
        zem = (
            bp
            + bv * tau[:, None]
            - p0
            - v0 * tau[:, None]
            + 0.5 * GRAVITY * tau[:, None] ** 2 * _Z
        )
        _i1, i2 = thrust_integrals(model, t0, tau, eta)
        return zem, np.linalg.norm(zem, axis=1) - i2

    if fixed_tau is None:
        tau_hi, _u = free_arrival_time(
            model, t0, p0, v0, bp, bv, eta=eta, tau_max=tau_max, d_tau=d_tau
        )
        ok = np.isfinite(tau_hi)
        hi = np.where(ok, tau_hi, 1.0)
        lo = np.maximum(hi - d_tau, 1e-3)
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            _zem, g = gap(mid)
            fits = g <= 0.0
            hi = np.where(fits, mid, hi)
            lo = np.where(fits, lo, mid)
        tau = hi
    else:
        tau = np.broadcast_to(np.asarray(fixed_tau, dtype=float), (n,)).copy()
        ok = np.ones(n, dtype=bool)
    zem, _g = gap(tau)
    i1, i2 = thrust_integrals(model, t0, tau, eta)
    zn = np.linalg.norm(zem, axis=1)
    u = zem / np.maximum(zn, 1e-12)[:, None]
    k = zn / np.maximum(i2, 1e-12)
    ok &= (k <= 1.0 + 1e-6) & (alpha(model, t0 + tau, eta) > 0.0)
    v_arr = v0 + (k * i1)[:, None] * u - GRAVITY * tau[:, None] * _Z
    return {
        "A": u,
        "B": np.zeros((n, 3)),
        "tau": np.where(ok, tau, np.inf),
        "ok": ok,
        "k": np.minimum(k, 1.0),
        "v_arr": v_arr,
        "res_r": np.zeros(n),
        "res_v": np.zeros(n),
    }


def solve_min_time(
    model,
    t0,
    p0,
    v0,
    bp,
    bv,
    v_rel,
    *,
    eta=ETA_PLAN,
    A=None,
    B=None,
    tau=None,
    iters=10,
    tol_r=0.05,
    tol_v=0.05,
    tau_max=8.0,
):
    """Saturated linear-tangent leg of minimum time, batched.

    ``A, B, tau`` warm-start the solve; without them the LQ grid supplies them.
    Returns a dict with ``A, B`` (``|A| = 1``), ``tau``, ``ok`` (converged, finite,
    inside ``tau_max``) and the final residual norms.
    """
    t0 = np.asarray(t0, dtype=float)
    p0, v0, bp, bv, v_rel = (
        np.asarray(x, dtype=float) for x in (p0, v0, bp, bv, v_rel)
    )
    n = t0.shape[0]
    if tau is None:
        tau_lq, A, B = cold_start(
            model, t0, p0, v0, bp, bv, v_rel, eta=eta, tau_max=tau_max
        )
        tau = 0.95 * tau_lq
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    tau = np.array(tau, dtype=float)
    live = (
        np.isfinite(tau)
        & np.all(np.isfinite(A), axis=1)
        & np.all(np.isfinite(B), axis=1)
    )
    A[~live] = _Z
    B[~live] = 0.0
    tau[~live] = 1.0
    A, B = _gauge(A, B)

    def residual(A, B, tau):
        s, a = _nodes(model, t0, tau, eta)
        Iv, Ir, u, wn = _integrals(A, B, tau, s, a)
        zem, zev = zem_zev(tau, p0, v0, bp, bv, v_rel)
        return np.concatenate([Iv - zev, Ir - zem], axis=1), (s, a, Iv, u, wn)

    R, cache = residual(A, B, tau)
    for _ in range(int(iters)):
        s, a, Iv, u, wn = cache
        J = np.zeros((n, 7, 7))
        J[:, :6, :6] = _jac_AB(A, B, tau, s, a, u, wn)
        wend = A + B * tau[:, None]
        uend = wend / np.maximum(np.linalg.norm(wend, axis=1), 1e-12)[:, None]
        aend = alpha(model, t0 + tau, eta)
        J[:, 0:3, 6] = aend[:, None] * uend - GRAVITY * _Z
        J[:, 3:6, 6] = Iv - (bv - v0 + GRAVITY * tau[:, None] * _Z)
        J[:, 6, 0:3] = A  # gauge: A . A_now stays 1
        rhs = -np.concatenate([R, np.zeros((n, 1))], axis=1)
        try:
            dx = np.linalg.solve(J, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            dx = np.stack(
                [np.linalg.lstsq(J[i], rhs[i], rcond=None)[0] for i in range(n)]
            )
        dx = np.where(np.isfinite(dx), dx, 0.0)
        # Damping: cap the time step, then halve while the residual grows.
        step = np.minimum(1.0, 0.3 * tau / np.maximum(np.abs(dx[:, 6]), 1e-12))
        base = np.linalg.norm(R, axis=1)
        best_A, best_B, best_tau, best_R, best_cache = A, B, tau, R, cache
        accepted = np.zeros(n, dtype=bool)
        for _half in range(4):
            A1 = A + step[:, None] * dx[:, 0:3]
            B1 = B + step[:, None] * dx[:, 3:6]
            tau1 = np.maximum(tau + step * dx[:, 6], MIN_TAU)
            A1, B1 = _gauge(A1, B1)
            R1, cache1 = residual(A1, B1, tau1)
            better = (np.linalg.norm(R1, axis=1) < base) & ~accepted
            best_A = np.where(better[:, None], A1, best_A)
            best_B = np.where(better[:, None], B1, best_B)
            best_tau = np.where(better, tau1, best_tau)
            best_R = np.where(better[:, None], R1, best_R)
            best_cache = tuple(
                np.where(better.reshape((-1,) + (1,) * (c1.ndim - 1)), c1, c0)
                for c0, c1 in zip(best_cache, cache1)
            )
            accepted |= better
            if accepted.all():
                break
            step = np.where(accepted, step, 0.5 * step)
        A, B, tau, R, cache = best_A, best_B, best_tau, best_R, best_cache
    rv = np.linalg.norm(R[:, 0:3], axis=1)
    rr = np.linalg.norm(R[:, 3:6], axis=1)
    ok = live & (rr < tol_r) & (rv < tol_v) & (tau <= tau_max) & (tau >= MIN_TAU)
    ok &= alpha(model, t0 + tau, eta) > 0.0
    return {"A": A, "B": B, "tau": tau, "ok": ok, "res_r": rr, "res_v": rv}


def solve_fixed_time(
    model,
    t0,
    p0,
    v0,
    bp,
    bv,
    v_rel,
    tau,
    *,
    eta=ETA_PLAN,
    A=None,
    B=None,
    k=None,
    iters=10,
    tol_r=0.05,
    tol_v=0.05,
):
    """Linear-tangent leg of fixed duration: unknowns ``A, B`` and throttle ``k``.

    ``k <= 1`` is feasible at this duration; ``k > 1`` says how far over the
    thrust bound the leg would need to go.
    """
    t0 = np.asarray(t0, dtype=float)
    p0, v0, bp, bv, v_rel = (
        np.asarray(x, dtype=float) for x in (p0, v0, bp, bv, v_rel)
    )
    tau = np.asarray(tau, dtype=float)
    n = t0.shape[0]
    zem, zev = zem_zev(tau, p0, v0, bp, bv, v_rel)
    s, a = _nodes(model, t0, tau, eta)
    if A is None:
        c0, c1 = lq_coeffs(tau, zem, zev)
        A, B = c0, c1
        amean = (GL_W[None, :] * a).sum(axis=1)
        k = np.maximum(
            np.linalg.norm(c0 + 0.5 * c1 * tau[:, None], axis=1), 1e-3
        ) / np.maximum(amean, 1e-9)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    k = np.array(k, dtype=float)
    A, B = _gauge(A, B)

    def residual(A, B, k):
        Iv, Ir, u, wn = _integrals(A, B, tau, s, a, k)
        return np.concatenate([Iv - zev, Ir - zem], axis=1), (Iv, Ir, u, wn)

    R, cache = residual(A, B, k)
    for _ in range(int(iters)):
        Iv, Ir, u, wn = cache
        J = np.zeros((n, 7, 7))
        J[:, :6, :6] = _jac_AB(A, B, tau, s, a, u, wn, k)
        kk = np.maximum(k, 1e-9)[:, None]
        J[:, 0:3, 6] = Iv / kk
        J[:, 3:6, 6] = Ir / kk
        J[:, 6, 0:3] = A
        rhs = -np.concatenate([R, np.zeros((n, 1))], axis=1)
        try:
            dx = np.linalg.solve(J, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            dx = np.stack(
                [np.linalg.lstsq(J[i], rhs[i], rcond=None)[0] for i in range(n)]
            )
        dx = np.where(np.isfinite(dx), dx, 0.0)
        step = np.ones(n)
        base = np.linalg.norm(R, axis=1)
        accepted = np.zeros(n, dtype=bool)
        nA, nB, nk, nR, ncache = A, B, k, R, cache
        for _half in range(4):
            A1, B1 = _gauge(
                A + step[:, None] * dx[:, 0:3], B + step[:, None] * dx[:, 3:6]
            )
            k1 = np.maximum(k + step * dx[:, 6], 1e-6)
            R1, cache1 = residual(A1, B1, k1)
            better = (np.linalg.norm(R1, axis=1) < base) & ~accepted
            nA = np.where(better[:, None], A1, nA)
            nB = np.where(better[:, None], B1, nB)
            nk = np.where(better, k1, nk)
            nR = np.where(better[:, None], R1, nR)
            ncache = tuple(
                np.where(better.reshape((-1,) + (1,) * (c1.ndim - 1)), c1, c0)
                for c0, c1 in zip(ncache, cache1)
            )
            accepted |= better
            if accepted.all():
                break
            step = np.where(accepted, step, 0.5 * step)
        A, B, k, R, cache = nA, nB, nk, nR, ncache
    rv = np.linalg.norm(R[:, 0:3], axis=1)
    rr = np.linalg.norm(R[:, 3:6], axis=1)
    ok = (rr < tol_r) & (rv < tol_v)
    return {"A": A, "B": B, "k": k, "ok": ok, "res_r": rr, "res_v": rv}


def solve_leg(
    model,
    t0,
    p0,
    v0,
    bp,
    bv,
    v_rel,
    *,
    eta=ETA_PLAN,
    tau_max=8.0,
    iters=8,
    exact=False,
    d_tau=0.05,
    max_steps=None,
):
    """Minimum-time leg: bounds, Newton, and a window search where Newton cannot vouch.

    Newton on ``tau`` is fast and usually right, but two things defeat it. The LQ
    leg spends thrust evenly rather than at the bound, so a fast pinned arrival
    often has no LQ time to start from. And the fixed-time throttle ``k(tau)``
    is not monotone: pinning the arrival velocity makes the feasible durations a
    window -- a single instant when the pinned velocity is the free-arrival one
    -- so a root found above the window's floor, or a bracket assuming "later is
    easier", both miss it.

    So: (1) :func:`time_bounds` rejects legs with no feasible time before any
    solve and gives the floor ``tau_lo``; (2) Newton from the LQ leg, accepted
    when it converges at or above the floor; (3) for the rest (all legs when
    ``exact``) ``k(tau)`` is followed up from the floor by continuation, stopping
    at the first ``k <= 1`` or once ``k`` has turned up past a minimum above 1;
    three bisection steps then tighten the window's floor. The returned leg meets
    its boundary conditions at a throttle ``k <= 1`` (reported as ``k``).
    """
    t0 = np.asarray(t0, dtype=float)
    p0, v0, bp, bv, v_rel = (
        np.asarray(x, dtype=float) for x in (p0, v0, bp, bv, v_rel)
    )
    n = t0.shape[0]
    tau_lo, _free, tau_lq, A0, B0 = time_bounds(
        model, t0, p0, v0, bp, bv, v_rel, eta=eta, tau_max=tau_max, d_tau=d_tau
    )
    sol = {
        "A": np.tile(_Z, (n, 1)),
        "B": np.zeros((n, 3)),
        "tau": np.full(n, np.inf),
        "ok": np.zeros(n, dtype=bool),
        "res_r": np.full(n, np.inf),
        "res_v": np.full(n, np.inf),
        "k": np.ones(n),
        "tau_lq": tau_lq,
        "tau_lo": tau_lo,
    }
    possible = np.flatnonzero(np.isfinite(tau_lo))
    if possible.size == 0:
        return sol
    a, b, c, d, e, f = (x[possible] for x in (t0, p0, v0, bp, bv, v_rel))
    lo = tau_lo[possible]
    # Newton, from the LQ leg where there is one, else from a stretch of the floor.
    start = np.where(
        np.isfinite(tau_lq[possible]), tau_lq[possible], np.minimum(1.2 * lo, tau_max)
    )
    zem, zev = zem_zev(start, b, c, d, e, f)
    cA, cB = lq_coeffs(start, zem, zev)
    nt = solve_min_time(
        model,
        a,
        b,
        c,
        d,
        e,
        f,
        eta=eta,
        A=cA,
        B=cB,
        tau=start,
        iters=iters,
        tau_max=tau_max,
    )
    lq_here = tau_lq[possible]
    good = nt["ok"] & (nt["tau"] >= lo - d_tau - 1e-9)
    # An LQ time is admissible, so a root above it is not the fastest leg.
    good &= ~np.isfinite(lq_here) | (nt["tau"] <= lq_here + 1e-9)
    rows = possible[good]
    for key in ("A", "B", "tau", "res_r", "res_v"):
        sol[key][rows] = nt[key][good]
    sol["ok"][rows] = True
    scan_local = np.flatnonzero(~good | bool(exact))
    if scan_local.size == 0:
        return sol
    idx0 = scan_local
    m = idx0.size
    aa, bb, cc, dd, ee, ff = (x[idx0] for x in (a, b, c, d, e, f))
    floor = np.maximum(lo[idx0] - d_tau, MIN_TAU)
    lq_s = lq_here[idx0]
    zem, zev = zem_zev(floor, bb, cc, dd, ee, ff)
    wA, wB = lq_coeffs(floor, zem, zev)
    wk = np.ones(m)
    kmin = np.full(m, np.inf)
    hit = np.zeros(m, dtype=bool)
    live = np.ones(m, dtype=bool)
    tau_prev = floor.copy()
    tau_hit = np.full(m, np.inf)
    hA, hB, hk = np.tile(_Z, (m, 1)), np.zeros((m, 3)), np.ones(m)
    steps = int(np.ceil(tau_max / d_tau)) if max_steps is None else int(max_steps)
    for j in range(steps):
        tau = floor + j * d_tau
        # k(tau_lq) <= 1 always (the LQ leg is admissible), so the walk ends there.
        cap = np.minimum(tau_max, np.where(np.isfinite(lq_s), lq_s + 1e-9, tau_max))
        todo = np.flatnonzero(live & ~hit & (tau <= cap))
        if todo.size == 0:
            break
        args = (aa[todo], bb[todo], cc[todo], dd[todo], ee[todo], ff[todo], tau[todo])
        warm = solve_fixed_time(
            model, *args, eta=eta, A=wA[todo], B=wB[todo], k=wk[todo], iters=5
        )
        redo = ~warm["ok"]
        if redo.any():
            r = np.flatnonzero(redo)
            cold = solve_fixed_time(model, *(x[r] for x in args), eta=eta, iters=6)
            for key in ("A", "B", "k", "ok", "res_r"):
                warm[key][r] = cold[key]
        ok1 = warm["ok"]
        upd = todo[ok1]
        wA[upd], wB[upd], wk[upd] = warm["A"][ok1], warm["B"][ok1], warm["k"][ok1]
        got = ok1 & (warm["k"] <= 1.0)
        sel = todo[got]
        hit[sel] = True
        tau_hit[sel], hA[sel], hB[sel], hk[sel] = (
            tau[sel],
            warm["A"][got],
            warm["B"][got],
            warm["k"][got],
        )
        tau_prev[todo[~got]] = tau[todo[~got]]
        kk = np.where(ok1, warm["k"], np.inf)
        kmin[todo] = np.minimum(kmin[todo], kk)
        # Past a minimum above 1 and climbing: no window ahead worth the walk.
        rising = ok1 & (kmin[todo] > 1.0) & (warm["k"] > kmin[todo] + 0.25)
        live[todo[rising]] = False
    lo_b = np.where(hit, np.maximum(tau_hit - d_tau, floor), floor)
    hi_b = tau_hit.copy()
    for _ in range(3):
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            break
        mid = 0.5 * (lo_b[idx] + hi_b[idx])
        fm = solve_fixed_time(
            model,
            aa[idx],
            bb[idx],
            cc[idx],
            dd[idx],
            ee[idx],
            ff[idx],
            mid,
            eta=eta,
            A=hA[idx],
            B=hB[idx],
            k=hk[idx],
            iters=5,
        )
        feas = fm["ok"] & (fm["k"] <= 1.0)
        up = idx[feas]
        hi_b[up], hA[up], hB[up], hk[up] = (
            mid[feas],
            fm["A"][feas],
            fm["B"][feas],
            fm["k"][feas],
        )
        lo_b[idx[~feas]] = mid[~feas]
    for i in range(m):
        row = possible[idx0[i]]
        if not hit[i]:
            continue
        if sol["ok"][row] and sol["tau"][row] <= hi_b[i] + 1e-6:
            continue  # Newton's root is already the floor
        sol["A"][row], sol["B"][row], sol["tau"][row] = hA[i], hB[i], hi_b[i]
        sol["k"][row] = hk[i]
        sol["res_r"][row] = 0.0
        sol["res_v"][row] = 0.0
        sol["ok"][row] = True
    return sol


def direction(A, B, s):
    """Thrust direction ``normalize(A + B*s)``; ``s`` is ``(N,)`` or ``(N, K)``."""
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        w = A + B * s[:, None]
    else:
        w = A[:, None, :] + B[:, None, :] * s[..., None]
    return w / np.maximum(np.linalg.norm(w, axis=-1), 1e-12)[..., None]


def rate_profile(A, B, tau):
    """``(peak direction rate rad/s, min |A+B s|/|A|, s at the peak)`` over the leg.

    ``|du/ds| = |A x B| / |A + B s|^2`` peaks where ``|A + B s|`` is smallest,
    at ``s* = clip(-A.B / |B|^2, 0, tau)``; a near-zero minimum is a flip.
    """
    bb = np.maximum((B * B).sum(axis=1), 1e-18)
    s_star = np.clip(-(A * B).sum(axis=1) / bb, 0.0, np.asarray(tau, dtype=float))
    w = A + B * s_star[:, None]
    wn2 = np.maximum((w * w).sum(axis=1), 1e-18)
    cross = np.linalg.norm(np.cross(A, B), axis=1)
    an = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    return cross / wn2, np.sqrt(wn2) / an, s_star


def profile(model, t0, p0, v0, A, B, tau, *, k=None, eta=ETA_PLAN, n=16):
    """Sample a leg: ``(s, p, v, u)`` with shapes ``(N, n+1)`` and ``(N, n+1, 3)``.

    Trapezoid cumulative integration on a uniform grid -- good to centimetres at
    ``n = 16`` for the smooth profiles here, and used for constraint checks only.
    """
    t0 = np.asarray(t0, dtype=float)
    tau = np.asarray(tau, dtype=float)
    frac = np.linspace(0.0, 1.0, int(n) + 1)
    s = tau[:, None] * frac[None, :]
    u = direction(A, B, s)
    scale = 1.0 if k is None else np.asarray(k, dtype=float)[:, None, None]
    f = scale * alpha(model, t0[:, None] + s, eta)[..., None] * u - GRAVITY * _Z
    h = (tau / int(n))[:, None, None]
    dv = 0.5 * (f[:, 1:] + f[:, :-1]) * h
    v = np.concatenate(
        [
            np.asarray(v0, float)[:, None],
            np.asarray(v0, float)[:, None] + np.cumsum(dv, axis=1),
        ],
        axis=1,
    )
    dp = 0.5 * (v[:, 1:] + v[:, :-1]) * h
    p = np.concatenate(
        [
            np.asarray(p0, float)[:, None],
            np.asarray(p0, float)[:, None] + np.cumsum(dp, axis=1),
        ],
        axis=1,
    )
    return s, p, v, u


def solve_two_leg(
    model,
    t0,
    p0,
    v0,
    bp1,
    bv1,
    bp2,
    bv2,
    *,
    eta=ETA_PLAN,
    init=None,
    iters=12,
    tol=0.05,
    tau_max=10.0,
):
    """Minimum-time pass through balloon 1 then balloon 2, velocities free.

    The passing velocity at balloon 1 is not chosen: it falls out of the
    necessary conditions. Across a free-velocity waypoint the velocity costate is
    continuous (so is the thrust direction) and the position costate may jump,
    by an amount orthogonal to the velocity there (Hamiltonian continuity); the
    final velocity is free, so the last leg's direction is constant. With
    ``u1 = normalize(A + B s)`` on leg 1 and ``u2 = c`` on leg 2:

        p(balloon 1 time) = balloon 1,  p(end) = balloon 2
        A + B*tau1 = c * tau2              (costate continuity)
        (B + c) . v(tau1) = 0              (Hamiltonian continuity)
        A . A_now = 1                      (scale)

    Eleven equations in ``A, B, c, tau1, tau2``; damped Newton with a forward-
    difference Jacobian, batched. ``init`` is ``(A, B, c, tau1, tau2)``; without
    it the free-arrival legs to 1 and then 2 seed it. Returns a dict with the
    solution, ``ok`` and ``v1`` (the passing velocity at balloon 1).
    """
    t0 = np.asarray(t0, dtype=float)
    p0, v0, bp1, bv1, bp2, bv2 = (
        np.asarray(x, dtype=float) for x in (p0, v0, bp1, bv1, bp2, bv2)
    )
    n = t0.shape[0]
    if init is None:
        f1 = free_leg(model, t0, p0, v0, bp1, bv1, eta=eta, tau_max=tau_max)
        tau1 = np.where(f1["ok"], f1["tau"], 2.0)
        p1 = bp1 + bv1 * tau1[:, None]
        v1 = np.where(f1["ok"][:, None], f1["v_arr"], v0)
        f2 = free_leg(
            model,
            t0 + tau1,
            p1,
            v1,
            bp2 + bv2 * tau1[:, None],
            bv2,
            eta=eta,
            tau_max=tau_max,
        )
        tau2 = np.where(f2["ok"], f2["tau"], 2.0)
        u1, u2 = f1["A"], f2["A"]
        c = u2 / np.maximum(tau2, 1e-3)[:, None]
        A = u1
        B = (c * tau2[:, None] - A) / np.maximum(tau1, 1e-3)[:, None]
    else:
        A, B, c, tau1, tau2 = (np.array(x, dtype=float) for x in init)
    x = np.concatenate([A, B, c, tau1[:, None], tau2[:, None]], axis=1)  # (n, 11)

    def residual(x):
        A, B, c = x[:, 0:3], x[:, 3:6], x[:, 6:9]
        tau1 = np.maximum(x[:, 9], MIN_TAU)
        tau2 = np.maximum(x[:, 10], MIN_TAU)
        s1, a1 = _nodes(model, t0, tau1, eta)
        Iv1, Ir1, _u, _wn = _integrals(A, B, tau1, s1, a1)
        v1 = v0 + Iv1 - GRAVITY * tau1[:, None] * _Z
        p1 = p0 + v0 * tau1[:, None] + Ir1 - 0.5 * GRAVITY * (tau1**2)[:, None] * _Z
        s2, a2 = _nodes(model, t0 + tau1, tau2, eta)
        i1, i2 = thrust_integrals(model, t0 + tau1, tau2, eta)
        u2 = c / np.maximum(np.linalg.norm(c, axis=1), 1e-12)[:, None]
        p2 = (
            p1
            + v1 * tau2[:, None]
            + i2[:, None] * u2
            - 0.5 * GRAVITY * (tau2**2)[:, None] * _Z
        )
        r_p1 = p1 - (bp1 + bv1 * tau1[:, None])
        r_p2 = p2 - (bp2 + bv2 * (tau1 + tau2)[:, None])
        r_c = (A + B * tau1[:, None]) - c * tau2[:, None]
        r_h = ((B + c) * v1).sum(axis=1, keepdims=True)
        return np.concatenate([r_p1, r_p2, r_c, r_h], axis=1), v1

    for _ in range(int(iters)):
        R, _v1 = residual(x)
        norm0 = np.linalg.norm(R[:, :6], axis=1) + np.linalg.norm(R[:, 6:], axis=1)
        J = np.zeros((n, 11, 11))
        h = 1e-5 * np.maximum(np.abs(x), 1.0)
        for k in range(11):
            xk = x.copy()
            xk[:, k] += h[:, k]
            Rk, _ = residual(xk)
            J[:, :10, k] = (Rk - R) / h[:, k : k + 1]
        J[:, 10, 0:3] = x[:, 0:3]  # scale: A . A_now fixed
        rhs = -np.concatenate([R, np.zeros((n, 1))], axis=1)
        try:
            dx = np.linalg.solve(J, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            dx = np.stack(
                [np.linalg.lstsq(J[i], rhs[i], rcond=None)[0] for i in range(n)]
            )
        dx = np.where(np.isfinite(dx), dx, 0.0)
        step = np.minimum(
            1.0,
            0.3
            * np.minimum(x[:, 9], x[:, 10])
            / np.maximum(np.maximum(np.abs(dx[:, 9]), np.abs(dx[:, 10])), 1e-12),
        )
        accepted = np.zeros(n, dtype=bool)
        best = x.copy()
        for _h in range(4):
            xt = x + step[:, None] * dx
            xt[:, 9:11] = np.maximum(xt[:, 9:11], MIN_TAU)
            an = np.maximum(np.linalg.norm(xt[:, 0:3], axis=1), 1e-12)[:, None]
            xt[:, 0:9] = xt[:, 0:9] / an
            Rt, _ = residual(xt)
            nt = np.linalg.norm(Rt[:, :6], axis=1) + np.linalg.norm(Rt[:, 6:], axis=1)
            better = (nt < norm0) & ~accepted
            best[better] = xt[better]
            accepted |= better
            if accepted.all():
                break
            step = np.where(accepted, step, 0.5 * step)
        x = best
    R, v1 = residual(x)
    ok = (np.linalg.norm(R[:, 0:3], axis=1) < tol) & (
        np.linalg.norm(R[:, 3:6], axis=1) < tol
    )
    # Costate rows are in the scale of |A| = 1: absolute tolerances.
    scale_h = (
        np.linalg.norm(x[:, 3:6], axis=1) + np.linalg.norm(x[:, 6:9], axis=1)
    ) * np.maximum(np.linalg.norm(v1, axis=1), 1.0)
    ok &= (np.linalg.norm(R[:, 6:9], axis=1) < 1e-3) & (
        np.abs(R[:, 9]) < 1e-3 * np.maximum(scale_h, 1.0)
    )
    ok &= (
        (x[:, 9] > MIN_TAU) & (x[:, 10] > MIN_TAU) & (x[:, 9] + x[:, 10] <= 2 * tau_max)
    )
    ok &= alpha(model, t0 + x[:, 9] + x[:, 10], eta) > 0.0
    return {
        "A": x[:, 0:3],
        "B": x[:, 3:6],
        "c": x[:, 6:9],
        "tau1": x[:, 9],
        "tau2": x[:, 10],
        "ok": ok,
        "v1": v1,
        "res": np.linalg.norm(R, axis=1),
    }
