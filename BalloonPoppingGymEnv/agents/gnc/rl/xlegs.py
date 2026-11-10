"""Moving-waypoint legs with a continuous thrust-direction sweep (drone-racing style: pass the balloon at a chosen time
while already turning toward the next one), for the beam planner's x rows (BeamPlanner.enable_x).

Leg to balloon b with pass time tk and exit thrust direction e: the thrust direction follows
    u(t) = q / |q|,  q = e + nu (tk - t),
a sweep along one great circle that ends on e at the pass (the time-optimal leg structure). For a fixed (tk, e) the
vector nu is the minimiser of the convex dual
    Psi(nu) = int_0^tau a_T(tk - s) |e + nu s| ds - Dp . nu + Rp |nu|,   Dp = b(tk) - p - v tau + g tau^2 / 2 z,
whose gradient is (thrust displacement) - Dp + Rp nu / |nu|: at the optimum the pass point lies Rp from the centre and
the leg maximises e . v(tk) over all thrust programmes that pass within Rp at tk. Point mass, full thrust (a_T = T / m
from the given thrust curve and mass), no aerodynamic force: the planner re-solves nu from the current state at every
decision step (feedback), and GivenSim scores the flown result with aero and the actuator lags.
Re-pointing: before the sweep the current nose is held for d = tau0 + angle(nose, u_start) / wr (the turn model of the
planner's viable-intercept rows); the command is the sweep direction at t + d (+ lead). Given parameters only.
"""

from __future__ import annotations

import numpy as np

G = 9.80665


def _unit(x):
    return x / np.maximum(np.sqrt((x * x).sum(-1)), 1e-12)[..., None]


def solve(
    p0, v0, t0, tk, B, e, Rp, a_fn, M=16, iters=10, tol=0.02, nu0=None, ls=6, rate=False
):
    """Rows: start (p0, v0 (n,3), t0 (n,)), pass time tk (n,), balloon centre B (n,3) at tk, unit exit direction e (n,3),
    pass radius Rp (scalar or (n,)). a_fn(t (n,M)) -> thrust acceleration at those absolute times.
    Returns (nu (n,3), ok (n,), v_arr (n,3), u_start (n,3) sweep direction at the leg start) [+ max sweep rate (n,) rad/s
    with rate=True]."""
    n = len(tk)
    Rp = np.broadcast_to(np.asarray(Rp, float), (n,))
    tau = np.maximum(tk - t0, 1e-3)
    fr = (np.arange(M) + 0.5) / M
    s = tau[:, None] * fr[None]  # time before the pass at each node, ascending
    aw = a_fn(tk[:, None] - s) * (tau / M)[:, None]
    I = (aw * s).sum(1)  # noqa: E741  thrust displacement capacity
    Dp = B - p0 - v0 * tau[:, None]
    Dp[:, 2] += 0.5 * G * tau**2
    dn = np.sqrt((Dp * Dp).sum(1))
    reach = (dn <= I + Rp - 1e-9) & (tk - t0 > 1e-3)
    inside = (
        np.sqrt(((I[:, None] * e - Dp) ** 2).sum(1)) <= Rp
    )  # constant e passes inside the sphere: nu = 0
    nu = np.where(
        reach[:, None], Dp / np.maximum(dn, 1e-9)[:, None] * (2.0 / tau)[:, None], 0.0
    )
    if nu0 is not None:
        good0 = np.isfinite(nu0).all(1)
        nz = np.where(good0[:, None], nu0, 0.0)
        good0 &= np.sqrt((nz * nz).sum(1)) * tau < 200.0
        nu = np.where((reach & good0)[:, None], nz, nu)
    nu[inside] = 0.0
    act = reach & ~inside
    eye = np.eye(3)[None]

    def psi(nu_, idx):
        q = e[idx][:, None, :] + nu_[:, None, :] * s[idx][..., None]
        return (
            (aw[idx] * np.sqrt((q * q).sum(-1))).sum(1)
            - (Dp[idx] * nu_).sum(1)
            + Rp[idx] * np.sqrt((nu_ * nu_).sum(1))
        )

    for _ in range(iters):
        idx = np.flatnonzero(act)
        if len(idx) == 0:
            break
        nu_i = nu[idx]
        si = s[idx]
        q = e[idx][:, None, :] + nu_i[:, None, :] * si[..., None]
        qn = np.sqrt((q * q).sum(-1))
        u = q / np.maximum(qn, 1e-12)[..., None]
        nn = np.sqrt((nu_i * nu_i).sum(1))
        nh = nu_i / np.maximum(nn, 1e-12)[:, None]
        grad = (
            (aw[idx][..., None] * si[..., None] * u).sum(1)
            - Dp[idx]
            + Rp[idx][:, None] * nh
        )
        done = np.sqrt((grad * grad).sum(1)) < tol
        act[idx[done]] = False
        keep = ~done
        if not keep.any():
            break
        idx, nu_i, u, qn, grad, nn, nh, si = (
            idx[keep],
            nu_i[keep],
            u[keep],
            qn[keep],
            grad[keep],
            nn[keep],
            nh[keep],
            si[keep],
        )
        cw = aw[idx] * si**2 / np.maximum(qn, 1e-9)
        H = -np.einsum("km,kmi,kmj->kij", cw, u, u)
        H += (cw.sum(1) * (1.0 + 1e-6))[:, None, None] * eye
        rn = Rp[idx] / np.maximum(nn, 1e-9)
        H += rn[:, None, None] * (eye - nh[:, :, None] * nh[:, None, :])
        fin = np.isfinite(H).all((1, 2)) & np.isfinite(grad).all(1)
        H = np.where(fin[:, None, None], H, eye)
        grad = np.where(fin[:, None], grad, 0.0)
        step = -np.linalg.solve(H, grad[..., None])[..., 0]
        cap = (
            200.0 / tau[idx]
        )  # beyond |nu| tau = 200 the sweep is a constant direction to 1e-4
        nn1 = np.sqrt(((nu_i + step) ** 2).sum(1))
        step *= np.minimum(1.0, cap / np.maximum(nn1, 1e-12))[:, None]
        act[idx[~fin]] = False
        f0 = psi(nu_i, idx)
        dec = (grad * step).sum(1)
        alpha = np.ones(len(idx))
        acc = np.zeros(len(idx), bool)
        for _ in range(ls):
            trial = nu_i + alpha[:, None] * step
            okk = (psi(trial, idx) <= f0 + 1e-4 * alpha * dec) & ~acc
            nu[idx[okk]] = trial[okk]
            acc |= okk
            if acc.all():
                break
            alpha = np.where(acc, alpha, 0.5 * alpha)
        na = ~acc
        nu[idx[na]] = nu_i[na] + alpha[na, None] * step[na]
    q = e[:, None, :] + nu[:, None, :] * s[..., None]
    u = q / np.maximum(np.sqrt((q * q).sum(-1)), 1e-12)[..., None]
    disp = (aw[..., None] * s[..., None] * u).sum(1)
    passd = np.sqrt(((disp - Dp) ** 2).sum(1))
    ok = reach & (passd <= Rp + tol + 1e-6) & np.isfinite(nu).all(1)
    v_arr = v0 + (aw[..., None] * u).sum(1)
    v_arr[:, 2] -= G * tau
    if rate:  # |du/dt| = |nu - (u . nu) u| / |q| at the nodes and at the pass (q = e)
        perp = nu[:, None, :] - (u * nu[:, None, :]).sum(-1)[..., None] * u
        r = np.sqrt((perp * perp).sum(-1)) / np.maximum(np.sqrt((q * q).sum(-1)), 1e-9)
        pe = nu - (e * nu).sum(1, keepdims=True) * e
        return nu, ok, v_arr, u[:, -1], np.maximum(r.max(1), np.sqrt((pe * pe).sum(1)))
    return nu, ok, v_arr, u[:, -1]


def tmin(p, v, t, nose, B_fn, a_fn, Rp, wr, tau0, T_max=6.0, dT=0.05, h=0.01):
    """Earliest pass time T (s after t, on a dT grid) of balloon B_fn(times (n,J)) -> (n,J,3) for a point mass at full
    thrust that holds the nose n0 for d = tau0 + angle(n0, Dp) / wr (thrust along n0) and may then thrust in any
    direction: |Dp - n0 H(d)| <= C(d) + Rp, H / C = displacement capacity of the hold / of the rest. nan where none."""
    n = len(t)
    T = np.arange(1, int(round(T_max / dT)) + 1) * dT  # (J,)
    K = int(np.ceil(T_max / h)) + 1
    am = a_fn(t[:, None] + (np.arange(K) + 0.5)[None] * h)  # (n,K)
    A1 = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(am * h, 1)], 1
    )  # A1(x) = int_0^x a
    A2 = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(am * ((np.arange(K) + 0.5) * h)[None] * h, 1)], 1
    )  # int_0^x a s

    def at(
        A, x
    ):  # linear interpolation of the cumulative tables at x (n,J) seconds; nan where x is nan
        bad = np.isnan(
            x
        )  # a nan nose or track point must give "no pass", not an out-of-range index
        f = np.clip(np.where(bad, 0.0, x) / h, 0, K)
        i = np.minimum(np.floor(f).astype(int), K - 1)
        w = f - i
        r = np.arange(n)[:, None]
        return np.where(bad, np.nan, A[r, i] * (1 - w) + A[r, i + 1] * w)

    Bt = B_fn(t[:, None] + T[None])
    Dp = Bt - p[:, None, :] - v[:, None, :] * T[None, :, None]
    Dp[:, :, 2] += 0.5 * G * T[None] ** 2
    dn = np.sqrt((Dp * Dp).sum(2))
    c = np.clip((Dp * nose[:, None, :]).sum(2) / np.maximum(dn, 1e-9), -1.0, 1.0)
    d = np.minimum(tau0 + np.arccos(c) / wr, T[None])
    Tt = np.broadcast_to(T[None], d.shape)
    Hd = Tt * at(A1, d) - at(A2, d)  # int_0^d a (T - s) ds
    Ct = Tt * at(A1, Tt) - at(A2, Tt) - Hd
    r = Dp - nose[:, None, :] * Hd[..., None]
    ok = np.sqrt((r * r).sum(2)) <= Ct + Rp
    any_ = ok.any(1)
    return np.where(any_, T[np.argmax(ok, 1)], np.nan)


def hold(p, v, t, d, n0, a_fn, K=4):
    """State after holding thrust along n0 for d s from (p, v, t) (K-node midpoint rule)."""
    fr = (np.arange(K) + 0.5) / K
    sh = d[:, None] * fr[None]
    aw = a_fn(t[:, None] + sh) * (d / K)[:, None]
    J = aw.sum(1)
    Ih = (aw * (d[:, None] - sh)).sum(1)
    p1 = p + v * d[:, None] + n0 * Ih[:, None]
    p1[:, 2] -= 0.5 * G * d**2
    v1 = v + n0 * J[:, None]
    v1[:, 2] -= G * d
    return p1, v1


def command(
    p, v, nose, t, tk, B, e, nu, Rp, a_fn, wr, tau0, lead=0.0, t_end=0.1, M=16, iters=10
):
    """Feedback form of the leg: re-solve nu from the current state (warm start nu) and return (unit thrust direction
    (n,3), ok (n,), nu (n,3) with the ok rows updated). Rows within t_end of the pass command e; rows past the pass or
    whose solve fails are not ok (the caller flies its fallback law)."""
    tau = tk - t
    out = e.copy()
    ok = tau > 0.0
    nu = nu.copy()
    live = np.flatnonzero(tau > t_end)
    if len(live) == 0:
        return out, ok, nu
    pl, vl, nl, tl, tkl, el = p[live], v[live], nose[live], t[live], tk[live], e[live]
    nw = nu[live]
    q0 = el + np.where(np.isfinite(nw), nw, 0.0) * tau[live][:, None]
    c = np.clip((nl * _unit(q0)).sum(1), -1.0, 1.0)
    d = np.minimum(tau0 + np.arccos(c) / wr, np.maximum(tau[live] - t_end, 0.0))
    p1, v1 = hold(pl, vl, tl, d, nl, a_fn)
    nus, oks, _, _ = solve(
        p1, v1, tl + d, tkl, B[live], el, Rp, a_fn, M=M, iters=iters, nu0=nw
    )
    sc = np.maximum(tkl - (tl + d + lead), 0.0)
    out[live] = np.where(
        oks[:, None], _unit(el + np.where(oks[:, None], nus, 0.0) * sc[:, None]), el
    )
    ok[live] = oks
    nu[live[oks]] = nus[oks]
    return out, ok, nu


def next_dir(xa, va, ta, bn_fn, a_fn, Rp, W=50, dtg=0.1):
    """Look-ahead exit direction: the thrust direction d' = Dp'/|Dp'| of the earliest constant-direction full-thrust
    pass of the next balloon from the arrival state (xa, va) at ta (rows), on a dtg grid of W steps. bn_fn(times (n,W))
    -> next-balloon positions (n,W,3). Returns (d' (n,3), found (n,))."""
    n = len(ta)
    tg = (np.arange(W) + 1) * dtg
    sm = (np.arange(W) + 0.5) * dtg
    a = a_fn(ta[:, None] + sm[None])
    A1 = np.cumsum(a * dtg, 1)
    A2 = np.cumsum(a * sm[None] * dtg, 1)
    I = tg[None] * A1 - A2  # noqa: E741
    bp = bn_fn(ta[:, None] + tg[None])
    Dp = bp - xa[:, None, :] - va[:, None, :] * tg[None, :, None]
    Dp[:, :, 2] += 0.5 * G * tg[None] ** 2
    dn = np.sqrt((Dp * Dp).sum(2))
    okk = dn <= I + Rp
    found = okk.any(1)
    i0 = np.where(found, okk.argmax(1), 0)
    d = Dp[np.arange(n), i0]
    return _unit(d), found


def floor_dirs(n, h_agl, vz):
    """The ground-safety floor of planner.guidance (the same rules), for directions computed outside it."""
    t_imp = np.where(vz < -0.5, h_agl / np.maximum(-vz, 1e-3), np.inf)
    el_min = np.where(
        (h_agl < 15.0) | (t_imp < 1.5),
        np.radians(60.0),
        np.where((h_agl < 30.0) | (t_imp < 3.0), np.radians(35.0), -np.pi / 2),
    )
    el = np.arcsin(np.clip(n[:, 2], -1, 1))
    fix = el < el_min
    if fix.any():
        n = n.copy()
        hz = np.arctan2(n[fix, 1], n[fix, 0])
        em = el_min[fix]
        n[fix] = np.stack(
            [np.cos(em) * np.cos(hz), np.cos(em) * np.sin(hz), np.sin(em)], 1
        )
    return n
