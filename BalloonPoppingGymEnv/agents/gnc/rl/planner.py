"""On-board rollout beam planner (receding horizon) over balloon target sequences.

Each candidate "fly to balloon b" segment is simulated with the SAME guidance law the agent executes (``guidance``)
through the given-only simulator (``GivenSim``), so a plan is what the rocket will actually fly. Every step of every
rollout runs the env's swept-segment pop test against all released balloons, so incidental pops count. Balloons follow
the predictor's tracks (``BeamPlanner.track``, observations only); not-yet-released balloons are not targets.

Pattern: receding-horizon beam search over discrete target choices (policy-guided beam search; tree search with a rollout
policy, cf. MCTS rollouts in Silver et al. 2016); the leaves are scored by depth-discounted pops, a time cost, a learned
critic and a reach term.
"""

from __future__ import annotations

import numpy as np

from . import features as F
from ..follow.ff_ref import FF_DIM, _clip, _cross, _l3, _l3sq, _norm, _nz, _sum3
from .given_sim import quat_rot

G = 9.80665
_EZ = np.array([0.0, 0.0, 1.0])


def _any(x):
    """np.any(x) for the scalars / row arrays guidance takes (without the wrapper's overhead: count_nonzero, the same
    nonzero test, NaN counting as True)."""
    return _nz(x) > 0 if isinstance(x, np.ndarray) else bool(np.any(x))


def _col(x):
    """np.reshape(x, (-1, 1)), as the view x[:, None] for a 1-D array (the same values)."""
    return (
        x[:, None]
        if isinstance(x, np.ndarray) and x.ndim == 1
        else np.reshape(x, (-1, 1))
    )


_R60, _R35, _MPI2 = (
    np.radians(60.0),
    np.radians(35.0),
    -np.pi / 2,
)  # guidance floor constants (the same values)


def _rows(x, n):
    """np.broadcast_to(np.reshape(x, -1), (n,)), as a plain view when x already has n rows."""
    if isinstance(x, np.ndarray) and x.shape == (n,):
        return x
    return np.broadcast_to(np.reshape(x, -1), (n,))


VI_T = np.arange(
    0.1, 6.0 + 1e-9, 0.05
)  # intercept-time grid of the viable-intercept search (s)
_VI_GT2 = (
    0.5 * G * VI_T[None, :] ** 2
)  # the gravity term of viable_dir, evaluated once (same values)


def viable_dir(pos, vel, tp, tv, nose, a_av, margin, wr=1.5, tau0=0.064):
    """Earliest viable constant-direction full-thrust intercept (RLUtilities is_viable pattern with a turn delay;
    time-optimal steering for bounded thrust against a constant-velocity target).
    dx(T) = tp + tv T - (pos + vel T + 1/2 g T^2); viable when T > tau and 2|dx| / (T - tau)^2 <= margin a_av,
    tau = tau0 + angle(nose, dx) / wr. Returns (unit direction of dx(T*) (R,3), T* (R,), nan if none)."""
    T = VI_T[None, :, None]
    dx = tp[:, None] + (tv - vel)[:, None] * T - pos[:, None]
    dx[:, :, 2] += _VI_GT2
    # the length-3 reductions as _l3sq / _l3 (the same values as add.reduce, see ff_ref._l3)
    dn = np.sqrt(_l3sq(dx * dx))  # = np.linalg.norm(dx, axis=2)
    cosang = _clip(_l3(dx * nose[:, None]) / np.maximum(dn, 1e-9), -1.0, 1.0)
    tau = tau0 + np.arccos(cosang) / wr
    te = VI_T[None, :] - tau
    ok = (te > 1e-3) & (
        2.0 * dn <= (margin * a_av)[:, None] * np.maximum(te, 1e-3) ** 2
    )
    anyok = ok.any(1)
    k = np.argmax(ok, axis=1)
    r = np.arange(len(pos))
    d = dx[r, k] / np.maximum(dn[r, k], 1e-9)[:, None]
    tstar = np.where(anyok, VI_T[k], np.nan)
    return d, tstar


def next_k(pos, vel, tp, tv, bp, bv, avail, tgt, k):
    """x rows: the k balloons with the earliest rough reach estimate (as BeamPlanner._cands) after tgt, per row, from tgt's
    position at the rough arrival time, flying at the current velocity. bp, bv (R,B,3) / (B,3) positions now and
    velocities; avail (R,B). Returns ids (R, k), -1 where fewer exist."""
    r = tp - pos
    d = np.linalg.norm(r, axis=1)
    vc = -(r * (tv - vel)).sum(1) / np.maximum(d, 1e-6)
    t1 = d / np.maximum(vc, 15.0)
    p1 = tp + tv * t1[:, None]
    rb = bp + bv[None] * t1[:, None, None] - p1[:, None]
    db = np.linalg.norm(rb, axis=2)
    vcb = -(rb * (bv[None] - vel[:, None])).sum(2) / np.maximum(db, 1e-6)
    est = np.where(avail, db / np.maximum(vcb, 15.0), np.inf)
    est[np.arange(len(pos)), tgt] = np.inf
    k = min(k, est.shape[1])
    ids = np.argsort(est, axis=1)[:, :k]
    return np.where(np.isfinite(np.take_along_axis(est, ids, 1)), ids, -1)


def los_fill_dir(n_reg, w, u, f, A, s_max=0.98):
    """Lean rows: line-of-sight fill of guidance's regular thrust direction, per row.
    w (R,3) = the law's clipped lead-pursuit acceleration + G z (the thrust acceleration it asks for), u (R,3) unit LOS
    (after aim offsets), A (R,) available specific thrust (m/s^2, > 0), f (R,) fill. The part of w normal to the LOS, w_perp,
    is kept (scaled down to at most s_max A when it is larger), the rest of the full thrust A goes along u:
        n_los = w_perp / A + sqrt(1 - |w_perp / A|^2) u,   n = unit((1 - f) n_reg + f n_los).
    Then A n_los - G z has the law's acceleration normal to u (feasible rows) and a component along u of
    A sqrt(1 - |w_perp / A|^2) - G u_z instead of the law's (nearly zero once on a collision course)."""
    wp = w - _l3(w * u)[:, None] * u
    s = wp / A[:, None]
    sn = _norm(s)
    s = s * (np.minimum(1.0, s_max / np.maximum(sn, 1e-12)))[:, None]
    n_los = s + np.sqrt(np.maximum(1.0 - _l3sq(s * s), 0.0))[:, None] * u
    nb = (1.0 - f)[:, None] * n_reg + f[:, None] * n_los
    return nb / np.maximum(_norm(nb), 1e-9)[:, None]


def guidance(
    pos,
    vel,
    tp,
    tv,
    h_agl,
    boost=0.0,
    amax=20.0,
    nav=2.0,
    tgo_min=0.8,
    off_up=0.0,
    off_side=0.0,
    vi=0.0,
    nose=None,
    a_av=None,
    vi_wr=1.5,
    vi_tau0=0.064,
    vi_blend=(0.25, 0.75),
    los_fill=0.0,
):
    """Lead pursuit (ZEM / t_go-closing) -> unit thrust dir (N,3).
    nav, tgo_min, boost, off_up, off_side, vi, los_fill may be scalars or per-row arrays (planner guidance variants).
    vi (per row) > 0: viable-intercept steering with that acceleration margin (viable_dir; needs nose, a_av), blended into
    the ZEM law as the intercept time T* drops from vi_blend[1] to vi_blend[0] s; rows with vi = 0 keep the ZEM law.
    los_fill (per row) f > 0: line-of-sight fill (lean rows; needs a_av = the available specific thrust A = thrust / mass
    at the current throttle, GivenSim.accel_avail): the regular direction n_reg is blended with n_los, the full-thrust
    direction that keeps the law's acceleration normal to the LOS and spends the rest along it (los_fill_dir), right after
    n_reg is formed; the vi blend and the ground floor follow. Rows with f = 0 or no usable A (None, nan, <= 1e-3) keep
    n_reg. Last, the ground-safety floor (the "loose" floor): the thrust elevation is at least 60 deg below 15 m AGL or
    within 1.5 s of impact, 35 deg below 30 m or within 3 s."""
    r = tp - pos
    # np.cross / np.linalg.norm(axis=1) as the same per-component expressions (_cross, _norm)
    if _any(off_up) or _any(
        off_side
    ):  # aim-point offset in the plane normal to the LOS
        los = r / np.maximum(_norm(r), 1e-6)[:, None]
        side = _cross(los, _EZ[None])
        side /= np.maximum(_norm(side), 1e-6)[:, None]
        up = _cross(side, los)
        r = r + _col(off_up) * up + _col(off_side) * side
    vrel = tv - vel
    dist = _norm(r)
    vc = -_l3(r * vrel) / np.maximum(dist, 1e-6)  # (.).sum(1) as _l3
    tgo = _clip(dist / np.maximum(vc, 12.0), tgo_min, 6.0)[:, None]
    a = _col(nav) * (r + vrel * tgo) / tgo**2
    a = a + _col(boost) * r / np.maximum(dist, 1e-6)[:, None]
    a = _clip(a, -amax, amax)
    a[:, 2] += G
    n = a / np.maximum(_norm(a), 1e-9)[:, None]
    if (
        _any(los_fill) and a_av is not None
    ):  # lean rows: line-of-sight fill (0 = none, n unchanged)
        lfm = _rows(los_fill, len(pos)).astype(float)
        A = _rows(a_av, len(pos)).astype(float)
        mf = (lfm > 0) & np.isfinite(A) & (A > 1e-3)
        if mf.any():
            u = r[mf] / np.maximum(dist[mf], 1e-6)[:, None]
            n[mf] = los_fill_dir(n[mf], a[mf], u, lfm[mf], A[mf])
    if _any(vi):
        vim = _rows(vi, len(pos)).astype(float)
        m = vim > 0
        a_m = _rows(a_av, len(pos))[m]
        d, tstar = viable_dir(
            pos[m], vel[m], tp[m], tv[m], nose[m], a_m, vim[m], vi_wr, vi_tau0
        )
        b0, b1 = (
            float(vi_blend[0]),
            float(vi_blend[1]),
        )  # the blend bounds as scalars (the same per-row values)
        w = _clip((tstar - b0) / (b1 - b0), 0.0, 1.0)
        w = np.where(np.isfinite(tstar), w, 0.0)[:, None]
        nb = w * d + (1.0 - w) * n[m]
        n[m] = nb / np.maximum(_norm(nb), 1e-9)[:, None]
    # ground-safety floor
    vz = vel[:, 2]
    if not _nz((h_agl < 30.0) | (vz < -0.5)):
        # no row is below 30 m AGL or descending faster than 0.5 m/s, so t_imp = inf and el_min = -pi/2 on every row (NaN
        # h_agl / vz included: their comparisons are False here and in the expressions below); n changes only where
        # el < -pi/2
        el = np.arcsin(_clip(n[:, 2], -1, 1))
        if not _nz(el < _MPI2):
            return n
    t_imp = np.where(vz < -0.5, h_agl / np.maximum(-vz, 1e-3), np.inf)
    el_min = np.where(
        (h_agl < 15.0) | (t_imp < 1.5),
        _R60,
        np.where((h_agl < 30.0) | (t_imp < 3.0), _R35, _MPI2),
    )
    el = np.arcsin(_clip(n[:, 2], -1, 1))
    fix = el < el_min
    if fix.any():
        hz = np.arctan2(n[fix, 1], n[fix, 0])
        e = el_min[fix]
        n[fix] = np.stack(
            [np.cos(e) * np.cos(hz), np.cos(e) * np.sin(hz), np.sin(e)], 1
        )
    return n


_EPS = 1e-6  # m, round-off margin of the pop-test bound (_deferred_pops)
_MARGIN = 1e-3  # m, extra slack of the bound (covers the rounding of the squared-distance expansion)


def _ff_split(F):
    """(N, 6) feedforward state -> (n_ref, w_ref), two C-contiguous (N, 3) copies."""
    return np.ascontiguousarray(F[:, :3]), np.ascontiguousarray(F[:, 3:])


def _seg_d2(a0, a1):
    """Squared closest distance of the swept segments with relative end points a0, a1 (n, 3): the env test's per-element
    arithmetic (dmin2 <= r2 between the rocket's and the balloon's step segments)."""
    da = a1 - a0
    s = _clip(-_sum3(a0 * da) / np.maximum(_sum3(da * da), 1e-12), 0, 1)
    return _sum3((a0 + s[:, None] * da) ** 2)


def _seg_d2T(a0, a1):
    """_seg_d2 for component-major relative end points a0, a1 (3, n) (contiguous rows instead of column views and an
    inner broadcast); the same per-element operations, (p0 + p1) + p2 sums as _sum3."""
    da = a1 - a0
    p = a0 * da
    r = da * da
    s = _clip(-((p[0] + p[1]) + p[2]) / np.maximum((r[0] + r[1]) + r[2], 1e-12), 0, 1)
    e = (a0 + s[None] * da) ** 2
    return (e[0] + e[1]) + e[2]


def _bal_seg(pl, t1, dt, j):
    """Balloon j[e] at the two ends of the step ending at t1[e] (E,): (b0, b1) (E, 3) each (track interpolation through
    one flat gather)."""
    tab, i, f = pl._tidx(np.concatenate([t1 - dt, t1]))
    B = tab.shape[1]
    tabf = tab.reshape(-1, 3)
    li = i * B + np.concatenate([j, j])
    b = tabf[li] * (1.0 - f)[:, None] + tabf[li + B] * f[:, None]
    E = len(j)
    return b[:E], b[E:]


class _TargetTrack:
    """Each rollout row's target balloon at both ends of every step, evaluated for CH steps at a time for the rows active
    then (the per-element expressions of _bal_seg). tk (L + 1, n0): the rows' time at the start of each step (tk[k + 1] =
    time after step k, the loop's running sum); tg (n0,) targets. Chunks of CH steps, both ends stored per step as one
    component-major (2, 3, n) block (one gather when rows finished), for _seg_d2T."""

    CH = 128

    def __init__(self, pl, tk, dt, tg):
        self.pl, self.tk, self.dt, self.tg = pl, tk, dt, tg
        self.L = len(tk) - 1
        self.lo = self.hi = 0
        self.rows = self.key = self.sel = None

    def _build(self, jk, c0):
        hi = min(jk + self.CH, self.L)
        t1 = self.tk[jk + 1 : hi + 1]
        tg = self.tg
        if c0 is not None:
            t1, tg = t1[:, c0], tg[c0]
        ch, n = t1.shape
        pl = self.pl
        # the interpolation of _bal_seg at t1 - dt and at t1, laid out (step, end, row)
        tt = np.empty((ch, 2, n))
        tt[:, 0] = t1 - self.dt
        tt[:, 1] = t1
        tab, i, f = pl._tidx(tt.reshape(-1))
        B = tab.shape[1]
        tabf = tab.reshape(-1, 3)
        li = i * B + np.tile(tg, 2 * ch)
        Bk = (tabf[li] * (1.0 - f)[:, None] + tabf[li + B] * f[:, None]).reshape(
            ch, 2, n, 3
        )
        self.Bk = np.ascontiguousarray(
            Bk.transpose(0, 1, 3, 2)
        )  # (step, end, component, row): (3, n) blocks
        self.lo, self.hi, self.rows, self.key = jk, hi, c0, c0

    def get(self, jk, c0):
        """(b0, b1) (3, n) component-major of the active rows at step jk (c0: their rows in tk / tg, None = all)."""
        if jk >= self.hi:
            self._build(jk, c0)
        Bj = self.Bk[jk - self.lo]  # (2, 3, n_chunk)
        if c0 is self.rows:
            return Bj[0], Bj[1]
        if (
            c0 is not self.key
        ):  # rows finished since the chunk was built: map the active rows into the chunk's rows
            self.sel = c0 if self.rows is None else np.searchsorted(self.rows, c0)
            self.key = c0
        G = Bj.take(self.sel, axis=2)
        return G[0], G[1]


def _deferred_pops(pl, rec, popped, rel_mask, r2, dt, W=10):
    """The incidental pops of a rollout, from its step records rec = [(ia, p0, p1, t1, target dmin2), ...] (active full row
    ids, step start / end positions, time after the step; rec not empty), written into popped (R, B) in place.

    The same set as the per-step test: (r, j) is popped when j is released and dmin2 <= r2 at a step where r is active
    (dmin2 by the per-step arithmetic, _seg_d2 / _bal_seg). Pairs are evaluated only where a bound cannot exclude them:
    per window of W steps and row active at its first step, P = the first step's start point, T = its start time t1 - dt,
    D = the largest distance of the row's step end points in the window from P (exact, from the records: every segment of
    the window lies in the ball |x - P| <= D), and a pair is kept unless |P - b_j(T)| > D + vmax_j W dt (1 + 1e-9) +
    radius + _EPS + _MARGIN (b_j is vmax_j-Lipschitz, |t - T| <= W dt on the window's segments). Rows with equal start times
    share every step time (the same running sums), so b_j(T) is evaluated once per time group and window."""
    L = len(rec)
    R, B = popped.shape
    ns = np.array([len(x[0]) for x in rec])
    RI = np.concatenate([x[0] for x in rec])
    P0 = np.concatenate([x[1] for x in rec])
    P1 = np.concatenate([x[2] for x in rec])
    T1 = np.concatenate([x[3] for x in rec])
    E = len(RI)
    SI = np.repeat(np.arange(L), ns)
    M = np.full((L, R), -1)
    M[SI, RI] = np.arange(E)
    nw = (L + W - 1) // W
    wq = SI // W
    refe = M[
        wq * W, RI
    ]  # each record's window reference (the row's record at the window's first step)
    d = P1 - P0[refe]
    # the per-(window, row) maximum through a dense (step, row) array and one max reduction (the same values as
    # np.maximum.at from zeros: every entry is a sum of squares >= 0, inactive entries stay 0)
    D2 = np.zeros((nw * W, R))
    D2[SI, RI] = _sum3(d * d)
    Dm2 = np.fmax.reduce(
        D2.reshape(nw, W, R), 1
    )  # fmax: a NaN step (blown-up row) must not hide the window's finite steps
    isref = SI % W == 0
    Pref = np.full((nw, R, 3), np.nan)
    Pref[wq[isref], RI[isref]] = P0[isref]
    Tref = np.full((nw, R), np.nan)
    Tref[wq[isref], RI[isref]] = T1[isref] - dt
    Dm = np.sqrt(Dm2) * (1.0 + 1e-12)
    vmax = pl._track_vmax()
    # one bound for all balloons, vmax = max_j vmax_j (a looser, still valid bound: the kept pairs are a superset of the
    # per-balloon bound's, and every kept pair is evaluated exactly below; fmax so that one NaN balloon speed leaves the
    # others' bound finite)
    cBm = (
        float(np.fmax.reduce(vmax)) * (W * dt * (1.0 + 1e-9))
        + (np.sqrt(r2) + _EPS + _MARGIN)
        if len(vmax)
        else 0.0
    )
    # time groups: rows whose times are equal at their first record share all later times
    t_first = np.full(R, np.nan)
    t_first[rec[0][0]] = rec[0][3]
    act = np.flatnonzero(np.isfinite(t_first))
    _, gi = np.unique(t_first[act], return_inverse=True)
    cw, cr, cj = [], [], []
    for g in range(gi.max() + 1 if len(gi) else 0):
        Rg = act[gi == g]
        vw = np.flatnonzero(np.isfinite(np.fmax.reduce(Tref[:, Rg], 1)))
        if not len(vw):
            continue
        Tg = np.fmax.reduce(Tref[np.ix_(vw, Rg)], 1)
        tr = Tref[np.ix_(vw, Rg)]
        assert ((tr == Tg[:, None]) | np.isnan(tr)).all()
        bg = pl._bp_all(Tg)  # (nv, B, 3)
        Pg = Pref[vw][:, Rg]  # (nv, ng, 3), nan rows inactive
        dots = np.matmul(Pg, bg.transpose(0, 2, 1))  # (nv, ng, B)
        # |P - b|^2 <= c^2 as |b|^2 - 2 P.b <= c^2 - |P|^2 with c = D + cBm per (window, row): the big (nv, ng, B) arrays
        # get 4 ops instead of 8 (rounding ~1e-9 m^2 at these coordinates, inside _MARGIN)
        c = Dm[vw][:, Rg] + cBm
        rhs = c * c - _l3sq(
            Pg * Pg
        )  # (nv, ng); nan for rows inactive in the window (no pair kept)
        dots *= -2.0
        dots += _l3sq(bg * bg)[:, None, :]
        m = (dots <= rhs[:, :, None]) & (rel_mask[None] & ~popped[Rg])[None]
        w_, r_, j_ = np.nonzero(m)
        if len(w_):
            cw.append(vw[w_])
            cr.append(Rg[r_])
            cj.append(j_)
    if not cw:
        return
    cw, cr, cj = np.concatenate(cw), np.concatenate(cr), np.concatenate(cj)
    st = cw[:, None] * W + np.arange(W)[None]  # the window's steps
    ok = st < L
    e = np.where(ok, M[np.minimum(st, L - 1), cr[:, None]], -1)
    q, o = np.nonzero(e >= 0)
    e = e[q, o]
    j = cj[q]
    b0, b1 = _bal_seg(pl, T1[e], dt, j)
    h = _seg_d2(P0[e] - b0, P1[e] - b1) <= r2
    if h.any():
        popped[RI[e[h]], j[h]] = True


# x-row settings (BeamPlanner.enable_x) with the only value this planner implements; other values select code that is not
# shipped: branch (x rows for a subset of the candidates), aim (chord aim only), keep (no keep row), force (level-1 child
# not forced to t2), wr (a re-pointing rate of its own), stick_fb (no sticky fallback), mode (waypoint rows), base (another
# base row; it must be a viable-intercept row, rollout passes x rows the step's rotation matrices)
X_FIXED = dict(
    branch=0,
    aim="pmp",
    keep=True,
    force=True,
    wr=0.0,
    stick_fb=True,
    mode="pass",
    base=(3.0, 0.3, 0.0, 0.0, 0.0, 0.9),
)


class BeamPlanner:
    def __init__(
        self,
        sim,
        radius,
        *,
        critic,
        beam=12,
        branch=10,
        depth=3,
        horizon=6.0,
        dt=0.02,
        dec_dt=0.1,
        lam=0.3,
        radius_margin=0.8,
        stick=0.2,
        variants=((2.0, 0.8, 0.0),),
        deep_var=1,
        deep_radius=3.0,
        long_horizon=10.0,
        low_pen=0.2,
        critic_beta=0.0,
        depth_gamma=1.0,
        reach_beta=0.0,
        reach_mode="soft",
        reach_k=3,
        reach_tau0=2.0,
        reach_tmax=6.0,
        reach_dt=0.2,
        reach_margin=0.9,
        reach_m=16,
    ):
        self.sim = sim
        # leaf reach term: thrust-limited reach times from each leaf end state to the available balloons, see _reach
        # (the soft form only)
        if reach_mode != "soft":
            raise ValueError(
                f"reach_mode={reach_mode!r}: this planner implements only reach_mode='soft'"
            )
        self.reach_beta, self.reach_k = float(reach_beta), int(reach_k)
        self.reach_tau0, self.reach_tmax, self.reach_dt = (
            float(reach_tau0),
            float(reach_tmax),
            float(reach_dt),
        )
        self.reach_margin, self.reach_m = float(reach_margin), int(reach_m)
        # leaf value: critic of a PPO policy trained in the given-only surrogate (training seeds only)
        self.critic = critic
        self.critic_beta = float(critic_beta)
        V = np.array(
            [tuple(v) + (0.0,) * (12 - len(v)) for v in variants], float
        )  # rows may differ in length
        # rows: (nav gain, t_go floor, boost m/s^2, aim offset up m, aim offset side m, viable-intercept margin,
        #        columns 6-10 unused (0; PlannerAgent rejects rows with other values),
        #        line-of-sight fill los_fill (lean rows, guidance / los_fill_dir; 0 = none))
        self.V = V
        self.vi_wr = 1.5  # rad/s thrust-redirect rate assumed by viable-intercept rows (PlannerAgent sets it)
        self.r2 = (radius * radius_margin) ** 2
        # deeper legs: one guidance variant and a relaxed 'reachable' radius (the flown leg is re-aimed later)
        self.deep_var, self.r2_deep = int(deep_var), float(deep_radius) ** 2
        self.long_horizon, self.low_pen = float(long_horizon), float(low_pen)
        # depth_gamma: pops first scored at search level d count gamma**d (deeper legs are less certain)
        self.depth_gamma = float(depth_gamma)
        self.beam, self.branch, self.depth = int(beam), int(branch), int(depth)
        self.horizon, self.dt, self.dec_n = (
            float(horizon),
            float(dt),
            max(1, int(round(dec_dt / dt))),
        )
        # dec_dt (PlannerAgent passes its decision interval, 0.1 s): the x rows' t_end default (enable_x)
        self.dec_dt = float(dec_dt)
        self.lam, self.stick = (
            float(lam),
            float(stick),
        )
        # lam_coast (PlannerAgent sets it): time after burnout costs lam_coast per s instead of the in-burn cost (the
        # burnout state fixes the ballistic coast, so its duration has no opportunity cost)
        self.lam_coast = None
        # lam_late, lam_late_t (PlannerAgent sets them): in-burn time cost for plans made at t0 - t_launch >= lam_late_t
        self.lam_late, self.lam_late_t = None, None
        # seq_log_at (PlannerAgent sets it for the tour decision; None = off): the first plan() call at or after seq_log_at
        # s after launch records its chosen leaf's whole sequence in seq_log: per leg (target, variant, end time after
        # launch, the balloons popped in the leg, end position); read-only, the plan and its result are unchanged
        self.seq_log_at = None
        self.seq_log = None
        # shoot_cur (None = no tour locked): the locked tour program dict(dirs (K, 3), t0 (s after launch), j1, j2, tour,
        # ...) that PlannerAgent sets and flies (coast_shoot.program_cmd at full throttle until burn_time, re-shot from its
        # estimate)
        self.shoot_cur = None
        self.last = None
        self.last_t2 = -1  # t2 of the chosen first move (x rows; -1 none)
        # track: predicted balloon positions (table (T+1, B, 3), t_start, dt) from predict.py, set by PlannerAgent before
        # every plan() call
        self.track = None
        # x rows (moving-waypoint sweep legs, xlegs.py): see enable_x (PlannerAgent calls it)
        self.xr = None
        self.xcode = None  # per variant row: -1 regular, j = x variant j, len(x variants) = the keep row
        self.nV0 = len(self.V)
        self.x_cur = None  # the x row flown now (PlannerAgent sets it before plan(); the keep row continues it)
        self.last_x = None  # parameters of the chosen first move when it is an x row

    def enable_x(self, cfg):
        """x rows: at search level 0, for each candidate balloon b, each x variant (0, slack s, beta) and each of the k2
        likeliest next balloons t2, one row that passes b at
            tk = t + T + s   (T = the point-mass minimum time with the hold model, xlegs.tmin)
        while sweeping its thrust direction onto the exit direction e (look-ahead aim: the thrust direction of the earliest
        constant-direction pass of t2 from the predicted arrival state; chord aim as the seed), flown in feedback
        (xlegs.command re-solved at every decision step); the base law (a viable-intercept variant row) is the fallback
        where the sweep has no solution. The level-1 child of an x row flies to its t2. keep: one more row that continues
        the x row flown now with its own (b, tk, e, t2). Rows are scored by the same rollouts and keys. The settings of
        X_FIXED take only their value there."""
        c = dict(cfg or {})
        for k, v in X_FIXED.items():
            if k in c and (tuple(c[k]) if isinstance(v, tuple) else c[k]) != v:
                raise ValueError(
                    f"xrows {k}={c[k]!r}: this planner implements only {k}={v!r}"
                )
        # x variant (m, s[, beta]): pass time tk = t + T + s with T = xlegs.tmin (m = 0); exit direction
        # e = unit((1 - beta) u_c + beta e_look), u_c = direction of the required displacement (no exit shaping at beta 0),
        # e_look = look-ahead / chord aim
        xv = np.array(
            [tuple(float(x) for x in r) + (1.0,) * (3 - len(r)) for r in c["variants"]],
            float,
        ).reshape(-1, 3)
        if (xv[:, 0] > 0).any():
            raise ValueError(
                "x variants with a viable-intercept margin m > 0 are not implemented (only m = 0, xlegs.tmin)"
            )
        base = X_FIXED["base"]
        self.xr = dict(
            vars=xv,
            k2=int(c.get("k2", 2)),
            rp=float(c.get("rp", 0.05)),  # pass offset from the balloon centre (m)
            lead=float(
                c.get("lead", 0.0)
            ),  # command the sweep direction this many s ahead
            chord_h=float(c.get("chord_h", 1.0)),
            t_min=float(c.get("t_min", 1.0)),  # x rows only from t_min s after launch
            M=int(c.get("M", 16)),
            # within t_end of the pass: command e (xlegs.command); default one decision step (dec_dt)
            t_end=float(c.get("t_end", self.dec_dt)),
            t_hold=float(
                c.get("t_hold", 0.4)
            ),  # failed solves this close to the pass keep e
            rate_cap=float(
                c.get("rate_cap", 0.0)
            ),  # rad/s: rows whose sweep turns faster are re-aimed at half beta, then dropped (0 = off)
        )
        self.nV0 = len(self.V)
        row = np.array(base + (0.0,) * (12 - len(base)), float)
        self.V = np.vstack([self.V, np.tile(row, (len(xv) + 1, 1))])
        self.xcode = np.concatenate([np.full(self.nV0, -1), np.arange(len(xv) + 1)])

    def _x_afn(self, tl):
        return lambda tt: self.sim.accel_avail(tt - tl, 1.0)

    def _x_track(self, tt, ids):
        """Predicted position of balloon ids[r] at time tt[r] (track)."""
        return self._bp_one(tt, ids)[0]

    def _x_rows(self, S, rr0, b0, avail, bv):
        """Level-0 x rows for the candidate (row rr0, balloon b0) pairs: (rr, tgt, var, t2, tk, B, e, nu, wp, nb) or
        None. The re-pointing rate of the hold model is vi_wr."""
        X = self.xr
        tl = S["t_launch"]
        t_now = float(S["t"][0])
        if t_now - tl < X["t_min"] or len(rr0) == 0:
            return None
        from . import xlegs as XL

        a_fn = self._x_afn(tl)
        wr, tau0 = self.vi_wr, self.sim.gtau
        burn_end = tl + self.sim.burn_time - 0.05
        pos, vel = S["pos"][rr0], S["vel"][rr0]
        nose = quat_rot(S["q"][rr0])[:, :, 2]
        tt = np.full(len(rr0), t_now)
        tp, tv = self._bp_one(tt, b0)
        bpn = self._bp_all(S["t"][:1])[0]
        nk = next_k(
            pos, vel, tp, tv, bpn[None], bv, avail[rr0], b0, X["k2"]
        )  # (n, k2), -1 none
        out = []
        tsm = None
        for j, (_, sl, _) in enumerate(X["vars"]):
            if tsm is None:  # the same minimum pass time for every variant (m = 0)
                tsm = XL.tmin(
                    pos,
                    vel,
                    tt,
                    nose,
                    lambda T: self._x_track(
                        T.ravel(), np.repeat(b0, T.shape[1])
                    ).reshape(T.shape + (3,)),
                    a_fn,
                    X["rp"],
                    wr,
                    tau0,
                )
            ts = tsm
            tk = tt + ts + sl
            ci, ki = np.nonzero(
                (nk >= 0) & (np.isfinite(ts) & (tk <= burn_end))[:, None]
            )
            if len(ci):
                out.append((ci, nk[ci, ki], tk[ci], np.full(len(ci), self.nV0 + j)))
        cur = self.x_cur
        if cur is not None and cur["tk"] > t_now + X["t_end"] and avail[0, cur["wp"]]:
            hit = np.flatnonzero(b0 == cur["wp"])
            t2c = int(cur["nb"])
            if len(hit) and (t2c < 0 or avail[0, t2c]):
                out.append(
                    (
                        hit[:1],
                        np.array([t2c]),
                        np.array([cur["tk"]]),
                        np.array([self.nV0 + len(X["vars"])]),
                    )
                )
        if not out:
            return None
        ci = np.concatenate([o[0] for o in out])
        t2 = np.concatenate([o[1] for o in out])
        tk = np.concatenate([o[2] for o in out])
        vr = np.concatenate([o[3] for o in out])
        tg = b0[ci]
        p, v, ns, tn = pos[ci], vel[ci], nose[ci], tt[ci]
        B = self._x_track(tk, tg)
        Rp = X["rp"]
        # hold the nose while turning toward the constant direction of the required displacement, then sweep
        tau = tk - tn
        Dp = B - p - v * tau[:, None]
        Dp[:, 2] += 0.5 * G * tau**2
        c = np.clip(
            (ns * Dp).sum(1) / np.maximum(np.sqrt((Dp * Dp).sum(1)), 1e-9), -1.0, 1.0
        )
        dh = np.minimum(tau0 + np.arccos(c) / wr, np.maximum(tau - X["t_end"], 0.0))
        p1, v1 = XL.hold(p, v, tn, dh, ns, a_fn)
        nu0 = np.full((len(tk), 3), np.nan)
        iskeep = vr == self.nV0 + len(X["vars"])
        if iskeep.any():
            nu0[iskeep] = cur["nu"]
        has2 = t2 >= 0
        e0 = np.tile(np.array([0.0, 0.0, 1.0]), (len(tk), 1))
        if has2.any():
            e0[has2] = XL._unit(
                self._x_track(tk[has2] + X["chord_h"], t2[has2]) - B[has2]
            )
        if iskeep.any():
            e0[iskeep] = cur["e"]
        beta = X["vars"][np.minimum(vr - self.nV0, len(X["vars"]) - 1), 2]
        Dh = B - p1 - v1 * (tk - tn - dh)[:, None]
        Dh[:, 2] += 0.5 * G * (tk - tn - dh) ** 2
        uc = XL._unit(Dh)  # constant-direction thrust of the remaining leg
        bl = lambda el, bt: XL._unit((1.0 - bt)[:, None] * uc + bt[:, None] * el)  # noqa: E731
        # seed: the chord aim; the look-ahead direction from the seed's arrival state
        nu, ok, va, _ = XL.solve(
            p1, v1, tn + dh, tk, B, bl(e0, beta), Rp, a_fn, M=X["M"], nu0=nu0
        )
        el = e0.copy()
        lk = ok & has2 & ~iskeep
        if lk.any():
            li = np.flatnonzero(lk)
            t2l = t2[li]
            dn, fd = XL.next_dir(
                B[li],
                va[li],
                tk[li],
                lambda T: self._x_track(T.ravel(), np.repeat(t2l, T.shape[1])).reshape(
                    T.shape + (3,)
                ),
                a_fn,
                Rp,
            )
            el[li[fd]] = dn[fd]
        nu0 = np.where(ok[:, None], nu, nu0)  # warm start from the seed
        e = np.where(iskeep[:, None], e0, bl(el, beta))
        nu, ok, _, _, rt = XL.solve(
            p1, v1, tn + dh, tk, B, e, Rp, a_fn, M=X["M"], nu0=nu0, rate=True
        )
        if X["rate_cap"] > 0:  # too fast a sweep: half the exit shaping, then drop
            fast = ok & (rt > X["rate_cap"]) & ~iskeep
            if fast.any():
                fi = np.flatnonzero(fast)
                e2 = XL._unit(0.5 * (uc[fi] + e[fi]))
                n2, ok2, _, _, r2_ = XL.solve(
                    p1[fi],
                    v1[fi],
                    tn[fi] + dh[fi],
                    tk[fi],
                    B[fi],
                    e2,
                    Rp,
                    a_fn,
                    M=X["M"],
                    nu0=nu[fi],
                    rate=True,
                )
                good = ok2 & (r2_ <= X["rate_cap"])
                e[fi[good]], nu[fi[good]] = e2[good], n2[good]
                ok[fi[~good]] = False
        g = np.flatnonzero(ok)
        if len(g) == 0:
            return None
        # (rows, target, variant, first-move code slot, tk, B, e, nu, sweep balloon wp, exit-aim balloon nb)
        return rr0[ci[g]], tg[g], vr[g], t2[g], tk[g], B[g], e[g], nu[g], tg[g], t2[g]

    def _x_step(self, xp, ia, cP, cV, cT, cN, Rm, tl, elev):
        """Decision step of the x rows among the active rows ia: cN (their commands) overwritten in place. Rows on the sweep
        (before tk, no failed solve) fly xlegs.command; a row whose solve fails outside t_hold leaves the sweep for good
        and flies its base law (cN as given). Rm: the step's rotation matrices of the active rows (rollout computes them
        whenever an x row is active: x rows are viable-intercept rows)."""
        xm = xp["on"][ia]
        if not xm.any():
            return
        from . import xlegs as XL

        X = self.xr
        ii = ia[xm]
        wa = np.flatnonzero(xm)
        nose = Rm[:, :, 2][xm]
        tt = cT[xm]
        tk = xp["tk"][ii]
        sw = np.flatnonzero(~xp["fb"][ii] & (tk > tt))  # rows on the sweep
        if len(sw):
            d, ok, nu = XL.command(
                cP[wa[sw]],
                cV[wa[sw]],
                nose[sw],
                tt[sw],
                tk[sw],
                xp["B"][ii[sw]],
                xp["e"][ii[sw]],
                xp["nu"][ii[sw]],
                X["rp"],
                self._x_afn(tl),
                self.vi_wr,
                self.sim.gtau,
                X["lead"],
                X["t_end"],
                X["M"],
            )
            xp["nu"][ii[sw]] = nu
            u_ = ok | (
                tk[sw] - tt[sw] < X["t_hold"]
            )  # a failed solve this close to the pass keeps e
            if not u_.all():  # failed rows leave the sweep for good
                xp["fb"][ii[sw[~u_]]] = True
            if u_.any():
                d = XL.floor_dirs(d, cP[wa[sw], 2] - elev, cV[wa[sw], 2])
                cN[wa[sw[u_]]] = d[u_]

    def _x_done(self, x0, fc, Bp, nV):
        """After plan(): last_x = the chosen first move's x-row parameters (None for a regular row)."""
        self.last_x = None
        if x0 is not None:
            xp, tg = x0
            w = np.flatnonzero(xp["on0"] & (xp["code"] == fc))
            if len(w):
                i = int(w[0])
                self.last_x = dict(
                    tgt=int(tg[i]),
                    tk=float(xp["tk"][i]),
                    B=xp["B"][i].copy(),
                    e=xp["e"][i].copy(),
                    nu=xp["nu0"][i].copy(),
                    t2=int(xp["t2"][i]),
                    var=int((fc // Bp) % nV),
                    wp=int(xp["wp"][i]),
                    nb=int(xp["nb"][i]),
                )

    def x_flight(self, pos, vel, nose, t, t_launch, xc, bpw=None, bvw=None):
        """In-flight x row (PlannerAgent): the decision-step command of the rollouts for one row. xc: last_x of the plan
        (its nu is the warm start, updated; fb set when it leaves the sweep). bpw, bvw (the waypoint balloon's observed
        position / velocity, passed by PlannerAgent) are not used: the x rows pass their own target. Returns the unit
        thrust direction or None (fly the base law)."""
        from . import xlegs as XL

        X = self.xr
        if not xc.get("fb", False) and xc["tk"] > t:
            d, ok, nu = XL.command(
                pos,
                vel,
                nose,
                np.array([t]),
                np.array([xc["tk"]]),
                xc["B"][None],
                xc["e"][None],
                xc["nu"][None],
                X["rp"],
                self._x_afn(t_launch),
                self.vi_wr,
                self.sim.gtau,
                X["lead"],
                X["t_end"],
                X["M"],
            )
            if ok[0] or xc["tk"] - t < X["t_hold"]:
                xc["nu"] = nu[0]
                return XL.floor_dirs(d, pos[:, 2] - self.sim.elevation, vel[:, 2])[0]
            xc["fb"] = True
        return None

    # -- balloon positions --------------------------------------------------------------------------
    def _tidx(self, tt):
        tab, ts, dtt = self.track
        f = (np.asarray(tt, float) - ts) / dtt
        i = _clip(np.floor(f).astype(int), 0, len(tab) - 2)
        return (
            tab,
            i,
            (f - i),
        )  # a outside [0, 1] extrapolates linearly from the end segments

    def _bp_all(self, tt):
        """Predicted positions of all balloons at per-row times tt: (R, B, 3)."""
        tab, i, a = self._tidx(tt)
        return tab[i] * (1.0 - a)[:, None, None] + tab[i + 1] * a[:, None, None]

    def _track_vmax(self):
        """(B,) largest segment speed of each balloon's predicted track (bound for _deferred_pops), cached per track."""
        if getattr(self, "_vmax_key", None) is not self.track:
            tab, _, dtt = self.track
            dd = np.diff(tab, axis=0)
            self._vmax = (
                np.sqrt(_l3sq(dd * dd)).max(0) / dtt * (1.0 + 1e-9)
                if len(tab) > 1
                else np.zeros(tab.shape[1])
            )
            self._vmax_key = self.track
        return self._vmax

    def _bp_one(self, tt, ids):
        """Predicted position and velocity of balloon ids[r] at time tt[r]: (R, 3), (R, 3)."""
        tab, i, a = self._tidx(tt)
        B = tab.shape[1]
        tabf = tab.reshape(
            -1, 3
        )  # tab[i, j] = tabf[i B + j] (same entries, one flat gather)
        li = i * B + ids
        p0, p1 = tabf[li], tabf[li + B]
        return p0 * (1.0 - a)[:, None] + p1 * a[:, None], (p1 - p0) / self.track[2]

    # -- candidates -------------------------------------------------------------
    def _cands(self, pos, vel, t, bv, avail):
        """Top-`branch` balloon ids per row by rough reach time. avail (R,B) bool."""
        bp = self._bp_all(t)
        r = bp - pos[:, None, :]
        d = np.sqrt(
            _l3sq(r * r)
        )  # = np.linalg.norm(r, axis=2) (_l3sq / _l3, see ff_ref._l3)
        vc = -_l3(r * (bv[None] - vel[:, None, :])) / np.maximum(d, 1e-6)
        est = d / np.maximum(vc, 15.0)
        est = np.where(avail, est, np.inf)
        k = min(self.branch, est.shape[1])
        idx = np.argpartition(est, k - 1, axis=1)[:, :k]
        ok = np.isfinite(np.take_along_axis(est, idx, axis=1))
        return idx, ok

    # -- rollouts -----------------------------------------------------------------
    def rollout(self, S, tgt, var, rel_mask, r2, resume=None, xp=None):
        """S: dict of row arrays (pos, vel, q, w, g, thr, t, popped (R,B), crash, optional ff).
        Fly each row toward its target until popped / horizon / ground. Returns new S.
        resume: (C, rows, cmd, k0) from self._cut of an earlier rollout of the same S, tgt, var and r2 with a shorter
        horizon; rows (ended there by the horizon alone, ascending) continue from that rollout's end state at step k0 with
        their last guidance command cmd, every other row keeps C's end state. A rollout started from S runs step k on
        exactly the same state, so the result is the same as a fresh rollout.
        xp (x rows, see enable_x; None = none): per-row sweep parameters (on, tk, B, e, nu, fb); at every decision step the
        rows with on set fly xlegs.command (nu updated in place) instead of their base law where it has a solution."""
        # The active rows are kept as compact working arrays and written back only when they finish. Each step tests only
        # the (row, target) pairs, which end rows (every target is a released balloon); the miss distance and the
        # incidental pops of every recorded step follow after the loop (_deferred_pops). Same arithmetic per element.
        src = S if resume is None else resume[0]
        pos, vel, q, w, g, thr = (
            src["pos"].copy(),
            src["vel"].copy(),
            src["q"].copy(),
            src["w"].copy(),
            src["g"].copy(),
            src["thr"].copy(),
        )
        t = src["t"].copy()
        popped = src["popped"].copy()
        crash = src["crash"].copy()
        # feedforward state (fresh rows start at the nose)
        ffs = src["ff"].copy() if "ff" in src else np.zeros((len(pos), FF_DIM))
        R = len(pos)
        miss = np.full(R, 1e9) if resume is None else src["miss"].copy()
        t_end = S["t"] + self.horizon
        elev = self.sim.elevation
        elev_g = elev + 0.5
        tl = S["t_launch"]
        dt = self.dt
        nsteps = int(round(self.horizon / self.dt))
        # compact state of the active (not finished) rows; ia maps them back to the full arrays
        k0 = 0
        if resume is None:
            ia = np.flatnonzero(~crash)
        else:
            ia, k0 = resume[1], int(resume[3])
        cP, cV, cQ, cW, cG, cTh = pos[ia], vel[ia], q[ia], w[ia], g[ia], thr[ia]
        cT, cPop, cTg, cVar, cTe = (
            t[ia],
            popped[ia],
            tgt[ia],
            var[ia],
            t_end[ia],
        )
        # the feedforward state as two contiguous arrays (n_ref, w_ref) inside the loop (ff_step's tuple state)
        cFF = _ff_split(ffs[ia])
        cN = np.zeros((len(ia), 3)) if resume is None else resume[2]
        cTe_m = cTe - 1e-9  # horizon test cT >= cTe - 1e-9
        ar = np.arange(len(ia))
        # the time-table values of every step at once. The loop's cT at the start of step k is the running sum
        # cT + dt + ... + dt, which cumsum reproduces term by term; c0 maps the active rows to the rows of this
        # precomputation (identity until the first row finishes). Without active rows or steps the loop does not run.
        n0, nk = len(ia), nsteps - k0
        pre_X = pre_hint = hz_any = None
        fast = False
        if n0 and nk > 0:
            # one more row than the step count: tk[k + 1] is cT after step k (horizon test tables below)
            tk = np.cumsum(np.concatenate([cT[None], np.full((nk, n0), dt)]), axis=0)
            trk = tk[:nk] - tl
            # every time-only quantity of the steps packed per (step, row) (GivenSim.step_tables), field-major
            # (F, nk, n0), so each step's block pre_X[:, k] has contiguous field rows
            pre_X, pre_B = self.sim.step_tables(trk, fm=True, dt=dt)
            # fast: T = T0 thr == T0 on every row and step, which holds when every row starts at thr == 1.0 (then
            # clip(1, 1 - s, 1 + s) keeps it at 1.0 while burning) and T0 == +0.0 at and after a row's first step without
            # burn (thr then decays to a finite value >= 0, and 0 * thr == +0.0)
            fast = (
                bool((cTh == 1.0).all())
                and self.sim.thr_rate * dt >= 0.0
                and bool(
                    (pre_X[0][np.logical_or.accumulate(~pre_B, axis=0)] == 0.0).all()
                )
            )
            # per step k, for all rows (then also every subset of them): 1 burning, -1 none burning, -2 past burn_end
            # (no thrust at k or later: GivenSim.step coast branch), 0 mixed (step decides)
            pre_hint = np.where(
                (trk > self.sim.burn_end).all(1),
                -2,
                np.where(pre_B.all(1), 1, np.where(pre_B.any(1), 0, -1)),
            ).tolist()
            # the horizon test cT >= cTe - 1e-9 of every step, and per step whether any row meets it
            hzt = tk[1:] >= cTe_m[None]
            hz_any = hzt.any(1).tolist()
        c0 = None  # None = all rows of the precomputation, in order
        vkey = None  # cVar the cached variant rows belong to
        ttk = None if pre_X is None else _TargetTrack(self, tk, dt, cTg)
        rec = []  # per step: (ia, p0, p1, t1, target dmin2)
        # rows that end by the horizon alone (resumable by a longer-horizon rollout, see resume); None = not resumable
        cut_r, cut_n = [], []
        for k in range(k0, nsteps):
            n = len(ia)
            if n == 0:
                break
            jk = k - k0
            tr = None
            tab = (
                (pre_X[:, jk], pre_B[jk], pre_hint[jk], fast)
                if c0 is None
                else (pre_X[:, jk].take(c0, axis=1), pre_B[jk][c0], pre_hint[jk], fast)
            )
            Rm = None
            # decision step; skipped when every row is past the burn with a finite throttle: then T = 0 on every row and
            # GivenSim.step takes its coast branch, where the thrust-direction command reaches no output
            if k % self.dec_n == 0 and not (
                pre_hint[k - k0] == -2 and bool(np.isfinite(cTh).all())
            ):
                tp, tv = self._bp_one(cT, cTg)
                if (
                    vkey is not cVar
                ):  # variant rows and their flags change only when rows finish (cached)
                    vv = self.V[cVar]
                    vi_on = bool((vv[:, 5] > 0).any())
                    lf_on = bool((vv[:, 11] > 0).any())
                    vcols = [vv[:, c] for c in range(12)]
                    vkey = cVar
                if vi_on:
                    # the nose from the step's own rotation matrix (quat_rot(q)[:, :, 2]), which the step then reuses;
                    # a_av from the step tables when T0 thr == T0
                    Rm = quat_rot(cQ)
                    a_av = tab[0][7] if fast else self.sim.accel_avail(cT - tl, cTh)
                    cN = guidance(
                        cP,
                        cV,
                        tp,
                        tv,
                        cP[:, 2] - elev,
                        boost=vcols[2],
                        nav=vcols[0],
                        tgo_min=vcols[1],
                        off_up=vcols[3],
                        off_side=vcols[4],
                        vi=vcols[5],
                        nose=Rm[:, :, 2],
                        a_av=a_av,
                        vi_wr=self.vi_wr,
                        vi_tau0=self.sim.gtau,
                        los_fill=vcols[11],
                    )
                elif lf_on:  # lean rows without viable-intercept terms: the regular law + the fill, with A as above
                    a_av = tab[0][7] if fast else self.sim.accel_avail(cT - tl, cTh)
                    cN = guidance(
                        cP,
                        cV,
                        tp,
                        tv,
                        cP[:, 2] - elev,
                        boost=vcols[2],
                        nav=vcols[0],
                        tgo_min=vcols[1],
                        off_up=vcols[3],
                        off_side=vcols[4],
                        a_av=a_av,
                        los_fill=vcols[11],
                    )
                else:
                    cN = guidance(
                        cP,
                        cV,
                        tp,
                        tv,
                        cP[:, 2] - elev,
                        boost=vcols[2],
                        nav=vcols[0],
                        tgo_min=vcols[1],
                        off_up=vcols[3],
                        off_side=vcols[4],
                    )
                if xp is not None:
                    self._x_step(xp, ia, cP, cV, cT, cN, Rm, tl, elev)
            p0 = cP
            # full throttle (thr_cmd 1.0); after the first step every row's ff state is an ff_step output (ff_step normed)
            cP, cV, cQ, cW, cG, cTh, cFF = self.sim.step(
                cP,
                cV,
                cQ,
                cW,
                cG,
                cTh,
                tr,
                cN,
                1.0,
                dt,
                ffs=cFF,
                ff_normed=k > k0,
                tab=tab,
                Rm=Rm,
            )
            cT = cT + dt
            gnd = cP[:, 2] < elev_g
            # (row, target) swept-segment test, the per-step arithmetic; hits are written into cPop. A row whose target is
            # popped finishes at once, so after the first step cPop[ar, cTg] is this step's target hit
            b0, b1 = ttk.get(k - k0, c0)  # (3, n) each
            dtg2 = _seg_d2T(p0.T - b0, cP.T - b1)  # component-major target test
            rec.append((ia, p0, cP, cT, dtg2))
            hit = dtg2 <= r2
            if _nz(hit):
                cPop[ar[hit], cTg[hit]] = True
                ended = (cPop[ar, cTg] | gnd) if k == k0 else (hit | gnd)
            else:
                ended = (cPop[ar, cTg] | gnd) if k == k0 else gnd
            if hz_any[k - k0]:
                hz = hzt[k - k0] if c0 is None else hzt[k - k0][c0]
            else:
                hz = None  # no row reaches the horizon in this step
            fin = ended if hz is None else (ended | hz)
            if _nz(fin):  # count_nonzero, the same test as fin.any()
                if cut_r is not None and hz is not None:
                    cm = hz & ~ended
                    if cm.any():
                        if k == nsteps - 1:
                            cut_r.append(ia[cm])
                            cut_n.append(cN[cm])
                        else:  # ended by the horizon before the last step: not resumable
                            cut_r = None
                j = ia[fin]
                pos[j], vel[j], q[j], w[j], g[j], thr[j] = (
                    cP[fin],
                    cV[fin],
                    cQ[fin],
                    cW[fin],
                    cG[fin],
                    cTh[fin],
                )
                t[j], popped[j], crash[j] = (
                    cT[fin],
                    cPop[fin],
                    gnd[fin],
                )
                ffs[j, :3], ffs[j, 3:] = cFF[0][fin], cFF[1][fin]
                kp = ~fin
                ia = ia[kp]
                cP, cV, cQ, cW, cG, cTh = (
                    cP[kp],
                    cV[kp],
                    cQ[kp],
                    cW[kp],
                    cG[kp],
                    cTh[kp],
                )
                cT, cPop, cTg, cVar, cTe, cTe_m, cN = (
                    cT[kp],
                    cPop[kp],
                    cTg[kp],
                    cVar[kp],
                    cTe[kp],
                    cTe_m[kp],
                    cN[kp],
                )
                cFF = (cFF[0][kp], cFF[1][kp])
                ar = ar[: len(ia)]
                c0 = kp.nonzero()[0] if c0 is None else c0[kp]
        if len(ia):  # rows still flying at the horizon step limit
            pos[ia], vel[ia], q[ia], w[ia], g[ia], thr[ia] = cP, cV, cQ, cW, cG, cTh
            t[ia], popped[ia] = cT, cPop
            ffs[ia, :3], ffs[ia, 3:] = cFF
            if cut_r is not None:
                cut_r.append(ia)
                cut_n.append(cN)
        if rec:
            # miss = the running minimum np.minimum(miss, sqrt(dtg2)) over each row's steps (in step order), then the
            # incidental pops of every recorded step
            np.minimum.at(
                miss,
                np.concatenate([x[0] for x in rec]),
                np.sqrt(np.concatenate([x[4] for x in rec])),
            )
            _deferred_pops(self, rec, popped, rel_mask, r2, dt)
        self._cut = None
        if cut_r is not None:
            rows = np.concatenate(cut_r) if cut_r else np.zeros(0, int)
            cmd = np.concatenate(cut_n) if cut_n else np.zeros((0, 3))
            o = np.argsort(rows, kind="stable")  # the order of ia (ascending row ids)
            self._cut = (rows[o], cmd[o], nsteps)
        out = dict(
            pos=pos,
            vel=vel,
            q=q,
            w=w,
            g=g,
            thr=thr,
            t=t,
            popped=popped,
            crash=crash,
            t_launch=S["t_launch"],
            miss=miss,
        )
        out["ff"] = ffs
        return out

    def _value(self, C, bp0, bv, rel_mask):
        """Critic estimate of future return from each row's end state (features.build as in training)."""
        d = self.critic
        R = len(C["pos"])
        nose = quat_rot(C["q"])[:, :, 2]
        t_abs = C["t"]
        t_rel = t_abs - C["t_launch"]
        mv = self._bp_all(t_abs)
        b_pos = np.where(rel_mask[None, :, None], mv, bp0[None])
        b_vel = np.broadcast_to(np.where(rel_mask[:, None], bv, 0.0)[None], b_pos.shape)
        st = np.where(C["popped"], 2, np.where(rel_mask[None], 1, 0))
        o, _, _ = F.build(
            C["pos"],
            C["vel"],
            nose,
            np.ones(R, bool),
            t_rel,
            t_abs,
            b_pos,
            b_vel,
            st,
            elevation=self.sim.elevation,
            burn_time=self.sim.burn_time,
            pad=np.array([0.0, 0.0, self.sim.elevation]),
        )
        return self._mlp(d, o)

    @staticmethod
    def _mlp(d, o):
        x = np.clip((o - d["obs_mean"]) / np.sqrt(d["obs_var"] + 1e-8), -10, 10)
        k = 0
        while f"v_W{k}" in d:
            x = x @ d[f"v_W{k}"].T + d[f"v_b{k}"]
            if f"v_W{k + 1}" in d:
                x = np.tanh(x)
            k += 1
        return x[:, 0]

    def _reach_times(self, C, bv, rel_mask):
        """Earliest time tau_j (s after each row's end state) at which balloon j can be reached with a fixed
        thrust direction at full given thrust. From (p, v, t) the reachable set after tau is the ball
            centre  p + v tau - 1/2 g z tau^2,   radius  margin * int_0^tau (tau - s) a_T(t + s) ds,
        a_T = T/m from the given thrust curve and mass (GivenSim tables), no turn delay.
        Balloon j is reachable at tau when |b_j(t + tau) - centre| <= radius + deep hit radius; tau_j is the
        first grid crossing, refined linearly. Only the reach_m candidates per row with the smallest rough
        reach estimate (as _cands) are evaluated. Returns (R, M) tau (inf = unreachable within reach_tmax)."""
        R = len(C["pos"])
        dtau = self.reach_dt
        taus = np.arange(1, int(round(self.reach_tmax / dtau)) + 1) * dtau  # (n,)
        n = len(taus)
        avail = rel_mask[None] & ~C["popped"]
        M = min(self.reach_m, int(rel_mask.sum()))
        if M == 0 or R == 0:
            return np.full((R, 1), np.inf)
        pos, vel, t_abs = C["pos"], C["vel"], C["t"]
        # candidate preselection (rough closing-speed reach estimate, as _cands)
        b_now = self._bp_all(t_abs)
        r = b_now - pos[:, None, :]
        d = np.sqrt(_l3sq(r * r))  # (.).sum(2) as _l3sq / _l3 (see ff_ref._l3)
        vc = -_l3(r * (bv[None] - vel[:, None, :])) / np.maximum(d, 1e-6)
        est = np.where(avail, d / np.maximum(vc, 15.0), np.inf)
        ids = (
            np.argpartition(est, M - 1, axis=1)[:, :M]
            if M < est.shape[1]
            else np.tile(np.arange(M), (R, 1))
        )
        valid = np.isfinite(np.take_along_axis(est, ids, 1))  # (R, M)
        # ball radius on the grid (midpoint rule on the given T/m table)
        t_rel = t_abs - C["t_launch"]
        s_mid = (np.arange(n) + 0.5) * dtau
        a = self.sim.accel_avail((t_rel[:, None] + s_mid[None]).ravel(), 1.0).reshape(
            R, n
        )
        A1 = np.cumsum(a * dtau, 1)
        A2 = np.cumsum(a * s_mid[None] * dtau, 1)
        rho = self.reach_margin * (taus[None] * A1 - A2)  # (R, n)
        ctr = pos[:, None, :] + vel[:, None, :] * taus[None, :, None]
        ctr[:, :, 2] -= 0.5 * G * taus[None] ** 2  # (R, n, 3)
        tt = t_abs[:, None] + taus[None]  # (R, n)
        tab, i, af = self._tidx(tt.ravel())
        af = af.reshape(R, n, 1, 1)
        # tab[i, j] gathered through one flat index (i B + j) into the (T B, 3) view: the same entries, faster
        B = tab.shape[1]
        tabf = tab.reshape(-1, 3)
        li = i.reshape(R, n, 1) * B + ids[:, None, :]
        bp = tabf[li] * (1.0 - af) + tabf[li + B] * af  # (R, n, M, 3)
        dx = bp - ctr[:, :, None, :]
        dn = np.sqrt(_l3sq(dx * dx))  # (R, n, M); (.).sum(3) as _l3sq
        rad = np.sqrt(self.r2_deep)
        gap = dn - rho[:, :, None] - rad  # <= 0: inside the ball
        ok = gap <= 0
        anyok = ok.any(1) & valid
        k = np.argmax(ok, axis=1)  # first grid hit (R, M)
        gk = np.take_along_axis(gap, k[:, None], 1)[:, 0]
        gp = np.take_along_axis(gap, np.maximum(k - 1, 0)[:, None], 1)[:, 0]
        fr = np.where(k > 0, gp / np.maximum(gp - gk, 1e-9), 1.0)  # linear crossing
        tj = np.where(k > 0, taus[np.maximum(k - 1, 0)] + fr * dtau, taus[0])
        tau = np.where(anyok, tj, np.inf)
        return tau

    def _reach(self, C, bv, rel_mask):
        """Leaf reach value from the reach times (soft form): the sum over the reach_k earliest balloons of
        exp(-tau_j / reach_tau0)."""
        tau = self._reach_times(C, bv, rel_mask)
        k = min(self.reach_k, tau.shape[1])
        tk = np.partition(tau, k - 1, axis=1)[:, :k]
        return np.exp(-tk / self.reach_tau0).sum(1)

    # -- search -----------------------------------------------------------------
    def plan(self, S0, bp0, bv, t0, rel_mask, cur=None):
        """S0: single-row state dict. Returns (best first move (target id, variant row) or None, info)."""
        S = {k: (v if k == "t_launch" else v.copy()) for k, v in S0.items()}
        x0 = None  # x rows of level 0: (xp, targets)
        first = np.full(1, -1)
        second = np.full(1, -1)
        lsec = []
        dpop = np.zeros(1)  # depth-discounted pops so far per row
        leaves = []
        slog = (
            self.seq_log_at is not None
            and self.seq_log is None
            and float(t0) - float(S0["t_launch"]) >= self.seq_log_at - 1e-9
        )
        H, lhist, lvt = (
            [()],
            [],
            [],
        )  # seq_log: per row the legs so far, per leaf its legs and its critic term
        for dlev in range(self.depth):
            avail = rel_mask[None] & ~S["popped"]
            idx, ok = self._cands(S["pos"], S["vel"], S["t"], bv, avail)
            if dlev == 1:  # x rows: the level-1 child must fly to the row's t2
                f2 = first % (len(rel_mask) + 1) - 1
                fr = np.flatnonzero(f2 >= 0)
                if len(fr):
                    idx[fr] = f2[fr][:, None]
                    ok[fr] = False
                    ok[fr, 0] = avail[fr, f2[fr]]
            rr, cc = np.nonzero(ok & ~S["crash"][:, None])
            if len(rr) == 0:
                break
            nV = len(self.V)
            xp = nxc = (
                None  # x rows: per-row sweep parameters for rollout / first-move codes with their t2
            )
            if dlev == 0:
                nR = self.nV0  # regular variant rows
                xs = self._x_rows(S, rr, idx[rr, cc], avail, bv)
                tgt = np.repeat(idx[rr, cc], nR)
                var = np.tile(np.arange(nR), len(rr))
                rr = np.repeat(rr, nR)
                if (
                    xs is not None
                ):  # x rows after the regular rows; their t2 enters the first-move code
                    xrr, xtg, xvr, xt2, xtk, xB, xe, xnu, xwp, xnb = xs
                    nreg, nx = len(tgt), len(xtg)
                    nxc = np.concatenate([np.full(nreg, -1), xt2])
                    tgt = np.concatenate([tgt, xtg])
                    var = np.concatenate([var, xvr])
                    rr = np.concatenate([rr, xrr])
                    on = np.concatenate([np.zeros(nreg, bool), np.ones(nx, bool)])

                    def pad(a, nreg=nreg):
                        return np.concatenate(
                            [np.full((nreg,) + a.shape[1:], np.nan), a]
                        )

                    def pdi(a, nreg=nreg):
                        return np.concatenate([np.full(nreg, -1), a])

                    xp = dict(
                        on=on,
                        on0=on.copy(),
                        fb=np.zeros(len(on), bool),
                        tk=pad(xtk),
                        B=pad(xB),
                        e=pad(xe),
                        nu=pad(xnu),
                        t2=pdi(xt2),
                        wp=pdi(xwp),
                        nb=pdi(xnb),
                    )
            else:
                tgt = idx[rr, cc]
                var = np.full(len(rr), min(self.deep_var, nV - 1))
            C = {
                k: (v if k == "t_launch" else v[rr])
                for k, v in S.items()
                if k != "miss"
            }
            Bp = (
                len(rel_mask) + 1
            )  # first move = (choice, t2): the x rows' t2 is part of it
            choice = (tgt * nV + var) * Bp + ((nxc + 1) if nxc is not None else 0)
            if xp is not None:
                xp["nu0"] = xp["nu"].copy()
                xp["code"] = choice
            fst = np.where(first[rr] >= 0, first[rr], choice)
            sec = tgt if dlev == 1 else second[rr]
            C0 = C
            C = self.rollout(
                C0,
                tgt,
                var,
                rel_mask,
                r2=self.r2 if dlev == 0 else self.r2_deep,
                xp=xp,
            )
            if (
                dlev == 0
                and self.long_horizon > self.horizon
                and not C["popped"][np.arange(len(tgt)), tgt].any()
            ):
                h = self.horizon
                cut = self._cut  # rows the short rollout ended by its horizon alone
                if (
                    xp is not None and cut is None
                ):  # a fresh long rollout: the x rows start over
                    xp["nu"], xp["on"], xp["fb"] = (
                        xp["nu0"].copy(),
                        xp["on0"].copy(),
                        np.zeros(len(xp["on0"]), bool),
                    )
                self.horizon = self.long_horizon  # nothing reachable: look further once
                # continue those rows from where the short rollout stopped instead of re-flying the first `horizon`
                # seconds (the same steps on the same state; see rollout(resume))
                C = self.rollout(
                    C0,
                    tgt,
                    var,
                    rel_mask,
                    r2=self.r2,
                    resume=None if cut is None else (C, cut[0], cut[1], cut[2]),
                    xp=xp,
                )
                self.horizon = h
            npop = C["popped"].sum(1) - S0["popped"].sum()
            if slog:
                newp = C["popped"] & ~C0["popped"]
                tr_ = C["t"] - C["t_launch"]
                Hc = [
                    H[r]
                    + (
                        (
                            int(tgt[i]),
                            int(var[i]),
                            float(tr_[i]),
                            [int(b) for b in np.flatnonzero(newp[i])],
                            [float(x) for x in C["pos"][i]],
                        ),
                    )
                    for i, r in enumerate(rr)
                ]
            dnew = dpop[rr] + self.depth_gamma**dlev * (
                C["popped"].sum(1) - C0["popped"].sum(1)
            )
            # lam_late: plans made at or after lam_late_t s after launch use lam_late as the in-burn time cost; time after
            # burnout costs lam_coast per s
            lam_b = self.lam if t0 - C["t_launch"] < self.lam_late_t else self.lam_late
            tb = C["t_launch"] + self.sim.burn_time
            tcost = lam_b * (
                np.minimum(C["t"], tb) - np.minimum(t0, tb)
            ) + self.lam_coast * (np.maximum(C["t"], tb) - np.maximum(t0, tb))
            key = dnew - tcost - 5.0 * C["crash"]
            vterm = self.critic_beta * np.where(
                C["crash"], 0.0, self._value(C, bp0, bv, rel_mask)
            )
            key = key + vterm
            if self.reach_beta:
                key = key + self.reach_beta * np.where(
                    C["crash"], 0.0, self._reach(C, bv, rel_mask)
                )
            if cur is not None:
                key = key + self.stick * (fst // Bp // len(self.V) == cur)
            # keep children that popped their target (or at least something)
            hit_t = C["popped"][np.arange(len(tgt)), tgt]
            if xp is not None:
                x0 = (xp, tgt.copy())
            keep = (hit_t | (npop > 0)) & ~C["crash"]
            for i in np.flatnonzero(keep):
                leaves.append(
                    (key[i], int(fst[i]), int(npop[i]), float(C["t"][i] - t0))
                )
                lsec.append(int(sec[i]))
                if slog:
                    lhist.append(Hc[i])
                    lvt.append(float(vterm[i]))
            if (
                dlev == 0
            ):  # fallback: nothing hits within the horizon -> closest approach, no crash
                for i in np.flatnonzero(~keep & ~C["crash"]):
                    low = max(
                        0.0, 30.0 - (C["pos"][i, 2] - self.sim.elevation)
                    )  # ending low = lost energy
                    leaves.append(
                        (
                            -10.0 - 0.1 * C["miss"][i] - tcost[i] - self.low_pen * low,
                            int(fst[i]),
                            0,
                            float(C["t"][i] - t0),
                        )
                    )
                    lsec.append(-1)
                    if slog:
                        lhist.append(Hc[i])
                        lvt.append(float(vterm[i]))
            if not keep.any():
                break
            ki = np.flatnonzero(keep)
            order = ki[np.argsort(-key[ki])]
            # diversity: at most 3 children per first choice at each level
            sel, cnt = [], {}
            for i in order:
                f = int(fst[i])
                c = cnt.get(f, 0)
                if c < 3:
                    sel.append(i)
                    cnt[f] = c + 1
                if len(sel) >= self.beam:
                    break
            sel = np.array(sel)
            S = {
                k: (v if k == "t_launch" else v[sel])
                for k, v in C.items()
                if k != "miss"
            }
            first = fst[sel]
            dpop = dnew[sel]
            second = sec[sel]
            if slog:
                H = [Hc[i] for i in sel]
        if not leaves:
            return None, None
        # value of a first target = best leaf key reachable through it (deeper leaves have more pops)
        bestf = {}
        bests = {}
        for (kv, f, n, dt_), sc in zip(leaves, lsec):
            if f not in bestf or kv > bestf[f][0]:
                bestf[f] = (kv, n, dt_)
                bests[f] = sc
        nV = len(self.V)
        Bp = len(rel_mask) + 1
        fc = max(bestf, key=lambda x: bestf[x][0])
        self.last_t2 = fc % Bp - 1
        f = fc // Bp
        self.last = (f, bestf[fc])
        if (
            slog
        ):  # the chosen first move's best leaf (the leaf whose key bestf[fc] holds)
            ib = max(
                (i for i, lv in enumerate(leaves) if lv[1] == fc),
                key=lambda i: leaves[i][0],
            )
            self.seq_log = dict(
                t_rel=float(t0) - float(S0["t_launch"]),
                first=int(f),
                key=float(leaves[ib][0]),
                npop=int(leaves[ib][2]),
                legs=[list(x) for x in lhist[ib]],
                n_leaves=len(leaves),
                vterm=lvt[ib],
            )
        self._x_done(x0, fc, Bp, nV)
        return (f // nV, f % nV), bestf[fc]
