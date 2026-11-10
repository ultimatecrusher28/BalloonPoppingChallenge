"""On-board rollout beam planner (receding horizon) over balloon target sequences.

Each candidate "fly to balloon b" segment is simulated with the SAME guidance law
the agent executes (``guidance``) through the given-only simulator (``GivenSim``),
so a plan is what the rocket will actually fly (the classical GNC's plan-vs-flight
gap was ~2 pops, research/LEGACY-VERDICTS.md §1). Every step of every rollout runs
the env's swept-segment pop test against all released balloons, so incidental pops
count. Balloons are predicted at their observed velocity (observations only;
not-yet-released balloons are not targets).

Pattern: receding-horizon beam search over discrete target choices (policy-guided
beam search / Go-Explore "return then explore", research/rl/pgbs.py; tree search
with a learned-free rollout policy, cf. MCTS rollouts in Silver et al. 2016).
"""

from __future__ import annotations

import numpy as np

from . import features as F
from ..follow.ff_ref import _add_reduce, _clip, _cross, _l3, _l3sq, _norm, _nz, _sum3
from .given_sim import quat_nose, quat_rot

G = 9.80665
_EZ = np.array([0.0, 0.0, 1.0])


def _any(x):
    """np.any(x) for the scalars / row arrays guidance takes (without the wrapper's overhead; round 2: count_nonzero,
    the same nonzero test, NaN counting as True)."""
    return _nz(x) > 0 if isinstance(x, np.ndarray) else bool(np.any(x))


def _col(x):
    """np.reshape(x, (-1, 1)), as the view x[:, None] for a 1-D array (round 2; the same values)."""
    return x[:, None] if isinstance(x, np.ndarray) and x.ndim == 1 else np.reshape(x, (-1, 1))


_R60, _R35, _MPI2 = np.radians(60.0), np.radians(35.0), -np.pi / 2  # guidance floor constants (the same values)


def _rows(x, n):
    """np.broadcast_to(np.reshape(x, -1), (n,)), as a plain view when x already has n rows."""
    if isinstance(x, np.ndarray) and x.shape == (n,):
        return x
    return np.broadcast_to(np.reshape(x, -1), (n,))


VI_T = np.arange(
    0.1, 6.0 + 1e-9, 0.05
)  # intercept-time grid of the viable-intercept search (s)
_VI_GT2 = 0.5 * G * VI_T[None, :] ** 2  # the gravity term of viable_dir / repoint_dir, evaluated once (same values)


def viable_dir(pos, vel, tp, tv, nose, a_av, margin, wr=1.5, tau0=0.064):
    """W3 P3b: earliest viable constant-direction full-thrust intercept (RLUtilities is_viable pattern with a
    turn delay; time-optimal steering for bounded thrust against a constant-velocity target).
    dx(T) = tp + tv T - (pos + vel T + 1/2 g T^2); viable when T > tau and 2|dx| / (T - tau)^2 <= margin a_av,
    tau = tau0 + angle(nose, dx) / wr. Returns (unit direction of dx(T*) (R,3), T* (R,), nan if none)."""
    T = VI_T[None, :, None]
    dx = tp[:, None] + (tv - vel)[:, None] * T - pos[:, None]
    dx[:, :, 2] += _VI_GT2
    # round 2: the length-3 reductions as _l3sq / _l3 (the same values as add.reduce, see ff_ref._l3)
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


def exit_thrust(e, a):
    """Unit thrust direction whose net acceleration (thrust a minus gravity) points along unit e (R,3);
    straight up where gravity cannot be cancelled along e."""
    ez = e[:, 2]
    disc = a**2 - G**2 * (1.0 - ez**2)
    s = -G * ez + np.sqrt(np.maximum(disc, 0.0))
    u = s[:, None] * e + np.array([0.0, 0.0, G])
    u = np.where((disc > 0)[:, None] & (s > 0)[:, None], u, np.array([0.0, 0.0, 1.0]))
    return u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)


def repoint_dir(
    pos, vel, tp, tv, pn, vn, nose, a_av, margin, rp, wr=1.5, tau0=0.064, lead=1.5
):
    """Stream M late re-point (M3 two-segment legs): thrust u1 now, then full thrust u2 over the last rp s,
    u2 = exit_thrust toward the next balloon (pn + vn (T + lead) - target position at T). For each T on VI_T,
    dx(T) as viable_dir; with T1 = T - tau - rp > 0 the leg is viable when
        |dx - 1/2 m a u2 rp^2| <= m a (T1^2 / 2 + T1 rp),   u1 = direction of the left side,
    and with T - tau <= rp (inside the re-point window) it is the plain viable_dir test, direction dx (which
    is ~u2 when phase 1 set it up). Returns (direction (R,3), T* (R,), nan if none) at the earliest viable T."""
    T = VI_T[None, :, None]
    dx = tp[:, None] + (tv - vel)[:, None] * T - pos[:, None]
    dx[:, :, 2] += _VI_GT2
    a = (margin * a_av)[:, None]
    e = (pn[:, None] + vn[:, None] * (T + lead)) - (tp[:, None] + tv[:, None] * T)
    e = e / np.maximum(np.linalg.norm(e, axis=2, keepdims=True), 1e-9)
    R, n = len(pos), len(VI_T)
    u2 = exit_thrust(e.reshape(-1, 3), np.repeat(a[:, 0], n)).reshape(R, n, 3)
    rpv = np.reshape(rp, (-1, 1))
    dx1 = dx - (0.5 * a * rpv**2)[:, :, None] * u2
    d1n = np.linalg.norm(dx1, axis=2)
    dn = np.linalg.norm(dx, axis=2)
    cos1 = np.clip((dx1 * nose[:, None]).sum(2) / np.maximum(d1n, 1e-9), -1.0, 1.0)
    cos0 = np.clip((dx * nose[:, None]).sum(2) / np.maximum(dn, 1e-9), -1.0, 1.0)
    te1 = VI_T[None, :] - tau0 - np.arccos(cos1) / wr - rpv  # phase-1 thrust time T1
    te0 = VI_T[None, :] - tau0 - np.arccos(cos0) / wr
    ph1 = te1 > 1e-3
    ok1 = ph1 & (d1n <= a * (0.5 * te1**2 + te1 * rpv))
    ok0 = ~ph1 & (te0 > 1e-3) & (2.0 * dn <= a * np.maximum(te0, 1e-3) ** 2)
    ok = ok1 | ok0
    anyok = ok.any(1)
    k = np.argmax(ok, axis=1)
    r = np.arange(R)
    d = np.where(
        ok1[r, k][:, None],
        dx1[r, k] / np.maximum(d1n[r, k], 1e-9)[:, None],
        dx[r, k] / np.maximum(dn[r, k], 1e-9)[:, None],
    )
    return d, np.where(anyok, VI_T[k], np.nan)


def zv_dir(pos, vel, tp, tv, pn, vn, tgo, zb, a_av, lead=1.5, fade=(0.2, 0.5)):
    """Round 10 terminal-velocity steering (ZEM/ZEV feedback, Guo/Hawkins/Wie 2013): thrust direction of
        a = 6 ZEM / t_go^2 - 2 ZEV / t_go,
    ZEM = tp + tv t_go - (pos + vel t_go - 1/2 g z t_go^2) (balloon drift and gravity included),
    ZEV = v* - (vel - g z t_go). Exit velocity v* = s* d*, d* = unit((1 - zb) u_los + zb u_next), u_los = unit
    of the approach (tp + tv t_go - pos), u_next = unit(pn + vn (t_go + lead) - (tp + tv t_go)) (zb -> 0 where
    there is no next balloon, pn nan); s* = straight-line constant-thrust exit speed, clipped to
    [|vel|, |vel| + a_av t_go]. The ZEV term fades out as t_go drops from fade[1] to fade[0] s (gains blow up).
    tgo, zb, a_av per row. Returns unit thrust directions (R,3) (fixed full thrust along it)."""
    t = np.maximum(tgo, 0.05)[:, None]
    pa = tp + tv * t
    zem = pa - (pos + vel * t)
    zem[:, 2] += 0.5 * G * t[:, 0] ** 2
    dl = pa - pos
    u_los = dl / np.maximum(np.linalg.norm(dl, axis=1, keepdims=True), 1e-6)
    has = np.isfinite(pn[:, 0])
    dn = np.where(
        has[:, None], np.nan_to_num(pn) + np.nan_to_num(vn) * (t + lead) - pa, u_los
    )
    u_n = dn / np.maximum(np.linalg.norm(dn, axis=1, keepdims=True), 1e-6)
    b = np.where(has, zb, 0.0)[:, None]
    d = (1.0 - b) * u_los + b * u_n
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    sp = np.linalg.norm(vel, axis=1)
    s = np.clip(
        2.0 * np.linalg.norm(dl, axis=1) / t[:, 0] - sp, sp, sp + a_av * t[:, 0]
    )
    zev = s[:, None] * d - vel
    zev[:, 2] += G * t[:, 0]
    w = np.clip((t[:, 0] - fade[0]) / (fade[1] - fade[0]), 0.0, 1.0)[:, None]
    a = 6.0 * zem / t**2 - w * 2.0 * zev / t
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def next_k(pos, vel, tp, tv, bp, bv, avail, tgt, k):
    """Round 10 pair rows: the k balloons with the earliest rough reach estimate after tgt (as next_after), per row.
    Returns ids (R, k), -1 where fewer exist."""
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


def next_after(pos, vel, tp, tv, bp, bv, avail, tgt):
    """Late re-point: the balloon to leave toward after tgt, per row = earliest rough reach time (as
    BeamPlanner._cands) from tgt's position at the rough arrival time, flying at the current velocity.
    bp, bv (R,B,3) / (B,3) positions now and velocities; avail (R,B). Returns ids (R,), -1 if none."""
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
    k = np.argmin(est, axis=1)
    return np.where(np.isfinite(est[np.arange(len(pos)), k]), k, -1)


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
    rp=0.0,
    pn=None,
    vn=None,
    rp_blend=(0.25, 0.75),
    floor=None,
    zb=0.0,
    zk=0.0,
    la=0.0,
    lt=1.2,
):
    """Lead pursuit (ZEM / t_go-closing, as features.base_action) -> unit thrust dir (N,3).
    nav, tgo_min, boost may be scalars or per-row arrays (planner guidance variants).
    vi (per row) > 0: W3 P3b viable-intercept steering with that acceleration margin (needs nose, a_av),
    blended into the ZEM law as the intercept time T* drops from vi_blend[1] to vi_blend[0] s; rows with
    vi = 0 (default) are the v7 law, unchanged.
    rp (per row) > 0 with vi > 0 and a next balloon (pn, vn per row, nan = none): repoint_dir instead of
    viable_dir, blended into ZEM over rp_blend (point-mass check: (0.05, 0.2) costs 0.4-0.7 s over two legs).
    floor (stream OP; None = unchanged): (el_lo deg, vz_min m/s, h_min m) relaxes the 60 deg near-ground floor
    (h_agl < 15 m) to el_lo while climbing (vz > vz_min) at h_agl >= h_min; the descent (t_imp) rules are kept.
    zk (per row) > 0: round 10 terminal-velocity rows (zv_dir) replace the law above; t_go = zk x the viable reach
    time (margin 0.9; the ZEM t_go where none), exit direction blend zb toward the next balloon (pn, vn; needs
    nose, a_av). Rows with zk = 0 (default) are unchanged.
    la (per row) > 0: round 10 late lateral push (chwang approach-shape profile): + la * ramp * e2 is added to the
    law's acceleration before normalising, e2 = unit component of the direction to the next balloon (pn, vn)
    normal to the LOS, ramp = clip((lt - t_go) / (lt - 0.5), 0, 1) (0 at t_go = lt s, full from 0.5 s)."""
    r = tp - pos
    # speed (research/sc40/speed): np.cross / np.linalg.norm(axis=1) as the same per-component expressions (_cross, _norm)
    if _any(off_up) or _any(off_side):  # aim-point offset in the plane normal to the LOS
        los = r / np.maximum(_norm(r), 1e-6)[:, None]
        side = _cross(los, _EZ[None])
        side /= np.maximum(_norm(side), 1e-6)[:, None]
        up = _cross(side, los)
        r = r + _col(off_up) * up + _col(off_side) * side
    vrel = tv - vel
    dist = _norm(r)
    vc = -_l3(r * vrel) / np.maximum(dist, 1e-6)  # (.).sum(1) as _l3 (round 2)
    tgo = _clip(dist / np.maximum(vc, 12.0), tgo_min, 6.0)[:, None]
    a = _col(nav) * (r + vrel * tgo) / tgo**2
    a = a + _col(boost) * r / np.maximum(dist, 1e-6)[:, None]
    a = _clip(a, -amax, amax)
    if _any(la) and pn is not None:
        lam_ = np.broadcast_to(np.reshape(la, -1), (len(pos),)).astype(float)
        ml = (lam_ > 0) & np.isfinite(pn[:, 0])
        if ml.any():
            los = r[ml] / np.maximum(dist[ml], 1e-6)[:, None]
            dn = (pn[ml] + vn[ml] * tgo[ml]) - (tp[ml] + tv[ml] * tgo[ml])
            e2 = dn - (dn * los).sum(1, keepdims=True) * los
            e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-6)
            ltm = np.broadcast_to(np.reshape(lt, -1), (len(pos),)).astype(float)[ml]
            ramp = np.clip((ltm - tgo[ml, 0]) / np.maximum(ltm - 0.5, 1e-3), 0.0, 1.0)
            a[ml] += (lam_[ml] * ramp)[:, None] * e2
    a[:, 2] += G
    n = a / np.maximum(_norm(a), 1e-9)[:, None]
    if _any(vi):
        vim = _rows(vi, len(pos)).astype(float)
        m = vim > 0
        a_m = _rows(a_av, len(pos))[m]
        d, tstar = viable_dir(
            pos[m], vel[m], tp[m], tv[m], nose[m], a_m, vim[m], vi_wr, vi_tau0
        )
        # round 2: the blend bounds as scalars (the same per-row values) unless re-point rows set their own
        b0, b1 = float(vi_blend[0]), float(vi_blend[1])
        if _any(rp):
            rpm = np.broadcast_to(np.reshape(rp, -1), (len(pos),)).astype(float)[m]
            mr = (rpm > 0) & np.isfinite(pn[m][:, 0])
            if mr.any():
                b0, b1 = np.full(len(d), b0), np.full(len(d), b1)
                im = np.flatnonzero(m)[mr]
                dr, tr = repoint_dir(
                    pos[im],
                    vel[im],
                    tp[im],
                    tv[im],
                    pn[im],
                    vn[im],
                    nose[im],
                    a_m[mr],
                    vim[m][mr],
                    rpm[mr],
                    vi_wr,
                    vi_tau0,
                )
                ok = np.isfinite(
                    tr
                )  # no viable re-point leg: keep the plain viable direction
                d[mr] = np.where(ok[:, None], dr, d[mr])
                tstar[mr] = np.where(ok, tr, tstar[mr])
                b0[mr] = np.where(ok, rp_blend[0], b0[mr])
                b1[mr] = np.where(ok, rp_blend[1], b1[mr])
        w = _clip((tstar - b0) / (b1 - b0), 0.0, 1.0)
        w = np.where(np.isfinite(tstar), w, 0.0)[:, None]
        nb = w * d + (1.0 - w) * n[m]
        n[m] = nb / np.maximum(_norm(nb), 1e-9)[:, None]
    if _any(zk):
        zkm = np.broadcast_to(np.reshape(zk, -1), (len(pos),)).astype(float)
        mz = zkm > 0
        a_z = np.broadcast_to(np.reshape(a_av, -1), (len(pos),))[mz]
        _, tstar = viable_dir(
            pos[mz],
            vel[mz],
            tp[mz],
            tv[mz],
            nose[mz],
            a_z,
            np.full(mz.sum(), 0.9),
            vi_wr,
            vi_tau0,
        )
        tz = np.where(np.isfinite(tstar), zkm[mz] * tstar, tgo[mz, 0])
        zbm = np.broadcast_to(np.reshape(zb, -1), (len(pos),)).astype(float)[mz]
        pnm = np.full((mz.sum(), 3), np.nan) if pn is None else pn[mz]
        vnm = np.zeros((mz.sum(), 3)) if vn is None else vn[mz]
        n[mz] = zv_dir(pos[mz], vel[mz], tp[mz], tv[mz], pnm, vnm, tz, zbm, a_z)
    # ground-safety floor ("loose", features.action_to_command)
    vz = vel[:, 2]
    if floor is None and not _nz((h_agl < 30.0) | (vz < -0.5)):
        # round 2: no row is below 30 m AGL or descending faster than 0.5 m/s, so t_imp = inf and el_min = -pi/2 on every
        # row (NaN h_agl / vz included: their comparisons are False here and in the expressions below); n changes only
        # where el < -pi/2
        el = np.arcsin(_clip(n[:, 2], -1, 1))
        if not _nz(el < _MPI2):
            return n
    t_imp = np.where(vz < -0.5, h_agl / np.maximum(-vz, 1e-3), np.inf)
    el_min = np.where(
        (h_agl < 15.0) | (t_imp < 1.5),
        _R60,
        np.where((h_agl < 30.0) | (t_imp < 3.0), _R35, _MPI2),
    )
    if floor is not None:
        el_lo, vz_min, h_min = floor
        climb = (h_agl < 15.0) & (t_imp >= 1.5) & (vz > vz_min) & (h_agl >= h_min)
        el_min = np.where(climb, np.minimum(el_min, np.radians(el_lo)), el_min)
    el = np.arcsin(_clip(n[:, 2], -1, 1))
    fix = el < el_min
    if fix.any():
        hz = np.arctan2(n[fix, 1], n[fix, 0])
        e = el_min[fix]
        n[fix] = np.stack(
            [np.cos(e) * np.cos(hz), np.cos(e) * np.sin(hz), np.sin(e)], 1
        )
    return n


class _PopTest:
    """Per-step swept-segment pop test of BeamPlanner.rollout, restricted to candidate (row, balloon) pairs.

    Exactly the env test (dmin2 <= r2 between the rocket's and the balloon's step segments, released and not yet
    popped balloons) with identical per-element arithmetic, but evaluated only for pairs a distance bound cannot
    exclude. Every `refresh` steps (or when a row breaks its budget) the distance Dref of each active row's
    segment start to every balloon is computed; until the next refresh, a pair is skipped only if
        Dref - Rcap - vmax * Tcap > radius + eps,
    where Rcap bounds the row's own displacement from its reference point (checked every step against the
    actual displacement) and vmax bounds the balloon's speed (largest segment speed of its piecewise-linear
    track, or |bv| for constant velocity). Any violated budget forces a refresh, then a full test, so a
    skipped pair can never be within the radius: hits and the target miss distance are bit-identical.

    Speed (research/sc40/speed): the budget is checked on each step's end point only (its start point is the previous
    step's end point, already checked against the same reference, or the reference itself right after a refresh), and
    the time-span check is implied by the refresh period (a refresh at most every REFRESH steps, so the span is at
    most REFRESH * dt < Tcap). The compact pair indices are cached until the active rows change or a refresh, both
    segment ends are evaluated in one pass, and hits are written into `pop` in place."""

    REFRESH = 10  # steps between candidate refreshes
    A_MAX = 100.0  # m/s^2, acceleration allowance in the displacement budget (checked, not trusted)
    EPS = 1e-6  # m, round-off margin

    def __init__(self, pl, bp0, bv, t0, rel_mask, r2, R):
        self.pl, self.bp0, self.bv, self.t0, self.rel, self.r2 = (
            pl,
            bp0,
            bv,
            t0,
            rel_mask,
            r2,
        )
        self.dt = pl.dt
        self.track = pl.track
        self.rad = float(np.sqrt(r2))
        self.Tc = self.REFRESH * self.dt * (1.0 + 1e-9)
        if self.track is None:
            self.vmax = np.sqrt((bv * bv).sum(1))
        else:
            self.vmax = pl._track_vmax()
        self.refP = np.zeros((R, 3))
        self.refT = np.zeros(R)
        self.Rcap = np.full(R, -1.0)
        self.inv = np.full(R, -1)
        self.rows = None
        self.since = 0
        self._ia = self._tg = None  # active rows / targets the cached compact indices below belong to

    def _b(self, tau):
        """All balloon positions at per-row times tau: (n, B, 3), the expressions of rollout's original test."""
        if self.track is None:
            return self.bp0[None] + self.bv[None] * (tau - self.t0)[:, None, None]
        return self.pl._bp_all(tau)

    def full(self, p0, p1, t1, pop, tg):
        dt = self.dt
        if self.track is None:
            b0 = self.bp0[None] + self.bv[None] * (t1 - dt - self.t0)[:, None, None]
            b1 = b0 + self.bv[None] * dt
        else:
            b0, b1 = self.pl._bp_all(t1 - dt), self.pl._bp_all(t1)
        a0 = p0[:, None, :] - b0
        a1 = p1[:, None, :] - b1
        da = a1 - a0
        s = np.clip(-(a0 * da).sum(2) / np.maximum((da * da).sum(2), 1e-12), 0, 1)
        dmin2 = ((a0 + s[..., None] * da) ** 2).sum(2)
        hit = (dmin2 <= self.r2) & self.rel[None] & ~pop
        return hit, dmin2[np.arange(len(p0)), tg]

    MARGIN = 1e-3  # m, extra slack of the refresh bound (covers the rounding of the squared-distance expansion)

    def _refresh(self, ia, p0, p1, tau0, pop):
        """New reference points / budgets and a candidate set that contains every pair that can come within the radius
        before the next refresh (a superset is enough: hits are evaluated exactly). A pair is kept unless
            |p0 - b(t')| > Rcap + vmax (Tcap + |tau0 - t'|) + radius + eps + MARGIN
        for a balloon position b(t') at a time t' near the row's tau0 (the tracks are vmax-Lipschitz). Speed
        (research/sc40/speed): when all rows share one time (level 0) t' = tau0 and the balloons are placed once, with
        |p0 - b|^2 = |p0|^2 - 2 p0.b + |b|^2 (rounding ~1e-9 m^2 here, far inside MARGIN); otherwise t' is the track-table
        time nearest each row (|tau0 - t'| <= half a table step inside the table) and b(t') is a table row; rows outside
        the table (round 2) use t' = tau0 itself, b(tau0) by the hit test's own interpolation formula."""
        sp = np.sqrt(((p1 - p0) ** 2).sum(1)) / self.dt
        rcap = sp * self.Tc + 0.5 * self.A_MAX * self.Tc**2 + 1.0
        self.refP[ia] = p0
        self.refT[ia] = tau0
        self.Rcap[ia] = rcap
        c0 = rcap + (self.rad + self.EPS + self.MARGIN)
        if self.track is None or (tau0 == tau0[0]).all():
            b = self._b(tau0[:1])[0]  # (B, 3) at the common time (constant velocity: exact per row below)
            if self.track is None and not (tau0 == tau0[0]).all():
                dd = p0[:, None, :] - self._b(tau0)
                d2 = _add_reduce(dd * dd, 2)
            else:
                d2 = (_add_reduce(p0 * p0, 1)[:, None] + _add_reduce(b * b, 1)[None]) - 2.0 * (p0 @ b.T)
            c = c0[:, None] + self.vmax[None] * self.Tc
        else:
            tab, ts, dtt = self.track
            ii = _clip(np.rint((tau0 - ts) / dtt).astype(int), 0, len(tab) - 1)
            off = np.abs(tau0 - (ts + ii * dtt))
            dd = p0[:, None, :] - tab[ii]
            # rows outside the table (past its end, where the tracks are extrapolated from the end segment): the reference
            # point is the track position at tau0 itself (the same interpolation formula as the hit test), so t' = tau0
            # and the bound is not widened by |tau0 - t'| (round 2; see NOTES.md "Round 2" for the superset argument)
            far = off > 0.5 * dtt
            if far.any():
                o = far.nonzero()[0]
                dd[o] = p0[o][:, None, :] - self.pl._bp_all(tau0[o])
                off[o] = 0.0
            d2 = _add_reduce(dd * dd, 2)
            c = c0[:, None] + self.vmax[None] * (self.Tc + off)[:, None]
        m = (d2 <= c * c) & self.rel[None] & ~pop
        rc, jj = np.nonzero(m)
        self.rows, self.jj = ia[rc], jj
        self.since = 0
        self._ia = None

    def _sync(self, ia, tg, ar):
        """Compact reference points / budgets and pair indices for the active rows ia (cached while ia and tg are the
        same arrays and no refresh happened)."""
        if self._ia is ia and self._tg is tg:
            return
        n = len(ia)
        self._rp, self._rcap = self.refP[ia], self.Rcap[ia]
        inv = self.inv
        inv[:] = -1
        inv[ia] = ar
        rc = inv[self.rows]
        keep = rc >= 0
        rc, jj = rc[keep], self.jj[keep]
        self._rc, self._jjc = rc, jj
        rcat, jcat = np.concatenate([rc, ar]), np.concatenate([jj, tg])
        self._rcat, self._jcat = rcat, jcat
        self._relc = self.rel[jcat]
        if self.track is not None:  # both segment ends at once: rows of [t1 - dt, t1] / [p0, p1]
            self._rc2, self._jj2 = np.concatenate([rcat, rcat + n]), np.concatenate([jcat, jcat])
        self._ia, self._tg = ia, tg

    def _ok(self, p1):
        d1 = p1 - self._rp
        return bool((np.sqrt(_sum3(d1 * d1)) <= self._rcap).all())

    def _pairs(self, p0, p1, t1):
        """dmin2 of the cached pairs (compact row, balloon), same per-element arithmetic as full()."""
        dt = self.dt
        rc, jj = self._rcat, self._jcat
        if self.track is None:
            b0 = self.bp0[jj] + self.bv[jj] * (t1 - dt - self.t0)[rc][:, None]
            b1 = b0 + self.bv[jj] * dt
            a0 = p0[rc] - b0
            a1 = p1[rc] - b1
        else:
            tab, i, f = self.pl._tidx(np.concatenate([t1 - dt, t1]))
            rc2 = self._rc2
            B = tab.shape[1]
            tabf = tab.reshape(-1, 3)  # tab[i, j] = tabf[i B + j]
            li, ff = i[rc2] * B + self._jj2, f[rc2]
            b = tabf[li] * (1.0 - ff)[:, None] + tabf[li + B] * ff[:, None]
            a = np.concatenate([p0, p1])[rc2] - b
            P = len(rc)
            a0, a1 = a[:P], a[P:]
        da = a1 - a0
        s = _clip(-_sum3(a0 * da) / np.maximum(_sum3(da * da), 1e-12), 0, 1)
        return _sum3((a0 + s[:, None] * da) ** 2)

    def step(self, ia, p0, p1, t1, pop, tg, ar):
        """ia: full row ids of the n active rows; p0, p1 (n, 3) step start / end; t1 (n,) time after the step;
        pop (n, B) popped before the step, updated in place with this step's hits; tg (n,) target ids; ar = arange(n).
        Returns the target dmin2 (n,)."""
        ok = False
        if self.rows is not None and self.since < self.REFRESH:
            self._sync(ia, tg, ar)
            ok = self._ok(p1)
        if not ok:
            self._refresh(ia, p0, p1, t1 - self.dt, pop)
            self._sync(ia, tg, ar)
            if not self._ok(p1):
                self.rows = None
                hit, dtg = self.full(p0, p1, t1, pop, tg)
                pop |= hit
                return dtg
        self.since += 1
        d2 = self._pairs(p0, p1, t1)
        h = (d2 <= self.r2) & self._relc  # candidate pairs, then (row, target) pairs; OR into pop
        if h.any():
            pop[self._rcat[h], self._jcat[h]] = True
        return d2[len(self._rc):]


def _ff_split(F):
    """(N, 6) feedforward state -> (n_ref, w_ref), two C-contiguous (N, 3) copies (round 2)."""
    return np.ascontiguousarray(F[:, :3]), np.ascontiguousarray(F[:, 3:])


def _seg_d2(a0, a1):
    """Squared closest distance of the swept segments with relative end points a0, a1 (n, 3): the per-element arithmetic
    of _PopTest._pairs (and, for every value that is not NaN, of _PopTest.full)."""
    da = a1 - a0
    s = _clip(-_sum3(a0 * da) / np.maximum(_sum3(da * da), 1e-12), 0, 1)
    return _sum3((a0 + s[:, None] * da) ** 2)


def _seg_d2T(a0, a1):
    """_seg_d2 for component-major relative end points a0, a1 (3, n) (round 2, second pass: contiguous rows instead of
    column views and an inner broadcast); the same per-element operations, (p0 + p1) + p2 sums as _sum3."""
    da = a1 - a0
    p = a0 * da
    r = da * da
    s = _clip(-((p[0] + p[1]) + p[2]) / np.maximum((r[0] + r[1]) + r[2], 1e-12), 0, 1)
    e = (a0 + s[None] * da) ** 2
    return (e[0] + e[1]) + e[2]


def _bal_seg(pl, bp0, bv, t0, t1, dt, j):
    """Balloon j[e] at the two ends of the step ending at t1[e] (E,): (b0, b1) (E, 3) each, the expressions of
    _PopTest._pairs (track interpolation through one flat gather, or constant velocity)."""
    if pl.track is None:
        b0 = bp0[j] + bv[j] * (t1 - dt - t0)[:, None]
        return b0, b0 + bv[j] * dt
    tab, i, f = pl._tidx(np.concatenate([t1 - dt, t1]))
    B = tab.shape[1]
    tabf = tab.reshape(-1, 3)
    li = i * B + np.concatenate([j, j])
    b = tabf[li] * (1.0 - f)[:, None] + tabf[li + B] * f[:, None]
    E = len(j)
    return b[:E], b[E:]


class _TargetTrack:
    """Round 2 (research/sc40/speed): each rollout row's target balloon at both ends of every step, evaluated for CH steps
    at a time for the rows active then (the same per-element expressions as the per-step pop test). tk (L + 1, n0): the
    rows' time at the start of each step (tk[k + 1] = time after step k, the loop's running sum); tg (n0,) targets.
    Second pass: chunks of 128 steps, both ends stored per step as one component-major (2, 3, n) block (one gather when
    rows finished), for _seg_d2T."""

    CH = 128

    def __init__(self, pl, bp0, bv, t0, tk, dt, tg):
        self.pl, self.bp0, self.bv, self.t0, self.tk, self.dt, self.tg = pl, bp0, bv, t0, tk, dt, tg
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
        if pl.track is None:  # constant velocity: _bal_seg's expressions (b1 = b0 + bv dt)
            b0, b1 = _bal_seg(pl, self.bp0, self.bv, self.t0, t1.ravel(), self.dt, np.tile(tg, ch))
            Bk = np.stack([b0.reshape(ch, n, 3), b1.reshape(ch, n, 3)], 1)
        else:  # track: the interpolation of _bal_seg at t1 - dt and at t1, laid out (step, end, row)
            tt = np.empty((ch, 2, n))
            tt[:, 0] = t1 - self.dt
            tt[:, 1] = t1
            tab, i, f = pl._tidx(tt.reshape(-1))
            B = tab.shape[1]
            tabf = tab.reshape(-1, 3)
            li = i * B + np.tile(tg, 2 * ch)
            Bk = (tabf[li] * (1.0 - f)[:, None] + tabf[li + B] * f[:, None]).reshape(ch, 2, n, 3)
        self.Bk = np.ascontiguousarray(Bk.transpose(0, 1, 3, 2))  # (step, end, component, row): (3, n) blocks
        self.lo, self.hi, self.rows, self.key = jk, hi, c0, c0

    def get(self, jk, c0):
        """(b0, b1) (3, n) component-major of the active rows at step jk (c0: their rows in tk / tg, None = all)."""
        if jk >= self.hi:
            self._build(jk, c0)
        Bj = self.Bk[jk - self.lo]  # (2, 3, n_chunk)
        if c0 is self.rows:
            return Bj[0], Bj[1]
        if c0 is not self.key:  # rows finished since the chunk was built: map the active rows into the chunk's rows
            self.sel = c0 if self.rows is None else np.searchsorted(self.rows, c0)
            self.key = c0
        G = Bj.take(self.sel, axis=2)
        return G[0], G[1]


def _deferred_pops(pl, rec, popped, rel_mask, r2, bp0, bv, t0, dt, W=10):
    """Round 2 (research/sc40/speed): the incidental pops of a rollout, from its step records rec = [(ia, p0, p1, t1), ...]
    (active full row ids, step start / end positions, time after the step), written into popped (R, B) in place.

    The same set as the per-step test: (r, j) is popped when j is released and dmin2 <= r2 at a step where r is active
    (dmin2 by the per-step arithmetic, _seg_d2 / _bal_seg). Pairs are evaluated only where a bound cannot exclude them:
    per window of W steps and row active at its first step, P = the first step's start point, T = its start time t1 - dt,
    D = the largest distance of the row's step end points in the window from P (exact, from the records: every segment of
    the window lies in the ball |x - P| <= D), and a pair is kept unless |P - b_j(T)| > D + vmax_j W dt (1 + 1e-9) +
    radius + EPS + MARGIN (b_j is vmax_j-Lipschitz, |t - T| <= W dt on the window's segments). Rows with equal start times
    share every step time (the same running sums), so b_j(T) is evaluated once per time group and window."""
    L = len(rec)
    if L == 0:
        return
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
    refe = M[wq * W, RI]  # each record's window reference (the row's record at the window's first step)
    d = P1 - P0[refe]
    # round 2 (second pass): the per-(window, row) maximum through a dense (step, row) array and one max reduction (the same
    # values as np.maximum.at from zeros: every entry is a sum of squares >= 0, inactive entries stay 0)
    D2 = np.zeros((nw * W, R))
    D2[SI, RI] = _sum3(d * d)
    Dm2 = np.fmax.reduce(D2.reshape(nw, W, R), 1)  # fmax: a NaN step (blown-up row) must not hide the window's finite steps
    isref = SI % W == 0
    Pref = np.full((nw, R, 3), np.nan)
    Pref[wq[isref], RI[isref]] = P0[isref]
    Tref = np.full((nw, R), np.nan)
    Tref[wq[isref], RI[isref]] = T1[isref] - dt
    Dm = np.sqrt(Dm2) * (1.0 + 1e-12)
    vmax = np.sqrt((bv * bv).sum(1)) if pl.track is None else pl._track_vmax()
    # round 2 (second pass): one bound for all balloons, vmax = max_j vmax_j (a looser, still valid bound: the kept pairs are a
    # superset of the per-balloon bound's, and every kept pair is evaluated exactly below; fmax so that one NaN balloon speed
    # leaves the others' bound finite, as in the per-balloon test)
    cBm = float(np.fmax.reduce(vmax)) * (W * dt * (1.0 + 1e-9)) + (np.sqrt(r2) + _PopTest.EPS + _PopTest.MARGIN) if len(vmax) else 0.0
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
        if pl.track is None:
            bg = bp0[None] + bv[None] * (Tg - t0)[:, None, None]
        else:
            bg = pl._bp_all(Tg)  # (nv, B, 3)
        Pg = Pref[vw][:, Rg]  # (nv, ng, 3), nan rows inactive
        dots = np.matmul(Pg, bg.transpose(0, 2, 1))  # (nv, ng, B)
        # |P - b|^2 <= c^2 as |b|^2 - 2 P.b <= c^2 - |P|^2 with c = D + cBm per (window, row): the big (nv, ng, B) arrays
        # get 4 ops instead of 8 (rounding ~1e-9 m^2 at these coordinates, inside MARGIN, as the expansion before)
        c = Dm[vw][:, Rg] + cBm
        rhs = c * c - _l3sq(Pg * Pg)  # (nv, ng); nan for rows inactive in the window (no pair kept)
        dots *= -2.0
        dots += _l3sq(bg * bg)[:, None, :]
        m = (dots <= rhs[:, :, None]) & (rel_mask[None] & ~popped[Rg])[None]
        w_, r_, j_ = np.nonzero(m)
        if len(w_):
            cw.append(vw[w_]); cr.append(Rg[r_]); cj.append(j_)
    if not cw:
        return
    cw, cr, cj = np.concatenate(cw), np.concatenate(cr), np.concatenate(cj)
    st = cw[:, None] * W + np.arange(W)[None]  # the window's steps
    ok = st < L
    e = np.where(ok, M[np.minimum(st, L - 1), cr[:, None]], -1)
    q, o = np.nonzero(e >= 0)
    e = e[q, o]
    j = cj[q]
    b0, b1 = _bal_seg(pl, bp0, bv, t0, T1[e], dt, j)
    h = _seg_d2(P0[e] - b0, P1[e] - b1) <= r2
    if h.any():
        popped[RI[e[h]], j[h]] = True


class BeamPlanner:
    def __init__(
        self,
        sim,
        radius,
        *,
        beam=12,
        branch=10,
        depth=3,
        horizon=6.0,
        dt=0.02,
        dec_dt=0.1,
        lam=0.3,
        boost=0.0,
        boost_t0=1e9,
        radius_margin=0.8,
        stick=0.2,
        variants=((2.0, 0.8, 0.0),),
        deep_var=1,
        deep_radius=3.0,
        long_horizon=10.0,
        low_pen=0.2,
        critic=None,
        critic_beta=0.0,
        coast=False,
        ground_vz=False,
        depth_gamma=1.0,
        reach_beta=0.0,
        reach_mode="soft",
        reach_k=3,
        reach_tau0=2.0,
        reach_tmax=6.0,
        reach_dt=0.2,
        reach_margin=0.9,
        reach_wr=0.0,
        reach_m=16,
        head_beta=0.0,
    ):
        self.sim = sim
        # round 9 leaf heading term (head_beta 0 = off, M2FB unchanged): + head_beta * cos(angle between the leaf
        # velocity and the direction to the earliest-reachable balloon's point at its reach time), see _reach_times
        self.head_beta = float(head_beta)
        self._head = None
        # M2 leaf cost-to-go (reach_beta 0 = off, v8 unchanged): thrust-limited reach times from each leaf
        # end state to the available balloons, see _reach
        self.reach_beta, self.reach_mode, self.reach_k = (
            float(reach_beta),
            str(reach_mode),
            int(reach_k),
        )
        self.reach_tau0, self.reach_tmax, self.reach_dt = (
            float(reach_tau0),
            float(reach_tmax),
            float(reach_dt),
        )
        self.reach_margin, self.reach_wr, self.reach_m = (
            float(reach_margin),
            float(reach_wr),
            int(reach_m),
        )
        # coast: plans that reach burnout end with an unpowered fall whose incidental pops are counted
        self.use_coast = bool(coast)
        # ground_vz: count ground contact as a crash only while descending (False = v7 behaviour, where every
        # rollout started on the pad or in the first ~0.7 s after launch "crashed" at once)
        self.ground_vz = bool(ground_vz)
        # floor: relaxed near-ground guidance floor while climbing (stream OP; None = v8, PlannerAgent sets it)
        self.floor = None
        # leaf value: critic of a PPO policy trained in the given-only surrogate (training seeds only)
        self.critic = critic
        self.critic_beta = float(critic_beta)
        V = np.array(
            [tuple(v) + (0.0,) * (11 - len(v)) for v in variants], float
        )  # rows may differ in length
        # rows: (nav gain, t_go floor, boost m/s^2, aim offset up m, aim offset side m, viable-intercept margin,
        #        late re-point window rp s (stream M, repoint_dir; 0 = off),
        #        round 10 exit-direction blend zb, t_go factor zk (zv_dir; zk 0 = off),
        #        round 10 late lateral push la m/s^2, onset lt s before the pop (la 0 = off))
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
        self.lam, self.boost, self.boost_t0, self.stick = (
            float(lam),
            float(boost),
            float(boost_t0),
            float(stick),
        )
        self.last = None
        # round 10: per first balloon, the second balloon of the best plan through it (previous plan call; -1 =
        # none). zv rows (zk > 0) shape their exit toward it instead of next_after when zv_hint is set
        self.hint = None
        self.zv_hint = False
        self.zv_hyst = 0.1
        # round 10 pair rows (zv_pair k > 0; 0 = off): at search level 0 every zv row is expanded once per each of
        # the k likeliest next balloons t2, shapes its exit toward t2, and its level-1 child must fly to t2, so the
        # exit shaping and the next balloon are scored together. last_t2: t2 of the chosen first move (-1 none)
        self.zv_pair = 0
        self.last_t2 = -1
        # track: optional predicted balloon positions (table (T+1, B, 3), t_start, dt) from predict.py; None keeps
        # the constant-velocity model bp0 + bv * (t - t0) (v7 behaviour, expressions unchanged below)
        self.track = None
        # oracle: optional balloon id order (offline diagnostic only, research/sc4/oracle_run.py; None = off). Each
        # search level may only fly to the first listed balloon its row has not popped
        self.oracle = None
        self.oracle_k = 3

    def _oracle_mask(self, popped, rel):
        """(R, B) bool: per row, the first oracle_k listed balloons that are released and not popped by the row;
        rows with none left are unrestricted (free planning, e.g. before the first listed balloon is released)."""
        m = np.zeros_like(popped)
        if len(self.oracle) == 0:
            return ~m
        left = ~popped[:, self.oracle] & rel[self.oracle][None]  # (R, K)
        rank = np.cumsum(left, axis=1)
        sel = left & (rank <= self.oracle_k)
        r, c = np.nonzero(sel)
        m[r, self.oracle[c]] = True
        m[~left.any(1)] = True
        return m

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
        """(B,) largest segment speed of each balloon's predicted track (bound for _PopTest), cached per track."""
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
        tabf = tab.reshape(-1, 3)  # tab[i, j] = tabf[i B + j] (same entries, one flat gather)
        li = i * B + ids
        p0, p1 = tabf[li], tabf[li + B]
        return p0 * (1.0 - a)[:, None] + p1 * a[:, None], (p1 - p0) / self.track[2]

    # -- candidates -------------------------------------------------------------
    def _cands(self, pos, vel, t, bp0, bv, t0, avail):
        """Top-`branch` balloon ids per row by rough reach time. avail (R,B) bool."""
        if self.track is None:
            bp = bp0[None] + bv[None] * (t - t0)[:, None, None]
        else:
            bp = self._bp_all(t)
        r = bp - pos[:, None, :]
        d = np.sqrt(_l3sq(r * r))  # = np.linalg.norm(r, axis=2) (round 2: _l3sq / _l3, see ff_ref._l3)
        vc = -_l3(r * (bv[None] - vel[:, None, :])) / np.maximum(d, 1e-6)
        est = d / np.maximum(vc, 15.0)
        est = np.where(avail, est, np.inf)
        k = min(self.branch, est.shape[1])
        idx = np.argpartition(est, k - 1, axis=1)[:, :k]
        ok = np.isfinite(np.take_along_axis(est, idx, axis=1))
        return idx, ok

    def _next_pv(self, pos, vel, tp, tv, t, bp0, bv, t0, avail, tgt):
        """Late re-point: position now and velocity of the balloon to leave toward after tgt (next_after);
        nan rows where there is none."""
        bp = (
            bp0[None] + bv[None] * (t - t0)[:, None, None]
            if self.track is None
            else self._bp_all(t)
        )
        nx = next_after(pos, vel, tp, tv, bp, bv, avail, tgt)
        ok = nx >= 0
        nxc = np.maximum(nx, 0)
        pn = np.where(ok[:, None], bp[np.arange(len(pos)), nxc], np.nan)
        vn = np.where(ok[:, None], bv[nxc], np.nan)
        return pn, vn

    # -- rollouts -----------------------------------------------------------------
    def rollout(self, S, tgt, var, bp0, bv, t0, rel_mask, r2=None, nxt=None, resume=None):
        """S: dict of row arrays (pos, vel, q, w, g, thr, t, popped (R,B), crash).
        Fly each row toward its target until popped / horizon / ground. Returns new S.
        resume (speed, research/sc40/speed): (C, rows, cmd, k0) from self._cut of an earlier rollout of the same S, tgt,
        var, r2 and nxt with a shorter horizon; rows (ended there by the horizon alone, ascending) continue from that
        rollout's end state at step k0 with their last guidance command cmd, every other row keeps C's end state. A
        rollout started from S runs step k on exactly the same state, so the result is the same as a fresh rollout."""
        # Speed (research/sc4/streams/planner_speed.md): the active rows are kept as compact working arrays
        # and written back only when they finish, instead of gathering / scattering every state array each
        # step; the swept-segment test evaluates only (row, balloon) pairs that a conservative distance
        # bound cannot exclude (_PopTest). Same arithmetic per element, so results are bit-identical.
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
        ffs = None
        if getattr(
            self.sim, "ff", False
        ):  # W3 P3a feedforward state (fresh rows start at the nose)
            from ..follow.ff_ref import FF_DIM

            ffs = src["ff"].copy() if "ff" in src else np.zeros((len(pos), FF_DIM))
        use_vi = bool(
            ((self.V[:, 5] > 0) | (self.V[:, 8] > 0) | (self.V[:, 9] > 0)).any()
        )
        R = len(pos)
        miss = np.full(R, 1e9) if resume is None else src["miss"].copy()
        r2 = self.r2 if r2 is None else r2
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
        cT, cPop, cTg, cVar, cTe, cMiss = (
            t[ia],
            popped[ia],
            tgt[ia],
            var[ia],
            t_end[ia],
            miss[ia],
        )
        cNx = None if nxt is None else nxt[ia]
        # round 2: the feedforward state as two contiguous arrays (n_ref, w_ref) inside the loop (ff_step's tuple state);
        # the same values, without per-step ops on (N, 3) column views of the (N, 6) state
        cFF = None if ffs is None else _ff_split(ffs[ia])
        cN = np.zeros((len(ia), 3)) if resume is None else resume[2]
        cTe_m = cTe - 1e-9  # horizon test cT >= cTe - 1e-9
        ar = np.arange(len(ia))
        # speed (research/sc40/speed): the time-table values of every step at once. The loop's cT at the start of step k
        # is the running sum cT + dt + ... + dt, which cumsum reproduces term by term; c0 maps the active rows to the
        # rows of this precomputation (identity until the first row finishes).
        n0, nk = len(ia), nsteps - k0
        pre_X = pre_hint = hz_any = None
        fast = False
        if n0 and nk > 0 and hasattr(self.sim, "step_tables"):
            # one more row than the step count: tk[k + 1] is cT after step k (round 2: horizon test tables below)
            tk = np.cumsum(np.concatenate([cT[None], np.full((nk, n0), dt)]), axis=0)
            trk = tk[:nk] - tl
            # round 2: every time-only quantity of the steps packed per (step, row) (GivenSim.step_tables)
            # second pass: field-major (F, nk, n0), so each step's block pre_X[:, k] has contiguous field rows
            pre_X, pre_B = self.sim.step_tables(trk, fm=True)
            # fast (round 2): T = T0 thr == T0 on every row and step, which holds when every row starts at thr == 1.0
            # (then clip(1, 1 - s, 1 + s) keeps it at 1.0 while burning) and T0 == +0.0 at and after a row's first step
            # without burn (thr then decays to a finite value >= 0, and 0 * thr == +0.0)
            fast = (
                bool((cTh == 1.0).all())
                and self.sim.thr_rate * dt >= 0.0
                and bool((pre_X[0][np.logical_or.accumulate(~pre_B, axis=0)] == 0.0).all())
            )
            # per step k, for all rows (then also every subset of them): 1 burning, -1 none burning, -2 past burn_time
            # (no thrust at k or later: GivenSim.step coast branch), 0 mixed (step decides)
            pre_hint = np.where(
                (trk > self.sim.burn_time).all(1),
                -2,
                np.where(pre_B.all(1), 1, np.where(pre_B.any(1), 0, -1)),
            ).tolist()
            # round 2: the horizon test cT >= cTe - 1e-9 of every step, and per step whether any row meets it
            hzt = tk[1:] >= cTe_m[None]
            hz_any = hzt.any(1).tolist()
        c0 = None  # None = all rows of the precomputation, in order
        vkey = None  # cVar the cached variant rows belong to
        # round 2, deferred pops: when nothing inside the loop reads a row's incidental pops (no variant row has the
        # next-balloon terms rp / zk / la, the only readers of cPop besides the target test), each step tests only the
        # (row, target) pairs, which end rows; the miss distance and the incidental pops of every recorded step follow
        # after the loop (_deferred_pops). Otherwise the per-step candidate test (_PopTest) runs as before.
        defer = (
            pre_X is not None
            and not bool(((self.V[:, 6] > 0) | (self.V[:, 8] > 0) | (self.V[:, 9] > 0)).any())
            and bool(rel_mask[cTg].all())
        )
        if defer:
            ttk = _TargetTrack(self, bp0, bv, t0, tk, dt, cTg)
            rec = []  # per step: (ia, p0, p1, t1, target dmin2)
        else:
            pt = _PopTest(self, bp0, bv, t0, rel_mask, r2, R)
        # rows that end by the horizon alone (resumable by a longer-horizon rollout, see resume); None = not resumable
        cut_r, cut_n = [], []
        for k in range(k0, nsteps):
            n = len(ia)
            if n == 0:
                break
            if pre_X is None:
                tr, tab = cT - tl, None
            else:
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
                pre_hint is not None
                and pre_hint[k - k0] == -2
                and bool(np.isfinite(cTh).all())
            ):
                if self.track is None:
                    tp, tv = bp0[cTg] + bv[cTg] * (cT - t0)[:, None], bv[cTg]
                else:
                    tp, tv = self._bp_one(cT, cTg)
                if vkey is not cVar:  # variant rows and their flags change only when rows finish (cached)
                    vv = self.V[cVar]
                    vi_on = use_vi and bool(((vv[:, 5] > 0) | (vv[:, 8] > 0) | (vv[:, 9] > 0)).any())
                    nx_on = vi_on and bool(((vv[:, 6] > 0) | (vv[:, 8] > 0) | (vv[:, 9] > 0)).any())
                    vcols = [vv[:, c] for c in range(11)]
                    vkey = cVar
                if vi_on:
                    # round 2: the nose from the step's own rotation matrix (quat_nose(q) == quat_rot(q)[:, :, 2], the
                    # same expressions), which the step then reuses; a_av from the step tables when T0 thr == T0
                    Rm = quat_rot(cQ)
                    a_av = tab[0][7] if fast else self.sim.accel_avail(cT - tl, cTh)
                    pn = vn = None
                    if nx_on:
                        pn, vn = self._next_pv(
                            cP, cV, tp, tv, cT, bp0, bv, t0, rel_mask[None] & ~cPop, cTg
                        )
                        if cNx is not None:  # pair rows: their own t2
                            hc = np.maximum(cNx, 0)
                            okh = ((vv[:, 8] > 0) | (vv[:, 9] > 0)) & (cNx >= 0)
                            if okh.any():
                                if self.track is None:
                                    bph, bvh = (
                                        bp0[hc] + bv[hc] * (cT - t0)[:, None],
                                        bv[hc],
                                    )
                                else:
                                    bph, bvh = self._bp_one(cT, hc)
                                pn = np.where(okh[:, None], bph, pn)
                                vn = np.where(okh[:, None], bvh, vn)
                        elif (
                            self.zv_hint and self.hint is not None
                        ):  # zv rows: second balloon of the last plan
                            h = self.hint[cTg]
                            hc = np.maximum(h, 0)
                            okh = (
                                (vv[:, 8] > 0)
                                & (h >= 0)
                                & rel_mask[hc]
                                & ~cPop[np.arange(n), hc]
                            )
                            if okh.any():
                                if self.track is None:
                                    bph, bvh = (
                                        bp0[hc] + bv[hc] * (cT - t0)[:, None],
                                        bv[hc],
                                    )
                                else:
                                    bph, bvh = self._bp_one(cT, hc)
                                pn = np.where(okh[:, None], bph, pn)
                                vn = np.where(okh[:, None], bvh, vn)
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
                        rp=vcols[6],
                        pn=pn,
                        vn=vn,
                        floor=self.floor,
                        zb=vcols[7],
                        zk=vcols[8],
                        la=vcols[9],
                        lt=vcols[10],
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
                        floor=self.floor,
                    )
            p0 = cP
            if cFF is None:  # full throttle (thr_cmd 1.0 = the former np.ones(n))
                cP, cV, cQ, cW, cG, cTh = self.sim.step(
                    cP, cV, cQ, cW, cG, cTh, tr, cN, 1.0, dt, tab=tab, Rm=Rm
                )
            else:  # after the first step every row's ff state is an ff_step output (ff_step normed)
                cP, cV, cQ, cW, cG, cTh, cFF = self.sim.step(
                    cP, cV, cQ, cW, cG, cTh, tr, cN, 1.0, dt, ffs=cFF, ff_normed=k > k0, tab=tab, Rm=Rm
                )
            cT = cT + dt
            gnd = cP[:, 2] < elev_g
            if self.ground_vz:  # a rocket still climbing off the pad is not a crash
                gnd &= cV[:, 2] < 0.0
            if defer:
                # (row, target) swept-segment test, the per-step arithmetic; hits are written into cPop. A row whose target
                # is popped finishes at once, so after the first step cPop[ar, cTg] is this step's target hit
                b0, b1 = ttk.get(k - k0, c0)  # (3, n) each
                dtg2 = _seg_d2T(p0.T - b0, cP.T - b1)  # round 2: component-major target test (same values)
                rec.append((ia, p0, cP, cT, dtg2))
                hit = dtg2 <= r2
                if _nz(hit):
                    cPop[ar[hit], cTg[hit]] = True
                    ended = (cPop[ar, cTg] | gnd) if k == k0 else (hit | gnd)
                else:
                    ended = (cPop[ar, cTg] | gnd) if k == k0 else gnd
            else:
                # swept-segment pop test vs released balloons (hits are written into cPop)
                dtg2 = pt.step(ia, p0, cP, cT, cPop, cTg, ar)
                cMiss = np.minimum(cMiss, np.sqrt(dtg2))
                ended = cPop[ar, cTg] | gnd
            if hz_any is None:
                hz = cT >= cTe_m
            elif hz_any[k - k0]:
                hz = hzt[k - k0] if c0 is None else hzt[k - k0][c0]
            else:
                hz = None  # no row reaches the horizon in this step
            fin = ended if hz is None else (ended | hz)
            if _nz(fin):  # round 2: count_nonzero, the same test as fin.any()
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
                if not defer:
                    miss[j] = cMiss[fin]
                    cMiss = cMiss[~fin]
                if cFF is not None:
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
                if cFF is not None:
                    cFF = (cFF[0][kp], cFF[1][kp])
                if cNx is not None:
                    cNx = cNx[kp]
                ar = ar[: len(ia)]
                c0 = kp.nonzero()[0] if c0 is None else c0[kp]
        if len(ia):  # rows still flying at the horizon step limit
            pos[ia], vel[ia], q[ia], w[ia], g[ia], thr[ia] = cP, cV, cQ, cW, cG, cTh
            t[ia], popped[ia] = cT, cPop
            if not defer:
                miss[ia] = cMiss
            if cFF is not None:
                ffs[ia, :3], ffs[ia, 3:] = cFF
            if cut_r is not None:
                cut_r.append(ia)
                cut_n.append(cN)
        if defer and rec:
            # miss = the running minimum np.minimum(miss, sqrt(dtg2)) over each row's steps (in step order), then the
            # incidental pops of every recorded step
            np.minimum.at(miss, np.concatenate([x[0] for x in rec]), np.sqrt(np.concatenate([x[4] for x in rec])))
            _deferred_pops(self, rec, popped, rel_mask, r2, bp0, bv, t0, dt)
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
        if ffs is not None:
            out["ff"] = ffs
        return out

    def coast(self, C, rows, bp0, bv, t0, rel_mask, dt=0.04, t_max=20.0):
        """Unpowered flight of `rows` from their state to the ground (TVC has no authority after
        burnout); returns the number of extra balloons each sweeps through (env pop test, full radius
        x margin). The state is not propagated: a coast is the end of a plan."""
        if len(rows) == 0:
            return np.zeros(0)
        pos, vel, q, w = (
            C["pos"][rows].copy(),
            C["vel"][rows].copy(),
            C["q"][rows].copy(),
            C["w"][rows].copy(),
        )
        g, thr = C["g"][rows].copy(), C["thr"][rows].copy()
        t = C["t"][rows].copy()
        popped = C["popped"][rows].copy()
        p0cnt = popped.sum(1)
        R = len(rows)
        done = np.zeros(R, bool)
        elev = self.sim.elevation
        nose = quat_rot(q)[:, :, 2]
        for _ in range(int(t_max / dt)):
            ia = np.flatnonzero(~done)
            if len(ia) == 0:
                break
            p_prev = pos[ia]
            out = self.sim.step(
                pos[ia],
                vel[ia],
                q[ia],
                w[ia],
                g[ia],
                thr[ia],
                t[ia] - C["t_launch"],
                nose[ia],
                np.ones(len(ia)),
                dt,
            )
            pos[ia], vel[ia], q[ia], w[ia], g[ia], thr[ia] = out
            t[ia] += dt
            if self.track is None:
                b0 = bp0[None] + bv[None] * (t[ia] - dt - t0)[:, None, None]
                b1 = b0 + bv[None] * dt
            else:
                b0, b1 = self._bp_all(t[ia] - dt), self._bp_all(t[ia])
            a0 = p_prev[:, None, :] - b0
            a1 = pos[ia][:, None, :] - b1
            da = a1 - a0
            s = np.clip(-(a0 * da).sum(2) / np.maximum((da * da).sum(2), 1e-12), 0, 1)
            dmin2 = ((a0 + s[..., None] * da) ** 2).sum(2)
            popped[ia] |= (dmin2 <= self.r2) & rel_mask[None]
            done[ia] |= (pos[ia, 2] < elev + 0.5) & (vel[ia, 2] < 0.0)
        return (popped.sum(1) - p0cnt).astype(float)

    def _value(self, C, bp0, bv, t0, rel_mask):
        """Critic estimate of future return from each row's end state (features.build as in training)."""
        d = self.critic
        R = len(C["pos"])
        nose = quat_rot(C["q"])[:, :, 2]
        t_abs = C["t"]
        t_rel = t_abs - C["t_launch"]
        mv = (
            bp0[None] + bv[None] * (t_abs - t0)[:, None, None]
            if self.track is None
            else self._bp_all(t_abs)
        )
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
        x = np.clip((o - d["obs_mean"]) / np.sqrt(d["obs_var"] + 1e-8), -10, 10)
        k = 0
        while f"v_W{k}" in d:
            x = x @ d[f"v_W{k}"].T + d[f"v_b{k}"]
            if f"v_W{k + 1}" in d:
                x = np.tanh(x)
            k += 1
        return x[:, 0]

    def _reach_times(self, C, bp0, bv, t0, rel_mask):
        """M2: earliest time tau_j (s after each row's end state) at which balloon j can be reached with a fixed
        thrust direction at full given thrust. From (p, v, t) the reachable set after tau is the ball
            centre  p + v tau - 1/2 g z tau^2,   radius  margin * int_0^tau (tau - s) a_T(t + s) ds,
        a_T = T/m from the given thrust curve and mass (GivenSim tables). Optional turn delay (reach_wr > 0):
        the radius uses tau - delta, delta = gimbal lag + angle(nose, dx) / reach_wr (as viable_dir).
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
            self._head = np.zeros(R)
            return np.full((R, 1), np.inf)
        pos, vel, t_abs = C["pos"], C["vel"], C["t"]
        # candidate preselection (rough closing-speed reach estimate, as _cands)
        if self.track is None:
            b_now = bp0[None] + bv[None] * (t_abs - t0)[:, None, None]
        else:
            b_now = self._bp_all(t_abs)
        r = b_now - pos[:, None, :]
        d = np.sqrt(_l3sq(r * r))  # round 2: (.).sum(2) as _l3sq / _l3 (see ff_ref._l3)
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
        if self.track is None:
            bp = bp0[ids][:, None] + bv[ids][:, None] * (tt - t0)[:, :, None, None]
        else:
            tab, i, af = self._tidx(tt.ravel())
            af = af.reshape(R, n, 1, 1)
            # tab[i, j] gathered through one flat index (i B + j) into the (T B, 3) view: the same entries, faster
            B = tab.shape[1]
            tabf = tab.reshape(-1, 3)
            li = i.reshape(R, n, 1) * B + ids[:, None, :]
            bp = tabf[li] * (1.0 - af) + tabf[li + B] * af  # (R, n, M, 3)
        dx = bp - ctr[:, :, None, :]
        dn = np.sqrt(_l3sq(dx * dx))  # (R, n, M); (.).sum(3) as _l3sq (round 2)
        rad = np.sqrt(self.r2_deep)
        if self.reach_wr > 0:
            nose = quat_rot(C["q"])[:, :, 2]
            cosang = np.clip(
                (dx * nose[:, None, None, :]).sum(3) / np.maximum(dn, 1e-9), -1.0, 1.0
            )
            delta = self.sim.gtau + np.arccos(cosang) / self.reach_wr
            fk = (taus[None, :, None] - delta) / dtau  # tau - delta in grid units
            k0 = np.clip(np.floor(fk).astype(int), 0, n)
            rho_p = np.concatenate([np.zeros((R, 1)), rho], 1)  # rho_p[k] = rho(k dtau)
            g0 = np.take_along_axis(rho_p, k0.reshape(R, -1), 1).reshape(k0.shape)
            g1 = np.take_along_axis(
                rho_p, np.minimum(k0 + 1, n).reshape(R, -1), 1
            ).reshape(k0.shape)
            fr = np.clip(fk - k0, 0.0, 1.0)
            rr = np.where(fk > 0, g0 * (1 - fr) + g1 * fr, 0.0)
            gap = dn - rr - rad
        else:
            gap = dn - rho[:, :, None] - rad  # <= 0: inside the ball
        ok = gap <= 0
        anyok = ok.any(1) & valid
        k = np.argmax(ok, axis=1)  # first grid hit (R, M)
        gk = np.take_along_axis(gap, k[:, None], 1)[:, 0]
        gp = np.take_along_axis(gap, np.maximum(k - 1, 0)[:, None], 1)[:, 0]
        fr = np.where(k > 0, gp / np.maximum(gp - gk, 1e-9), 1.0)  # linear crossing
        tj = np.where(k > 0, taus[np.maximum(k - 1, 0)] + fr * dtau, taus[0])
        tau = np.where(anyok, tj, np.inf)
        if (
            self.head_beta
        ):  # heading cos toward the earliest-reachable balloon (0 for rows with none)
            r_ = np.arange(R)
            m_ = np.argmin(tau, 1)
            dp = bp[r_, k[r_, m_], m_] - pos
            sp = np.linalg.norm(vel, axis=1) * np.linalg.norm(dp, axis=1)
            self._head = np.where(
                np.isfinite(tau[r_, m_]) & (sp > 1e-6),
                (vel * dp).sum(1) / np.maximum(sp, 1e-9),
                0.0,
            )
        return tau

    def _reach(self, C, bp0, bv, t0, rel_mask):
        """M2 leaf value from the reach times: 'min' -> -min_j tau_j (capped at reach_tmax + 1);
        'soft' -> sum over the reach_k earliest balloons of exp(-tau_j / reach_tau0)."""
        tau = self._reach_times(C, bp0, bv, t0, rel_mask)
        if self.reach_mode == "min":
            return -np.minimum(tau.min(1), self.reach_tmax + 1.0)
        k = min(self.reach_k, tau.shape[1])
        tk = np.partition(tau, k - 1, axis=1)[:, :k]
        return np.exp(-tk / self.reach_tau0).sum(1)

    # -- search -----------------------------------------------------------------
    def plan(self, S0, bp0, bv, t0, rel_mask, cur=None):
        """S0: single-row state dict. Returns (best first target id or None, info)."""
        S = {k: (v if k == "t_launch" else v.copy()) for k, v in S0.items()}
        first = np.full(1, -1)
        second = np.full(1, -1)
        lsec = []
        dpop = np.zeros(1)  # depth-discounted pops so far per row
        leaves = []
        for dlev in range(self.depth):
            avail = rel_mask[None] & ~S["popped"]
            if self.oracle is not None:
                avail &= self._oracle_mask(S["popped"], rel_mask)
            idx, ok = self._cands(S["pos"], S["vel"], S["t"], bp0, bv, t0, avail)
            if (
                self.zv_pair and dlev == 1
            ):  # pair rows: the level-1 child must fly to the row's t2
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
            nxt = None
            if dlev == 0:
                tgt = np.repeat(idx[rr, cc], nV)
                var = np.tile(np.arange(nV), len(rr))
                rr = np.repeat(rr, nV)
                if (
                    self.zv_pair
                ):  # pair rows: each zv row once per likely next balloon t2
                    zv = (self.V[var, 8] > 0) | (self.V[var, 9] > 0)
                    keep0 = ~zv
                    tz, vz, rz = tgt[zv], var[zv], rr[zv]
                    if self.track is None:
                        bpn = bp0 + bv * (S["t"][0] - t0)
                        tp_, tv_ = bpn[tz], bv[tz]
                    else:
                        bpn = self._bp_all(S["t"][:1])[0]
                        tp_, tv_ = self._bp_one(np.full(len(tz), S["t"][0]), tz)
                    nk = next_k(
                        S["pos"][rz],
                        S["vel"][rz],
                        tp_,
                        tv_,
                        bpn[None],
                        bv,
                        avail[rz],
                        tz,
                        self.zv_pair,
                    )
                    jj, kk = np.nonzero(nk >= 0)
                    tgt = np.concatenate([tgt[keep0], tz[jj]])
                    var = np.concatenate([var[keep0], vz[jj]])
                    rr = np.concatenate([rr[keep0], rz[jj]])
                    nxt = np.concatenate([np.full(keep0.sum(), -1), nk[jj, kk]])
            else:
                tgt = idx[rr, cc]
                var = np.full(len(rr), min(self.deep_var, nV - 1))
            C = {
                k: (v if k == "t_launch" else v[rr])
                for k, v in S.items()
                if k != "miss"
            }
            Bp = (
                len(rel_mask) + 1 if self.zv_pair else 1
            )  # first move = (choice, t2) when pair rows are on
            choice = (tgt * nV + var) * Bp + ((nxt + 1) if nxt is not None else 0)
            fst = np.where(first[rr] >= 0, first[rr], choice)
            sec = tgt if dlev == 1 else second[rr]
            C0 = C
            C = self.rollout(
                C0,
                tgt,
                var,
                bp0,
                bv,
                t0,
                rel_mask,
                r2=self.r2 if dlev == 0 else self.r2_deep,
                nxt=nxt,
            )
            if (
                dlev == 0
                and self.long_horizon > self.horizon
                and not C["popped"][np.arange(len(tgt)), tgt].any()
            ):
                h = self.horizon
                cut = self._cut  # rows the short rollout ended by its horizon alone
                self.horizon = self.long_horizon  # nothing reachable: look further once
                # speed (research/sc40/speed): continue those rows from where the short rollout stopped instead of
                # re-flying the first `horizon` seconds (the same steps on the same state; see rollout(resume))
                C = self.rollout(
                    C0,
                    tgt,
                    var,
                    bp0,
                    bv,
                    t0,
                    rel_mask,
                    r2=self.r2,
                    nxt=nxt,
                    resume=None if cut is None else (C, cut[0], cut[1], cut[2]),
                )
                self.horizon = h
            npop = C["popped"].sum(1) - S0["popped"].sum()
            dnew = dpop[rr] + self.depth_gamma**dlev * (
                C["popped"].sum(1) - C0["popped"].sum(1)
            )
            past = np.zeros(len(tgt), bool)
            if self.use_coast:  # burnt out: the rest of the flight is ballistic, count it and stop expanding
                past = (C["t"] - C["t_launch"] >= self.sim.burn_time) & ~C["crash"]
                if past.any():
                    npop = npop.astype(float)
                    npop[past] += self.coast(
                        C, np.flatnonzero(past), bp0, bv, t0, rel_mask
                    )
            key = (
                (npop if self.depth_gamma == 1.0 else dnew)
                - self.lam * (C["t"] - t0)
                - 5.0 * C["crash"]
            )
            if self.critic is not None and self.critic_beta:
                key = key + self.critic_beta * np.where(
                    C["crash"] | past, 0.0, self._value(C, bp0, bv, t0, rel_mask)
                )
            if self.reach_beta:
                key = key + self.reach_beta * np.where(
                    C["crash"] | past, 0.0, self._reach(C, bp0, bv, t0, rel_mask)
                )
                if self.head_beta:
                    key = key + self.head_beta * np.where(
                        C["crash"] | past, 0.0, self._head
                    )
            if cur is not None:
                key = key + self.stick * (fst // Bp // len(self.V) == cur)
            # keep children that popped their target (or at least something)
            hit_t = C["popped"][np.arange(len(tgt)), tgt]
            keep = (hit_t | (npop > 0)) & ~C["crash"]
            for i in np.flatnonzero(keep):
                leaves.append(
                    (key[i], int(fst[i]), int(npop[i]), float(C["t"][i] - t0))
                )
                lsec.append(int(sec[i]))
            if (
                dlev == 0
            ):  # fallback: nothing hits within the horizon -> closest approach, no crash
                for i in np.flatnonzero(~keep & ~C["crash"]):
                    low = max(
                        0.0, 30.0 - (C["pos"][i, 2] - self.sim.elevation)
                    )  # ending low = lost energy
                    leaves.append(
                        (
                            -10.0
                            - 0.1 * C["miss"][i]
                            - self.lam * (C["t"][i] - t0)
                            - self.low_pen * low,
                            int(fst[i]),
                            0,
                            float(C["t"][i] - t0),
                        )
                    )
                    lsec.append(-1)
            if not (keep & ~past).any():
                break
            ki = np.flatnonzero(keep & ~past)
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
        if not leaves:
            return None, None
        # value of a first target = best leaf key reachable through it (deeper leaves have more pops)
        bestf = {}
        bests = {}
        for (kv, f, n, dt_), sc in zip(leaves, lsec):
            if f not in bestf or kv > bestf[f][0]:
                bestf[f] = (kv, n, dt_)
                bests[f] = sc
        fc = max(bestf, key=lambda x: bestf[x][0])
        nV = len(self.V)
        Bp = len(rel_mask) + 1 if self.zv_pair else 1
        self.last_t2 = fc % Bp - 1 if self.zv_pair else -1
        f = fc // Bp
        self.last = (f, bestf[fc])
        if self.zv_hint:  # best second balloon per first balloon over leaves that have one; the previous
            hk = {}  # hint is kept while its best leaf is within zv_hyst of the best (no flip on near-ties)
            for (kv, ff, _, _), t2 in zip(leaves, lsec):
                if t2 >= 0:
                    d2 = hk.setdefault(ff // nV, {})
                    if t2 not in d2 or kv > d2[t2]:
                        d2[t2] = kv
            old = self.hint
            self.hint = np.full(len(rel_mask), -1)
            for t1, d2 in hk.items():
                b2 = max(d2, key=d2.get)
                if (
                    old is not None
                    and old[t1] in d2
                    and d2[old[t1]] >= d2[b2] - self.zv_hyst
                ):
                    b2 = old[t1]
                self.hint[t1] = b2
        return (f // nV, f % nV), bestf[fc]
