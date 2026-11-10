"""Balloon track predictor from observations only: altitude wind profile + self-tuned smoothing.

The horizontal wind is estimated as a function of altitude, U(z) = (u, v), from every balloon velocity
observed so far (released, airborne balloons are wind tracers; all past samples are kept as running sums on
a 1 m altitude grid). The profile is a Whittaker-Eilers penalised smoother (Eilers 2003, "A perfect
smoother"): minimise sum_w (y - u)^2 + lam * |D2 u|^2, a banded linear system. Nothing about the
simulator's gust model is assumed (discussion #168): the smoothing strength `lam` and the blend time `tau`
(how fast a balloon's own observed velocity gives way to the profile) are chosen online by prequential
error, i.e. how well each setting would have predicted the balloons' current positions from stored
snapshots 2 s and 4 s ago.

Prediction: z(t) = z0 + vz0 * t (own rise rate); horizontal velocity
v_h(t) = U(z(t)) + (v_h0 - U(z0)) * exp(-t / tau), integrated at `dt`.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded

LAMS = (1.0, 10.0, 100.0, 1e3, 1e4)  # smoothing candidates (grid units: 1 m)
TAUS = (0.5, 2.0, 8.0)  # blend-time candidates (s)
SNAP_DT, SNAP_AGES = 0.5, (2.0, 4.0)  # snapshots kept for the prequential score


class WindProfilePredictor:
    def __init__(self, z_min, z_max=900.0, dz=1.0, min_vz=0.5):
        self.z0, self.dz = float(z_min), float(dz)
        self.n = int(np.ceil((z_max - z_min) / dz)) + 1
        self.zg = self.z0 + np.arange(self.n) * self.dz
        self.sw = np.zeros(self.n)
        self.su = np.zeros(self.n)
        self.sv = np.zeros(self.n)
        self.min_vz = float(min_vz)
        self.lam, self.tau = 100.0, 2.0
        self.U = None  # (n, 2) current profile
        self.snaps = []  # (t, pos, vel, released)
        self.score = np.zeros((len(LAMS), len(TAUS)))
        self._last_snap = -1e9
        # banded D2'D (pentadiagonal), upper form for solveh_banded
        n = self.n
        if n < 5:
            raise ValueError("altitude grid too small")
        d0 = np.full(n, 6.0)
        d0[[0, -1]] = 1.0
        d0[[1, -2]] = 5.0  # main diagonal
        d1 = np.full(n - 1, -4.0)
        d1[[0, -1]] = -2.0  # first off-diagonal
        self._DtD = np.zeros((3, n))
        self._DtD[2] = d0
        self._DtD[1, 1:] = d1
        self._DtD[0, 2:] = 1.0

    # -- data -----------------------------------------------------------------------------------
    def observe(self, t, pos, vel, released):
        """Add one observation of all balloons (call every decision step, e.g. 0.1 s)."""
        m = released & (np.abs(vel[:, 2]) > self.min_vz)
        if m.any():
            f = (pos[m, 2] - self.z0) / self.dz
            i = np.clip(np.floor(f).astype(int), 0, self.n - 2)
            a = np.clip(f - i, 0.0, 1.0)
            for idx, w in ((i, 1.0 - a), (i + 1, a)):
                np.add.at(self.sw, idx, w)
                np.add.at(self.su, idx, w * vel[m, 0])
                np.add.at(self.sv, idx, w * vel[m, 1])
        if t - self._last_snap >= SNAP_DT - 1e-9:
            self._last_snap = t
            self.snaps.append((float(t), pos.copy(), vel.copy(), released.copy()))
            keep = max(SNAP_AGES) + SNAP_DT
            self.snaps = [s for s in self.snaps if t - s[0] <= keep + 1e-9]

    def _profile(self, lam):
        ab = lam * self._DtD.copy()
        ab[2] += self.sw + 1e-9
        u = solveh_banded(
            ab, np.stack([self.su, self.sv], 1), lower=False, check_finite=False
        )
        return u

    # -- fitting --------------------------------------------------------------------------------
    def refit(self, t, pos, vel, released):
        """Re-solve the profile and re-select (lam, tau) by prequential track error. Call at plan time."""
        if self.sw.sum() < 20:
            self.U = None
            return
        profiles = [self._profile(lam) for lam in LAMS]
        cur = released & (np.abs(vel[:, 2]) > self.min_vz)
        for age in SNAP_AGES:
            old = [s for s in self.snaps if abs((t - s[0]) - age) < SNAP_DT / 2]
            if not old:
                continue
            ts, p0, v0, r0 = old[0]
            m = r0 & cur & (np.abs(v0[:, 2]) > self.min_vz)
            if m.sum() < 3:
                continue
            prs = self._propagate_all(profiles, TAUS, p0[m], v0[m], t - ts, 0.1)
            pm = pos[m]
            for i in range(len(profiles)):
                for j in range(len(TAUS)):
                    self.score[i, j] = 0.9 * self.score[i, j] + np.mean(
                        np.linalg.norm(prs[i, j] - pm, axis=1)
                    )
        i, j = (
            np.unravel_index(np.argmin(self.score), self.score.shape)
            if self.score.any()
            else (2, 1)
        )
        self.lam, self.tau = LAMS[i], TAUS[j]
        self.U = profiles[i]

    def wind_at(self, z):
        if self.U is None:
            return None
        return np.stack(
            [np.interp(z, self.zg, self.U[:, 0]), np.interp(z, self.zg, self.U[:, 1])],
            -1,
        )

    def wind_table(self, z_lo=None):
        """The profile as a lookup table (z, u, v) restricted to [z_lo, the highest altitude any balloon has reached].
        A reader using np.interp then holds the end values outside the range, so neither the D2 continuation above the
        data nor anything below z_lo is handed on. None without a profile or with fewer than 2 grid points in range."""
        if self.U is None:
            return None
        k = np.nonzero(self.sw > 0)[0]
        if not len(k):
            return None
        lo = self.zg[0] if z_lo is None else float(z_lo)
        m = (self.zg >= lo) & (self.zg <= self.zg[k[-1]])
        if m.sum() < 2:
            return None
        return self.zg[m], self.U[m, 0], self.U[m, 1]

    # -- prediction -----------------------------------------------------------------------------
    @staticmethod
    def _ztrack(z0, vz, n, dt):
        """Altitudes of _propagate_all / table: z_k (k = 0..n, z_{k+1} = z_k + vz dt, a sequential running sum) and the
        mid-step altitudes zm_k = z_k + 0.5 dt vz (k = 0..n-1)."""
        zk = np.cumsum(
            np.concatenate([z0[None], np.broadcast_to(vz * dt, (n, len(z0)))]), axis=0
        )
        return zk, zk[:-1] + 0.5 * dt * vz

    def _propagate_all(self, profiles, taus, p0, v0, horizon, dt):
        """The prediction model with profile U and blend time tau, for every profile and tau at once: (len(profiles),
        len(taus), M, 3) positions after horizon s in steps of about dt, z_{k+1} = z_k + vz dt and
        p_{k+1} = p_k + (U(zm_k) + (v_h0 - U(z_0)) exp(-(k + 0.5) dt / tau)) dt (zm_k the mid-step altitude). The
        altitude track does not depend on U or tau, so each profile is interpolated once for all steps; the horizontal
        positions are the sequential sums, taken by cumsum."""
        n = max(1, int(round(horizon / dt)))
        dt = horizon / n
        p = p0.astype(float)
        M = len(p)
        zk, zm = self._ztrack(p[:, 2], v0[:, 2], n, dt)
        zf = zm.ravel()
        nU, nT = len(profiles), len(taus)
        dev = np.empty((nU, M, 2))
        um = np.empty((nU, n, M, 2))
        for i, U in enumerate(profiles):
            for c in (0, 1):
                dev[i, :, c] = v0[:, c] - np.interp(p[:, 2], self.zg, U[:, c])
                um[i, :, :, c] = np.interp(zf, self.zg, U[:, c]).reshape(n, M)
        e = np.array(
            [[np.exp(-(k + 0.5) * dt / tau) for tau in taus] for k in range(n)]
        )  # (n, nT)
        d = (
            um[:, None] + dev[:, None, None] * e.T[None, :, :, None, None]
        ) * dt  # (nU, nT, n, M, 2)
        xy0 = np.broadcast_to(p[:, :2], (nU, nT, 1, M, 2))
        xy = np.cumsum(np.concatenate([xy0, d], axis=2), axis=2)[:, :, -1]
        out = np.empty((nU, nT, M, 3))
        out[..., :2] = xy
        out[..., 2] = zk[-1]
        return out

    def table(self, pos, vel, released, horizon=20.0, dt=0.1):
        """Predicted positions (T+1, B, 3) at times 0, dt, ..., horizon from now. Unreleased balloons and
        the no-profile case fall back to constant velocity (the planner's previous model)."""
        T = int(round(horizon / dt))
        tt = np.arange(T + 1) * dt
        out = pos[None] + vel[None] * tt[:, None, None]
        if self.U is None:
            return out
        m = released & (np.abs(vel[:, 2]) > self.min_vz)
        if not m.any():
            return out
        p = pos[m].astype(float)
        v0 = vel[m]
        u0 = self.wind_at(p[:, 2])
        dev = v0[:, :2] - u0
        # the step model (zm, wind_at(zm), p += ... per step) as one interpolation of all mid-step altitudes and
        # sequential running sums (cumsum)
        M = len(p)
        zk, zm = self._ztrack(p[:, 2], v0[:, 2], T, dt)
        wm = self.wind_at(zm.ravel()).reshape(T, M, 2)
        e = np.array([np.exp(-(k + 0.5) * dt / self.tau) for k in range(T)])
        d = (wm + dev[None] * e[:, None, None]) * dt
        res = np.empty((T, M, 3))
        res[:, :, :2] = np.cumsum(np.concatenate([p[None, :, :2], d]), axis=0)[1:]
        res[:, :, 2] = zk[1:]
        out[1:, m] = res
        return out
