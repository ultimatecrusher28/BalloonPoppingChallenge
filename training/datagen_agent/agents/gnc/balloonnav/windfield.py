"""windfield: altitude wind profile from balloon drift + per-balloon identify.

Balloons reach terminal velocity quickly, so their horizontal velocity tracks
the local wind. u(z) and v(z) are estimated SEPARATELY (plan spec) from
causal samples aggregated into altitude cells (100 Hz samples of one balloon
are not independent; a cell carries one aggregate per balloon):

- baseline: inverse-variance weighted cell means (always available)
- main: weighted smoothing spline over cell aggregates
  (scipy.interpolate.make_smoothing_spline) when >= 4 distinct cells

Above/below the covered band the profile falls back to the nearest cell with
inflated uncertainty, and ``coverage()`` says so (WindCoverage metadata).

The candidate extends ``extrapolate``: short horizons keep the per-balloon
constant-acceleration model (measured p90 0.13 m at 8 s); beyond that the
balloon's horizontal velocity relaxes toward the wind profile at its
predicted altitude, and ``identify()`` reports each balloon's rise rate
(buoyancy/mass proxy; MC draws mass with 25% sigma).
"""

from __future__ import annotations

import numpy as np

from ..interfaces import BalloonParams, WindCoverage
from ..registry import register
from .extrapolate import ExtrapolatePredictor, MAX_EXTRAP_S

CELL_M = 25.0  # altitude cell size (10-50 m per plan)
MIN_CELLS_SPLINE = 4
RELAX_S = 6.0  # horizontal velocity relaxation time toward wind


@register("balloonnav", "windfield")
class WindfieldPredictor(ExtrapolatePredictor):
    def __init__(self, given_parameters, model, **kwargs):
        super().__init__(given_parameters, model, **kwargs)
        self.elevation = float(given_parameters["environment"]["elevation"])
        # cell index -> {bid: (sum_w, sum_wu, sum_wv, n)}
        self._cells: dict[int, dict[int, list]] = {}
        self._rise: dict[int, list] = {}
        self._spline_u = None
        self._spline_v = None
        self._spline_dirty = True
        self._steps = 0

    # -- ingestion -----------------------------------------------------
    def update(self, observation: dict) -> None:
        super().update(observation)
        self._steps += 1
        if self._steps % 10:  # 10 Hz aggregation is plenty
            return
        live = np.nonzero(self.status == 1)[0]
        for bid in live:
            z = float(self.states[bid, 2])
            u, v, w = (float(x) for x in self.states[bid, 3:6])
            cell = int((z - self.elevation) // CELL_M)
            slot = self._cells.setdefault(cell, {}).setdefault(
                int(bid), [0.0, 0.0, 0.0, 0]
            )
            slot[0] += 1.0
            slot[1] += u
            slot[2] += v
            slot[3] += 1
            rise = self._rise.setdefault(int(bid), [0.0, 0])
            rise[0] += w
            rise[1] += 1
        self._spline_dirty = True

    # -- profile -------------------------------------------------------
    def _cell_aggregates(self):
        zs, us, vs, ws = [], [], [], []
        for cell, per_bid in sorted(self._cells.items()):
            if not per_bid:
                continue
            uu = [s[1] / s[0] for s in per_bid.values()]
            vv = [s[2] / s[0] for s in per_bid.values()]
            zs.append(self.elevation + (cell + 0.5) * CELL_M)
            us.append(float(np.mean(uu)))
            vs.append(float(np.mean(vv)))
            ws.append(float(len(per_bid)))  # distinct balloons = weight
        return (np.asarray(zs), np.asarray(us), np.asarray(vs), np.asarray(ws))

    def _rebuild(self):
        self._spline_dirty = False
        zs, us, vs, ws = self._cell_aggregates()
        self._prof = (zs, us, vs)
        self._spline_u = self._spline_v = None
        if len(zs) >= MIN_CELLS_SPLINE:
            try:
                from scipy.interpolate import make_smoothing_spline

                self._spline_u = make_smoothing_spline(zs, us, w=ws)
                self._spline_v = make_smoothing_spline(zs, vs, w=ws)
            except Exception:
                pass

    def wind_at(self, z: float) -> np.ndarray:
        """(u, v) at altitude z: spline inside coverage, nearest cell
        outside (the caller sees the band through coverage())."""
        if self._spline_dirty:
            self._rebuild()
        zs, us, vs = self._prof
        if len(zs) == 0:
            return np.zeros(2)
        z_c = float(np.clip(z, zs[0], zs[-1]))
        if self._spline_u is not None:
            return np.array([float(self._spline_u(z_c)), float(self._spline_v(z_c))])
        return np.array([float(np.interp(z_c, zs, us)), float(np.interp(z_c, zs, vs))])

    def coverage(self) -> WindCoverage:
        if self._spline_dirty:
            self._rebuild()
        zs, _us, _vs = self._prof
        distinct = len({b for per in self._cells.values() for b in per})
        if len(zs) == 0:
            return WindCoverage(
                z_lo=self.elevation, z_hi=self.elevation, n_balloons=0, quality="prior"
            )
        quality = "good" if len(zs) >= MIN_CELLS_SPLINE and distinct >= 8 else "sparse"
        return WindCoverage(
            z_lo=float(zs[0]), z_hi=float(zs[-1]), n_balloons=distinct, quality=quality
        )

    def identify(self, balloon_id: int) -> BalloonParams | None:
        rise = self._rise.get(int(balloon_id))
        if not rise or rise[1] < 10:
            return None
        w_mean = rise[0] / rise[1]
        return BalloonParams(
            balloon_id=int(balloon_id),
            buoyancy_mass_ratio=float(w_mean),
            drag_scale=1.0,
            confidence=min(rise[1] / 100.0, 1.0),
        )

    # -- prediction ----------------------------------------------------
    def predict(self, balloon_id: int, t: float):
        out = super().predict(balloon_id, t)
        if not isinstance(out, tuple):
            return out
        dt_raw = max(float(t - self.t), 0.0)
        if dt_raw <= MAX_EXTRAP_S:
            return out  # short horizon: measured-good model
        p, v = out
        # Beyond the const-accel horizon, relax the horizontal velocity
        # toward the wind at the predicted altitude over RELAX_S.
        wind = self.wind_at(float(p[2]))
        extra = dt_raw - MAX_EXTRAP_S
        k = 1.0 - np.exp(-extra / RELAX_S)
        v_adj = v.copy()
        v_adj[0:2] = (1 - k) * v[0:2] + k * wind
        p_adj = p.copy()
        p_adj[0:2] += 0.5 * (v_adj[0:2] - v[0:2]) * extra
        return p_adj, v_adj
