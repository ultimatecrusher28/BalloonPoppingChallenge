"""extrapolate: per-balloon short-horizon extrapolation.

Balloon observations are ground truth, so no filtering: hold the latest
(pos, vel) per balloon and extrapolate with a smoothed acceleration estimate
(EMA of the velocity difference). Serves as the terminal-guidance predictor
and as the comparison baseline for the wind-field candidate.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import BalloonPredictor, ReleasePoint
from ..registry import register

ACCEL_EMA = 0.2  # smoothing for the per-balloon accel estimate
MAX_EXTRAP_S = 8.0  # clamp the horizon; beyond this the estimate is stale


@register("balloonnav", "extrapolate")
class ExtrapolatePredictor(BalloonPredictor):
    def __init__(self, given_parameters, model, **_kwargs):
        self.t = None
        self.states = None  # (N, 6) latest observation
        self.status = None  # (N,) 0 ground / 1 released / 2 popped
        self.accel = None  # (N, 3) EMA of dv/dt per balloon

    def update(self, observation: dict) -> None:
        t = float(observation["simulation_time"])
        states = np.asarray(observation["balloon_states"], float)
        status = np.asarray(observation["balloon_status"], int).reshape(-1)
        if self.accel is None:
            self.accel = np.zeros((states.shape[0], 3))
        elif self.t is not None and t > self.t:
            dv = (states[:, 3:6] - self.states[:, 3:6]) / (t - self.t)
            live = (status == 1) & (self.status == 1)
            self.accel[live] = (1 - ACCEL_EMA) * self.accel[live] + ACCEL_EMA * dv[live]
        self.t, self.states, self.status = t, states, status

    def predict(self, balloon_id: int, t: float):
        status = int(self.status[balloon_id])
        if status == 2:
            return None
        if status == 0:
            return ReleasePoint(
                balloon_id=balloon_id, position=self.states[balloon_id, :3].copy()
            )
        # Constant acceleration up to MAX_EXTRAP_S, constant velocity beyond
        # (NEVER freeze: planners look 25+ s ahead, and a frozen target sits
        # tens of meters below where the balloon will be).
        dt_raw = max(float(t - self.t), 0.0)
        dt = min(dt_raw, MAX_EXTRAP_S)
        extra = dt_raw - dt
        p = self.states[balloon_id, :3]
        v = self.states[balloon_id, 3:6]
        a = self.accel[balloon_id]
        v_end = v + a * dt
        return (p + v * dt + 0.5 * a * dt * dt + v_end * extra), v_end

    # -- batch helper for planners ------------------------------------
    def released_ids(self) -> np.ndarray:
        return np.nonzero(self.status == 1)[0]

    def paths(self, ids: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """(len(ids), len(ts), 3) predicted positions at absolute times ts.

        Constant acceleration up to MAX_EXTRAP_S, constant velocity beyond
        (balloons drift steadily; measured p90 error 0.13 m at 8 s, and the
        far tail self-corrects through in-flight re-solving)."""
        dt_raw = np.asarray(ts, float) - self.t
        dt = np.clip(dt_raw, 0.0, MAX_EXTRAP_S)[None, :, None]
        extra = np.clip(dt_raw - MAX_EXTRAP_S, 0.0, None)[None, :, None]
        p = self.states[ids, None, :3]
        v = self.states[ids, None, 3:6]
        a = self.accel[ids, None, :]
        return p + v * dt + 0.5 * a * dt * dt + (v + a * MAX_EXTRAP_S) * extra
