"""Shared Reference implementations (see interfaces.Reference)."""

from __future__ import annotations

import numpy as np


class SampledReference:
    """A reference trajectory stored as samples on a uniform-ish time grid.

    Times are seconds SINCE LAUNCH. ``state(t)`` linearly interpolates
    (pos, vel, accel, body_axis); the axis is re-normalized after
    interpolation. Beyond the last sample the reference holds its end state.
    """

    def __init__(self, ts, pos, vel, accel, axis):
        self.ts = np.asarray(ts, float)
        self.pos = np.asarray(pos, float)
        self.vel = np.asarray(vel, float)
        self.accel = np.asarray(accel, float)
        self.axis = np.asarray(axis, float)

    @property
    def t_end(self) -> float:
        return float(self.ts[-1])

    def _interp(self, arr, t):
        return np.stack([np.interp(t, self.ts, arr[:, i]) for i in range(3)], axis=-1)

    def state(self, t):
        t = np.clip(t, self.ts[0], self.ts[-1])
        pos = self._interp(self.pos, t)
        vel = self._interp(self.vel, t)
        accel = self._interp(self.accel, t)
        axis = self._interp(self.axis, t)
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = np.where(
            norm > 1e-9,
            axis / np.where(norm > 1e-9, norm, 1.0),
            np.array([0.0, 0.0, 1.0]),
        )
        return pos, vel, accel, axis
