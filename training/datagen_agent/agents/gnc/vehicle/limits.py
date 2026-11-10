"""Measured limit tables for planning and the inner loop.

Numbers come from ``research/limits/data`` (rerunnable measurements). The
files are read when present; otherwise the embedded snapshot (same numbers)
applies, so the agent stays self-contained for submission.

Currently served:

- ``lateral_disturbance(speed, aoa_deg)``: unmodeled lateral acceleration
  (aero normal force) the simple thrust-axis model does NOT predict, from
  probe-accel-residual (REPORT section 1.3). The inner loop treats it as a
  disturbance bound; ignoring it costs 20-30% of the commanded acceleration
  above 30 m/s.
- ``SLEW_REST_TO_REST_90_S``: measured attitude-turn timescale for cost
  models (a large turn costs about half a balloon of tour rhythm).

The full speed x AoA lateral-ACCEL AUTHORITY table (what the vehicle CAN pull,
not just the model error) is measurement group 2 and will extend this module.
"""

from __future__ import annotations

import json
import os

import numpy as np

# Measured on the legacy branch (slew_bangbang): angular accel 7.45-8.2 rad/s^2,
# rest-to-rest 90 deg in about 1.0 s at 60 deg/s gimbal-rate-limited TVC.
SLEW_REST_TO_REST_90_S = 1.0
GIMBAL_RATE_DEG_S = 60.0
GIMBAL_MAX_DEG = 15.0
AOA_LIFT_SATURATION_DEG = 18.0

_DATA_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "research",
        "limits",
        "data",
    )
)

# Embedded snapshot of probe-accel-residual: mean unmodeled lateral accel
# (m/s^2) per speed bin (aoa-aggregated; the aoa axis refines it when the
# data file is available).
_RESIDUAL_SPEED_EDGES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 150.0]
_RESIDUAL_LATERAL = [0.15, 0.6, 1.35, 3.0, 4.8, 4.8]


# Measured turn-time anchors (slew_bangbang, thr=1.0): angle deg -> settle s.
# Roughly linear 0.7/1.1/1.4 s at 30/90/150 deg across tb and speed.
_TURN_ANGLE_DEG = [0.0, 30.0, 90.0, 150.0, 180.0]
_TURN_TIME_S = [0.25, 0.70, 1.10, 1.40, 1.55]

# Embedded snapshot of lateral_accel_table (thr=1.0, alpha=18 deg boundary):
# speed -> (a_lat authority m/s^2, unmodeled lateral resid m/s^2).
_LAT_SPEED = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]
_LAT_A18 = [5.40, 5.81, 6.18, 6.61, 7.11, 7.72, 8.98, 10.08, 10.90]
_LAT_RESID18 = [0.17, 0.46, 0.89, 1.44, 2.12, 2.97, 4.72, 6.26, 7.53]


def turn_time(angle_rad: float) -> float:
    """Measured rest-to-rest attitude turn time (s) for a given angle."""
    deg = abs(np.degrees(angle_rad))
    return float(np.interp(deg, _TURN_ANGLE_DEG, _TURN_TIME_S))


class LimitTables:
    """Lookup object; construct once per agent and share."""

    def __init__(self, data_dir: str = _DATA_DIR):
        self._cells = None
        self._lat_rows = None
        path = os.path.join(data_dir, "probe-accel-residual.json")
        try:
            with open(path) as f:
                self._cells = json.load(f)["cells"]
        except (OSError, ValueError, KeyError):
            self._cells = None
        try:
            with open(os.path.join(data_dir, "lateral_accel_table.json")) as f:
                self._lat_rows = [
                    r for r in json.load(f) if r.get("ok") and r["throttle"] == 1.0
                ]
        except (OSError, ValueError, KeyError):
            self._lat_rows = None

    def a_lat_max(
        self, speed: float, aoa_deg: float = AOA_LIFT_SATURATION_DEG
    ) -> float:
        """Measured velocity-turn acceleration (m/s^2) at this speed and AoA
        (default: the 18 deg lift-saturation boundary)."""
        if self._lat_rows is not None:
            pts = sorted(
                (r["v"], r["a_lat"])
                for r in self._lat_rows
                if abs(r["alpha_cmd"] - aoa_deg) < 2.5
            )
            if pts:
                vs, als = zip(*pts)
                return float(np.interp(speed, vs, als))
        return float(np.interp(speed, _LAT_SPEED, _LAT_A18))

    def lateral_use(self, vel, accel, *, speed_floor: float = 8.0):
        """Velocity-turning demand as a fraction of the measured authority.

        ``vel`` and ``accel`` are (N, 3) samples of the same trajectory; the
        returned array holds one ratio per sample with |v| above
        ``speed_floor`` (below that a flight path is not a meaningful thing
        to turn, and the a_lat table is not measured there).

        Only the component of acceleration PERPENDICULAR to the velocity is
        charged: the along-track part changes speed, not heading, and the
        a_lat table measures heading change.

        One definition, three consumers - the planners' feasibility gates and
        ``metrics.plan_metrics`` - so a ratio on a bench row and a ratio
        inside a search mean the same thing.
        """
        vel = np.atleast_2d(np.asarray(vel, dtype=float))
        accel = np.atleast_2d(np.asarray(accel, dtype=float))
        speed = np.linalg.norm(vel, axis=1)
        live = speed > speed_floor
        if not live.any():
            return np.empty(0)
        vh = vel[live] / speed[live][:, None]
        a = accel[live]
        lat = np.linalg.norm(a - (a * vh).sum(axis=1)[:, None] * vh, axis=1)
        cap = np.array([self.a_lat_max(float(x)) for x in speed[live]])
        return lat / np.maximum(cap, 1e-6)

    def lateral_disturbance(self, speed: float, aoa_deg: float | None = None) -> float:
        """Unmodeled lateral accel (m/s^2) at this flight condition."""
        if self._lat_rows is not None and aoa_deg is not None:
            pts = sorted(
                (r["v"], r["resid_lat"])
                for r in self._lat_rows
                if abs(r["alpha_true"] - aoa_deg) < 4.0
            )
            if len(pts) >= 3:
                vs, res = zip(*pts)
                return float(np.interp(speed, vs, res))
        if self._cells is not None and aoa_deg is not None:
            best, best_score = None, None
            for c in self._cells:
                in_speed = c["speed_lo"] <= speed < c["speed_hi"]
                in_aoa = c["aoa_lo"] <= aoa_deg < c["aoa_hi"]
                if in_speed and in_aoa:
                    return float(c["resid_lat_mean"])
                if in_speed:
                    score = abs(0.5 * (c["aoa_lo"] + c["aoa_hi"]) - aoa_deg)
                    if best_score is None or score < best_score:
                        best, best_score = c, score
            if best is not None:
                return float(best["resid_lat_mean"])
        idx = int(
            np.clip(
                np.digitize(speed, _RESIDUAL_SPEED_EDGES) - 1,
                0,
                len(_RESIDUAL_LATERAL) - 1,
            )
        )
        return _RESIDUAL_LATERAL[idx]
