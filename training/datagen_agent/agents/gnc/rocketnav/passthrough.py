"""passthrough: GNSS read-through + gyro-integrated attitude.

Exact in sc0/sc1 (zero sensor noise); the permanent fallback elsewhere. The
attitude starts from the launch inclination/heading and integrates body rates;
position/velocity pass the GNSS solution straight through.

Sensor layout (100 Hz, nan before launch):
``[gyroX..Z (body), accX..Z (body), posX..Z, velX..Z (launch frame)]``.
"""

from __future__ import annotations

import numpy as np

from ....envs.balloon_world import get_initial_attitude
from ..interfaces import LaunchDecision, RocketEstimate, RocketNav
from ..registry import register


def _quat_mult(q, r):
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array(
        [
            q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3,
            q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2,
            q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1,
            q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0,
        ]
    )


@register("rocketnav", "passthrough")
class PassthroughNav(RocketNav):
    def __init__(self, given_parameters, model, **_kwargs):
        self.dt = float(given_parameters["simulation"]["time_step"])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_t = None
        self._started = False

    def on_launch(self, decision: LaunchDecision, t_launch: float) -> None:
        e = get_initial_attitude(decision.inclination_deg, decision.heading_deg)
        self.quat = np.asarray(e, float)
        self._started = True
        self._last_t = None

    def update(self, observation: dict, *, deadline: float) -> RocketEstimate:
        t = float(observation["simulation_time"])
        sens = np.asarray(observation["rocket_sensors"], float)
        omega = sens[0:3]
        pos = sens[6:9]
        vel = sens[9:12]

        if np.isfinite(omega).all() and self._started:
            dt = self.dt if self._last_t is None else max(t - self._last_t, 0.0)
            angle = np.linalg.norm(omega) * dt
            if angle > 1e-12:
                axis = omega / np.linalg.norm(omega)
                dq = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
                self.quat = _quat_mult(self.quat, dq)
                self.quat /= np.linalg.norm(self.quat)
            self._last_t = t

        ok = np.isfinite(pos).all()
        return RocketEstimate(
            t=t,
            position=pos if ok else np.zeros(3),
            velocity=vel if ok else np.zeros(3),
            quaternion=self.quat.copy(),
            omega=omega if np.isfinite(omega).all() else np.zeros(3),
            quality={
                "attitude": 0.0 if self._started else np.inf,
                "position": 0.0 if ok else np.inf,
                "velocity": 0.0 if ok else np.inf,
            },
            source="passthrough",
        )
