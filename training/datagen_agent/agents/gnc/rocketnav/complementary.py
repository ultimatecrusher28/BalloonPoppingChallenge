"""complementary: Mahony-style attitude filter, mekf's fallback/diagnostic.

Gyro integration corrected at low gain toward an acceleration reference
built from GNSS velocity: the world-frame specific force is the SLOPE of a
trailing least-squares fit over ``window_s`` of velocity samples minus
gravity (NEVER a raw 100 Hz difference: with 0.05 m/s GNSS noise that would
be ~7 m/s^2 of noise). The correction vector is the Mahony error
e = f_body x R^T f_world_ref (Mahony/Hamel/Pflimlin 2008); roll about the
thrust axis stays unobservable, which the consumers tolerate.

Position/velocity pass GNSS through with a light EMA.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ....envs.balloon_world import get_initial_attitude
from ..interfaces import LaunchDecision, RocketEstimate, RocketNav
from ..registry import register
from ..vehicle.rocket_model import GRAVITY

_G_VEC = np.array([0.0, 0.0, -GRAVITY])


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


def _quat_to_matrix(q):
    e0, e1, e2, e3 = q
    return np.array(
        [
            [1 - 2 * (e2**2 + e3**2), 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)],
            [2 * (e1 * e2 + e0 * e3), 1 - 2 * (e1**2 + e3**2), 2 * (e2 * e3 - e0 * e1)],
            [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), 1 - 2 * (e1**2 + e2**2)],
        ]
    )


@register("rocketnav", "complementary")
class ComplementaryNav(RocketNav):
    def __init__(
        self,
        given_parameters,
        model,
        *,
        kp=0.5,
        ki=0.02,
        window_s=0.5,
        vel_ema=0.3,
        **_kwargs,
    ):
        self.dt = float(given_parameters["simulation"]["time_step"])
        self.kp = float(kp)
        self.ki = float(ki)
        self.n_win = max(int(window_s / self.dt), 5)
        self.vel_ema = float(vel_ema)
        self._started = False
        self._last_t = None

    def on_launch(self, decision: LaunchDecision, t_launch: float) -> None:
        self.q = np.asarray(
            get_initial_attitude(decision.inclination_deg, decision.heading_deg), float
        )
        self.bias = np.zeros(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self._vel_hist: deque = deque(maxlen=self.n_win)
        self._started = True
        self._last_t = None

    def _accel_reference(self):
        """World specific force from the trailing velocity fit, or None."""
        if len(self._vel_hist) < self.n_win:
            return None
        ts = np.array([h[0] for h in self._vel_hist])
        vs = np.array([h[1] for h in self._vel_hist])
        ts = ts - ts[-1]
        slope = np.polynomial.polynomial.polyfit(ts, vs, 1)[1]  # dv/dt (3,)
        return slope - _G_VEC  # specific force = a - g

    def update(self, observation: dict, *, deadline: float) -> RocketEstimate:
        t = float(observation["simulation_time"])
        sens = np.asarray(observation["rocket_sensors"], float)
        if not self._started or not np.isfinite(sens[0:3]).all():
            return RocketEstimate(
                t=t,
                position=np.zeros(3),
                velocity=np.zeros(3),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                omega=np.zeros(3),
                quality={"attitude": np.inf, "position": np.inf, "velocity": np.inf},
                source="complementary",
            )
        dt = self.dt if self._last_t is None else max(t - self._last_t, 1e-4)
        self._last_t = t
        gyro = sens[0:3]
        accel = sens[3:6]

        corr = np.zeros(3)
        f_ref = self._accel_reference()
        f_norm = np.linalg.norm(accel)
        if f_ref is not None and f_norm > 1.0 and np.linalg.norm(f_ref) > 1.0:
            rot = _quat_to_matrix(self.q)
            v_body_ref = rot.T @ (f_ref / np.linalg.norm(f_ref))
            e = np.cross(accel / f_norm, v_body_ref)
            corr = self.kp * e
            self.bias -= self.ki * e * dt

        w = gyro - self.bias + corr
        angle = np.linalg.norm(w) * dt
        if angle > 1e-12:
            axis = w / np.linalg.norm(w)
            dq = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
            self.q = _quat_mult(self.q, dq)
            self.q /= np.linalg.norm(self.q)

        if np.isfinite(sens[6:12]).all():
            gnss_p, gnss_v = sens[6:9], sens[9:12]
            self.p = (
                (1 - self.vel_ema) * self.p + self.vel_ema * gnss_p
                if np.linalg.norm(self.p) > 0
                else gnss_p.copy()
            )
            self.v = (1 - self.vel_ema) * self.v + self.vel_ema * gnss_v
            self._vel_hist.append((t, gnss_v.copy()))

        return RocketEstimate(
            t=t,
            position=self.p.copy(),
            velocity=self.v.copy(),
            quaternion=self.q.copy(),
            omega=gyro - self.bias,
            quality={
                "attitude": 0.05 if f_ref is not None else 0.2,
                "position": 2.0,
                "velocity": 0.1,
            },
            source="complementary",
        )
