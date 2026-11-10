"""Inner loop: acceleration command -> TVC + throttle (CompensatedInnerLoop, feedforward path).

The desired force is m * (a_cmd + g z) minus the aero normal force (GivenLimits) and the drag; the throttle comes from its
magnitude, and the nose is steered toward its direction through the square-root braking attitude reference
(follow/ff_ref.py, the same law as the planner's GivenSim) and the exact inversion of the given gimbal actuator.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import InnerLoop, RocketEstimate
from ..vehicle.rocket_model import GRAVITY
from .ff_ref import FF_DIM, FFParams, ff_step

_Z = np.array([0.0, 0.0, 1.0])


def quat_to_matrix(q):
    e0, e1, e2, e3 = q
    return np.array(
        [
            [1 - 2 * (e2**2 + e3**2), 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)],
            [2 * (e1 * e2 + e0 * e3), 1 - 2 * (e1**2 + e3**2), 2 * (e2 * e3 - e0 * e1)],
            [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), 1 - 2 * (e1**2 + e2**2)],
        ]
    )


def _action(gx, gy, throttle):
    return {
        "launch": False,
        "launch_inclination_heading": np.array([90.0, 0.0]),
        "tvc": np.array([gx, gy], dtype=float),
        "roll": 0.0,
        "throttle": float(throttle),
    }


class CompensatedInnerLoop(InnerLoop):
    """Aero-normal-force compensation from the given-parameter model (GivenLimits), throttle commands clamped to the
    actuator's rate limit, and the gimbal command from the feedforward attitude reference with the exact inversion of
    the given gimbal actuator (first-order lag, then rate limit, then range)."""

    def __init__(
        self,
        given_parameters,
        model,
        *,
        tables,
        ff_model,
        k_att=2.0,
        w_max=2.0,
        kp_rate=25.0,
        ff=True,
    ):
        if ff is not True:
            raise ValueError(
                "this agent's inner loop is the feedforward path only (loop_kwargs ff: true)"
            )
        self.model = model
        self.tables = tables
        self.k_att = float(k_att)
        self.w_max = float(w_max)  # rad/s body-rate command clamp
        self.kp_rate = float(kp_rate)  # deg of gimbal per rad/s rate error
        self.t_launch = None
        self.g_act = np.zeros(2)
        self.thr_act = 1.0
        # Actuator model from given_parameters: rate limits, the gimbal's first-order lag and its range.
        ctrl = dict(given_parameters.get("rocket", {}).get("control", {}))
        dt = float(given_parameters["simulation"]["time_step"])
        self.dt = dt
        self.gimbal_step = float(ctrl.get("gimbal_rate_limit", 60.0)) * dt
        self.throttle_step = float(ctrl.get("throttle_rate_limit", 2.0)) * dt
        # feedforward (follow/ff_ref.py; mirrored in GivenSim): g_act holds the modelled actuator OUTPUT; ff_model
        # provides inertia(t_rel) and lever from given parameters (the planner's GivenSim)
        self.ff = True
        self.gmax = float(ctrl["max_gimbal_angle"])
        gtau = float(ctrl.get("gimbal_time_constant", 0.0) or 0.0)
        self.g_alpha = dt / (gtau + dt)
        self.ff_model = ff_model
        self.ffp = FFParams(gmax=self.gmax)
        self.ff_state = np.zeros(FF_DIM)

    def on_launch(self, t_launch: float) -> None:
        self.t_launch = float(t_launch)
        self.g_act = np.zeros(2)
        self.thr_act = 1.0
        self.ff_state = np.zeros(FF_DIM)

    def _p3a_gimbal(self, n_des, rotation, thrust, t_rel, omega, flags):
        """Cascade on the feedforward reference -> desired actuator output -> exact actuator inversion. Same law as
        GivenSim.step (ff); returns the command u."""
        omega = np.asarray(omega, float)
        n_att = np.asarray(n_des, float)
        st, n_ref, w_ff, d_ff = ff_step(
            self.ff_state[None],
            n_att[None],
            rotation[None],
            np.array([thrust]),
            np.array([self.ff_model.inertia(t_rel)]),
            self.ff_model.lever,
            self.dt,
            self.ffp,
        )
        self.ff_state = st[0]
        n_att = n_ref[0]
        d = rotation.T @ n_att
        ang = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
        h = float(np.hypot(d[0], d[1]))
        err = np.zeros(2) if h < 1e-12 else np.array([-d[1] / h * ang, d[0] / h * ang])
        flags["tilt_err_deg"] = float(
            np.degrees(np.arccos(np.clip((rotation.T @ n_des)[2], -1.0, 1.0)))
        )
        w_cmd = np.clip(self.k_att * err, -self.w_max, self.w_max)
        g_des = np.clip(
            self.kp_rate * (w_cmd + w_ff[0] - omega) + d_ff[0],
            -self.gmax,
            self.gmax,
        )
        y = self.g_act
        step = (
            0.999 * self.gimbal_step
        )  # stay inside the actuator's own rate limit (float margin)
        y_des = np.clip(g_des, y - step, y + step)
        u = np.clip(y + (y_des - y) / self.g_alpha, -self.gmax, self.gmax)
        y_new = y + self.g_alpha * (
            u - y
        )  # the actuator model (IIR, then rate limit, then range)
        y_new = np.clip(
            np.clip(y_new, y - self.gimbal_step, y + self.gimbal_step),
            -self.gmax,
            self.gmax,
        )
        self.g_act = y_new
        return u

    def solve(self, accel_cmd, estimate: RocketEstimate):
        t_rel = 0.0 if self.t_launch is None else estimate.t - self.t_launch
        self.tables.set_context(t_rel, float(estimate.position[2]))
        m = float(self.model.mass(t_rel))
        thrust_full = float(self.model.thrust(t_rel))
        rotation = quat_to_matrix(estimate.quaternion)
        nose = rotation[:, 2]
        vel = estimate.velocity
        speed = float(np.linalg.norm(vel))

        flags = {
            "burnout": thrust_full <= 1e-9,
            "throttle_sat": False,
            "tilt_err_deg": 0.0,
            "aoa_deg": 0.0,
        }
        # Aero normal force: acts perpendicular to the body axis toward the
        # nose's off-velocity side; magnitude from the given-parameter model.
        aero = np.zeros(3)
        if speed > 5.0:
            vh = vel / speed
            aoa = float(np.degrees(np.arccos(np.clip(nose @ vh, -1.0, 1.0))))
            flags["aoa_deg"] = round(aoa, 1)
            lat_dir = nose - (nose @ vh) * vh
            norm = np.linalg.norm(lat_dir)
            if norm > 1e-6:
                aero = self.tables.lateral_disturbance(speed, aoa) * lat_dir / norm
            drag = (
                -0.5
                * float(self.model.density(estimate.position[2]))
                * float(self.model.drag_coeff(t_rel))
                * self.model.ref_area
                * speed
                * vel
                / m
            )
        else:
            drag = np.zeros(3)

        force = m * (np.asarray(accel_cmd, float) + GRAVITY * _Z - aero - drag)
        force_mag = float(np.linalg.norm(force))
        if flags["burnout"] or force_mag < 1e-9:
            self.thr_act = max(0.0, self.thr_act - self.throttle_step)
            achievable = aero + drag - GRAVITY * _Z
            return _action(0.0, 0.0, self.thr_act), achievable, flags

        thr_des = force_mag / thrust_full
        flags["throttle_sat"] = thr_des > 1.0
        # Below 12 m/s the T/W margin (1.21 at ignition) leaves no room to
        # throttle down: a plan asking for less than full thrust here cannot
        # even hold the climb (measured: thr 0.78 never lifted off). Fly full
        # throttle; the leg re-solve absorbs the early arrival.
        if speed < 12.0:
            thr_des = 1.0
        thr_des = float(np.clip(thr_des, 0.0, 1.0))
        thr_cmd = float(
            np.clip(
                thr_des,
                self.thr_act - self.throttle_step,
                self.thr_act + self.throttle_step,
            )
        )
        self.thr_act = thr_cmd
        thr_u = thr_cmd

        n_des = force / force_mag
        cmd = self._p3a_gimbal(
            n_des, rotation, thrust_full * thr_cmd, t_rel, estimate.omega[:2], flags
        )

        achievable = (thrust_full * thr_cmd / m) * nose + aero + drag - GRAVITY * _Z
        return _action(float(cmd[0]), float(cmd[1]), thr_u), achievable, flags
