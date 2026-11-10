"""inner-loop-rough: acceleration command -> TVC + throttle.

Skeleton inversion: desired force = m*(a_cmd + g z); throttle from its
magnitude, attitude PD (on the body-frame pointing error) steers the nose
toward its direction. Reports the honestly achievable acceleration along the
CURRENT nose axis and saturation flags, per the InnerLoop contract.

The wave-1 candidate adds the 2D limits-table compensation and separates
gimbal rate-limit feedforward from the sc3/4 low-pass lag.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import InnerLoop, RocketEstimate
from ..registry import register
from ..vehicle.limits import GIMBAL_MAX_DEG
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


@register("inner_loop", "inner-loop-rough")
class RoughInnerLoop(InnerLoop):
    def __init__(self, given_parameters, model, *, kp=1.2, kd=0.35, **_kwargs):
        self.model = model
        self.kp = float(kp)
        self.kd = float(kd)
        self.t_launch = None

    def on_launch(self, t_launch: float) -> None:
        self.t_launch = float(t_launch)

    def solve(self, accel_cmd, estimate: RocketEstimate):
        t_rel = 0.0 if self.t_launch is None else estimate.t - self.t_launch
        m = float(self.model.mass(t_rel))
        thrust_full = float(self.model.thrust(t_rel))
        rotation = quat_to_matrix(estimate.quaternion)
        nose = rotation[:, 2]

        force = m * (np.asarray(accel_cmd, float) + GRAVITY * _Z)
        force_mag = float(np.linalg.norm(force))
        flags = {
            "burnout": thrust_full <= 1e-9,
            "throttle_sat": False,
            "tilt_err_deg": 0.0,
        }

        if flags["burnout"] or force_mag < 1e-9:
            action = _action(0.0, 0.0, 0.0)
            achievable = -GRAVITY * _Z
            return action, achievable, flags

        throttle = force_mag / thrust_full
        flags["throttle_sat"] = throttle > 1.0
        throttle = float(np.clip(throttle, 0.0, 1.0))

        n_des = force / force_mag
        # Pointing error in body frame (nose = body +Z), legacy-verified form.
        d = rotation.T @ n_des
        ang = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
        h = float(np.hypot(d[0], d[1]))
        ex, ey = (0.0, 0.0) if h < 1e-12 else (-d[1] / h * ang, d[0] / h * ang)
        flags["tilt_err_deg"] = float(np.degrees(ang))
        gx = float(
            np.clip(
                np.degrees(self.kp * ex - self.kd * estimate.omega[0]),
                -GIMBAL_MAX_DEG,
                GIMBAL_MAX_DEG,
            )
        )
        gy = float(
            np.clip(
                np.degrees(self.kp * ey - self.kd * estimate.omega[1]),
                -GIMBAL_MAX_DEG,
                GIMBAL_MAX_DEG,
            )
        )

        achievable = thrust_full * throttle / m * nose - GRAVITY * _Z
        return _action(gx, gy, throttle), achievable, flags


def _action(gx, gy, throttle):
    return {
        "launch": False,
        "launch_inclination_heading": np.array([90.0, 0.0]),
        "tvc": np.array([gx, gy], dtype=float),
        "roll": 0.0,
        "throttle": float(throttle),
    }


@register("inner_loop", "inner-loop-compensated")
class CompensatedInnerLoop(InnerLoop):
    """Wave-1 inversion: aero-normal-force compensation from the measured 2D
    table, gimbal commands shaped inside the 60 deg/s rate limit (bang-bang
    with braking on large errors, PD capture near alignment), throttle
    commands clamped to the actuator's 0.020/step rate limit.

    Measured basis (research/limits): unmodeled lateral accel 0.5-7.5 m/s^2
    over 10-60 m/s at the 18 deg AoA boundary; gimbal ramp 0.21 s; peak
    angular accel about 12.7 rad/s^2.
    """

    GIMBAL_STEP_DEG = 0.6  # 60 deg/s at 100 Hz
    THROTTLE_STEP = 0.020  # env actuator rate limit per step
    BANG_ERR_DEG = 6.0  # above this, time-optimal-ish bang; below, PD

    # Cascade gains (legacy-validated): attitude error -> clamped body-rate
    # command -> rate loop -> gimbal. A plain PD limit-cycled at 200-340
    # deg/s under gimbal saturation + the 0.25 s actuator sweep.
    def __init__(
        self,
        given_parameters,
        model,
        *,
        k_att=2.0,
        w_max=2.0,
        kp_rate=25.0,
        wdot_est=8.0,
        use_lead=False,
        tables=None,
        act_inv=False,
        ff=False,
        ff_model=None,
        thr_inv=False,
        **_kwargs,
    ):
        self.model = model
        if tables is not None:
            self.tables = tables
        else:
            from ..vehicle.limits import LimitTables

            self.tables = LimitTables()
        self.k_att = float(k_att)
        self.w_max = float(w_max)  # rad/s body-rate command clamp
        self.kp_rate = float(kp_rate)  # deg of gimbal per rad/s rate error
        self.wdot_est = float(wdot_est)
        self.t_launch = None
        self.g_act = np.zeros(2)
        self.thr_act = 1.0
        # Actuator model from given_parameters (legal): rate limits replace
        # the hardcoded steps, and the first-order lags get an inverse lead
        # compensation, SEPARATE from the rate-limit handling (interface
        # contract; sc1/2 have tau=0 and the lead is a no-op).
        ctrl = dict(given_parameters.get("rocket", {}).get("control", {}))
        dt = float(given_parameters["simulation"]["time_step"])
        self.dt = dt
        self.gimbal_step = float(ctrl.get("gimbal_rate_limit", 60.0)) * dt
        self.throttle_step = float(ctrl.get("throttle_rate_limit", 2.0)) * dt
        # Lead compensation off by default: paired sc3 measured -0.30 (the
        # derivative term amplifies estimate noise); rate limits stay in.
        use = bool(use_lead)
        self.tau_gimbal = (
            float(ctrl.get("gimbal_time_constant", 0.0) or 0.0) if use else 0.0
        )
        self.tau_throttle = (
            float(ctrl.get("throttle_time_constant", 0.0) or 0.0) if use else 0.0
        )
        self._g_des_prev = np.zeros(2)
        self._thr_des_prev = 1.0
        # W3 P3a (follow/ff_ref.py; mirrored in GivenSim). act_inv: exact inversion of the given gimbal
        # actuator (lag then rate limit); g_act then holds the modelled actuator OUTPUT. ff: setpoint
        # feedforward (implies act_inv); ff_model provides inertia(t_rel) and lever from given parameters
        # (the planner's GivenSim). Both off = v7.
        self.ff = bool(ff)
        self.act_inv = bool(act_inv) or self.ff
        self.gmax = float(ctrl.get("max_gimbal_angle", GIMBAL_MAX_DEG))
        gtau = float(ctrl.get("gimbal_time_constant", 0.0) or 0.0)
        self.g_alpha = dt / (gtau + dt)
        # thr_inv: the same exact inversion for the given throttle actuator (IIR lag, then rate limit, then
        # range). thr_act then holds the modelled throttle OUTPUT; GivenSim already assumes this output
        # (rate-limited, no lag), so the planner needs no change. Off = v7 behaviour.
        self.thr_inv = bool(thr_inv)
        self.thr_range = tuple(float(x) for x in ctrl.get("throttle_range", (0.0, 1.0)))
        ttau = float(ctrl.get("throttle_time_constant", 0.0) or 0.0)
        self.t_alpha = dt / (ttau + dt)
        if self.ff:
            if ff_model is None:
                raise ValueError("ff=True needs ff_model (inertia(t_rel), lever)")
            self.ff_model = ff_model
            self.ffp = FFParams(
                gmax=self.gmax,
                **{k: v for k, v in _kwargs.items() if k.startswith("ff_")},
            )
            self.ff_state = np.zeros(FF_DIM)

    def _lead_gimbal(self, desired):
        """First-order inverse: u = y + tau*dy/dt, clipped to the range."""
        if self.tau_gimbal <= 0.0:
            return desired
        out = desired + self.tau_gimbal * (desired - self._g_des_prev) / self.dt
        self._g_des_prev = desired.copy()
        return np.clip(out, -GIMBAL_MAX_DEG, GIMBAL_MAX_DEG)

    def _lead_throttle(self, desired):
        if self.tau_throttle <= 0.0:
            return desired
        out = desired + self.tau_throttle * (desired - self._thr_des_prev) / self.dt
        self._thr_des_prev = desired
        return float(np.clip(out, 0.0, 1.0))

    def on_launch(self, t_launch: float) -> None:
        self.t_launch = float(t_launch)
        self.g_act = np.zeros(2)
        self.thr_act = 1.0
        if self.ff:
            self.ff_state = np.zeros(FF_DIM)

    def _axis_cmds(self, err, omega):
        """Cascade: attitude error -> clamped body-rate command -> rate loop.
        The rate clamp bounds how violently the attitude loop may ask to
        turn, which is what keeps the loop out of the saturation limit
        cycle a plain PD fell into."""
        cmds = []
        for e, w in zip(err, omega):
            w_cmd = float(np.clip(self.k_att * e, -self.w_max, self.w_max))
            cmds.append(
                float(
                    np.clip(self.kp_rate * (w_cmd - w), -GIMBAL_MAX_DEG, GIMBAL_MAX_DEG)
                )
            )
        return cmds

    def _p3a_gimbal(self, n_des, rotation, thrust, t_rel, omega, flags):
        """W3 P3a: cascade (optionally on the feedforward reference) -> desired actuator output -> exact
        actuator inversion. Same law as GivenSim.step(act_inv/ff); returns the command u."""
        omega = np.asarray(omega, float)
        n_att = np.asarray(n_des, float)
        if self.ff:
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
        if self.ff:
            g_des = np.clip(
                self.kp_rate * (w_cmd + w_ff[0] - omega) + d_ff[0],
                -self.gmax,
                self.gmax,
            )
        else:
            g_des = np.clip(self.kp_rate * (w_cmd - omega), -self.gmax, self.gmax)
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
        if hasattr(self.tables, "set_context"):
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
        # nose's off-velocity side; magnitude from the measured table.
        aero = np.zeros(3)
        if speed > 5.0:
            vh = vel / speed
            aoa = float(np.degrees(np.arccos(np.clip(nose @ vh, -1.0, 1.0))))
            flags["aoa_deg"] = round(aoa, 1)
            lat_dir = nose - (nose @ vh) * vh
            norm = np.linalg.norm(lat_dir)
            if norm > 1e-6:
                # Discussion #165 given-mode: the model needs t_rel/altitude
                # context (thrust/mass and ISA density both depend on it)
                # before a lookup; legacy LimitTables has no such method, so
                # this is a no-op there and the legacy path is unaffected.
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
        if self.thr_inv:
            lo, hi = self.thr_range
            y = self.thr_act
            y_des = float(
                np.clip(
                    thr_des,
                    max(lo, y - self.throttle_step),
                    min(hi, y + self.throttle_step),
                )
            )
            u = float(
                np.clip(y + (y_des - y) / self.t_alpha, lo, hi)
            )  # action sent to the env
            y_new = y + self.t_alpha * (u - y)  # the actuator model
            self.thr_act = float(
                np.clip(
                    np.clip(y_new, y - self.throttle_step, y + self.throttle_step),
                    lo,
                    hi,
                )
            )
            thr_cmd = self.thr_act  # modelled output (force)
            thr_u = u
        else:
            thr_des = self._lead_throttle(float(np.clip(thr_des, 0.0, 1.0)))
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
        if self.act_inv:
            cmd = self._p3a_gimbal(
                n_des, rotation, thrust_full * thr_cmd, t_rel, estimate.omega[:2], flags
            )
        else:
            d = rotation.T @ n_des
            ang = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
            h = float(np.hypot(d[0], d[1]))
            err = (0.0, 0.0) if h < 1e-12 else (-d[1] / h * ang, d[0] / h * ang)
            flags["tilt_err_deg"] = float(np.degrees(ang))
            raw = self._lead_gimbal(
                np.asarray(self._axis_cmds(err, estimate.omega[:2]))
            )
            # Rate-limit-aware: command only what the actuator can reach this step.
            cmd = np.clip(
                raw, self.g_act - self.gimbal_step, self.g_act + self.gimbal_step
            )
            self.g_act = cmd

        achievable = (thrust_full * thr_cmd / m) * nose + aero + drag - GRAVITY * _Z
        return _action(float(cmd[0]), float(cmd[1]), thr_u), achievable, flags
