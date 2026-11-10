"""mekf: multiplicative error-state Kalman filter with a model-aided pitch/yaw rate.

States: nominal (p, v, q, b_g, b_a, omega_xy, d_xy); error (dp, dv, dtheta, db_g, db_a, domega_xy, dd_xy), 19 in all,
attitude error as a body-frame rotation vector injected by RIGHT multiplication (Sola 2017, arXiv:1711.02508).
Joseph-form updates, covariance reset Jacobian after injection, quaternion normalised every step. Sensor layout:
[gyro(3) body, accel(3) body specific force, GNSS pos(3), GNSS vel(3) launch frame], 100 Hz, nan before launch.

Everything comes from ``given_parameters``:
- Noise: per-sample sigma = noise density * sqrt(sampling rate), as the env builds its sensors; bias random walks and
  GNSS accuracies (altitude separate) as given. A scenario without gyro noise is not supported.
- Launch state: position at the pad (0, 0, environment.elevation), the frame GNSS and the balloons use; attitude from
  the launch decision; biases zero (the given constant bias is 0 and the sensors are built at launch). The launch
  velocity is not known (the env adds a numerical 0.23 m/s in its first step), so the first valid sample only
  initialises the state: velocity from the GNSS velocity with the given accuracy, position and bias variances from the
  elapsed step.
- Pitch/yaw body rate omega_xy and a first-order Gauss-Markov disturbance angular acceleration d_xy are states. The rate
  is propagated with the given-parameter moment model (``rate_model.RateModel``: TVC, aerodynamic lift with rate
  damping, inertia, ISA atmosphere, wind 0) and the outputs of the given actuator models driven by the commands the env
  applied (``set_actuation``); the gyro x/y is a measurement; the roll rate stays gyro-driven; attitude uses the
  trapezoid of the rate. The disturbance power q and correlation time tau are estimated online by a multiple-model bank
  (``rate_model.GmBank``) whose likelihood fades with a ``MEMORY_S`` memory.
- Gravity at the launch site (``rate_model.local_gravity``: the given value if any, else WGS84 normal gravity at the
  given latitude and the current altitude), as the env uses.
- Position update p += v dt + a dt^2 / 2 (the trapezoid of the old and new velocity), with the matching covariance terms.
- Accelerometer plausibility check: a sample above the largest specific force the given vehicle can produce is replaced
  by the model specific force, with the velocity process noise of a uniform error inside that limit (the env produces
  one numerical spike of about 1087 m/s^2 at motor cutoff).

``set_actuation(tvc_deg, throttle)`` must be called once per step, before ``update``, with the command the env applied
over the coming interval (the previous action, from the step after the launch step on); ``needs_actuation`` says so.
"""

from __future__ import annotations

import numpy as np

from ....envs.balloon_world import get_initial_attitude
from ..interfaces import LaunchDecision, RocketEstimate, RocketNav
from ..registry import register
from .rate_model import GivenActuator, GmBank, RateModel, local_gravity

# Memory of the disturbance likelihood (s). The unmodelled angular acceleration is not stationary over the burn, and
# about 10 s of data pins ln q to +-0.5 (Fisher estimate for the given gyro noise).
MEMORY_S = 10.0
NX = 19  # error states: dp, dv, dtheta, db_g, db_a (15), domega_xy 15:17, dd_xy 17:19


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


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotvec_to_quat(phi):
    angle = float(np.linalg.norm(phi))
    if angle < 1e-12:
        return np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]])
    axis = phi / angle
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


@register("rocketnav", "mekf")
class MekfNav(RocketNav):
    needs_actuation = True

    def __init__(self, given_parameters, model=None, **_kwargs):
        gp = given_parameters
        s = gp["rocket"]["sensors"]
        if not s["gyro_noise_density"] > 0:
            raise ValueError(
                "mekf needs gyro noise: the disturbance grid spans [RW_g^2, sigma_g^2 / dt]"
            )
        rt = float(s["sampling_rate"]) ** 0.5
        self.dt = float(gp["simulation"]["time_step"])
        self.sig_g = float(s["gyro_noise_density"] * rt)
        self.sig_a = float(s["accelerometer_noise_density"] * rt)
        self.r_pos = float(s["gnss_position_accuracy"]) ** 2
        self.r_alt = float(s["gnss_altitude_accuracy"]) ** 2
        self.r_vel = float(s["gnss_velocity_accuracy"]) ** 2
        self.q_bg = float(s["gyro_random_walk_density"]) ** 2
        self.q_ba = float(s["accelerometer_random_walk_density"]) ** 2
        self.rw_g = float(np.sqrt(self.q_bg))
        self.p0 = np.array([0.0, 0.0, float(gp["environment"]["elevation"])])
        # gravity of the altitude above sea level, which is p_z (the frame starts at the pad elevation)
        self.gravity = local_gravity(gp)
        self.rm = RateModel(gp)
        self._ctrl = gp["rocket"]["control"]
        self._started = False
        self._last_t = None

    def on_launch(self, decision: LaunchDecision, t_launch: float) -> None:
        self.p = self.p0.copy()
        self.v = np.zeros(3)
        self.q = np.asarray(
            get_initial_attitude(decision.inclination_deg, decision.heading_deg), float
        )
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.w = np.zeros(2)
        self.dd = np.zeros(2)
        self.t_launch = float(t_launch)
        self._started = True
        self._last_t = None
        self._first = True
        self._burning_prev = True
        c = self._ctrl
        gmax = c["max_gimbal_angle"]
        self._act_g = GivenActuator(
            c.get("gimbal_time_constant"),
            c["gimbal_rate_limit"],
            -gmax,
            gmax,
            0.0,
            self.dt,
            2,
        )
        lo, hi = c["throttle_range"]
        self._act_t = GivenActuator(
            c.get("throttle_time_constant"),
            c["throttle_rate_limit"],
            lo,
            hi,
            1.0,
            self.dt,
            1,
        )
        self._g = np.zeros(2)
        self._thr = 1.0
        self.bank = GmBank(self.dt, self.sig_g, self.rw_g, self.rm.burn, MEMORY_S)
        q_hat, tau_hat, _ = self.bank.estimate()
        self._gm = (q_hat, tau_hat)

    def set_actuation(self, tvc_deg, throttle):
        """Advance the given actuator models with the command the env applied over the coming interval."""
        self._g = self._act_g.apply(np.ravel(np.asarray(tvc_deg, float))[:2])
        self._thr = float(
            self._act_t.apply(np.ravel(np.asarray(throttle, float))[:1])[0]
        )

    # -- filter steps -------------------------------------------------------
    def _init_known(self, sens, dt):
        """First valid sample: state at this sample, no propagation or update."""
        vel = sens[9:12]
        self.v = vel.copy() if np.isfinite(vel).all() else np.zeros(3)
        P = np.zeros((NX, NX))
        P[0:3, 0:3] = np.eye(3) * self.r_vel * dt * dt
        P[3:6, 3:6] = np.eye(3) * self.r_vel
        P[9:12, 9:12] = np.eye(3) * self.q_bg * dt
        P[12:15, 12:15] = np.eye(3) * self.q_ba * dt
        q_hat, tau_hat = self._gm
        P[17:19, 17:19] = np.eye(2) * q_hat / (2 * tau_hat)
        self.P = P

    def _propagate(self, gyro, accel, dt, tm0):
        rot = _quat_to_matrix(self.q)
        f = accel - self.ba
        extra_qv = 0.0
        i, burning = self.rm.index(tm0)
        z = float(self.p[2])
        # accelerometer plausibility check
        a_norm = float(np.linalg.norm(accel))
        fmax = self.rm.f_max(
            float(np.linalg.norm(self.v)), i, z, float(np.linalg.norm(self.w))
        )
        if a_norm > fmax:
            f = self.rm.specific_force(
                self.v, rot, self.w, self._g, self._thr, i, burning, z
            )
            extra_qv = (fmax + float(np.linalg.norm(f))) ** 2 / 5.0 * dt * dt
        # pitch/yaw rate from the moment model; the disturbance likelihood restarts at motor cutoff
        if self._burning_prev and not burning:
            self.bank.reset_weights()
        self._burning_prev = burning
        q_hat, tau_hat = self._gm
        phi = np.exp(-dt / tau_hat)
        s2 = q_hat / (2 * tau_hat)
        u = self.rm.ang_accel(self.v, rot, self.w, self._g, self._thr, i, burning, z)
        w_new = self.w + dt * (u + self.dd)
        wz = gyro[2] - self.bg[2]
        w = 0.5 * (
            np.array([self.w[0], self.w[1], wz]) + np.array([w_new[0], w_new[1], wz])
        )
        self._u = u
        self.w = w_new
        self.dd = phi * self.dd

        g_vec = np.array([0.0, 0.0, -self.gravity(float(self.p[2]))])
        acc_w = rot @ f + g_vec
        self.p = self.p + self.v * dt + 0.5 * acc_w * dt * dt
        self.v = self.v + acc_w * dt
        self.q = _quat_mult(self.q, _rotvec_to_quat(w * dt))
        self.q /= np.linalg.norm(self.q)

        F = np.eye(NX)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -rot @ _skew(f) * dt
        F[3:6, 12:15] = -rot * dt
        F[0:3, 6:9] = -0.5 * rot @ _skew(f) * dt * dt
        F[0:3, 12:15] = -0.5 * rot * dt * dt
        F[6:9, 6:9] = np.eye(3) - _skew(w) * dt
        F[6:8, 15:17] = np.eye(2) * dt
        F[8, 11] = -dt
        F[15:17, 17:19] = np.eye(2) * dt
        F[17:19, 17:19] = np.eye(2) * phi
        Q = np.zeros((NX, NX))
        Q[3:6, 3:6] = np.eye(3) * ((self.sig_a * dt) ** 2 + extra_qv)
        Q[9:12, 9:12] = np.eye(3) * self.q_bg * dt
        Q[12:15, 12:15] = np.eye(3) * self.q_ba * dt
        Q[8, 8] = (self.sig_g * dt) ** 2
        Q[17:19, 17:19] = np.eye(2) * s2 * (1 - phi**2)
        self.P = F @ self.P @ F.T + Q

    def _update(self, sens, gnss_ok):
        """Gyro x/y and, when valid, GNSS position and velocity in one Joseph-form update."""
        H = np.zeros((2, NX))
        H[:, 15:17] = np.eye(2)
        H[:, 9:11] = np.eye(2)
        rows_h = [H]
        rows_y = [sens[0:2] - (self.w + self.bg[:2])]
        rows_r = [self.sig_g**2] * 2
        if gnss_ok:
            H = np.zeros((6, NX))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)
            rows_h.append(H)
            rows_y.append(np.concatenate([sens[6:9] - self.p, sens[9:12] - self.v]))
            rows_r += [self.r_pos, self.r_pos, self.r_alt] + [self.r_vel] * 3
        H = np.vstack(rows_h)
        y = np.concatenate(rows_y)
        R = np.diag(rows_r)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(len(y)))
        dx = K @ y
        IKH = np.eye(NX) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        dtheta = dx[6:9]
        self.q = _quat_mult(self.q, _rotvec_to_quat(dtheta))
        self.q /= np.linalg.norm(self.q)
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]
        self.w = self.w + dx[15:17]
        self.dd = self.dd + dx[17:19]
        G = np.eye(NX)
        G[6:9, 6:9] = np.eye(3) - 0.5 * _skew(dtheta)  # covariance reset
        self.P = G @ self.P @ G.T

    def _estimate(self, t, gyro):
        return RocketEstimate(
            t=t,
            position=self.p.copy(),
            velocity=self.v.copy(),
            quaternion=self.q.copy(),
            omega=np.array([self.w[0], self.w[1], gyro[2] - self.bg[2]]),
            quality={
                "attitude": float(np.sqrt(np.trace(self.P[6:9, 6:9]))),
                "position": float(np.sqrt(np.trace(self.P[0:3, 0:3]))),
                "velocity": float(np.sqrt(np.trace(self.P[3:6, 3:6]))),
            },
            source="mekf",
        )

    # -- interface -----------------------------------------------------
    def update(self, observation: dict, *, deadline: float) -> RocketEstimate:
        t = float(observation["simulation_time"])
        sens = np.asarray(observation["rocket_sensors"], float)
        if not self._started or not np.isfinite(sens[0:6]).all():
            return RocketEstimate(
                t=t,
                position=np.zeros(3),
                velocity=np.zeros(3),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                omega=np.zeros(3),
                quality={"attitude": np.inf, "position": np.inf, "velocity": np.inf},
                source="mekf",
            )
        dt = self.dt if self._last_t is None else max(t - self._last_t, 1e-4)
        self._last_t = t
        gyro, accel = sens[0:3], sens[3:6]
        if self._first:
            self._first = False
            self._init_known(sens, dt)
            return self._estimate(t, gyro)
        # motor clock: the env starts the burn one step after the launch step, so the interval ending at t starts at
        # motor time t - t_launch - 2 dt
        self._propagate(gyro, accel, dt, t - self.t_launch - 2 * dt)
        self.bank.step(self._u, gyro[0:2])
        q_hat, tau_hat, _ = self.bank.estimate()
        self._gm = (q_hat, tau_hat)
        self._update(sens, bool(np.isfinite(sens[6:12]).all()))
        return self._estimate(t, gyro)
