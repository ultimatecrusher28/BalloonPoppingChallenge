"""Model-aided body-rate estimation pieces for MekfNav.

Physics from given parameters only, the same as ``gnc/rl/given_sim.py`` ``GivenSim.step`` (tables are taken from a
``GivenSim`` built with a table step equal to the simulation step):
- angular acceleration of pitch/yaw: TVC sin(g) T thr lever, per-surface Barrowman lift with omega x r damping and the
  pi/10 attack-angle cap, lateral inertia I(t), ISA atmosphere, wind 0;
- specific force (body frame): thrust with the TVC direction of the env (x = -T sin g_y, y = T sin g_x), body-axis drag,
  surface lift, over the mass m(t);
- f_max: the largest specific force the given vehicle can produce (thrust limit, aerodynamic force at the attack-angle
  cap and the current speed, and the rotational term at the accelerometer, which sits at the centre of dry mass).
Actuators: the env's first-order lag, then rate limit, then range clamp (ActiveRocketPy ``Actuator``), driven by the
commands the agent sent.
Disturbance parameters: a bank of per-axis 2-state filters (s = rate + gyro bias, d = unmodelled angular acceleration,
first-order Gauss-Markov) on a log grid of (q, tau), q = 2 sigma^2 tau the low-frequency power. The grid is
[RW_g^2, sigma_g^2 / dt] x [dt, burn_time], spaced at most x2, log-uniform prior; the log-likelihood fades with a
finite memory (the unmodelled angular acceleration is not stationary over the burn).
Motor clock: the env starts the burn one step after the launch step, so motor time = t - t_launch - dt and the burn
window is (0, burn_time]; the propagation interval ending at observation time t starts at motor time t - t_launch - 2 dt.
"""

from __future__ import annotations

import math

import numpy as np


def local_gravity(given_parameters):
    """Gravity magnitude (m/s^2) as a function of altitude above sea level at the launch site: the given value if the
    given parameters carry one (environment.gravity), else the WGS84 normal gravity (Somigliana formula with the
    second-order height correction, as RocketPy's Environment uses when no gravity is set) at the given latitude."""
    env = given_parameters["environment"]
    if env.get("gravity") is not None:
        g = float(env["gravity"])
        return lambda h: g
    a, f, m_rot = 6378137.0, 1 / 298.257223563, 3.449786506841e-3  # WGS84
    g_e, k, e2 = 9.7803253359, 1.931852652458e-3, 6.694379990141e-3
    s2 = math.sin(math.radians(float(env["latitude"]))) ** 2
    g0 = g_e * (1 + k * s2) / math.sqrt(1 - e2 * s2)
    c1 = 2.0 / a * (1 + f + m_rot - 2 * f * s2)
    return lambda h: g0 * (1 - c1 * h + 3 * h * h / (a * a))


class GivenActuator:
    """One actuator channel set per env step: first-order lag (alpha = 1 / (1 + tau * rate)), rate limit, clamp."""

    def __init__(self, tau, rate_limit, lo, hi, init, dt, n):
        rate = 1.0 / dt
        self.alpha = 1.0 / (1.0 + float(tau) * rate) if tau else 1.0
        self.step = float(rate_limit) / rate if rate_limit is not None else np.inf
        self.lo, self.hi = float(lo), float(hi)
        self.y = np.full(n, float(init))

    def apply(self, u):
        u = np.asarray(u, float).reshape(self.y.shape)
        v = self.alpha * u + (1.0 - self.alpha) * self.y
        ch = v - self.y
        v = np.where(np.abs(ch) > self.step, self.y + np.sign(ch) * self.step, v)
        self.y = np.clip(v, self.lo, self.hi)
        return self.y.copy()


class RateModel:
    """Given-parameter tables and single-state models (motor-clock table index ``i``, altitude ``z`` above sea level)."""

    def __init__(self, given_parameters):
        from ..rl.given_sim import GivenSim

        gp = given_parameters
        self.dt = float(gp["simulation"]["time_step"])
        s = GivenSim(gp, dt_tab=self.dt)
        self.burn = s.burn_time
        self.n_tab = len(s.t_tab)
        self.mass, self.thrust, self.rcm, self.icm = (
            s.mass_t,
            s.thrust_t,
            s.rcm_t,
            s.icm_t,
        )
        self.cp = np.asarray(s.cp, float)
        self.cla = [np.asarray(c, float) for c in s.cla]
        self.mach, self.cd_on, self.cd_off = s.mach, s.cd_on, s.cd_off
        self.S, self.lever = float(s.S), float(s.lever)
        self.z0, self.rho_tab, self.a_tab = float(s.z_tab[0]), s.rho_tab, s.a_tab
        ctrl = gp["rocket"]["control"]
        self.gmax = float(ctrl["max_gimbal_angle"])
        self.thr_max = float(ctrl["throttle_range"][1])
        self.T_max = float(np.max(self.thrust))
        self.cap = math.pi / 10.0

    def index(self, tm0):
        """Table index and thrust-on flag for a propagation interval starting at motor time tm0."""
        i = int(round(max(tm0, 0.0) / self.dt))
        return min(i, self.n_tab - 1), (tm0 > -1e-9) and (tm0 < self.burn - 1e-9)

    def atm(self, z):
        zi = min(max(int((z - self.z0) / 10.0), 0), len(self.rho_tab) - 1)
        return float(self.rho_tab[zi]), float(self.a_tab[zi])

    def _surfaces(self, fb, w_xy, rcm, rho, a_s):
        """Lift components (lx, ly) and moments (M1, M2) of all surfaces, as GivenSim.step (one row)."""
        fb0, fb1, fb2 = fb
        w0, w1 = w_xy
        lx = ly = M1 = M2 = 0.0
        for k in range(len(self.cp)):
            cpz = self.cp[k] - rcm
            sv0 = fb0 - w1 * cpz
            sv1 = fb1 + w0 * cpz
            ss = math.sqrt(sv0 * sv0 + sv1 * sv1 + fb2 * fb2)
            lat = math.hypot(sv0, sv1)
            c = min(max(-fb2 / max(ss, 1e-9), -1.0), 1.0)
            aoa = min(math.acos(c), self.cap)
            lift = (
                0.5
                * rho
                * ss
                * ss
                * self.S
                * float(np.interp(ss / a_s, self.mach, self.cla[k]))
                * aoa
            )
            inv = lift / lat if lat > 1e-9 else 0.0
            lx += inv * sv0
            ly += inv * sv1
            M1 += -cpz * inv * sv1
            M2 += cpz * inv * sv0
        return lx, ly, M1, M2

    def ang_accel(self, vel, rot, w_xy, g_deg, thr, i, burning, z):
        """Pitch/yaw angular acceleration (rad/s^2) from the given model, wind 0."""
        rho, a_s = self.atm(z)
        T = float(self.thrust[i]) * thr if burning else 0.0
        fb = rot.T @ (-np.asarray(vel, float))
        _, _, M1, M2 = self._surfaces(fb, w_xy, float(self.rcm[i]), rho, a_s)
        M1 += math.sin(math.radians(g_deg[0])) * T * self.lever
        M2 += math.sin(math.radians(g_deg[1])) * T * self.lever
        i_lat = float(self.icm[i])
        return np.array([M1 / i_lat, M2 / i_lat])

    def specific_force(self, vel, rot, w_xy, g_deg, thr, i, burning, z):
        """Model specific force in the body frame (m/s^2)."""
        rho, a_s = self.atm(z)
        T = float(self.thrust[i]) * thr if burning else 0.0
        fb = rot.T @ (-np.asarray(vel, float))
        speed = float(np.linalg.norm(fb))
        lx, ly, _, _ = self._surfaces(fb, w_xy, float(self.rcm[i]), rho, a_s)
        cd = float(
            np.interp(speed / a_s, self.mach, self.cd_on if burning else self.cd_off)
        )
        sgx, sgy = math.sin(math.radians(g_deg[0])), math.sin(math.radians(g_deg[1]))
        # env TVC force (ActiveRocketPy flight.py thrust1/2/3): body x -T sin g_y, body y T sin g_x, the rest axial
        F = np.array(
            [
                -T * sgy + lx,
                T * sgx + ly,
                T * math.sqrt(max(1.0 - sgx * sgx - sgy * sgy, 0.0))
                - 0.5 * rho * speed * speed * self.S * cd,
            ]
        )
        return F / float(self.mass[i])

    def f_max(self, speed, i, z, w_xy_norm):
        """Largest specific-force magnitude the given vehicle can produce at this speed (m/s^2)."""
        rho, a_s = self.atm(z)
        m, i_lat, rcm = float(self.mass[i]), float(self.icm[i]), abs(float(self.rcm[i]))
        qS = 0.5 * rho * speed * speed * self.S
        M = speed / a_s
        lift = [qS * float(np.interp(M, self.mach, c)) * self.cap for c in self.cla]
        drag = qS * max(
            float(np.interp(M, self.mach, self.cd_on)),
            float(np.interp(M, self.mach, self.cd_off)),
        )
        f_thr = self.T_max * self.thr_max / m
        f_aero = (sum(lift) + drag) / m
        wdot_max = (
            self.T_max * self.thr_max * self.lever * math.sin(math.radians(self.gmax))
            + sum(L * abs(cp - float(self.rcm[i])) for L, cp in zip(lift, self.cp))
        ) / i_lat
        return f_thr + f_aero + (wdot_max + w_xy_norm * w_xy_norm) * rcm


class GmBank:
    """Multiple-model likelihood over a (q, tau) grid of the Gauss-Markov disturbance, per axis x and y (pooled)."""

    def __init__(self, dt, sig_g, rw_g, burn_time, memory_s):
        q_lo, q_hi = rw_g**2, sig_g**2 / dt
        t_lo, t_hi = dt, float(burn_time)
        nq = int(math.ceil(math.log2(q_hi / q_lo))) + 1
        nt = int(math.ceil(math.log2(t_hi / t_lo))) + 1
        Q, TAU = np.meshgrid(
            np.geomspace(q_lo, q_hi, nq), np.geomspace(t_lo, t_hi, nt), indexing="ij"
        )
        q, tau = Q.ravel(), TAU.ravel()
        self.lq, self.lt = np.log(q), np.log(tau)
        self.s2 = q / (2.0 * tau)
        self.phi = np.exp(-dt / tau)
        self.qd = self.s2 * (1.0 - self.phi**2)
        self.dt, self.R, self.qs = dt, sig_g**2, rw_g**2 * dt
        self.decay = math.exp(-dt / memory_s)
        self.shape = (nq, nt)
        G = len(q)
        self.s = np.zeros((2, G))
        self.d = np.zeros((2, G))
        self.P11 = np.zeros((2, G))
        self.P12 = np.zeros((2, G))
        self.P22 = np.tile(self.s2, (2, 1))
        self.L = np.zeros(G)

    def step(self, u, y):
        """u: model angular acceleration (2,), y: gyro x/y (2,)."""
        dt = self.dt
        self.s = self.s + dt * (np.asarray(u, float)[:, None] + self.d)
        self.d = self.phi * self.d
        P11 = self.P11 + 2 * dt * self.P12 + dt * dt * self.P22 + self.qs
        P12 = self.phi * (self.P12 + dt * self.P22)
        P22 = self.phi**2 * self.P22 + self.qd
        S = P11 + self.R
        nu = np.asarray(y, float)[:, None] - self.s
        K1, K2 = P11 / S, P12 / S
        self.s = self.s + K1 * nu
        self.d = self.d + K2 * nu
        self.P22 = P22 - K2 * P12
        self.P12 = (1 - K1) * P12
        self.P11 = (1 - K1) * P11
        self.L = self.decay * self.L - 0.5 * np.sum(np.log(S) + nu * nu / S, axis=0)

    def estimate(self):
        """(q_hat, tau_hat) = exp of the posterior mean of the logs; also the posterior weights."""
        w = np.exp(self.L - self.L.max())
        w /= w.sum()
        return float(np.exp(w @ self.lq)), float(np.exp(w @ self.lt)), w

    def reset_weights(self):
        self.L[:] = 0.0
