"""Batched numpy surrogate of rocket + closed inner loop, for RL training.

Translational: thrust along the nose (throttle * T(t) / m), axial drag,
aero normal force q*S*Cn_alpha(M)*alpha/m (alpha clipped at 18 deg, same as
GivenLimits.lateral_disturbance), gravity. All tables from given_parameters
(export_vehicle.py).

Rotational: the closed-loop response of CompensatedInnerLoop + TVC is modelled
as a cascade with fitted constants (fit_sysid.py against real-env flights):
    w_des = clip(k_att * theta, w_max) about axis (nose x n_cmd)
    dw    = clip(w_des - w, alpha_scale * T*sin(gmax)*arm/I * dt)  + aero moment
where the aero moment term c_aero * q * sin(aoa) rotates the nose toward the
velocity (weathercock, c_aero > 0) or away (< 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


@dataclass
class Params:
    k_att: float = 2.5498   # fit_sysid.py, 20 real flights (2026-09-23)
    w_max: float = 3.6386
    alpha_scale: float = 0.8751
    c_aero: float = 0.0031
    w_damp: float = 0.2599       # extra rate damping (1/s)
    # fitted aero (fit_aero.py): axial = qS/m*(ax0 + ax2 sin^2 a) along -v,
    # normal = qS/m*(n1 sin a cos a + n2 sin a |sin a|) along the nose's lateral dir.
    ax0: float = 0.751
    ax2: float = 2.319
    n1: float = 5.467
    n2: float = -1.038
    thrust_k: float = 1.0        # thrust multiplier (domain randomization)


class Vehicle:
    def __init__(self, scenario=1):
        d = np.load(HERE / f"vehicle_sc{scenario}.npz")
        self.d = {k: d[k] for k in d.files}
        self.t_tab = self.d["t"]
        self.dt_tab = float(self.t_tab[1] - self.t_tab[0])
        self.elevation = float(self.d["elevation"])
        self.burn_time = float(self.d["burn_time"])
        self.g = float(self.d["g"])
        self.S = float(self.d["ref_area"])
        self.alpha_k = float(np.sin(self.d["gimbal_max_rad"]) * self.d["arm"] / self.d["inertia"])
        self.radius = float(self.d["radius"])
        wf = HERE / f"wind_sc{scenario}.npz"
        if wf.exists():
            w = np.load(wf); self.wz, self.wu, self.wv = w["z"], w["u"], w["v"]
        else:
            self.wz, self.wu, self.wv = np.array([0.0, 1.0]), np.zeros(2), np.zeros(2)

    def wind(self, z):
        return np.stack([np.interp(z, self.wz, self.wu), np.interp(z, self.wz, self.wv), np.zeros_like(z)], -1)

    def _tab(self, name, t):
        i = np.clip((t / self.dt_tab).astype(int), 0, len(self.t_tab) - 1)
        return self.d[name][i]

    def thrust(self, t):
        return np.where((t >= 0) & (t <= self.burn_time), self._tab("thrust", np.maximum(t, 0)), 0.0)

    def mass(self, t):
        return self._tab("mass", np.maximum(t, 0))

    def cd(self, t):
        return self._tab("cd", np.maximum(t, 0))

    def rho_a(self, z):
        zz = self.d["z"]
        i = np.clip(((z - zz[0]) / (zz[1] - zz[0])).astype(int), 0, len(zz) - 1)
        return self.d["rho"][i], self.d["a_sound"][i]

    def cn_alpha(self, mach):
        return np.interp(mach, self.d["mach"], self.d["cn_alpha"])


def _unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-9)


def step(veh: Vehicle, prm: Params, pos, vel, nose, w, t_rel, n_cmd, thr, dt):
    """One integration step for N rockets (all arrays (N,3)/(N,)). Returns new state
    and the acceleration used."""
    T = veh.thrust(t_rel) * thr
    m = veh.mass(t_rel)
    rho, a_s = veh.rho_a(pos[:, 2])
    vair = vel - veh.wind(pos[:, 2])
    speed = np.linalg.norm(vair, axis=1)
    vh = vair / np.maximum(speed, 1e-9)[:, None]
    q = 0.5 * rho * speed ** 2
    cosa = np.clip((nose * vh).sum(1), -1.0, 1.0)
    aoa = np.arccos(cosa)
    lat = nose - cosa[:, None] * vh
    lat_u = _unit(lat)
    sa = np.sin(aoa)
    qs = q * veh.S / m
    a_norm = qs * (prm.n1 * sa * cosa + prm.n2 * sa * np.abs(sa))
    a_drag = qs * (prm.ax0 + prm.ax2 * sa * sa)
    acc = (T / m)[:, None] * nose + a_norm[:, None] * lat_u - a_drag[:, None] * vh
    acc[:, 2] -= veh.g

    # attitude: cascade toward n_cmd
    axis = np.cross(nose, n_cmd)
    s = np.linalg.norm(axis, axis=1)
    theta = np.arctan2(s, np.clip((nose * n_cmd).sum(1), -1, 1))
    axis_u = axis / np.maximum(s, 1e-9)[:, None]
    w_des = np.clip(prm.k_att * theta, 0.0, prm.w_max)[:, None] * axis_u
    dw_max = prm.alpha_scale * veh.alpha_k * T * dt
    dw = w_des - w
    nrm = np.linalg.norm(dw, axis=1)
    dw = dw * np.minimum(1.0, dw_max / np.maximum(nrm, 1e-9))[:, None]
    # weathercock: rotate nose toward velocity
    wc_axis = np.cross(nose, vh)
    w_new = w + dw + (prm.c_aero * q * dt)[:, None] * wc_axis - prm.w_damp * w * dt
    # keep w perpendicular to nose (no roll channel)
    w_new -= (w_new * nose).sum(1)[:, None] * nose

    vel_new = vel + acc * dt
    pos_new = pos + 0.5 * (vel + vel_new) * dt
    ang = np.linalg.norm(w_new, axis=1) * dt
    ax = w_new / np.maximum(np.linalg.norm(w_new, axis=1), 1e-9)[:, None]
    # Rodrigues rotation of nose about ax by ang
    c, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
    nose_new = nose * c + np.cross(ax, nose) * sn + ax * (ax * nose).sum(1)[:, None] * (1 - c)
    nose_new = _unit(nose_new)
    return pos_new, vel_new, nose_new, w_new, acc


# ---------------------------------------------------------------------------
# Rigid-body attitude + vectorized CompensatedInnerLoop (higher fidelity).
# Rotational fit (fit_rot.py, 24 real flights, R2 0.97):
#   dw_x = cx*T sin(g0) arm/I - ax*qS vby/|v| - dqx*q*wx - dx*wx
#   dw_y = cy*T sin(g1) arm/I + ay*qS vbx/|v| - dqy*q*wy - dy*wy
@dataclass
class RigidParams:
    cx: float = 0.923
    cy: float = 0.9205
    ax: float = 0.0217
    ay: float = 0.0218
    dqx: float = 3.0e-4
    dqy: float = 2.0e-4
    dx: float = 0.0589
    dy: float = 0.1123
    # CompensatedInnerLoop defaults (agent side, not fitted)
    k_att: float = 2.0
    w_max: float = 2.0
    kp_rate: float = 25.0
    gimbal_max_deg: float = 15.0
    gimbal_step_deg: float = 0.6
    throttle_step: float = 0.02


def quat_rot(q):
    e0, e1, e2, e3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (e2**2 + e3**2), 2 * (e1*e2 - e0*e3), 2 * (e1*e3 + e0*e2)], -1),
        np.stack([2 * (e1*e2 + e0*e3), 1 - 2 * (e1**2 + e3**2), 2 * (e2*e3 - e0*e1)], -1),
        np.stack([2 * (e1*e3 - e0*e2), 2 * (e2*e3 + e0*e1), 1 - 2 * (e1**2 + e2**2)], -1)], 1)


def quat_from_nose(nose):
    """Minimal-rotation quaternion taking body +Z to ``nose`` (zero roll-ish)."""
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(np.broadcast_to(z, nose.shape), nose)
    s = np.linalg.norm(ax, axis=1); c = nose[:, 2]
    ang = np.arctan2(s, c)
    axu = ax / np.maximum(s, 1e-12)[:, None]
    q = np.concatenate([np.cos(ang / 2)[:, None], axu * np.sin(ang / 2)[:, None]], 1)
    q[s < 1e-12] = np.array([1.0, 0, 0, 0])
    return q


def step_rigid(veh: Vehicle, prm: Params, rp: RigidParams, pos, vel, q, w, g, thr, t_rel, n_cmd, thr_cmd, dt=0.01):
    """One 100 Hz step. q (N,4) body->world, w (N,2) body rates x,y, g (N,2) gimbal deg,
    thr (N,) actual throttle. Returns pos, vel, q, w, g, thr, nose."""
    Rm = quat_rot(q)
    nose = Rm[:, :, 2]
    # --- inner loop (CompensatedInnerLoop.solve, force direction = n_cmd)
    speed_g = np.linalg.norm(vel, axis=1)
    burn = veh.thrust(t_rel) > 1e-9
    tgt = np.where(speed_g < 12.0, 1.0, thr_cmd)
    thr = np.where(burn, np.clip(tgt, thr - rp.throttle_step, thr + rp.throttle_step),
                   np.maximum(0.0, thr - rp.throttle_step))
    d = np.einsum("nji,nj->ni", Rm, n_cmd)
    ang = np.arccos(np.clip(d[:, 2], -1, 1)); h = np.hypot(d[:, 0], d[:, 1])
    err = np.where(h[:, None] < 1e-12, 0.0, np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(h, 1e-12)[:, None] * ang[:, None])
    w_cmd = np.clip(rp.k_att * err, -rp.w_max, rp.w_max)
    g_des = np.clip(rp.kp_rate * (w_cmd - w), -rp.gimbal_max_deg, rp.gimbal_max_deg)
    g_new = np.clip(g_des, g - rp.gimbal_step_deg, g + rp.gimbal_step_deg)
    g = np.where(burn[:, None], g_new, 0.0)
    # --- forces (translational model of step())
    T = veh.thrust(t_rel) * thr * prm.thrust_k
    m = veh.mass(t_rel)
    rho, _ = veh.rho_a(pos[:, 2])
    vair = vel - veh.wind(pos[:, 2])
    speed = np.linalg.norm(vair, axis=1)
    vh = vair / np.maximum(speed, 1e-9)[:, None]
    qd = 0.5 * rho * speed ** 2
    cosa = np.clip((nose * vh).sum(1), -1.0, 1.0)
    sa = np.sqrt(np.maximum(0.0, 1 - cosa ** 2))
    lat_u = nose - cosa[:, None] * vh
    lat_u /= np.maximum(np.linalg.norm(lat_u, axis=1), 1e-9)[:, None]
    qs = qd * veh.S / m
    acc = (T / m)[:, None] * nose + (qs * (prm.n1 * sa * cosa + prm.n2 * sa * sa))[:, None] * lat_u \
        - (qs * (prm.ax0 + prm.ax2 * sa * sa))[:, None] * vh
    acc[:, 2] -= veh.g
    # --- rotation
    I = float(veh.d["inertia"]); arm = float(veh.d["arm"])
    vb = np.einsum("nji,nj->ni", Rm, vair)
    qsv = qd * veh.S / np.maximum(speed, 1e-3)
    gr = np.radians(g)
    dwx = rp.cx * T * np.sin(gr[:, 0]) * arm / I - rp.ax * qsv * vb[:, 1] - rp.dqx * qd * w[:, 0] - rp.dx * w[:, 0]
    dwy = rp.cy * T * np.sin(gr[:, 1]) * arm / I + rp.ay * qsv * vb[:, 0] - rp.dqy * qd * w[:, 1] - rp.dy * w[:, 1]
    w = w + np.stack([dwx, dwy], 1) * dt
    # --- integrate
    vel_new = vel + acc * dt
    pos = pos + 0.5 * (vel + vel_new) * dt
    om = np.concatenate([w, np.zeros((w.shape[0], 1))], 1)
    an = np.linalg.norm(om, axis=1) * dt
    axu = om / np.maximum(np.linalg.norm(om, axis=1), 1e-12)[:, None]
    dq = np.concatenate([np.cos(an / 2)[:, None], axu * np.sin(an / 2)[:, None]], 1)
    e0, e1, e2, e3 = q.T; f0, f1, f2, f3 = dq.T
    q = np.stack([e0*f0 - e1*f1 - e2*f2 - e3*f3, e0*f1 + e1*f0 + e2*f3 - e3*f2,
                  e0*f2 - e1*f3 + e2*f0 + e3*f1, e0*f3 + e1*f2 - e2*f1 + e3*f0], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return pos, vel_new, q, w, g, thr, quat_rot(q)[:, :, 2]
