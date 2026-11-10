"""Given-parameters-only numpy surrogate (SPEC T2; Discussions #150/#165).

Every physical quantity comes from a RocketPy ``Rocket`` rebuilt from
``given_parameters`` exactly as the official env builds it
(``balloon_world.__init_rocket_simulation``; all keys used are in the given
list): mass(t), thrust(t), CM offset(t), lateral inertia(t), nozzle lever,
per-surface CP and Barrowman Cn_alpha(M), drag Cd(M) power on/off.
No constant is fitted to simulator states.

Dynamics mirror ActiveRocketPy ``Flight.u_dot_generalized`` (flight.py):
  * drag  R3 = -q S Cd along the body axis (q from the free-stream speed);
  * lift per surface, body frame, from the surface's local stream velocity
    (v_body + w x r_cp, so aero damping is included), attack angle capped at
    pi/10, lift = q S Cn_alpha(M) * alpha along the stream's lateral part,
    moment (-cpz*Ly, cpz*Lx);
  * TVC moment M1 += sin(gx) T lever, M2 += sin(gy) T lever
    (lever = nozzle_to_cdm); lateral thrust force negligible (the official
    code adds sin(g) newtons, not T sin(g));
  * throttle scales thrust; engine off after burn_time.
Omitted (small, documented): jet damping, CM-shift inertial terms,
buoyancy, Coriolis. Wind: supplied per episode (estimated from balloon
drift by the caller) or zero.

The inner loop is the vectorised CompensatedInnerLoop of surrogate.step_rigid
(agent-side gains, not fitted) plus the given gimbal first-order lag
(gimbal_time_constant; null in sc1/sc2, 0.064 s in sc3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # training/ppo/ (it ran as methods/v7/offline/, parents[2])


def build_rocket(gp):
    """RocketPy Rocket from given_parameters, same construction as the env."""
    from rocketpy import (CylindricalTank, Fluid, HybridMotor,
                          MassFlowRateBasedTank, Rocket)
    R = gp["rocket"]; tk = R["tank"]; mo = R["motor"]; b = R["rocket_body"]
    tank = MassFlowRateBasedTank(
        name="oxidizer_tank", geometry=CylindricalTank(radius=tk["radius"], height=tk["height"]),
        flux_time=(0.0, tk["flux_time"]), initial_liquid_mass=tk["initial_liquid_mass"],
        initial_gas_mass=tk["initial_gas_mass"], liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=tk["liquid_mass_flow_rate_out"], gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0, liquid=Fluid(name=tk["liquid"], density=tk["liquid_density"]),
        gas=Fluid(name=tk["gas"], density=tk["gas_density"]))
    motor = HybridMotor(
        thrust_source=mo["thrust_source"], dry_mass=mo["dry_mass"], dry_inertia=tuple(mo["dry_inertia"]),
        center_of_dry_mass_position=mo["center_of_dry_mass_position"], burn_time=(0.0, mo["burn_time"]),
        reshape_thrust_curve=False, grain_number=mo["grain_number"], grain_separation=mo["grain_separation"],
        grain_outer_radius=mo["grain_outer_radius"], grain_initial_inner_radius=mo["grain_initial_inner_radius"],
        grain_initial_height=mo["grain_initial_height"], grain_density=mo["grain_density"],
        nozzle_radius=mo["nozzle_radius"], throat_radius=mo["throat_radius"], interpolation_method="linear",
        nozzle_position=mo["nozzle_position"], grains_center_of_mass_position=mo["grains_center_of_mass_position"],
        coordinate_system_orientation="nozzle_to_combustion_chamber")
    motor.add_tank(tank=tank, position=tk["tank_position"])
    r = Rocket(radius=b["radius"], mass=b["mass"], inertia=tuple(b["inertia"]),
               center_of_mass_without_motor=b["center_of_mass_without_motor"],
               power_off_drag=b["power_off_drag"], power_on_drag=b["power_on_drag"],
               coordinate_system_orientation="tail_to_nose", volume=b["volume"])
    r.add_motor(motor, position=mo["motor_position"])
    r.add_nose(length=R["nose"]["length"], kind=R["nose"]["kind"], position=R["nose"]["position"])
    f = R["fins"]
    if f["useFins"]:
        r.add_trapezoidal_fins(n=f["n"], span=f["span"], root_chord=f["root_chord"],
                               tip_chord=f["tip_chord"], position=f["position"])
    return r, motor


def export(scenario=1):
    """Write vehicle_given_sc{N}.npz (tables only; no RocketPy at train time)."""
    sys.path.insert(0, str(ROOT))
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    sys.path.insert(0, str(HERE))  # the training-time vehicle modules (training/ppo/agent_modules), not the finals agent
    from agent_modules.rocket_model import RocketModel, GRAVITY
    from agent_modules import given_limits as glm
    _, gp = load_scenario_parameters(scenario)
    r, motor = build_rocket(gp)
    rm = RocketModel(gp)
    burn = float(gp["rocket"]["motor"]["burn_time"])
    t = np.arange(0.0, burn + 40.0, 0.01)
    mass = np.array([float(r.total_mass(x)) for x in t])
    thrust = np.array([float(motor.thrust(x)) if x <= burn else 0.0 for x in t])
    rcm = np.array([float(r.com_to_cdm_function(x)) for x in t])
    i11 = np.array([float(r.get_inertia_tensor_at_time(x)[0][0]) for x in t])
    i_cm = i11 - mass * rcm ** 2
    mach = np.linspace(0.0, 0.6, 61)
    cps, cla = [], []
    for s, _pos in r.aerodynamic_surfaces:
        cps.append(float(r.surfaces_cp_to_cdm[s][2]))
        cla.append([float(s.clalpha(x)) for x in mach])
    z = np.arange(0.0, 2500.0, 10.0) + float(gp["environment"]["elevation"])
    ctrl = gp["rocket"]["control"]
    out = HERE / f"vehicle_given_sc{scenario}.npz"
    np.savez(out, t=t, thrust=thrust, mass=mass, rcm=rcm, i_cm=i_cm, mach=mach,
             cp=np.array(cps), cla=np.array(cla),
             cd_on=np.array([float(r.power_on_drag(x)) for x in mach]),
             cd_off=np.array([float(r.power_off_drag(x)) for x in mach]),
             z=z, rho=np.array([rm.density(x) for x in z]),
             a_sound=np.array([glm._isa_speed_of_sound(x) for x in z]),
             ref_area=float(r.area), lever=float(r.nozzle_to_cdm), burn_time=burn,
             elevation=float(gp["environment"]["elevation"]), g=GRAVITY,
             gimbal_max_deg=float(ctrl["max_gimbal_angle"]), gimbal_rate=float(ctrl["gimbal_rate_limit"]),
             gimbal_tau=float(ctrl["gimbal_time_constant"] or 0.0),
             throttle_rate=float(ctrl["throttle_rate_limit"]),
             throttle_tau=float(ctrl["throttle_time_constant"] or 0.0),
             radius=float(gp["balloon"]["radius"]),
             # compatibility keys used by env_vec / features (surrogate.Vehicle API)
             inertia=float(i_cm[0]), arm=float(r.nozzle_to_cdm),
             gimbal_max_rad=float(np.radians(ctrl["max_gimbal_angle"])),
             cn_alpha=np.array(cla).sum(0), cd=np.where(t <= burn, float(r.power_on_drag(0.05)),
                                                       float(r.power_off_drag(0.05))))
    print("wrote", out, "mass0", mass[0], "I0", i_cm[0], "cps", cps, "lever", r.nozzle_to_cdm)


class GivenVehicle:
    """Same public API as surrogate.Vehicle, backed by vehicle_given_sc{N}.npz."""

    def __init__(self, scenario=1):
        d = np.load(HERE / f"vehicle_given_sc{scenario}.npz")
        self.d = {k: d[k] for k in d.files}
        self.t_tab = self.d["t"]; self.dt_tab = float(self.t_tab[1] - self.t_tab[0])
        self.elevation = float(self.d["elevation"]); self.burn_time = float(self.d["burn_time"])
        self.g = float(self.d["g"]); self.S = float(self.d["ref_area"])
        self.radius = float(self.d["radius"]); self.lever = float(self.d["lever"])
        self.cp = self.d["cp"]; self.cla = self.d["cla"]; self.mach = self.d["mach"]
        self.alpha_k = float(np.sin(self.d["gimbal_max_rad"]) * self.lever / self.d["i_cm"][0])
        self.wz, self.wu, self.wv = np.array([0.0, 1.0]), np.zeros(2), np.zeros(2)

    def set_wind(self, z, u, v):
        self.wz, self.wu, self.wv = np.asarray(z, float), np.asarray(u, float), np.asarray(v, float)

    def wind(self, z):
        return np.stack([np.interp(z, self.wz, self.wu), np.interp(z, self.wz, self.wv), np.zeros_like(z)], -1)

    def _tab(self, name, t):
        i = np.clip((np.asarray(t) / self.dt_tab).astype(int), 0, len(self.t_tab) - 1)
        return self.d[name][i]

    def thrust(self, t):
        return np.where((t >= 0) & (t <= self.burn_time), self._tab("thrust", np.maximum(t, 0)), 0.0)

    def mass(self, t):
        return self._tab("mass", np.maximum(t, 0))

    def cd(self, t):
        return self._tab("cd", np.maximum(t, 0))

    def i_cm(self, t):
        return self._tab("i_cm", np.maximum(t, 0))

    def rcm(self, t):
        return self._tab("rcm", np.maximum(t, 0))

    def rho_a(self, z):
        zz = self.d["z"]
        i = np.clip(((z - zz[0]) / (zz[1] - zz[0])).astype(int), 0, len(zz) - 1)
        return self.d["rho"][i], self.d["a_sound"][i]

    def cn_alpha(self, mach):
        return np.interp(mach, self.mach, self.d["cn_alpha"])


def _quat_rot(q):
    e0, e1, e2, e3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (e2**2 + e3**2), 2 * (e1*e2 - e0*e3), 2 * (e1*e3 + e0*e2)], -1),
        np.stack([2 * (e1*e2 + e0*e3), 1 - 2 * (e1**2 + e3**2), 2 * (e2*e3 - e0*e1)], -1),
        np.stack([2 * (e1*e3 - e0*e2), 2 * (e2*e3 + e0*e1), 1 - 2 * (e1**2 + e2**2)], -1)], 1)


def step_given(veh: GivenVehicle, prm, rp, pos, vel, q, w, g, thr, t_rel, n_cmd, thr_cmd, dt=0.01,
               g_act=None):
    """Drop-in for surrogate.step_rigid (same signature/returns). ``prm`` is used
    only for ``thrust_k`` (domain randomisation); ``rp`` for the agent-side
    inner-loop gains. ``g`` is the actual gimbal (deg)."""
    Rm = _quat_rot(q)
    nose = Rm[:, :, 2]
    speed_g = np.linalg.norm(vel, axis=1)
    burn = veh.thrust(t_rel) > 1e-9
    # --- throttle actuator (rate limit, given)
    thr_step = float(veh.d["throttle_rate"]) * dt
    tgt = np.where(speed_g < 12.0, 1.0, thr_cmd)
    thr = np.where(burn, np.clip(tgt, thr - thr_step, thr + thr_step), np.maximum(0.0, thr - thr_step))
    # --- CompensatedInnerLoop (agent side): attitude error -> rate cmd -> gimbal cmd
    d = np.einsum("nji,nj->ni", Rm, n_cmd)
    ang = np.arccos(np.clip(d[:, 2], -1, 1)); h = np.hypot(d[:, 0], d[:, 1])
    err = np.where(h[:, None] < 1e-12, 0.0,
                   np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(h, 1e-12)[:, None] * ang[:, None])
    w_cmd = np.clip(rp.k_att * err, -rp.w_max, rp.w_max)
    gmax = float(veh.d["gimbal_max_deg"])
    g_des = np.clip(rp.kp_rate * (w_cmd - w), -gmax, gmax)
    # agent rate-limits its own command; env actuator: rate limit then first-order lag
    g_step = float(veh.d["gimbal_rate"]) * dt
    g_cmd = np.clip(g_des, g - g_step, g + g_step)
    tau = float(veh.d["gimbal_tau"])
    g_new = g_cmd if tau <= 0 else g + (g_cmd - g) * (dt / (tau + dt))
    g = np.where(burn[:, None], g_new, 0.0)
    # --- forces in body frame
    T = veh.thrust(t_rel) * thr * getattr(prm, "thrust_k", 1.0)
    m = veh.mass(t_rel); I = veh.i_cm(t_rel); rcm = veh.rcm(t_rel)
    rho, a_s = veh.rho_a(pos[:, 2])
    wind = veh.wind(pos[:, 2])
    fs = wind - vel                                   # free-stream velocity (world)
    fs_speed = np.linalg.norm(fs, axis=1)
    mach = fs_speed / a_s
    qd = 0.5 * rho * fs_speed ** 2
    S = veh.S
    cd = np.where(burn, np.interp(mach, veh.mach, veh.d["cd_on"]), np.interp(mach, veh.mach, veh.d["cd_off"]))
    fs_b = np.einsum("nji,nj->ni", Rm, fs)            # body-frame free stream
    F = np.zeros_like(vel); M1 = np.zeros(len(vel)); M2 = np.zeros(len(vel))
    F[:, 2] = -qd * S * cd + T                         # drag + thrust (axial)
    w3 = np.concatenate([w, np.zeros((len(w), 1))], 1)
    for k in range(len(veh.cp)):
        cpz = veh.cp[k] - rcm                           # CP relative to CM, body z
        r_cp = np.zeros_like(vel); r_cp[:, 2] = cpz
        sv = fs_b - np.cross(w3, r_cp)                  # comp stream = wind_b - (v_b + w x r)
        ss = np.linalg.norm(sv, axis=1)
        lat = np.hypot(sv[:, 0], sv[:, 1])
        aoa = np.minimum(np.arccos(np.clip(-sv[:, 2] / np.maximum(ss, 1e-9), -1, 1)), np.pi / 10)
        cl = np.interp(ss / a_s, veh.mach, veh.cla[k]) * aoa
        lift = 0.5 * rho * ss ** 2 * S * cl
        lx = np.where(lat > 1e-9, lift * sv[:, 0] / np.maximum(lat, 1e-9), 0.0)
        ly = np.where(lat > 1e-9, lift * sv[:, 1] / np.maximum(lat, 1e-9), 0.0)
        F[:, 0] += lx; F[:, 1] += ly
        M1 += -cpz * ly; M2 += cpz * lx
    gr = np.radians(g)
    M1 += np.sin(gr[:, 0]) * T * veh.lever
    M2 += np.sin(gr[:, 1]) * T * veh.lever
    acc = np.einsum("nij,nj->ni", Rm, F) / m[:, None]
    acc[:, 2] -= veh.g
    w = w + np.stack([M1, M2], 1) / I[:, None] * dt
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
    return pos, vel_new, q, w, g, thr, _quat_rot(q)[:, :, 2]


if __name__ == "__main__":
    export(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
