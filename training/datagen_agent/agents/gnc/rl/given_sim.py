"""Runtime, given-parameters-only vectorised rocket simulator for on-board planning.

Physics built only from ``given_parameters`` (Discussion #150/#165): a RocketPy
``Rocket`` is rebuilt from the given rocket sections exactly as the official env
builds it, and tabulated (mass, thrust, CM offset, lateral inertia, nozzle lever,
per-surface CP / Barrowman Cn_alpha(M), drag Cd(M)). Dynamics mirror
ActiveRocketPy ``Flight.u_dot_generalized``: body-axis drag, per-surface lift with
omega x r damping and the pi/10 attack-angle cap, TVC moment sin(g)*T*lever.
The inner loop is the agent's own CompensatedInnerLoop (gains copied, vectorised)
plus the given gimbal lag. Wind is estimated online from observed balloon drift.
Checked (not fitted) against official flights in research/rl/check_given.py:
0.26 m mean position error 2 s ahead (research/f1/RESULTS_F1.md, T2).
Same physics as research/rl/surrogate_given.py (training side).
"""

from __future__ import annotations

import numpy as np

from ..vehicle.rocket_model import RocketModel, GRAVITY
from ..vehicle import given_limits as glm
from ..follow.ff_ref import FFParams, ff_step, ff_step_off, act_inv_y, _norm, _clip, _interp, _einsum, _nz


def _build_rocket(gp):
    from rocketpy import (
        CylindricalTank,
        Fluid,
        HybridMotor,
        MassFlowRateBasedTank,
        Rocket,
    )

    R = gp["rocket"]
    tk = R["tank"]
    mo = R["motor"]
    b = R["rocket_body"]
    tank = MassFlowRateBasedTank(
        name="oxidizer_tank",
        geometry=CylindricalTank(radius=tk["radius"], height=tk["height"]),
        flux_time=(0.0, tk["flux_time"]),
        initial_liquid_mass=tk["initial_liquid_mass"],
        initial_gas_mass=tk["initial_gas_mass"],
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=tk["liquid_mass_flow_rate_out"],
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=Fluid(name=tk["liquid"], density=tk["liquid_density"]),
        gas=Fluid(name=tk["gas"], density=tk["gas_density"]),
    )
    motor = HybridMotor(
        thrust_source=mo["thrust_source"],
        dry_mass=mo["dry_mass"],
        dry_inertia=tuple(mo["dry_inertia"]),
        center_of_dry_mass_position=mo["center_of_dry_mass_position"],
        burn_time=(0.0, mo["burn_time"]),
        reshape_thrust_curve=False,
        grain_number=mo["grain_number"],
        grain_separation=mo["grain_separation"],
        grain_outer_radius=mo["grain_outer_radius"],
        grain_initial_inner_radius=mo["grain_initial_inner_radius"],
        grain_initial_height=mo["grain_initial_height"],
        grain_density=mo["grain_density"],
        nozzle_radius=mo["nozzle_radius"],
        throat_radius=mo["throat_radius"],
        interpolation_method="linear",
        nozzle_position=mo["nozzle_position"],
        grains_center_of_mass_position=mo["grains_center_of_mass_position"],
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    motor.add_tank(tank=tank, position=tk["tank_position"])
    r = Rocket(
        radius=b["radius"],
        mass=b["mass"],
        inertia=tuple(b["inertia"]),
        center_of_mass_without_motor=b["center_of_mass_without_motor"],
        power_off_drag=b["power_off_drag"],
        power_on_drag=b["power_on_drag"],
        coordinate_system_orientation="tail_to_nose",
        volume=b["volume"],
    )
    r.add_motor(motor, position=mo["motor_position"])
    r.add_nose(
        length=R["nose"]["length"],
        kind=R["nose"]["kind"],
        position=R["nose"]["position"],
    )
    f = R.get("fins", {})
    if f.get("useFins", False):
        r.add_trapezoidal_fins(
            n=f["n"],
            span=f["span"],
            root_chord=f["root_chord"],
            tip_chord=f["tip_chord"],
            position=f["position"],
        )
    return r, motor


# quat_rot (speed, research/sc40/speed): all products e_i e_j come from one outer product (E[4 i + j] = e_i e_j); the
# nine entries, in row-major order, are diagonal 1 - 2 (s_a + s_b) and off-diagonal 2 (p_ab -+ p_cd), as before
_QR_A = np.array([10, 6, 7, 6, 5, 11, 7, 11, 5])  # s2 p12 p13 | p12 s1 p23 | p13 p23 s1
_QR_B = np.array([15, 3, 2, 3, 15, 1, 2, 1, 10])  # s3 p03 p02 | p03 s3 p01 | p02 p01 s2
_QR_S = np.array([1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])  # x + (-y) == x - y exactly
_QR_D = np.array([True, False, False, False, True, False, False, False, True])
# round 2: the outer product's entries E[4 i + j] = e_i e_j that the A / B gathers pick, as index pairs (A entries, then B)
_QR_I = np.concatenate([_QR_A // 4, _QR_B // 4])
_QR_J = np.concatenate([_QR_A % 4, _QR_B % 4])


def quat_rot(q):
    """Body->world rotation matrices (N, 3, 3), C-contiguous. Each entry is the same expression as before (bit-identical):
    m00 = 1 - 2 (e2^2 + e3^2), m01 = 2 (e1 e2 - e0 e3), ...; the products come from one outer product."""
    N = len(q)
    # round 2: only the 18 products the gathers below need, e_i e_j (the same products as the outer product's entries),
    # from two C-ordered np.take gathers instead of the (N, 4, 4) broadcast and two gathers of it
    # (the ndarray.take method is the function np.take calls, without its Python wrapper)
    X = q.take(_QR_I, axis=1) * q.take(_QR_J, axis=1)  # (N, 18): E[A] then E[B]
    m = 2 * (X[:, :9] + X[:, 9:] * _QR_S)  # (N, 9) C-contiguous: 2 (.) everywhere
    d = m[:, ::4]  # the diagonal entries 0, 4, 8: 1 - 2 (s_a + s_b), in place (layout kept)
    np.subtract(1.0, d, out=d)
    return m.reshape(N, 3, 3)


def quat_nose(q):
    """Nose (body z) in the world frame, = quat_rot(q)[:, :, 2] (same expressions, bit-identical)."""
    e0, e1, e2, e3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.empty((len(q), 3))
    n[:, 0] = 2 * (e1 * e3 + e0 * e2)
    n[:, 1] = 2 * (e2 * e3 - e0 * e1)
    n[:, 2] = 1 - 2 * (e1**2 + e2**2)
    return n


def quat_from_nose(nose):
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(np.broadcast_to(z, nose.shape), nose)
    s = np.linalg.norm(ax, axis=1)
    c = nose[:, 2]
    ang = np.arctan2(s, c)
    axu = ax / np.maximum(s, 1e-12)[:, None]
    q = np.concatenate([np.cos(ang / 2)[:, None], axu * np.sin(ang / 2)[:, None]], 1)
    q[s < 1e-12] = np.array([1.0, 0, 0, 0])
    return q


class GivenSim:
    """Tables + batched step. Construct once in the agent's __init__ (prepare phase)."""

    K_ATT, W_MAX, KP_RATE = 2.0, 2.0, 25.0  # CompensatedInnerLoop defaults (agent side)

    def __init__(
        self,
        gp,
        dt_tab=0.05,
        k_att=2.0,
        w_max=2.0,
        kp_rate=25.0,
        act_inv=False,
        ff=False,
        axial=False,
        **ff_kwargs,
    ):
        self.K_ATT, self.W_MAX, self.KP_RATE = (
            float(k_att),
            float(w_max),
            float(kp_rate),
        )
        # W3 P3a (follow/ff_ref.py), mirrors CompensatedInnerLoop(act_inv=, ff=): exact gimbal-actuator
        # inversion (g is then the actuator OUTPUT) and setpoint feedforward (needs the ff state, see step).
        # Both off = v7 (bit-identical).
        self.ff = bool(ff)
        self.act_inv = bool(act_inv) or self.ff
        # axial (round 11; False = unchanged): axial thrust T sqrt(1 - sin^2 g_x - sin^2 g_y) as the env's TVC model
        # (ActiveRocketPy flight.py thrust3) instead of T, i.e. the thrust lost to a deflected nozzle
        self.axial = bool(axial)
        r, motor = _build_rocket(gp)
        rm = RocketModel(gp)
        self.burn_time = float(gp["rocket"]["motor"]["burn_time"])
        self.elevation = float(gp["environment"]["elevation"])
        self.g = GRAVITY
        t = np.arange(0.0, self.burn_time + 40.0, dt_tab)
        self.t_tab, self.dt_tab = t, dt_tab
        self.mass_t = np.array([float(r.total_mass(x)) for x in t])
        self.thrust_t = np.array(
            [float(motor.thrust(x)) if x <= self.burn_time else 0.0 for x in t]
        )
        rcm = np.array([float(r.com_to_cdm_function(x)) for x in t])
        i11 = np.array([float(r.get_inertia_tensor_at_time(x)[0][0]) for x in t])
        self.rcm_t = rcm
        self.icm_t = i11 - self.mass_t * rcm**2
        self.mach = np.linspace(0.0, 0.6, 61)
        self.cp, self.cla = [], []
        for s, _pos in r.aerodynamic_surfaces:
            self.cp.append(float(r.surfaces_cp_to_cdm[s][2]))
            self.cla.append(np.array([float(s.clalpha(x)) for x in self.mach]))
        self.cd_on = np.array([float(r.power_on_drag(x)) for x in self.mach])
        self.cd_off = np.array([float(r.power_off_drag(x)) for x in self.mach])
        self.S = float(r.area)
        self.lever = float(r.nozzle_to_cdm)
        self.z_tab = np.arange(0.0, 3000.0, 10.0) + self.elevation
        self.rho_tab = np.array([rm.density(x) for x in self.z_tab])
        self.a_tab = np.array([glm._isa_speed_of_sound(x) for x in self.z_tab])
        ctrl = gp["rocket"]["control"]
        self.gmax = float(ctrl["max_gimbal_angle"])
        self.grate = float(ctrl["gimbal_rate_limit"])
        self.gtau = float(ctrl.get("gimbal_time_constant") or 0.0)
        self.thr_rate = float(ctrl["throttle_rate_limit"])
        self.wz = np.array([0.0, 1.0])
        self.wu = np.zeros(2)
        self.wv = np.zeros(2)
        self.ffp = FFParams(gmax=self.gmax, **ff_kwargs) if self.ff else None
        self._speed_tables()

    def _speed_tables(self):
        """Stacked copies of the lookup tables used by step (speed, research/sc40/speed): one gather per step for the
        time tables (mass, CM offset, lateral inertia) and one for the atmosphere (density, speed of sound); the values
        are the same table entries."""
        self._tab3 = np.stack([self.mass_t, self.rcm_t, self.icm_t], 1)
        self._atm2 = np.stack([self.rho_tab, self.a_tab], 1)
        # round 2: density, speed of sound and the step's 0.5 * rho (the same product, per table entry)
        self._atm3 = np.stack([self.rho_tab, self.a_tab, 0.5 * self.rho_tab], 1)
        self._cp_arr = np.array(self.cp, float)

    def accel_avail(self, t_rel, thr=1.0):
        """Thrust acceleration T*thr/m at t_rel (s after launch), per row; 0 outside the burn."""
        tr = np.asarray(t_rel, float)
        T0 = np.where(
            (tr >= 0) & (tr <= self.burn_time),
            self._tab(self.thrust_t, np.maximum(tr, 0)),
            0.0,
        )
        return T0 * thr / self._tab(self.mass_t, np.maximum(tr, 0))

    def inertia(self, t_rel):
        """Lateral inertia about the CM at t_rel (s after launch), for the loop's feedforward."""
        return float(self._tab(self.icm_t, max(float(t_rel), 0.0)))

    # -- wind from balloon drift (observations) ---------------------------------
    def update_wind(self, b_pos, b_vel, released):
        m = released & (np.abs(b_vel[:, 2]) > 0.5)
        if m.sum() < 5:
            return
        z, u, v = b_pos[m, 2], b_vel[m, 0], b_vel[m, 1]
        o = np.argsort(z)
        z, u, v = z[o], u[o], v[o]
        edges = np.arange(np.floor(z[0] / 25) * 25, z[-1] + 25, 25.0)
        zc, uc, vc = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            k = (z >= a) & (z < b)
            if k.any():
                zc.append(z[k].mean())
                uc.append(u[k].mean())
                vc.append(v[k].mean())
        if len(zc) >= 1:
            self.wz, self.wu, self.wv = np.array(zc), np.array(uc), np.array(vc)

    def _tab(self, arr, t):
        i = np.clip((np.asarray(t) / self.dt_tab).astype(int), 0, len(self.t_tab) - 1)
        return arr[i]

    def time_tables(self, t_rel):
        """The per-row time-table values step uses, for t_rel of any shape: (T0 thrust, burn flag, (mass, CM offset,
        lateral inertia) (..., 3)). step(pre=...) takes them precomputed (the same expressions, evaluated once)."""
        if "_tab3" not in self.__dict__:
            self._speed_tables()
        ti = _clip(
            (np.maximum(t_rel, 0) / self.dt_tab).astype(int), 0, len(self.t_tab) - 1
        )
        T0 = np.where((t_rel >= 0) & (t_rel <= self.burn_time), self.thrust_t[ti], 0.0)
        return T0, T0 > 1e-9, self._tab3[ti]

    # fields of step_tables (round 2): 0 T0, 1 mass, 2 CM offset, 3 lateral inertia, 4 a_max, 5 (2 brake) a_max,
    # 6 max(T0 lever, 1e-9), 7 T0 / mass, then per surface k: cp_k - rcm (K), (cp_k - rcm) * -1 (K), (cp_k - rcm) * 1 (K)
    NT = 8

    def step_tables(self, t_rel, fm=False):
        """Round 2 (research/sc40/speed): the time-only per-row quantities step uses, for t_rel of any shape, packed as
        (..., NT + 3 K) floats (fields above), plus the burn flags. Fields 4-7 are the values of ff_step's a_max,
        2 brake a_max, max(T lever, 1e-9) and of accel_avail(t_rel, thr) when T = T0 thr equals T0 on every row (a
        caller checks it: thr == 1 while burning, T0 == 0 once a row has stopped burning). Same expressions, evaluated
        once for all steps of a rollout instead of once per step. fm=True (round 2, second pass): field-major, X (NT + 3 K,
        ...) with one field per leading index (the same values, X_fm[f] == X[..., f]), the layout step(tab=) takes."""
        if "_atm3" not in self.__dict__:
            self._speed_tables()
        # round 2 (second pass): every field is an elementwise function of (T0, the time-table row ti), and T0 is either
        # thrust_t[ti] (inside the burn window [0, burn_time]) or +0.0 (outside, np.where's 0.0). The fields of both
        # cases are evaluated once per table index by the same expressions (_step_fields, cached in _srows) and gathered
        # here: the same values as evaluating them per (step, row).
        rows = self.__dict__.get("_srows")
        if rows is None:
            Lt = len(self.t_tab)
            rows = self._srows = self._step_fields(np.concatenate([self.thrust_t, np.zeros(Lt)]),
                                                   np.concatenate([self._tab3, self._tab3]))
        Lt = len(self.t_tab)
        ti = _clip((np.maximum(t_rel, 0) / self.dt_tab).astype(int), 0, Lt - 1)  # as time_tables
        out = ~((t_rel >= 0) & (t_rel <= self.burn_time))  # outside the burn window: the time_tables test, negated
        idx = ti + Lt * out
        if fm:
            rT = self.__dict__.get("_srowsT")
            if rT is None:
                rT = self._srowsT = np.ascontiguousarray(rows.T)
            X = rT.take(idx, axis=1)
            return X, X[0] > 1e-9
        X = rows[idx]
        return X, X[..., 0] > 1e-9  # burn = T0 > 1e-9, as time_tables

    def _step_fields(self, T0, tb):
        """step_tables' fields for T0 (...,) and time-table rows tb (..., 3) (mass, CM offset, lateral inertia)."""
        K = len(self.cp)
        NT = self.NT
        X = np.empty(np.shape(T0) + (NT + 3 * K,))
        X[..., 0] = T0
        X[..., 1:4] = tb
        if self.ff:
            p = self.ffp
            a_max = p.afrac * T0 * self.lever * p.sin_gmax / np.maximum(tb[..., 2], 1e-6)  # as ff_step (T = T0)
            X[..., 4] = a_max
            X[..., 5] = 2.0 * p.brake * a_max  # ff_step's sqrt(2.0 * p.brake * a_max * th) argument before * th
            X[..., 6] = np.maximum(T0 * self.lever, 1e-9)
        else:
            X[..., 4:7] = np.nan
        X[..., 7] = T0 / tb[..., 0]  # accel_avail(t_rel, thr) = T0 thr / mass with T0 thr == T0
        cpz = self._cp_arr - tb[..., 1:2]  # cp_k - rcm, as step's cpz
        X[..., NT : NT + K] = cpz
        X[..., NT + K : NT + 2 * K] = cpz * -1.0  # cpz[:, None, :] * _NEG_POS_COL, row -1 then row +1
        X[..., NT + 2 * K :] = cpz * 1.0
        return X

    def step(self, pos, vel, q, w, g, thr, t_rel, n_cmd, thr_cmd, dt, ffs=None, ff_normed=False, pre=None, tab=None, Rm=None):
        """ffs: (N, FF_DIM) feedforward state, used when self.ff (then returned as a 7th element).
        thr_cmd: (N,) or a scalar (1.0 = full throttle, as the planner's rollouts command).
        pre: optional time_tables(t_rel) for these rows (t_rel is then not used); ff_normed: see ff_step(normed).
        tab (round 2): optional (X, burn, hint, fast) with X, burn = step_tables(t_rel, fm=True) for these rows, X field-major
        (NT + 3 K, N) with contiguous field rows (t_rel and pre are then not used), hint as in pre, fast = True when the caller guarantees T = T0 thr == T0 on every row and
        thr == 1.0 on every burning row (rollouts check it once per rollout); Rm: optional quat_rot(q), already computed.
        Speed (research/sc40/speed): every element is computed by the same sequence of operations as the earlier
        per-component code (bit-identical); x/y (and per-surface) components are processed as columns of one array, and
        the einsum operands keep their earlier shapes and C-contiguous layout."""
        if "_atm3" not in self.__dict__:  # an instance built before _speed_tables existed (e.g. copied state)
            self._speed_tables()
        N = len(pos)
        if Rm is None:
            Rm = np.ascontiguousarray(quat_rot(q))
        # one table index for all time tables (as _tab, evaluated once)
        X = None
        fast = False
        if tab is not None:  # round 2: step_tables fields, field-major (one row per field)
            X, burn, bhint, fast = tab
            T0, m, rcm, i_lat = X[0], X[1], X[2], X[3]
        else:
            if pre is None:
                T0, burn, tb = self.time_tables(t_rel)
                bhint = None
            else:  # bhint: 1 all rows burning, -1 none, -2 every row past burn_time (no thrust now or later), 0 / None: look
                T0, burn, tb = pre[:3]
                bhint = pre[3] if len(pre) > 3 else None
            m, rcm, i_lat = tb[:, 0], tb[:, 1], tb[:, 2]
        perm = bhint == -2
        allb = bhint == 1
        thr_step = self.thr_rate * dt
        if fast and allb:
            pass  # clip(1.0, thr - s, thr + s) == thr == 1.0 on every (burning) row: thr is unchanged
        else:
            lo = thr - thr_step
            if perm:  # np.where(burn, ..., maximum(0, lo)) with burn False on every row
                thr = np.maximum(0.0, lo)
            else:
                if isinstance(thr_cmd, float) and thr_cmd == 1.0:
                    tgt = 1.0  # where(speed < 12, 1, thr_cmd) is 1 everywhere
                else:
                    tgt = np.where(_norm(vel) < 12.0, 1.0, thr_cmd)
                if allb:  # np.where with burn True on every row (round 2)
                    thr = _clip(tgt, lo, thr + thr_step)
                else:
                    thr = np.where(burn, _clip(tgt, lo, thr + thr_step), np.maximum(0.0, lo))
        use_ff = self.ff and ffs is not None
        n_att = n_cmd
        # fast: T0 thr == T0 on every row (thr == 1 while burning, T0 == +0.0 and thr finite >= 0 otherwise)
        T = T0 if fast else T0 * thr
        # every row past the burn: the gimbal output where(burn, ., 0) is 0, and with T <= 1e-9 on every row the
        # feedforward state is fully reset (ff_step_off gives ff_step's new state); n_cmd and the attitude / gimbal law
        # then reach no output, so they are skipped (speed, research/sc40/speed). With fast, T = T0 = +0.0 past burn_time.
        coast = perm and (fast or not use_ff or bool((T <= 1e-9).all()))
        if coast:
            if use_ff:
                ffs = ff_step_off(Rm, dt, split=isinstance(ffs, tuple))
            g = np.zeros((N, 2))
        else:
            if use_ff:
                ffs, n_att, w_ff, d_ff = ff_step(
                    ffs, n_cmd, Rm, T, i_lat, self.lever, dt, self.ffp, normed=ff_normed,
                    tabs=(X[4], X[5], X[6]) if (fast and allb) else None,
                )
            d = _einsum("nji,nj->ni", Rm, np.ascontiguousarray(n_att))  # einsum operands C-ordered (as before)
            ang = np.arccos(_clip(d[:, 2], -1, 1))
            h = np.hypot(d[:, 0], d[:, 1])
            if _nz(h < 1e-12):
                err = np.where(
                    h[:, None] < 1e-12,
                    0.0,
                    d[:, _YX] * _NEG_POS / np.maximum(h, 1e-12)[:, None] * ang[:, None],
                )  # (-d1, d0) / h * ang
            else:  # round 2: no row has h < 1e-12, so where(False, 0.0, x) == x and maximum(h, 1e-12) == h (NaN included)
                err = d[:, _YX] * _NEG_POS / h[:, None] * ang[:, None]
            w_cmd = _clip(self.K_ATT * err, -self.W_MAX, self.W_MAX)
            if use_ff:
                g_des = _clip(
                    self.KP_RATE * (w_cmd + w_ff - w) + d_ff, -self.gmax, self.gmax
                )
            else:
                g_des = _clip(self.KP_RATE * (w_cmd - w), -self.gmax, self.gmax)
            gs = self.grate * dt
            if self.act_inv:
                g_new = act_inv_y(g_des, g, gs, dt / (self.gtau + dt), self.gmax)
            else:
                g_cmd = _clip(g_des, g - gs, g + gs)
                g_new = (
                    g_cmd if self.gtau <= 0 else g + (g_cmd - g) * (dt / (self.gtau + dt))
                )
            g = g_new if allb else np.where(burn[:, None], g_new, 0.0)
        pz = pos[:, 2]
        zi = _clip(
            ((pz - self.z_tab[0]) / 10.0).astype(int), 0, len(self.z_tab) - 1
        )
        at = self._atm3[zi]
        rho, a_s, hrS = at[:, 0], at[:, 1], at[:, 2]  # hrS = 0.5 * rho (table entry)
        fs = np.empty((N, 3))  # wind - vel, wind = (u(z), v(z), 0)
        fs[:, 0] = _interp(pz, self.wz, self.wu)
        fs[:, 1] = _interp(pz, self.wz, self.wv)
        fs[:, 2] = 0.0
        fs -= vel
        fs_speed = _norm(fs)
        mach = fs_speed / a_s
        qd = hrS * fs_speed**2
        if bhint == 1 or (bhint is None or bhint == 0) and burn.all():
            cd = _interp(mach, self.mach, self.cd_on)
        elif bhint == -1 or bhint == -2 or not burn.any():
            cd = _interp(mach, self.mach, self.cd_off)
        else:
            cd = np.where(
                burn,
                _interp(mach, self.mach, self.cd_on),
                _interp(mach, self.mach, self.cd_off),
            )
        fs_b = _einsum("nji,nj->ni", Rm, np.ascontiguousarray(fs))
        # aerodynamic surfaces, all at once. Round 2 (second pass): per-surface arrays laid out (component, surface, row),
        # (2, K, N) / (K, N), so every op runs on contiguous rows (the earlier (N, 2, K) layout had inner loops of length
        # K); the same per-element operations as the earlier loop over k
        K = len(self.cp)
        if X is None:
            # cpz_k - rcm, times -1 / +1 (the earlier cpz[:, None, :] * (-1, +1)), as the step_tables fields
            CPZN = (self._cp_arr[:, None] - rcm[None]) * _NP3  # (2, K, N)
        else:
            CPZN = X[self.NT + K :].reshape(2, K, N)
        # w x r with r = (0, 0, cpz): sv = fb_xy + (-w_y, w_x) cpz = fb_c + w_YX[c] (cpz NP[c]); (w NP) cpz == w (cpz NP)
        # exactly (NP = -1, +1 is a sign)
        SV = fs_b.T[:2, None, :] + w.T[_YX][:, None, :] * CPZN
        SQ = SV * SV
        fb2 = fs_b[:, 2]
        ss = np.sqrt(SQ[0] + SQ[1] + (fb2 * fb2)[None])  # (K, N)
        lat = np.hypot(SV[0], SV[1])
        aoa = np.minimum(
            np.arccos(_clip(-fb2[None] / np.maximum(ss, 1e-9), -1, 1)), np.pi / 10
        )
        xm = ss / a_s[None]
        cla = np.empty((K, N))
        for k in range(K):
            cla[k] = _interp(xm[k], self.mach, self.cla[k])
        lift = hrS[None] * ss**2 * self.S * cla * aoa
        if _nz(lat > 1e-9) == lat.size:  # round 2: where(True, x, 0.0) == x and maximum(lat, 1e-9) == lat on every entry
            inv = lift / lat
        else:
            inv = np.where(lat > 1e-9, lift / np.maximum(lat, 1e-9), 0.0)
        LXY = inv[None] * SV  # (lx, ly) per surface
        MXY = LXY[_YX] * CPZN  # (-cpz ly, cpz lx) per surface
        # F and M were accumulated from 0 surface by surface; F is assembled field-major, then copied C-ordered (N, 3) for
        # the einsum (as the earlier concatenate)
        Ft = np.empty((3, N))
        Fxy = 0.0 + LXY[:, 0]
        Mxy = 0.0 + MXY[:, 0]
        for k in range(1, K):
            Fxy = np.add(Fxy, LXY[:, k], out=Ft[:2]) if k == K - 1 else Fxy + LXY[:, k]
            Mxy = Mxy + MXY[:, k]
        if K == 1:
            Ft[:2] = Fxy
        np.add(-qd * self.S * cd, T, out=Ft[2])
        F = np.ascontiguousarray(Ft.T)
        Mxy = np.ascontiguousarray(Mxy.T)  # (N, 2)
        gr = np.radians(g)
        if self.axial:
            sx, sy = np.sin(gr[:, 0]), np.sin(gr[:, 1])
            F[:, 2] += T * (np.sqrt(np.maximum(1.0 - sx * sx - sy * sy, 0.0)) - 1.0)
        Mxy = Mxy + np.sin(gr) * T[:, None] * self.lever
        acc = _einsum("nij,nj->ni", Rm, np.ascontiguousarray(F)) / m[:, None]
        acc[:, 2] -= self.g
        w = w + Mxy / i_lat[:, None] * dt
        vel_new = vel + acc * dt
        # round 2: (0.5 s) dt == s (0.5 dt): scaling by 2^-1 is exact and commutes with rounding (0.5 dt is exact); this
        # holds for every s that is not a nonzero subnormal (|s| < 2.2e-308 m/s), which a velocity sum never is here
        pos = pos + (vel + vel_new) * (0.5 * dt)
        w2 = w * w
        wn0 = np.sqrt(w2[:, 0] + w2[:, 1])
        wn = np.maximum(wn0, 1e-12)
        half = wn0 * (0.5 * dt)  # round 2: (wn0 dt) / 2 == wn0 (dt / 2), the same power-of-two argument as pos
        # (f0, f1, f2) = (cos(an / 2), w_x s, w_y s), s = sin(an / 2) / wn
        # round 2: f with its negated entries f1 * -1, f2 * -1 appended (the products f_F * S of before; * 1.0 is the
        # identity), so the signed gather is one plain gather
        ws = w * (np.sin(half) / wn)[:, None]
        fq = np.concatenate([np.cos(half)[:, None], ws, ws * -1.0], 1)
        # quaternion product q (x) f, component c = (X1 Y1 +- X2 Y2) +- X3 Y3 (signs folded into the f gather; exact)
        P = q[:, _QP_E] * fq[:, _QP_G]
        q = (P[:, 0] + P[:, 1]) + P[:, 2]
        qq = q * q
        q /= np.sqrt(qq[:, 0] + qq[:, 1] + qq[:, 2] + qq[:, 3])[:, None]
        if use_ff:
            return pos, vel_new, q, w, g, thr, ffs
        return pos, vel_new, q, w, g, thr


_YX = np.array([1, 0])
_NEG_POS = np.array([-1.0, 1.0])
# round 2: column gathers of the surface velocity for K surfaces in (component, surface) order
_SV_W = {K: np.repeat(np.array([1, 0]), K) for K in range(1, 6)}
_SV_CK = {K: np.repeat(np.array([0, 1]), K) for K in range(1, 6)}
_NEG_POS_COL = np.array([[-1.0], [1.0]])
_NP3 = np.array([-1.0, 1.0]).reshape(2, 1, 1)  # round 2: the (-1, +1) factor of the (2, K, N) surface layout
# q (x) f with f = (f0, f1, f2, 0):  q0' = e0 f0 - e1 f1 - e2 f2,  q1' = e0 f1 + e1 f0 - e3 f2,
#                                    q2' = e0 f2 + e2 f0 + e3 f1,  q3' = e3 f0 + e1 f2 - e2 f1
_QP_E = np.array([[0, 0, 0, 3], [1, 1, 2, 1], [2, 3, 3, 2]])
_QP_F = np.array([[0, 1, 2, 0], [1, 0, 0, 2], [2, 2, 1, 1]])
_QP_S = np.array([[1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, -1.0]])
# round 2: the signed gather into (f0, f1, f2, f1 * -1, f2 * -1): f_k with sign -1 is column k + 2 (f0 is never negated)
assert (_QP_F[_QP_S < 0] > 0).all()
_QP_G = np.where(_QP_S < 0, _QP_F + 2, _QP_F)
