"""Rocket mass/thrust/drag model built from ``given_parameters`` only.

The mass(t) and (full-throttle) thrust(t) curves are extracted *exactly* by
rebuilding the same RocketPy hybrid motor the environment uses (see
``balloon_world.BalloonPoppingEnv.__init_rocket_simulation``) from the parameters
the agent is legally given, then tabulating on a fine grid. The point-mass
surrogate then reads these tables — no RocketPy in the rollout hot loop.

Throttle convention (competition quirk): throttle scales effective Isp at
*constant mass flow*, so ``thrust(t, u) = u * thrust_full(t)`` while ``mass(t)``
is throttle-independent (the engine still cuts off at ``burn_time``).
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.80665  # m/s^2, matches RocketPy's nominal sea-level g

# International Standard Atmosphere (troposphere) constants
_ISA_T0 = 288.15  # K
_ISA_L = 0.0065  # K/m lapse rate
_ISA_RHO0 = 1.225  # kg/m^3
_ISA_R = 287.05  # J/(kg K)
_ISA_EXP = GRAVITY / (_ISA_R * _ISA_L) - 1.0  # ~4.256


def isa_density(altitude_m):
    """ISA troposphere air density (kg/m^3) at geometric altitude above MSL.

    A standard-atmosphere approximation: the agent is *not* given the ensemble
    atmosphere file, so this is the best legal density model. Wind is unknown
    here and inferred separately from balloon drift (balloonnav/windfield)."""
    h = np.asarray(altitude_m, dtype=float)
    base = np.clip(1.0 - _ISA_L * h / _ISA_T0, 1e-6, None)
    return _ISA_RHO0 * base**_ISA_EXP


class RocketModel:
    """Tabulated mass(t), thrust(t), and a drag/density model for the rocket.

    Parameters
    ----------
    given_parameters : dict
        The scenario ``given_parameters`` dict passed to the agent.
    n_grid : int
        Number of time samples for the mass/thrust tables over [0, max_time].
    """

    def __init__(self, given_parameters: dict, n_grid: int = 2000) -> None:
        gp = given_parameters
        self.given_parameters = gp

        body = gp["rocket"]["rocket_body"]
        motor = gp["rocket"]["motor"]
        sim = gp["simulation"]
        env = gp["environment"]

        self.body_mass = float(body["mass"])
        self.radius = float(body["radius"])
        self.ref_area = np.pi * self.radius**2
        self.cd_power_on = float(body["power_on_drag"])
        self.cd_power_off = float(body["power_off_drag"])

        self.burn_time = float(motor["burn_time"])
        self.thrust_full = float(motor["thrust_source"])  # constant thrust source
        self.motor_dry_mass = float(motor["dry_mass"])

        self.elevation = float(env["elevation"])
        self.max_time = float(sim["max_time"])
        self.dt = float(sim["time_step"])

        self._t_grid = np.linspace(0.0, self.max_time, int(n_grid))
        self._build_tables()

    # ------------------------------------------------------------------
    def _build_tables(self) -> None:
        """Tabulate full-throttle thrust(t) and total rocket mass(t)."""
        motor = self._build_motor()

        ts = self._t_grid
        try:  # RocketPy Functions accept array input
            thrust = np.asarray(motor.thrust(ts), dtype=float).reshape(-1)
            prop = np.asarray(motor.propellant_mass(ts), dtype=float).reshape(-1)
        except Exception:  # fall back to per-sample evaluation
            thrust = np.array([float(motor.thrust(t)) for t in ts])
            prop = np.array([float(motor.propellant_mass(t)) for t in ts])

        # After burn time the motor produces no thrust regardless of throttle.
        thrust = np.where(ts <= self.burn_time, thrust, 0.0)
        # Total rocket mass = structural body + motor dry + propellant remaining.
        self._thrust_grid = np.clip(thrust, 0.0, None)
        self._mass_grid = (
            self.body_mass + self.motor_dry_mass + np.clip(prop, 0.0, None)
        )

        self.dry_mass = float(self._mass_grid[-1])  # burnout mass
        self.wet_mass = float(self._mass_grid[0])

    def _build_motor(self):
        """Rebuild the env's hybrid motor from given_parameters (one-time)."""
        from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
        from rocketpy.motors.tank import MassFlowRateBasedTank

        m = self.given_parameters["rocket"]["motor"]
        tk = self.given_parameters["rocket"]["tank"]

        gas = Fluid(name=tk["gas"], density=tk["gas_density"])
        liq = Fluid(name=tk["liquid"], density=tk["liquid_density"])
        shape = CylindricalTank(radius=tk["radius"], height=tk["height"])
        tank = MassFlowRateBasedTank(
            name="oxidizer_tank",
            geometry=shape,
            flux_time=(0.0, tk["flux_time"]),
            initial_liquid_mass=tk["initial_liquid_mass"],
            initial_gas_mass=tk["initial_gas_mass"],
            liquid_mass_flow_rate_in=0,
            liquid_mass_flow_rate_out=tk["liquid_mass_flow_rate_out"],
            gas_mass_flow_rate_in=0,
            gas_mass_flow_rate_out=0,
            liquid=liq,
            gas=gas,
        )
        motor = HybridMotor(
            thrust_source=m["thrust_source"],
            dry_mass=m["dry_mass"],
            dry_inertia=tuple(m["dry_inertia"]),
            center_of_dry_mass_position=m["center_of_dry_mass_position"],
            burn_time=(0.0, m["burn_time"]),
            reshape_thrust_curve=False,
            grain_number=m["grain_number"],
            grain_separation=m["grain_separation"],
            grain_outer_radius=m["grain_outer_radius"],
            grain_initial_inner_radius=m["grain_initial_inner_radius"],
            grain_initial_height=m["grain_initial_height"],
            grain_density=m["grain_density"],
            nozzle_radius=m["nozzle_radius"],
            throat_radius=m["throat_radius"],
            interpolation_method="linear",
            nozzle_position=m["nozzle_position"],
            grains_center_of_mass_position=m["grains_center_of_mass_position"],
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )
        motor.add_tank(tank=tank, position=tk["tank_position"])
        return motor

    # ------------------------------------------------------------------
    # Vectorized accessors (t may be scalar or array)
    # ------------------------------------------------------------------
    def mass(self, t):
        """Total rocket mass (kg) at time(s) t since launch."""
        return np.interp(t, self._t_grid, self._mass_grid)

    def thrust(self, t, throttle=1.0):
        """Thrust magnitude (N) at time(s) t for the given throttle in [0, 1]."""
        return throttle * np.interp(t, self._t_grid, self._thrust_grid)

    def density(self, z):
        """Air density (kg/m^3) at ENU up-coordinate z (≈ altitude MSL)."""
        return isa_density(z)

    def drag_coeff(self, t):
        """Drag coefficient: power-on during burn, power-off after."""
        return np.where(
            np.asarray(t) <= self.burn_time, self.cd_power_on, self.cd_power_off
        )
