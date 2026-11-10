"""Vehicle approximation reconstructed from given geometry, thrust and mass.

Provenance review: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165
This mode avoids the legacy measured tables; it is not organizer certification.
RocketPy NoseCone and TrapezoidalFins supply lift slopes from given geometry.
Normal force uses q * reference_area * clalpha * alpha, with the pi/10 cap
in ActiveRocketPy/rocketpy/rocket/aero_surface/aero_surface.py,
AeroSurface.compute_forces_and_moments. Density uses the ISA approximation.

Turn time approximates a rest-to-rest bang-bang maneuver using given TVC
range, thrust, nozzle arm, body plus dry-motor inertia and gimbal travel rate.
It omits propellant/parallel-axis inertia and uses body CG for vehicle CG;
these errors have not been quantified. It also omits aero torque and current
angular velocity. The lateral-force envelope includes passive lift, which is
not independently controllable after burnout. The existing inner loop's
force-direction approximation remains in place. Consumers prime time and
altitude before using this shared model; initial context is launch/ground.
"""

from __future__ import annotations

import numpy as np

from .rocket_model import isa_density

_ISA_T0_K = 288.15  # ISA sea-level temperature, K (standard atmosphere)
_ISA_LAPSE_K_M = 0.0065  # ISA troposphere lapse rate, K/m
_AIR_GAMMA = 1.4  # ratio of specific heats for air
_AIR_R = 287.05  # J/(kg K), specific gas constant for dry air

# Evaluation angle of attack when a caller does not specify one. Matches the
# point the other consumers already query by not overriding the default
# (mirrors LimitTables.a_lat_max's default), so a legacy-vs-given comparison
# asks the same physical question. NOT a fitted saturation point: this
# aerodynamic cap comes from the public AeroSurface implementation.
_DEFAULT_QUERY_AOA_DEG = 18.0


def _isa_speed_of_sound(altitude_m: float) -> float:
    """ISA troposphere speed of sound (m/s); same standard atmosphere as
    ``rocket_model.isa_density``, so density and Mach stay consistent."""
    t = max(_ISA_T0_K - _ISA_LAPSE_K_M * float(altitude_m), 1.0)
    return float(np.sqrt(_AIR_GAMMA * _AIR_R * t))


def _build_aero_surfaces(given_parameters: dict, radius: float):
    """Nose cone (required) and fin set (optional) from given geometry only,
    built with RocketPy's own public aero-surface classes so ``clalpha`` is
    RocketPy's formula, never a hand re-derivation or a fitted number."""
    from rocketpy import NoseCone, TrapezoidalFins

    rocket_cfg = given_parameters["rocket"]
    nose_cfg = rocket_cfg["nose"]
    nose = NoseCone(
        length=float(nose_cfg["length"]),
        kind=str(nose_cfg["kind"]),
        base_radius=radius,
        rocket_radius=radius,
    )
    fins_cfg = rocket_cfg.get("fins", {})
    fins = None
    if fins_cfg.get("useFins", False):
        fins = TrapezoidalFins(
            n=int(fins_cfg["n"]),
            root_chord=float(fins_cfg["root_chord"]),
            tip_chord=float(fins_cfg["tip_chord"]),
            span=float(fins_cfg["span"]),
            rocket_radius=radius,
        )
    return nose, fins


class GivenLimits:
    """given-parameters-only stand-in for ``vehicle.limits.LimitTables``.

    Constructed once per agent (``agent.py``, ``limits_mode="given"``) and
    shared by the beam planner, the feasible follower, and the compensated
    inner loop — one physics model, three consumers, the same sharing
    pattern ``LimitTables`` uses. Exposes the same method names/signatures
    those consumers already call (``a_lat_max``, ``lateral_use``,
    ``lateral_disturbance``) plus ``turn_time`` and ``set_context`` (see
    module docstring). Deliberately does NOT subclass or construct
    ``LimitTables``: it never touches ``research/limits/data``.
    """

    def __init__(self, given_parameters: dict, model, *, turn_model="ramp"):
        if turn_model not in {"ramp", "slew"}:
            raise ValueError("given turn_model must be ramp or slew")
        self.turn_model = turn_model
        gp = given_parameters
        self.given_parameters = gp
        self.model = model
        self.ref_area = float(model.ref_area)
        self.elevation = float(gp["environment"]["elevation"])

        body = gp["rocket"]["rocket_body"]
        motor = gp["rocket"]["motor"]
        ctrl = gp["rocket"]["control"]

        radius = float(body["radius"])
        self._nose, self._fins = _build_aero_surfaces(gp, radius)

        self._gimbal_max_rad = float(np.radians(float(ctrl["max_gimbal_angle"])))
        self._gimbal_sin = float(np.sin(self._gimbal_max_rad))
        gimbal_rate_rad_s = float(np.radians(float(ctrl["gimbal_rate_limit"])))
        self._gimbal_rate = gimbal_rate_rad_s
        self._t_ramp = 2.0 * self._gimbal_max_rad / max(gimbal_rate_rad_s, 1e-9)

        # Axial distance from the nozzle (TVC pivot) to the vehicle's center
        # of mass: see the "Limitations" note in the module docstring re:
        # the motor's own CG/inertia contribution being omitted.
        self._arm_m = abs(
            float(motor["motor_position"])
            + float(motor["nozzle_position"])
            - float(body["center_of_mass_without_motor"])
        )
        self._inertia_yaw = max(
            float(body["inertia"][0]) + float(motor["dry_inertia"][0]), 1e-9
        )

        self._t_rel = 0.0
        self._altitude = self.elevation

    # -- context -----------------------------------------------------------
    def set_context(self, t_rel: float, altitude: float) -> None:
        """Prime the flight context (time since launch, altitude MSL) the
        next lookup uses for thrust/mass (via ``model``) and density (via
        ISA). Optional: unset, lookups use the construction-time default
        ``(0.0, elevation)``, which is what makes the zero-thrust /
        zero-AoA cases well-defined even before any caller has flown."""
        self._t_rel = float(t_rel)
        self._altitude = float(altitude)

    # -- aero ----------------------------------------------------------------
    def _cn_alpha(self, mach: float) -> float:
        cn = float(self._nose.clalpha(mach))
        if self._fins is not None:
            cn += float(self._fins.clalpha(mach))
        return cn

    def lateral_disturbance(self, speed, aoa_deg=None) -> float:
        """Analytic aero normal-force acceleration (m/s^2) at this speed and
        AoA: ``N = q * S_ref * Cn_alpha(mach) * alpha``, ``a = N / mass(t_rel)``.
        Zero at zero AoA or zero speed or zero mass — unlike the fitted
        table's speed-only fallback, which returns a nonzero residual even
        when ``aoa_deg`` is None."""
        speed = float(speed)
        if aoa_deg is None or speed <= 0.0:
            return 0.0
        mass = float(self.model.mass(self._t_rel))
        if mass <= 1e-9:
            return 0.0
        mach = speed / _isa_speed_of_sound(self._altitude)
        alpha_rad = float(np.clip(np.radians(float(aoa_deg)), -np.pi / 10, np.pi / 10))
        cn = self._cn_alpha(mach) * alpha_rad
        q = 0.5 * float(isa_density(self._altitude)) * speed * speed
        return float(q * self.ref_area * cn / mass)

    def a_lat_max(self, speed, aoa_deg=_DEFAULT_QUERY_AOA_DEG) -> float:
        """Powered TVC plus passive aerodynamic acceleration envelope (m/s^2): the TVC lateral
        thrust component (zero once ``model.thrust(t_rel)`` is zero, e.g.
        post-burnout) plus the aero contribution at ``aoa_deg``."""
        thrust = float(self.model.thrust(self._t_rel))
        mass = float(self.model.mass(self._t_rel))
        tvc = thrust * self._gimbal_sin / mass if mass > 1e-9 else 0.0
        return float(tvc + self.lateral_disturbance(speed, aoa_deg))

    def lateral_use(self, vel, accel, *, speed_floor: float = 8.0):
        """Same definition as ``LimitTables.lateral_use``: the perpendicular
        (velocity-turning) component of ``accel`` as a fraction of
        ``a_lat_max`` at that sample's speed. Duplicated rather than
        inherited so this class never touches ``LimitTables``."""
        vel = np.atleast_2d(np.asarray(vel, dtype=float))
        accel = np.atleast_2d(np.asarray(accel, dtype=float))
        speed = np.linalg.norm(vel, axis=1)
        live = speed > speed_floor
        if not live.any():
            return np.empty(0)
        vh = vel[live] / speed[live][:, None]
        a = accel[live]
        lat = np.linalg.norm(a - (a * vh).sum(axis=1)[:, None] * vh, axis=1)
        cap = np.array([self.a_lat_max(float(x)) for x in speed[live]])
        return lat / np.maximum(cap, 1e-6)

    # -- attitude ------------------------------------------------------------
    def turn_time(self, angle_rad: float) -> float:
        """Approximate rest-to-rest attitude turn time (s): bang-bang under
        the current TVC-torque angular acceleration (from ``model.thrust``
        at the primed context, so it honestly goes to infinity with zero
        thrust — matching the measured finding that coast authority is
        effectively zero, handoff.md section 3, "Phase C"), plus the
        gimbal's own travel time across the bang-bang switch."""
        angle = abs(float(angle_rad))
        if angle == 0.0:
            return 0.0
        thrust = float(self.model.thrust(self._t_rel))
        torque_max = thrust * self._gimbal_sin * self._arm_m
        alpha_max = torque_max / self._inertia_yaw
        if alpha_max <= 1e-9:
            return float("inf")
        if self.turn_model == "slew":
            # Rest-to-rest, zero initial/final gimbal, bounded acceleration
            # and jerk. Integrating four jerk ramps gives theta=2*j*tj**3.
            # With saturated acceleration and plateau ta:
            # theta=alpha*(ta+tj)*(ta+2*tj), duration=4*tj+2*ta.
            # cos(max_gimbal) bounds the sin(gimbal) torque slope from below.
            jerk = (
                thrust
                * self._arm_m
                / self._inertia_yaw
                * self._gimbal_rate
                * np.cos(self._gimbal_max_rad)
            )
            if jerk <= 1e-9:
                return float("inf")
            ramp = alpha_max / jerk
            threshold = 2.0 * alpha_max * ramp**2
            if angle <= threshold:
                return float(4.0 * (angle / (2.0 * jerk)) ** (1.0 / 3.0))
            return float(ramp + np.sqrt(ramp**2 + 4.0 * angle / alpha_max))
        return float(2.0 * np.sqrt(angle / alpha_max) + self._t_ramp)
