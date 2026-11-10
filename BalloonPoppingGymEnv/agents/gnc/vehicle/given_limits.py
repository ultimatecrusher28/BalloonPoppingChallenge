"""Lateral aerodynamic force model reconstructed from the given geometry, thrust and mass (discussion #165).

RocketPy's NoseCone and TrapezoidalFins supply the lift slopes from the given geometry. The normal force is
q * reference_area * clalpha * alpha with the pi/10 cap of ActiveRocketPy's AeroSurface.compute_forces_and_moments; the
density and the speed of sound use the ISA approximation. The consumers (the inner loop and accel_for_thrust_dir) prime the
time since launch and the altitude (set_context) before a lookup.
"""

from __future__ import annotations

import numpy as np

from .rocket_model import isa_density

_ISA_T0_K = 288.15  # ISA sea-level temperature, K (standard atmosphere)
_ISA_LAPSE_K_M = 0.0065  # ISA troposphere lapse rate, K/m
_AIR_GAMMA = 1.4  # ratio of specific heats for air
_AIR_R = 287.05  # J/(kg K), specific gas constant for dry air


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
    """The lateral aerodynamic force model from given parameters only, shared by the inner loop and the thrust-direction
    inversion (accel_for_thrust_dir)."""

    def __init__(self, given_parameters: dict, model):
        gp = given_parameters
        self.given_parameters = gp
        self.model = model
        self.ref_area = float(model.ref_area)
        self.elevation = float(gp["environment"]["elevation"])

        body = gp["rocket"]["rocket_body"]
        radius = float(body["radius"])
        self._nose, self._fins = _build_aero_surfaces(gp, radius)

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
        Zero at zero AoA or zero speed or zero mass."""
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
