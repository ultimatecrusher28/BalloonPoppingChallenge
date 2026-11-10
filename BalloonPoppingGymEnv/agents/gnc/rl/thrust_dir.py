"""Thrust-direction command -> CompensatedInnerLoop acceleration command.

A learned guidance policy commands a unit thrust direction ``n`` (launch
frame) and a throttle. ``CompensatedInnerLoop.solve`` expects a kinematic
acceleration and inverts ``force = m * (a + g z - aero - drag)``; supplying
``a = thr*T/m * n - g z + aero + drag`` (same aero/drag terms the loop uses)
makes the requested force exactly ``thr * T * n``. Given-parameter model only.
"""

from __future__ import annotations

import numpy as np

from ..follow.inner_loop import quat_to_matrix
from ..vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])


def accel_for_thrust_dir(n_cmd, throttle, estimate, model, tables, t_rel):
    n = np.asarray(n_cmd, float)
    n = n / max(np.linalg.norm(n), 1e-9)
    if hasattr(tables, "set_context"):
        tables.set_context(t_rel, float(estimate.position[2]))
    m = float(model.mass(t_rel))
    thrust = float(model.thrust(t_rel))
    vel = np.asarray(estimate.velocity, float)
    speed = float(np.linalg.norm(vel))
    aero = np.zeros(3)
    drag = np.zeros(3)
    if speed > 5.0:
        nose = quat_to_matrix(estimate.quaternion)[:, 2]
        vh = vel / speed
        aoa = float(np.degrees(np.arccos(np.clip(nose @ vh, -1.0, 1.0))))
        lat = nose - (nose @ vh) * vh
        ln = np.linalg.norm(lat)
        if ln > 1e-6:
            aero = tables.lateral_disturbance(speed, aoa) * lat / ln
        drag = (
            -0.5
            * float(model.density(estimate.position[2]))
            * float(model.drag_coeff(t_rel))
            * model.ref_area
            * speed
            * vel
            / m
        )
    return float(throttle) * thrust / m * n - GRAVITY * _Z + aero + drag
