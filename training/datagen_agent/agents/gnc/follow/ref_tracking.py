"""ref-tracking: PD + feedforward follower for optimized references.

The wave-2 pair of trajectory-opt (plan spec): a collocation reference
already satisfies the actuator's direction-rate bound, so the follower's job
is TRACKING, not re-solving; the min-time re-solve follower measured -0.40
paired when flying refit plans because its commands fight the boundary-tight
reference. Law: a_cmd = a_ref + Kp (p_ref - p) + Kd (v_ref - v), with the
leg windows only advancing attribution and the ground guard shared with
leg-resolve.
"""

from __future__ import annotations

import numpy as np

from .. import registry
from ..interfaces import Follower, PlanViolation, RocketEstimate
from ..vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])


@registry.register("follower", "ref-tracking")
class RefTrackingFollower(Follower):
    GROUND_MARGIN_M = 12.0

    def __init__(
        self,
        given_parameters,
        model,
        *,
        inner_loop="inner-loop-compensated",
        inner_loop_kwargs=None,
        kp=4.0,
        kd=3.5,
        **_kwargs,
    ):
        self.model = model
        self.inner = registry.build(
            "inner_loop",
            inner_loop,
            given_parameters,
            model,
            **dict(inner_loop_kwargs or {}),
        )
        self.kp = float(kp)
        self.kd = float(kd)
        self.elevation = float(given_parameters["environment"]["elevation"])
        self.t_launch = None
        self._violations: list[PlanViolation] = []
        self._last_violation_t = -np.inf

    def on_launch(self, decision, t_launch: float) -> None:
        self.t_launch = float(t_launch)
        if hasattr(self.inner, "on_launch"):
            self.inner.on_launch(t_launch)

    def drain_violations(self):
        out, self._violations = self._violations, []
        return out

    def _ground_guarded(self, a_cmd, estimate, t_rel):
        z0 = float(estimate.position[2])
        vz = float(estimate.velocity[2])
        az = float(a_cmd[2])
        z_min = min(z0 + vz * d + 0.5 * az * d * d for d in (0.5, 1.0, 1.5, 2.0))
        if z_min > self.elevation + self.GROUND_MARGIN_M or vz > 5.0:
            return a_cmd
        avail = float(self.model.thrust(t_rel)) / float(self.model.mass(t_rel))
        return np.array(
            [
                -0.25 * estimate.velocity[0],
                -0.25 * estimate.velocity[1],
                0.97 * avail - GRAVITY,
            ]
        )

    def act(
        self, estimate: RocketEstimate, plan, predictor, *, deadline: float
    ) -> dict:
        t = estimate.t
        t_rel = t - (self.t_launch if self.t_launch is not None else t)
        if plan is not None and t_rel <= plan.ref.t_end:
            p_ref, v_ref, a_ref, _axis = plan.ref.state(t_rel)
            err = p_ref - estimate.position
            a_cmd = a_ref + self.kp * err + self.kd * (v_ref - estimate.velocity)
            err_n = float(np.linalg.norm(err))
            if err_n > 25.0 and t - self._last_violation_t > 1.0:
                self._violations.append(
                    PlanViolation(leg_id=-1, t=t, dimension="position", magnitude=err_n)
                )
                self._last_violation_t = t
        else:
            a_cmd = -GRAVITY * _Z  # coast
        a_cmd = self._ground_guarded(a_cmd, estimate, t_rel)
        action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
        return action
