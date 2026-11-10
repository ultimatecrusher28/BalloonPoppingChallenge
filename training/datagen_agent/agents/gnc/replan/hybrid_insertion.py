"""hybrid-insertion: periodic remaining-tour re-search + violation response.

Two triggers, per the module contract (new balloons and infeasible plans are
the replanner's job; small execution deviation is the follower's):

- every ``replan_every`` seconds: newly released balloons have appeared and
  the chained leg model has drifted; re-search the remaining tour from the
  estimated state (the wave-1 debugging showed the plan after 1-2 passes is
  stale enough to cascade into misses).
- immediately on a reported timing violation.

The new plan keeps the immutable LaunchDecision and is adopted whenever its
swept-pop score is at least the incumbent's remaining score.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import Plan, Replanner
from ..registry import register

_Z = np.array([0.0, 0.0, 1.0])


@register("replanner", "hybrid-insertion")
class HybridInsertion(Replanner):
    def __init__(
        self,
        given_parameters,
        model,
        *,
        replan_every=1.0,
        planner="sequence-legs-beam",
        planner_kwargs=None,
        **_kwargs,
    ):
        from .. import registry

        self.model = model
        self.replan_every = float(replan_every)
        self.planner = registry.build(
            "launchplan", planner, given_parameters, model, **dict(planner_kwargs or {})
        )
        self._t_launch = None
        self._last = None
        # Searches actually run vs plans actually swapped in: the pair says
        # whether re-planning is buying anything or just spending CPU.
        self.proposed = 0
        self.adopted = 0
        # Churn, not just count: dropping targets the incumbent still claimed
        # is the plan "jumping around", and a change of the IMMEDIATE next
        # target is the one that moves the follower's leg cursor mid-pass.
        self.targets_dropped = 0
        self.next_target_switches = 0

    def on_launch(self, decision, t_launch: float) -> None:
        self._t_launch = float(t_launch)
        self._last = None

    def update(
        self,
        plan: Plan,
        observation: dict,
        estimate,
        predictor,
        violations,
        *,
        deadline: float,
    ) -> Plan:
        if self._t_launch is None or plan is None or plan.launch is None:
            return plan
        t = float(estimate.t)
        due = self._last is None or t - self._last >= self.replan_every
        if not (due or violations):
            return plan
        self._last = t
        t_rel = t - self._t_launch
        if t_rel > self.model.burn_time - 1.5:
            return plan  # nothing plannable on the last thrust breath
        from ..follow.inner_loop import quat_to_matrix

        axis = quat_to_matrix(estimate.quaternion)[:, 2]
        new = self.planner.replan_from(
            estimate.position, estimate.velocity, axis, t_rel, plan.launch, predictor
        )
        if new is None:
            return plan
        self.proposed += 1
        new_score = getattr(self.planner, "last_sweep_score", 0)
        # Strictly better only: adopting ties re-plans every second and
        # thrashes the follower's leg cursor mid-pass.
        if new_score > self._remaining_score(plan, predictor, t):
            self.adopted += 1
            self._count_churn(plan, new, t)
            return new
        return plan

    def _count_churn(self, plan: Plan, new: Plan, t: float) -> None:
        old_ahead = [p.balloon_id for p in plan.pops if p.t >= t]
        new_ahead = [p.balloon_id for p in new.pops if p.t >= t]
        self.targets_dropped += len(set(old_ahead) - set(new_ahead))
        if old_ahead and new_ahead and old_ahead[0] != new_ahead[0]:
            self.next_target_switches += 1

    @staticmethod
    def _remaining_score(plan: Plan, predictor, t: float) -> int:
        """The incumbent's claim: planned pops still ahead and alive."""
        alive = 0
        for p in plan.pops:
            if p.t >= t and int(predictor.status[p.balloon_id]) == 1:
                alive += 1
        return alive
