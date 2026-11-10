"""Opt-in reset of terminal freeze state when the replanner replaces a plan."""

from .. import registry
from .leg_resolve import LegResolveFeasible


@registry.register("follower", "leg-resolve-fresh-freeze")
class LegResolveFreshFreeze(LegResolveFeasible):
    """A leg index identifies a leg only within its own plan.

    Without invalidation, replacing a plan can reuse an old target's frozen
    acceleration when the new plan reaches the same index. Reset only the
    freeze key: all guidance, direction settling and planning are unchanged.
    """

    def _sync_plan(self, plan):
        changed = plan is not None and id(plan) != getattr(self, "_plan_id", None)
        super()._sync_plan(plan)
        if changed:
            self._frozen_leg = None
