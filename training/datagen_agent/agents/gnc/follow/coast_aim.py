"""leg-resolve-coast-aim: spend the burn's last seconds aiming the coast.

``leg-resolve-feasible`` stops flying the moment the plan's legs run out: it
tracks the reference tail, then commands zero net thrust. Measured on sc1
seeds 0-19 that happens about 3.4 s before burnout, and the roughly 10 s of
ballistic flight that follows (the episode ends at ground impact, not at
``max_time``) scores nothing.

This candidate adds ONE branch, taken only when the parent would have gone
idle: with legs exhausted and burn time left, solve a thrust-then-ballistic
intercept (``coast_intercept``) and fly it. Arrival may be after burnout,
which is the part the plan cannot express - ``beam.py``'s ``_leg`` rejects
any leg with ``t0 + tau > burn_time`` by construction.

Everything else - leg advance, tube check, min-time endgame, freeze,
commit-and-turn, ground guard - is inherited untouched. With
``coast_enable=False`` the branch is never entered and the class is the
parent, which is the off-proof this is benched against.
"""

from __future__ import annotations

import numpy as np

from .. import registry
from .coast_intercept import select_target, solve_a_net, thrust_need
from .leg_resolve import LegResolveFeasible


@registry.register("follower", "leg-resolve-coast-aim")
class LegResolveCoastAim(LegResolveFeasible):
    """Parent plus a terminal thrust-then-ballistic intercept.

    ``window_s``      how early before burnout the branch may engage. It only
                      ever engages with no live leg, so this bounds the work,
                      not the behaviour.
    ``max_coast_s``   how far past burnout an arrival may be planned. 2.0 s is
                      where the measured point-mass coast residual (p50 1.48 m
                      at 2 s) spends the 1.5 m pop radius, REPORT section
                      7.14 - a measured horizon, NOT a tuned number.
    ``allow_coast``   False restricts arrivals to the burn itself, which
                      isolates "use the leftover thrust" from "score after
                      burnout" in a paired bench.
    ``fix_late_legs`` re-solve a LIVE leg whose min-time arrival is past
                      burnout as thrust-then-coast. The parent flies such a
                      leg with a command it cannot finish, so this changes
                      behaviour only where the parent was already writing a
                      cheque the engine cannot cash.
    """

    def __init__(
        self,
        given_parameters,
        model,
        *,
        coast_enable=True,
        window_s=8.0,
        max_coast_s=2.0,
        allow_powered=True,
        allow_coast=True,
        max_candidates=24,
        reselect_s=1.0,
        fix_late_legs=True,
        late_margin_s=0.1,
        **kwargs,
    ):
        super().__init__(given_parameters, model, **kwargs)
        self.coast_enable = bool(coast_enable)
        self.window_s = float(window_s)
        self.max_coast_s = float(max_coast_s)
        self.allow_powered = bool(allow_powered)
        self.allow_coast = bool(allow_coast)
        self.max_candidates = int(max_candidates)
        self.reselect_s = float(reselect_s)
        self.fix_late_legs = bool(fix_late_legs)
        self.late_margin_s = float(late_margin_s)
        self.burn_time = float(model.burn_time)
        self._coast_target = None  # (bid, t_arrive)
        self._coast_picked_t = -np.inf
        self.coast_engaged_t = None  # diagnostics only
        self.late_leg_fixes = 0  # diagnostics only

    # -- the two new branches -------------------------------------------
    def act(self, estimate, plan, predictor, *, deadline: float) -> dict:
        if self.coast_enable and not getattr(self, "_climbing", False):
            t_rel = estimate.t - (
                self.t_launch if self.t_launch is not None else estimate.t
            )
            s_left = self.burn_time - t_rel
            if 1e-3 < s_left <= self.window_s:
                leg = (
                    None
                    if plan is None
                    else self._advance_legs(estimate.t, plan, predictor)
                )
                action = (
                    self._coast_act(estimate, predictor, t_rel, s_left)
                    if leg is None
                    else self._late_leg_act(estimate, predictor, leg, t_rel, s_left)
                )
                if action is not None:
                    return action
        return super().act(estimate, plan, predictor, deadline=deadline)

    def _late_leg_act(self, estimate, predictor, leg, t_rel, s_left):
        """A live leg whose min-time arrival falls PAST burnout.

        ``intercept.min_time_accel`` holds its acceleration to arrival, which
        is a command the vehicle cannot complete once the engine has cut: the
        parent flies the first ``s_left`` seconds of it and then whatever
        ballistic arc that happens to leave. Re-solving the same target as
        thrust-then-coast keeps the same intent but makes the whole command
        flyable. Returns None - deferring to the parent, unchanged - whenever
        the arrival is inside the burn or no coast-aware solution exists.
        """
        if not (self.allow_coast and self.fix_late_legs):
            return None
        from .intercept import min_time_accel

        avail = float(self.model.thrust(t_rel)) / float(self.model.mass(t_rel))
        if avail <= 1e-6:
            return None
        bid = leg.window.balloon_id
        sol = min_time_accel(
            estimate.position,
            estimate.velocity,
            estimate.t,
            lambda t_abs: predictor.predict(bid, t_abs),
            avail,
        )
        if sol is None or sol[0] <= s_left + self.late_margin_s:
            return None
        pick = select_target(
            estimate.position,
            estimate.velocity,
            estimate.t,
            s_left,
            avail,
            [bid],
            predictor.predict,
            self.model,
            max_coast_s=self.max_coast_s,
            allow_powered=False,
            allow_coast=True,
        )
        if pick is None:
            return None
        self.late_leg_fixes += 1
        if self.coast_engaged_t is None:
            self.coast_engaged_t = float(estimate.t)
        a_cmd = self._ground_guarded(pick[2], estimate, t_rel)
        action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
        return action

    # -- helpers --------------------------------------------------------
    def _candidates(self, estimate, predictor):
        """Released, unpopped balloons, nearest first.

        Reads ``predictor.status`` and ``predictor.states``, both of which
        come from ``observation`` - the same source the legs use.
        """
        live = np.flatnonzero(np.asarray(predictor.status) == 1)
        if live.size == 0:
            return live
        pos = np.asarray(predictor.states)[live, :3]
        order = np.argsort(np.linalg.norm(pos - estimate.position, axis=1))
        return live[order[: self.max_candidates]]

    def _coast_act(self, estimate, predictor, t_rel, s_left):
        avail = float(self.model.thrust(t_rel)) / float(self.model.mass(t_rel))
        if avail <= 1e-6:
            return None
        tgt = self._coast_target
        if tgt is not None:
            bid, t_arrive = tgt
            stale = int(predictor.status[bid]) != 1 or t_arrive - estimate.t <= 0.05
            if stale:
                tgt = self._coast_target = None
        if tgt is None and estimate.t - self._coast_picked_t >= self.reselect_s:
            self._coast_picked_t = estimate.t
            pick = select_target(
                estimate.position,
                estimate.velocity,
                estimate.t,
                s_left,
                avail,
                self._candidates(estimate, predictor),
                predictor.predict,
                self.model,
                max_coast_s=self.max_coast_s,
                allow_powered=self.allow_powered,
                allow_coast=self.allow_coast,
            )
            if pick is None:
                return None
            self._coast_target = (pick[0], pick[1])
            if self.coast_engaged_t is None:
                self.coast_engaged_t = float(estimate.t)
            tgt = self._coast_target
        if tgt is None:
            return None

        bid, t_arrive = tgt
        target = predictor.predict(int(bid), t_arrive)
        if not isinstance(target, tuple):
            self._coast_target = None
            return None
        sol = solve_a_net(
            estimate.position,
            estimate.velocity,
            target[0],
            t_arrive - estimate.t,
            s_left,
            self.model,
        )
        if sol is None:
            self._coast_target = None
            return None
        a_net, miss = sol
        if miss > 0.5 or thrust_need(a_net) > avail:
            # The aim has drifted out of the envelope; drop it and let the
            # next step re-select rather than fly a command that misses.
            self._coast_target = None
            return None
        a_cmd = self._ground_guarded(a_net, estimate, t_rel)
        action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
        return action


@registry.register("follower", "leg-resolve-burn-tail")
def _burn_tail_factory(given_parameters, model, **kwargs):
    """coast-aim with arrivals restricted to the burn: the control arm that
    separates "use the leftover thrust" from "score after burnout"."""
    kwargs.setdefault("allow_coast", False)
    return LegResolveCoastAim(given_parameters, model, **kwargs)
