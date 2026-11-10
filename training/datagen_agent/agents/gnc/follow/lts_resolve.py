"""Opt-in free-arrival thrust-integral follower experiment.

Uses vehicle.lts.free_leg, the same constant-direction family (B=0) as the
LTS planner with arrival_aim=None. Unlike the baseline, it integrates future
thrust/mass and omits the explicit turn-drift refinement. This does NOT test
curved LTS flight or prove that model mismatch causes the score deficit.

Endgame freeze, direction settling, ground guard and min-time next-target
lookahead remain inherited. Thus this is consistency of the current-leg
solve only. It does not follow a pinned arrival-velocity LTS plan.

The horizon matches the baseline intercept's 26 seconds, avoiding a separate
8-second horizon change. Search resolution and work are bounded. Deadline
checks prevent starting further LTS solves; the bounded vectorized solve is
not preemptible. On deadline exhaustion use the baseline solve.
"""

from __future__ import annotations

import time

import numpy as np

from .. import registry
from ..vehicle import lts
from .leg_resolve import LegResolveFeasible

_Z = np.array([0.0, 0.0, 1.0])


@registry.register("follower", "leg-resolve-lts")
class LegResolveLts(LegResolveFeasible):
    """Swap only the current-leg intercept model; default OFF."""

    def __init__(
        self, given_parameters, model, *, lts_tau_max=26.0, lts_d_tau=0.05, **kwargs
    ):
        super().__init__(given_parameters, model, **kwargs)
        self.lts_tau_max = float(lts_tau_max)
        self.lts_d_tau = float(lts_d_tau)
        if not 0.05 <= self.lts_tau_max <= 26.0:
            raise ValueError("lts_tau_max must be in [0.05, 26]")
        if not 0.01 <= self.lts_d_tau <= 0.25:
            raise ValueError("lts_d_tau must be in [0.01, 0.25]")

    def _solve_intercept(
        self, estimate, t, t_rel, avail, bid, predictor, *, deadline=float("inf")
    ):
        """Free-arrival LTS leg from the current state to ``bid``.

        Returns ``(tau, a_cmd)`` at ``s=0`` of the solved leg (the command
        to fly THIS step; the leg is re-solved fresh next step) or None when
        no feasible arrival exists inside ``lts_tau_max``. ``a_cmd`` uses the
        same net-acceleration convention as the parent's ``min_time_accel``
        (gravity already folded in), so it drops straight into the shared
        endgame/freeze/ground-guard machinery below in ``act()``.
        """
        if time.perf_counter() >= deadline:
            return super()._solve_intercept(
                estimate, t, t_rel, avail, bid, predictor, deadline=deadline
            )
        if avail <= 1e-6:
            return None
        target = predictor.predict(bid, t)
        if not isinstance(target, tuple):
            return None
        bp, bv = target
        if bp is None or bv is None:
            return None
        if not np.isfinite(np.r_[bp, bv]).all():
            return None
        sol = lts.free_leg(
            self.model,
            np.array([t_rel]),
            estimate.position[None],
            estimate.velocity[None],
            np.asarray(bp, float)[None],
            np.asarray(bv, float)[None],
            eta=self.eta,
            tau_max=self.lts_tau_max,
            d_tau=self.lts_d_tau,
        )
        if not bool(sol["ok"][0]):
            return None
        tau = float(sol["tau"][0])
        u0 = np.asarray(sol["A"][0], float)  # B = 0: direction is constant
        k0 = float(np.asarray(sol["k"])[0])
        alpha0 = float(lts.alpha(self.model, t_rel, self.eta))
        a_cmd = k0 * alpha0 * u0 - self._g * _Z
        return tau, a_cmd


@registry.register("follower", "leg-resolve-lts-turn")
class LegResolveLtsTurn(LegResolveLts):
    """Single-factor follow-up: restore the baseline's turn-drift heuristic.

    Like the baseline, drift position at unchanged velocity while turning,
    then re-solve to the future target using the current burn-age model.
    This deliberately retains that approximation to isolate the heuristic.
    """

    def _solve_intercept(
        self, estimate, t, t_rel, avail, bid, predictor, *, deadline=float("inf")
    ):
        from dataclasses import replace
        from .inner_loop import quat_to_matrix
        from ..vehicle.limits import turn_time

        sol = super()._solve_intercept(
            estimate, t, t_rel, avail, bid, predictor, deadline=deadline
        )
        if sol is None or time.perf_counter() >= deadline:
            return sol
        nose = quat_to_matrix(estimate.quaternion)[:, 2]
        need = sol[1] + self._g * _Z
        need /= max(np.linalg.norm(need), 1e-9)
        t_turn = turn_time(float(np.arccos(np.clip(nose @ need, -1, 1))))
        if t_turn > 0.3:
            drift = replace(
                estimate, position=estimate.position + estimate.velocity * t_turn
            )
            refined = super()._solve_intercept(
                drift, t + t_turn, t_rel, avail, bid, predictor, deadline=deadline
            )
            if refined is not None:
                return t_turn + refined[0], refined[1]
        return sol
