"""leg-resolve-rough: re-solve the current leg every step (skeleton).

Each control step the active leg's intercept is re-solved from the LATEST
rocket estimate and balloon prediction (constant-acceleration solution to
arrive at the window centre), so execution deviation is corrected here, not
by the replanner. Outside the leg's tube it emits a PlanViolation and keeps
flying. With legs exhausted (or absent) it tracks the reference tail, then
cuts throttle.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import Follower, PlanViolation, RocketEstimate
from .. import registry
from ..vehicle.rocket_model import GRAVITY

_Z = np.array([0.0, 0.0, 1.0])
MIN_T_GO = 0.15
ACCEL_CAP = 30.0  # m/s^2 command cap; the inner loop saturates honestly
VIOLATION_COOLDOWN_S = 1.0


@registry.register("follower", "leg-resolve-rough")
class LegResolveRough(Follower):
    def __init__(
        self,
        given_parameters,
        model,
        *,
        inner_loop="inner-loop-rough",
        inner_loop_kwargs=None,
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
        self.t_launch = None
        self.leg_idx = 0
        self._violations: list[PlanViolation] = []
        self._last_violation_t = -np.inf

    def on_launch(self, decision, t_launch: float) -> None:
        self.t_launch = float(t_launch)
        self.leg_idx = 0
        if hasattr(self.inner, "on_launch"):
            self.inner.on_launch(t_launch)

    # -- helpers -------------------------------------------------------
    def _sync_plan(self, plan):
        """A replanner may swap the plan; the leg cursor follows the plan."""
        if plan is not None and id(plan) != getattr(self, "_plan_id", None):
            self._plan_id = id(plan)
            self.leg_idx = 0

    def _advance_legs(self, t, plan, predictor):
        self._sync_plan(plan)
        legs = plan.legs or []
        while self.leg_idx < len(legs):
            leg = legs[self.leg_idx]
            status = int(predictor.status[leg.window.balloon_id])
            if status == 2 or t > leg.window.t_close:
                self.leg_idx += 1
                continue
            return leg
        return None

    def drain_violations(self):
        out, self._violations = self._violations, []
        return out

    # -- acting --------------------------------------------------------
    def act(
        self, estimate: RocketEstimate, plan, predictor, *, deadline: float
    ) -> dict:
        t = estimate.t
        t_rel = t - (self.t_launch if self.t_launch is not None else t)
        leg = None if plan is None else self._advance_legs(t, plan, predictor)

        if leg is not None:
            t_aim = 0.5 * (leg.window.t_open + leg.window.t_close)
            t_go = max(t_aim - t, MIN_T_GO)
            target = predictor.predict(leg.window.balloon_id, t + t_go)
            if isinstance(target, tuple):
                zem = target[0] - (estimate.position + estimate.velocity * t_go)
                a_cmd = 2.0 * zem / (t_go * t_go)
                mag = np.linalg.norm(a_cmd)
                if mag > ACCEL_CAP:
                    a_cmd *= ACCEL_CAP / mag
                self._check_tube(t, t_rel, leg, plan, estimate)
                action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
                return action

        # No live leg: track the reference tail while it lasts, then coast.
        if plan is not None and t_rel <= plan.ref.t_end:
            p_ref, v_ref, a_ref, _axis = plan.ref.state(t_rel)
            a_cmd = (
                a_ref
                + 2.0 * (p_ref - estimate.position)
                + 3.0 * (v_ref - estimate.velocity)
            )
            action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
            return action
        action, _achievable, _flags = self.inner.solve(-GRAVITY * _Z, estimate)
        return action  # zero net thrust: coast

    def _check_tube(self, t, t_rel, leg, plan, estimate):
        p_ref, _v, _a, _axis = plan.ref.state(t_rel)
        err = float(np.linalg.norm(estimate.position - p_ref))
        if err > leg.tube(t_rel) and t - self._last_violation_t > VIOLATION_COOLDOWN_S:
            self._violations.append(
                PlanViolation(
                    leg_id=leg.leg_id, t=t, dimension="position", magnitude=err
                )
            )
            self._last_violation_t = t


@registry.register("follower", "leg-resolve-feasible")
class LegResolveFeasible(LegResolveRough):
    """Wave-1 leg re-solve: the aim time slides inside the intercept window
    until the constant-acceleration solve fits the measured envelope (thrust
    magnitude + lateral authority), and an aim past the window's close is a
    timing PlanViolation flown best-effort. Defaults to the compensated inner
    loop; everything else (leg advance, tube check) inherits from rough.
    """

    LATE_GRACE_S = 4.0  # a missed window's balloon stays the target this
    # long past t_close before the leg is abandoned
    FREEZE_S = 1.2  # endgame: hold the frozen command open-loop

    # ``tables``/``turn_time``: injectable, default to a fresh LimitTables()
    # and vehicle.limits.turn_time (bit-identical to the pre-injection
    # behaviour). agent.py's limits_mode="given" (Discussion #165) injects a
    # shared GivenLimits instead, so this candidate never constructs a
    # legacy LimitTables in that mode.
    # eta 1.0: the PLANNER reserves margin (its own eta < 1); the follower's
    # feasibility test must accept anything physically flyable, or a plan
    # solved at the planner's margin is rejected the moment execution error
    # touches it and every aim slides late.
    # mid_phase "min-time": re-solve the intercept every step (wave-1 proven)
    # mid_phase "track-ref": PD+feedforward on the plan reference while far
    # from the window, min-time+freeze endgame unchanged (the wave-2 graft:
    # an optimized reference is worth following mid-leg, but the proven
    # terminal machinery stays).
    def __init__(
        self,
        given_parameters,
        model,
        *,
        inner_loop="inner-loop-compensated",
        inner_loop_kwargs=None,
        eta=1.0,
        use_terminal_zem=False,
        mid_phase="min-time",
        track_kp=4.0,
        track_kd=3.5,
        tables=None,
        turn_time=None,
        recover_intercept_windows=False,
        **kwargs,
    ):
        super().__init__(
            given_parameters,
            model,
            inner_loop=inner_loop,
            inner_loop_kwargs=inner_loop_kwargs,
            **kwargs,
        )
        self.recover_intercept_windows = bool(recover_intercept_windows)
        self.intercept_stats = {}
        self.mid_phase = str(mid_phase)
        self.track_kp = float(track_kp)
        self.track_kd = float(track_kd)
        from ..vehicle.rocket_model import GRAVITY as _G

        if tables is not None:
            self.tables = tables
        else:
            from ..vehicle.limits import LimitTables

            self.tables = LimitTables()
        if turn_time is not None:
            self.turn_time_fn = turn_time
        else:
            from ..vehicle.limits import turn_time as _default_turn_time

            self.turn_time_fn = _default_turn_time
        self.eta = float(eta)
        self.use_terminal_zem = bool(use_terminal_zem)
        self.elevation = float(given_parameters["environment"]["elevation"])
        self._need_dir = None
        self._g = _G

    def _advance_legs(self, t, plan, predictor):
        self._sync_plan(plan)
        legs = plan.legs or []
        while self.leg_idx < len(legs):
            leg = legs[self.leg_idx]
            status = int(predictor.status[leg.window.balloon_id])
            if status == 2 or t > leg.window.t_close + self.LATE_GRACE_S:
                self.leg_idx += 1
                continue
            return leg
        return None

    TURN_TRANSIENT_S = 0.4  # the gimbal cannot re-point instantly; model
    # the first 0.4 s of a committed turn as thrust
    # still along the current nose axis

    def _sweeps_current(self, estimate, predictor, bid, a_next, tau):
        """Does the path under the next leg's command still sweep within the
        pop radius of the CURRENT balloon? Two-phase model: thrust along the
        current nose during the turn transient, then the new command."""
        from .inner_loop import quat_to_matrix

        nose = quat_to_matrix(estimate.quaternion)[:, 2]
        t_rel = estimate.t - (
            self.t_launch if self.t_launch is not None else estimate.t
        )
        a_now = float(self.model.thrust(t_rel)) / float(
            self.model.mass(t_rel)
        ) * nose - self._g * np.array([0.0, 0.0, 1.0])
        d_min = np.inf
        p = estimate.position.copy()
        v = estimate.velocity.copy()
        s = 0.0
        for _ in range(int(min(tau + 0.4, 2.0) / 0.05)):
            a = a_now if s < self.TURN_TRANSIENT_S else a_next
            p = p + v * 0.05 + 0.5 * a * 0.0025
            v = v + a * 0.05
            s += 0.05
            target = predictor.predict(bid, estimate.t + s)
            if isinstance(target, tuple):
                d_min = min(d_min, float(np.linalg.norm(p - target[0])))
        return d_min <= 1.0

    def _feasible(self, a_cmd, estimate, t_rel):
        """Demanded thrust accel within eta*T/m. No separate lateral cap:
        TVC turns the thrust vector anywhere within about a second (the
        transient is the planner's turn_time and the continuous re-solve),
        and the measured f_perp keeps growing past the 18 deg AoA boundary,
        so a sustained-lateral cap under-states the vehicle and slid every
        aim late."""
        avail = (
            self.eta * float(self.model.thrust(t_rel)) / float(self.model.mass(t_rel))
        )
        need = np.asarray(a_cmd) + np.array([0.0, 0.0, self._g])
        return bool(np.linalg.norm(need) <= avail)

    GUARD_EXIT_SPEED = 12.0  # hold the rail axis at full thrust until here:
    # the measured reach-set protocol, stable from
    # 60 deg up; ZEM at low speed noses over.

    def on_launch(self, decision, t_launch: float) -> None:
        super().on_launch(decision, t_launch)
        from ....envs.balloon_world import get_initial_attitude
        from .inner_loop import quat_to_matrix

        q = np.asarray(
            get_initial_attitude(decision.inclination_deg, decision.heading_deg), float
        )
        self._rail_axis = quat_to_matrix(q)[:, 2]
        self._climbing = True

    GROUND_MARGIN_M = 12.0

    def _ground_guarded(self, a_cmd, estimate, t_rel):
        """Never fly a command whose short-horizon path floors out: a chased
        low balloon that is missed otherwise drags the rocket into the ground
        (observed: crash 6 s after a 3 m miss). Overrides with a climb."""
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
                0.97 * avail - self._g,
            ]
        )

    def _solve_intercept(
        self, estimate, t, t_rel, avail, bid, predictor, *, deadline=float("inf")
    ):
        """``(tau, a_cmd)`` for a min-time intercept of ``bid``, or None if
        unreachable inside the primitive's horizon.

        Shared min-time primitive: earliest full-thrust arrival. Arriving
        before the window is FINE (the balloon is physically there); a leg
        the primitive cannot reach inside its horizon returns None, so the
        caller abandons it with a violation and moves to the next leg.

        Turn-transient-aware refinement (the planner's own leg model): when
        the needed thrust direction is far from the current nose, the naive
        solve's arrival keeps receding while we turn. Solve instead from the
        post-turn drifted state and command THAT direction, which also holds
        it steady through the turn.

        Overridable: a follower whose plan was built with a different leg
        primitive re-solves with that primitive instead here, so the plan
        and the execution agree on what a leg costs (see
        ``follow/lts_resolve.py``). The "commit-and-turn" lookahead in
        ``act()`` below calls ``min_time_accel`` directly rather than through
        this hook - it peeks at the NEXT leg before this one's arrival, and
        routing it through an overridden primitive is a different change
        than this hook covers.
        """
        if hasattr(self.tables, "set_context"):
            self.tables.set_context(t_rel, float(estimate.position[2]))
        from .intercept import min_time_accel

        sol = (
            None
            if avail <= 1e-6
            else min_time_accel(
                estimate.position,
                estimate.velocity,
                t,
                lambda t_abs: predictor.predict(bid, t_abs),
                avail,
                recover_windows=self.recover_intercept_windows,
                stats=self.intercept_stats,
            )
        )
        if sol is not None:
            from .inner_loop import quat_to_matrix

            nose = quat_to_matrix(estimate.quaternion)[:, 2]
            need = sol[1] + self._g * np.array([0.0, 0.0, 1.0])
            need /= max(np.linalg.norm(need), 1e-9)
            t_turn = self.turn_time_fn(float(np.arccos(np.clip(nose @ need, -1, 1))))
            if t_turn > 0.3:
                drift = estimate.position + estimate.velocity * t_turn
                sol2 = min_time_accel(
                    drift,
                    estimate.velocity,
                    t + t_turn,
                    lambda t_abs: predictor.predict(bid, t_abs),
                    avail,
                    recover_windows=self.recover_intercept_windows,
                    stats=self.intercept_stats,
                )
                if sol2 is not None:
                    sol = (t_turn + sol2[0], sol2[1])
        return sol

    def act(self, estimate, plan, predictor, *, deadline: float) -> dict:
        t = estimate.t
        t_rel = t - (self.t_launch if self.t_launch is not None else t)
        leg = None if plan is None else self._advance_legs(t, plan, predictor)

        if getattr(self, "_climbing", False):
            if np.linalg.norm(estimate.velocity) >= self.GUARD_EXIT_SPEED:
                self._climbing = False
            else:
                avail = float(self.model.thrust(t_rel)) / float(self.model.mass(t_rel))
                a_cmd = avail * self._rail_axis - np.array([0.0, 0.0, self._g])
                action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
                return action

        while leg is not None:
            avail = (
                self.eta
                * float(self.model.thrust(t_rel))
                / float(self.model.mass(t_rel))
            )
            bid = leg.window.balloon_id
            sol = self._solve_intercept(
                estimate, t, t_rel, avail, bid, predictor, deadline=deadline
            )
            if sol is None:
                if t - self._last_violation_t > VIOLATION_COOLDOWN_S:
                    self._violations.append(
                        PlanViolation(
                            leg_id=leg.leg_id,
                            t=t,
                            dimension="timing",
                            magnitude=float("inf"),
                        )
                    )
                    self._last_violation_t = t
                self.leg_idx += 1
                leg = self._advance_legs(t, plan, predictor)
                continue
            tau, a_cmd = sol
            if (
                t + tau > leg.window.t_close
                and t - self._last_violation_t > VIOLATION_COOLDOWN_S
            ):
                self._violations.append(
                    PlanViolation(
                        leg_id=leg.leg_id,
                        t=t,
                        dimension="timing",
                        magnitude=t + tau - leg.window.t_close,
                    )
                )
                self._last_violation_t = t

            # Endgame direction settling: inside 2.5 s the commanded thrust
            # direction may rotate at most 40 deg/s, so the nose is settled
            # before the freeze (misalignment at freeze onset was the
            # 3 m terminal miss: 10-20 deg of thrust error over 1.2 s).
            if tau < 2.5:
                need = a_cmd + self._g * np.array([0.0, 0.0, 1.0])
                mag = float(np.linalg.norm(need))
                if mag > 1e-9:
                    new_dir = need / mag
                    dt_act = max(t - getattr(self, "_dir_t", t), 0.005)
                    if (
                        getattr(self, "_dir_leg", None) == self.leg_idx
                        and self._need_dir is not None
                    ):
                        ang = float(
                            np.arccos(np.clip(self._need_dir @ new_dir, -1.0, 1.0))
                        )
                        max_step = np.radians(40.0) * dt_act
                        if ang > max_step:
                            axis_r = np.cross(self._need_dir, new_dir)
                            axis_r /= max(np.linalg.norm(axis_r), 1e-12)
                            c, s = np.cos(max_step), np.sin(max_step)
                            new_dir = (
                                self._need_dir * c
                                + np.cross(axis_r, self._need_dir) * s
                            )
                            a_cmd = mag * new_dir - self._g * np.array([0.0, 0.0, 1.0])
                    self._need_dir = new_dir
                    self._dir_leg = self.leg_idx
                    self._dir_t = t
            else:
                self._need_dir = None

            # Terminal ZEM handoff: engages when the perpendicular correction
            # is affordable (required accel vs budget, per plan; no hardcoded
            # switch time). Offline rig (terminal_guidance_test): ZEM N=3
            # p50 miss 1.50 m / pop 50% vs min-time+freeze 1.98 m / 32%.
            # In-flight handoff at tau<2.0 with the affordability gate alone
            # measured -2.50 paired (0/18/2) despite the law winning the
            # offline rig: the ZEM's own closest-approach t_go and the
            # min-time arrival often disagree mid-leg, and P-perp drops
            # arrival-time control. Off by default; the gated variant below
            # additionally requires the two clocks to agree.
            handed_off = False
            if self.use_terminal_zem and tau < 1.2:
                from .terminal_zem import terminal_accel

                term = terminal_accel(
                    estimate.position,
                    estimate.velocity,
                    t,
                    lambda t_abs: predictor.predict(bid, t_abs),
                    avail,
                )
                if term is not None and abs(term[0] - tau) < 0.4 * tau + 0.1:
                    tau, a_cmd = term
                    handed_off = True
            if not handed_off and tau < self.FREEZE_S:
                # Freeze the endgame: near arrival the re-solve rotates its
                # command faster than the gimbal follows (measured 2-5 m
                # misses); the frozen constant-acceleration arc sweeps
                # through on its own (legacy freeze_s, re-validated here).
                if getattr(self, "_frozen_leg", None) != self.leg_idx:
                    self._frozen_leg = self.leg_idx
                    self._frozen_cmd = a_cmd.copy()
                a_cmd = self._frozen_cmd

            # Wave-2 graft: while far from the window, follow the plan's
            # reference instead of re-solving (an optimized ref encodes a
            # better path than the const-accel chase); the endgame below
            # stays min-time+freeze either way.
            if (
                self.mid_phase == "track-ref"
                and not handed_off
                and tau > 1.5
                and plan is not None
                and t_rel <= plan.ref.t_end
            ):
                p_ref, v_ref, a_ref, _axis = plan.ref.state(t_rel)
                a_cmd = (
                    a_ref
                    + self.track_kp * (p_ref - estimate.position)
                    + self.track_kd * (v_ref - estimate.velocity)
                )

            # Commit-and-turn: a pop is swept geometry, so once the path
            # under the NEXT leg's command still passes within the pop radius
            # of the current balloon, turn now instead of waiting for the
            # hit. This is where the per-balloon rhythm is won (the dead
            # post-pop turn was measured at 0.7-1.4 s).
            legs = plan.legs or []
            if tau < 1.5 and self.leg_idx + 1 < len(legs):
                nxt = legs[self.leg_idx + 1]
                nbid = nxt.window.balloon_id
                if int(predictor.status[nbid]) == 1:
                    from .intercept import min_time_accel

                    sol_n = min_time_accel(
                        estimate.position,
                        estimate.velocity,
                        t,
                        lambda t_abs: predictor.predict(nbid, t_abs),
                        avail,
                        recover_windows=self.recover_intercept_windows,
                        stats=self.intercept_stats,
                    )
                    if sol_n is not None and self._sweeps_current(
                        estimate, predictor, bid, sol_n[1], tau
                    ):
                        a_n = self._ground_guarded(sol_n[1], estimate, t_rel)
                        action, _a, _f = self.inner.solve(a_n, estimate)
                        return action
            self._check_tube(t, t_rel, leg, plan, estimate)
            a_cmd = self._ground_guarded(a_cmd, estimate, t_rel)
            action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
            return action

        if plan is not None and t_rel <= plan.ref.t_end:
            p_ref, v_ref, a_ref, _axis = plan.ref.state(t_rel)
            a_cmd = (
                a_ref
                + 2.0 * (p_ref - estimate.position)
                + 3.0 * (v_ref - estimate.velocity)
            )
            a_cmd = self._ground_guarded(a_cmd, estimate, t_rel)
            action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
            return action
        a_cmd = self._ground_guarded(
            -self._g * np.array([0.0, 0.0, 1.0]), estimate, t_rel
        )
        action, _achievable, _flags = self.inner.solve(a_cmd, estimate)
        return action


@registry.register("follower", "leg-resolve-track-ref")
def _track_ref_factory(given_parameters, model, **kwargs):
    """leg-resolve-feasible with the mid-leg phase following the plan ref
    (the wave-2 graft for optimized references)."""
    kwargs.setdefault("mid_phase", "track-ref")
    return LegResolveFeasible(given_parameters, model, **kwargs)
