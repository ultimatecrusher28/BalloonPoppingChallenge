"""GncAgent: wires the five modules together.

Candidates are chosen by name from ``agent_kwargs`` (see ``registry.py``).
The agent owns the shared Budget (wall-time accounting per module, exported
via ``budget_report()``), the launch handshake (the launch step must co-issue
inclination/heading; TVC is ignored on that step), and the action-validity
guard (a NaN command would be dropped by the env; the last good action is
replayed instead).
"""

from __future__ import annotations

import numpy as np

from ...envs.balloon_world import check_action
from ..base_agent import BaseAgent
from . import follow, replan  # noqa: F401 (register)
from . import balloonnav as _balloonnav  # noqa: F401 (register)
from . import launchplan as _launchplan  # noqa: F401 (register)
from . import rocketnav as _rocketnav  # noqa: F401 (register)
from . import registry
from .budget import Budget
from .metrics import plan_metrics
from .vehicle.rocket_model import RocketModel


def _idle(launch=False, attitude=(90.0, 0.0), throttle=0.0):
    return {
        "launch": bool(launch),
        "launch_inclination_heading": np.array(attitude, dtype=float),
        "tvc": np.zeros(2),
        "roll": 0.0,
        "throttle": float(throttle),
    }


# limits_mode="given" (Discussion #165): the candidates GivenLimits was
# written to inject into. Anything else would either silently fall back to
# constructing a legacy LimitTables() inside the candidate (defeating the
# whole point) or simply ignore the injected object, so unknown/incompatible
# selections raise instead of degrading quietly.
_GIVEN_MODE_LAUNCHPLAN = {"sequence-legs-beam"}
_GIVEN_MODE_FOLLOWER = {"leg-resolve-feasible", "leg-resolve-track-ref"}
_GIVEN_MODE_INNER_LOOP = {"inner-loop-compensated"}


def _validate_given_mode(
    planner_name: str, follower: str, follower_kwargs: dict
) -> None:
    if planner_name not in _GIVEN_MODE_LAUNCHPLAN:
        raise ValueError(
            f"limits_mode='given' does not support launchplan={planner_name!r}; "
            f"supported: {sorted(_GIVEN_MODE_LAUNCHPLAN)}"
        )
    if follower not in _GIVEN_MODE_FOLLOWER:
        raise ValueError(
            f"limits_mode='given' does not support follower={follower!r}; "
            f"supported: {sorted(_GIVEN_MODE_FOLLOWER)}"
        )
    inner_loop = follower_kwargs.get("inner_loop", "inner-loop-compensated")
    if inner_loop not in _GIVEN_MODE_INNER_LOOP:
        raise ValueError(
            f"limits_mode='given' does not support inner_loop={inner_loop!r}; "
            f"supported: {sorted(_GIVEN_MODE_INNER_LOOP)}"
        )


class GncAgent(BaseAgent):
    """Five-module GNC agent; only observation + given_parameters are read."""

    def __init__(
        self,
        given_parameters,
        *,
        rocketnav="passthrough",
        balloonnav="extrapolate",
        launchplan="sequence-legs-greedy",
        follower="leg-resolve-rough",
        replanner=None,
        rocketnav_kwargs=None,
        balloonnav_kwargs=None,
        launchplan_kwargs=None,
        follower_kwargs=None,
        replanner_kwargs=None,
        limits_mode="legacy",
        given_turn_model="ramp",
        run_prepare=True,
        plan_call_cap_s=5.0,
        replan_call_cap_s=0.5,
        nav_call_cap_s=0.05,
        follow_call_cap_s=0.1,
        **_kwargs,
    ):
        super().__init__(given_parameters)
        gp = given_parameters
        self.model = RocketModel(gp)
        self.limits_mode = str(limits_mode)
        self.budget = Budget()
        self._caps = {
            "launchplan": float(plan_call_cap_s),
            "replan": float(replan_call_cap_s),
            "rocketnav": float(nav_call_cap_s),
            "follow": float(follow_call_cap_s),
        }

        launchplan_kwargs = dict(launchplan_kwargs or {})
        follower_kwargs = dict(follower_kwargs or {})
        replanner_kwargs = dict(replanner_kwargs or {})
        self._tables = None
        if self.limits_mode == "given":
            from .vehicle.given_limits import GivenLimits

            _validate_given_mode(launchplan, follower, follower_kwargs)
            # One shared physics-derived limits object (no legacy
            # LimitTables, no research/limits/data read) for the beam
            # planner, the feasible follower, and its compensated inner
            # loop. Injection takes precedence over any caller-supplied
            # "tables"/"turn_time" so the "no legacy LimitTables" invariant
            # holds regardless of what an eval config also happens to pass.
            if replanner not in (None, "", "none", "hybrid-insertion"):
                raise ValueError(
                    "given limits require hybrid-insertion or no replanner"
                )
            if (
                replanner == "hybrid-insertion"
                and replanner_kwargs.get("planner", "sequence-legs-beam")
                != "sequence-legs-beam"
            ):
                raise ValueError("given limits require a beam replanner")
            self._tables = GivenLimits(gp, self.model, turn_model=given_turn_model)
            if replanner == "hybrid-insertion":
                nested = dict(replanner_kwargs.get("planner_kwargs") or {})
                nested.update(tables=self._tables, turn_time=self._tables.turn_time)
                replanner_kwargs["planner_kwargs"] = nested
            launchplan_kwargs["tables"] = self._tables
            launchplan_kwargs["turn_time"] = self._tables.turn_time
            follower_kwargs["tables"] = self._tables
            follower_kwargs["turn_time"] = self._tables.turn_time
            inner_loop_kwargs = dict(follower_kwargs.get("inner_loop_kwargs") or {})
            inner_loop_kwargs["tables"] = self._tables
            follower_kwargs["inner_loop_kwargs"] = inner_loop_kwargs
        elif self.limits_mode != "legacy":
            raise ValueError(
                f"unknown limits_mode {limits_mode!r}; expected 'legacy' or 'given'"
            )

        build = registry.build
        self.nav = build(
            "rocketnav", rocketnav, gp, self.model, **dict(rocketnav_kwargs or {})
        )
        self.predictor = build(
            "balloonnav", balloonnav, gp, self.model, **dict(balloonnav_kwargs or {})
        )
        self.planner = build(
            "launchplan", launchplan, gp, self.model, **launchplan_kwargs
        )
        self.follower = build("follower", follower, gp, self.model, **follower_kwargs)
        self.replanner = (
            None
            if replanner in (None, "", "none")
            else build(
                "replanner", replanner, gp, self.model, **dict(replanner_kwargs or {})
            )
        )

        self.candidates = {
            "rocketnav": rocketnav,
            "balloonnav": balloonnav,
            "launchplan": launchplan,
            "follower": follower,
            "replanner": replanner or "none",
            "limits_mode": self.limits_mode,
        }
        if run_prepare:  # free window: __init__ runs before the env wall clock
            for module in self._modules():
                module.prepare(gp)

        # Diagnostics read by the balloon-scope harness / logger.
        self.plan = None
        self._t_launch = None
        self.attitude = None
        self.launch_depth = 0
        # First-order quantities of the flown plan (see metrics.py), merged
        # into the balloon-scope bench row by ``bench_metrics``. The burnout
        # state is the endgame half: a tour that burns out diving leaves the
        # balloon field within seconds and scores nothing after cutoff.
        self._plan_metrics: dict = {}
        self._burnout: dict = {}
        self.violation_log: list = []
        self._armed = False
        self._last_good = None
        self._estimate = None

    def _modules(self):
        out = [self.nav, self.predictor, self.planner, self.follower]
        if self.replanner is not None:
            out.append(self.replanner)
        return out

    # -- main loop -----------------------------------------------------
    def get_action(self, observation):
        if not self._armed:
            self.budget.arm()
            self._armed = True
        t = float(observation["simulation_time"])

        with self.budget.charge("balloonnav"):
            self.predictor.update(observation)

        if self._t_launch is None:
            with self.budget.charge("launchplan"):
                plan = self.planner.plan(
                    observation,
                    self.predictor,
                    deadline=self.budget.deadline(self._caps["launchplan"]),
                )
            if plan is None or plan.launch is None or t < plan.launch.t_launch:
                return _idle()
            return self._launch(plan, t)

        with self.budget.charge("rocketnav"):
            self._estimate = self.nav.update(
                observation, deadline=self.budget.deadline(self._caps["rocketnav"])
            )
        if not self._burnout and t - self._t_launch >= self.model.burn_time:
            self._burnout = {
                "burnout_vz": round(float(self._estimate.velocity[2]), 2),
                "burnout_z": round(float(self._estimate.position[2]), 1),
                "burnout_speed": round(
                    float(np.linalg.norm(self._estimate.velocity)), 2
                ),
            }
        violations = self.follower.drain_violations()
        self.violation_log.extend(violations)
        if self.replanner is not None:
            with self.budget.charge("replan"):
                self.plan = self.replanner.update(
                    self.plan,
                    observation,
                    self._estimate,
                    self.predictor,
                    violations,
                    deadline=self.budget.deadline(self._caps["replan"]),
                )
        with self.budget.charge("follow"):
            action = self.follower.act(
                self._estimate,
                self.plan,
                self.predictor,
                deadline=self.budget.deadline(self._caps["follow"]),
            )

        if check_action(action):  # non-empty: some field is unusable
            return self._last_good if self._last_good is not None else _idle()
        self._last_good = action
        return action

    def _launch(self, plan, t):
        self.plan = plan
        self._t_launch = t
        decision = plan.launch
        self.attitude = (decision.inclination_deg, decision.heading_deg)
        self.launch_depth = len(plan.pops)
        self._plan_metrics = plan_metrics(
            plan, self.model.burn_time, self.model, tables=self._tables
        )
        for module in self._modules():
            module.on_launch(decision, t)
        return _idle(launch=True, attitude=self.attitude, throttle=1.0)

    # -- diagnostics (debug/analysis only, never consumed in-loop) -----
    @property
    def bench_metrics(self) -> dict:
        """Plan-side quantities, the burnout state, and the replan ledger.

        Read once at the end of an episode by the balloon-scope harness. The
        replan pair matters on its own: a previous architecture proposed 14.7
        re-plans an episode and adopted 0.3 of them, so its search cost was
        almost entirely wasted.
        """
        out = dict(self._plan_metrics)
        out.update(self._burnout)
        consumers = {
            "planner": self.planner,
            "follower": self.follower,
            "replanner": getattr(self.replanner, "planner", None),
        }
        for label, consumer in consumers.items():
            stats = getattr(consumer, "intercept_stats", {})
            for metric in ("attempts", "hits"):
                out[f"window_recovery_{label}_{metric}"] = stats.get(metric, 0)
        r = self.replanner
        if r is not None and getattr(r, "proposed", None) is not None:
            out["replans_proposed"] = int(r.proposed)
            out["replans_adopted"] = int(r.adopted)
            for name in ("targets_dropped", "next_target_switches"):
                if getattr(r, name, None) is not None:
                    out[name] = int(getattr(r, name))
        return out

    def budget_report(self) -> dict:
        report = self.budget.report()
        report["candidates"] = dict(self.candidates)
        report["violations"] = len(self.violation_log)
        return report

    @property
    def estimated_states(self):
        e = self._estimate
        if e is None:
            return None
        return np.concatenate([e.position, e.velocity, e.quaternion, e.omega])
