"""Module contracts for the five-module GNC agent.

Five modules, each with swappable candidates behind these interfaces:

- ``LaunchPlanner``   (launchplan/) launch decision + initial tour. The launch
  decision is a JOINT choice of time x attitude x first target: probes showed
  rail tilt alone buys ~0.2 s of first-pop time while first-target choice owes
  ~4 s (research/limits/REPORT.md section 1.2), and waiting until the field is
  nearly fully released costs nothing (+0.70 pops paired, section 1.1).
- ``Replanner``       (replan/) reacts to newly released balloons. Correcting
  execution deviation is NOT its job; that belongs to the follower, and a
  deviation the follower cannot absorb surfaces as a ``PlanViolation``.
- ``RocketNav``       (rocketnav/) rocket state estimation from sensors only.
- ``BalloonPredictor``(balloonnav/) balloon prediction. Balloon observations
  are ground truth in all scored scenarios, so this is a predictor, not a
  filter.
- ``Follower``        (follow/) flies the plan; owns the inner loop that turns
  an acceleration command into TVC/throttle against the measured limits table
  (unmodeled lateral aero reaches 20-30% of the commanded acceleration above
  30 m/s, section 1.3).

Shared rules:

- Every planning/estimation call takes a ``deadline`` (a value of
  ``time.perf_counter()``) from the shared wall-time budget (``budget.py``);
  the episode is truncated wholesale at the env's 600 s wall limit, which is
  NOT in given_parameters and must be kept with margin.
- Heavy precomputation belongs in ``prepare()``: agent ``__init__`` and env
  ``reset`` run outside the wall-time budget (a legal free window).
- All scoring and search objectives count swept-segment pops against ALL
  released balloons (``pops.py`` mirrors the env test) - never leg counts.
- Candidates are chosen by name through ``registry.py``; composite candidates
  (e.g. an estimator with a fallback) are themselves registered names, so any
  combination can be A/B-compared from an eval config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


# ----------------------------------------------------------------- launch
@dataclass(frozen=True)
class LaunchDecision:
    """The one-shot launch command. Frozen: a replanner cannot alter it by
    type, and after the launch step the env ignores it anyway (the launch step
    must co-issue inclination/heading and its TVC command is discarded; the
    ~2 steps after launch are near-uncontrolled - that responsibility sits
    with the launch planner, not the follower).

    ``max_rail_tilt_deg`` is explicit because the env allows 0-90 deg while
    the legacy code silently self-imposed 15. If clamping changes the
    attitude, the planner must RE-SOLVE the first leg for the clamped
    attitude, not just clip the number (REPORT section 1.2).
    """

    t_launch: float  # simulation time to issue the launch step
    inclination_deg: float  # 90 = vertical
    heading_deg: float  # compass bearing of the lean
    max_rail_tilt_deg: float  # cap the solution was solved under
    first_target: int | None = None  # balloon id the first leg aims at


@dataclass(frozen=True)
class ReleasePoint:
    """Pre-release knowledge of a balloon: its release POINT is visible from
    t=0 (scattered ~ +/-100 m around the site); its release TIME is not
    observable until status flips. ``predict()`` returns this for status==0
    so planners can reason about future targets without inventing a time.
    """

    balloon_id: int
    position: np.ndarray  # (3,) launch-frame release point
    release_time_known: bool = False


# ----------------------------------------------------------------- plan parts
@dataclass
class InterceptWindow:
    """A pop opportunity as an inequality, not a point: any approach passing
    within ``miss_radius`` of the balloon during [t_open, t_close] with an
    arrival speed/axis inside the bands scores. Point-target equality
    constraints (7 equalities for a 1.5 m sphere) were the legacy mistake.
    """

    balloon_id: int
    t_open: float
    t_close: float
    balloon_state_at: Callable[[float], tuple[np.ndarray, np.ndarray]]
    # (pos, vel) of the balloon at time t, from the predictor snapshot the
    # plan was built against.
    miss_radius: float = 1.5
    arrival_speed_band: tuple[float, float] | None = None  # m/s (lo, hi)
    arrival_axis_band: tuple[np.ndarray, float] | None = None
    # (unit vector, max angle rad): approach-direction cone, when it matters.


@dataclass
class PopExpectation:
    """One pop the plan claims, at when and with how much geometric margin.
    The list is the plan's own scorecard; realized pops are attributed against
    it for the execution-gap ledger.
    """

    balloon_id: int
    t: float
    margin_m: float  # predicted closest-approach margin (< 1.5)


class Reference(Protocol):
    """Time-parameterized reference trajectory: the follower's contract.

    ``state(t)`` returns (pos, vel, accel, body_axis) at time-since-launch t
    (scalars or vectorized over an array of t). Legs are only one way to build
    a Reference; followers and replanners must be written against
    (ref, pops), never against leg internals.
    """

    def state(self, t): ...
    @property
    def t_end(self) -> float: ...


@dataclass
class Leg:
    """One planned intercept leg (sequence-legs construction detail).

    ``tube(t)`` is the per-time allowed tracking error radius: inside the tube
    the follower absorbs deviations silently; outside it the follower emits a
    ``PlanViolation`` and keeps flying its best effort while the replanner
    decides. ``slew_time_required`` is the attitude turn time this leg needs
    at its start (rest-to-rest 90 deg is about 1.0 s), so cost models and
    feasibility checks share one number.
    """

    leg_id: int
    window: InterceptWindow
    t_start: float
    t_end: float
    u_start: np.ndarray  # (3,) body-axis unit vector at t_start
    u_end: np.ndarray  # (3,) body-axis unit vector at t_end
    slew_time_required: float
    tube: Callable[[float], float]


@dataclass(frozen=True)
class PlanViolation:
    """The follower left a leg's tube (or timing/attitude band): the fact is
    attributed, logged, and handed to the replanner. It is a report, not an
    exception - flight continues.
    """

    leg_id: int
    t: float
    dimension: str  # "position" | "timing" | "attitude"
    magnitude: float  # how far outside, in the dimension's unit


@dataclass
class Plan:
    """What the launch planner and replanner exchange with the follower.

    ``ref`` + ``pops`` is the contract; ``legs`` is optional structure for
    candidates that have it (sequence-legs) and None for those that don't
    (trajectory-opt). ``launch`` is None once airborne.
    """

    launch: LaunchDecision | None
    ref: Reference
    pops: list[PopExpectation]
    legs: list[Leg] | None = None


# ----------------------------------------------------------------- estimates
@dataclass
class RocketEstimate:
    """13-state estimate plus honesty about its error.

    ``quality`` holds expected 1-sigma error scales (attitude in rad,
    position in m, velocity in m/s) - floats, not booleans, because the
    legacy failure mode was an attitude that stayed finite while wrong by
    36-145 deg (p90, sc2/3). Consumers gate on scale, and composite
    candidates (mekf+complementary-fallback) switch on it. ``source`` names
    the candidate that produced this estimate for attribution.
    """

    t: float  # simulation time of validity
    position: np.ndarray  # (3,) launch frame
    velocity: np.ndarray  # (3,) launch frame
    quaternion: np.ndarray  # (4,) scalar-first body->world
    omega: np.ndarray  # (3,) body rates rad/s
    quality: dict[str, float] = field(
        default_factory=lambda: {"attitude": 0.0, "position": 0.0, "velocity": 0.0}
    )
    source: str = ""


@dataclass
class BalloonParams:
    """Per-balloon identified physical parameters (MC draws mass with 25%
    sigma; drag coefficient is not in given_parameters). Identified from the
    observed truth trajectory; ``confidence`` in [0, 1] reflects how much
    trajectory the fit has seen.
    """

    balloon_id: int
    buoyancy_mass_ratio: float
    drag_scale: float
    confidence: float


@dataclass
class WindCoverage:
    """What the wind-field estimate actually knows: outside
    [z_lo, z_hi] the field falls back to the prior with inflated
    uncertainty, and consumers must see that, not a silent extrapolation.
    """

    z_lo: float
    z_hi: float
    n_balloons: int  # distinct balloons feeding the fit
    quality: str  # "prior" | "sparse" | "good"


# ----------------------------------------------------------------- module ABCs
class GncModule(ABC):
    """Base for all five modules: construction is cheap, ``prepare()`` is the
    expensive precompute run in the free window (before the wall clock
    starts), and it must be safe to skip (a candidate that needs it declares
    so and degrades if skipped).
    """

    def prepare(self, given_parameters: dict) -> None:
        """Heavy one-time precomputation; default is a no-op."""

    def on_launch(self, decision: LaunchDecision, t_launch: float) -> None:
        """Called once when the launch step is issued. The launch attitude is
        the estimator's initial condition (sensors are nan before launch) and
        the follower's time origin; default is a no-op."""


class RocketNav(GncModule):
    """Rocket state estimation. Consumes only ``observation`` content."""

    @abstractmethod
    def update(self, observation: dict, *, deadline: float) -> RocketEstimate:
        """Advance the estimate to the observation's time and return it."""


class BalloonPredictor(GncModule):
    """Balloon prediction from ground-truth balloon observations."""

    @abstractmethod
    def update(self, observation: dict) -> None:
        """Ingest this step's balloon_states/balloon_status."""

    @abstractmethod
    def predict(self, balloon_id: int, t: float):
        """(pos, vel) at simulation time t for a released balloon
        (status==1); a ``ReleasePoint`` for status==0 (release time unknown);
        None for a popped/landed balloon (status==2).
        """

    def identify(self, balloon_id: int) -> BalloonParams | None:
        """Identified per-balloon parameters, when the candidate builds them."""
        return None

    def coverage(self) -> WindCoverage | None:
        """Wind-field coverage metadata, when the candidate estimates wind."""
        return None


class LaunchPlanner(GncModule):
    """Owns the launch decision and the initial plan. Called every pre-launch
    step; returns None to keep waiting (default posture: wait late,
    REPORT section 1.1) or a Plan whose ``launch`` is set.
    """

    @abstractmethod
    def plan(
        self, observation: dict, predictor: BalloonPredictor, *, deadline: float
    ) -> Plan | None: ...


class Replanner(GncModule):
    """In-flight replanning around newly released balloons and reported
    violations. Must return a Plan with the same (immutable) launch field;
    returning the input plan unchanged is the common case.
    """

    @abstractmethod
    def update(
        self,
        plan: Plan,
        observation: dict,
        estimate: RocketEstimate,
        predictor: BalloonPredictor,
        violations: list[PlanViolation],
        *,
        deadline: float,
    ) -> Plan: ...


class Follower(GncModule):
    """Flies the plan's reference. Owns execution-deviation correction and
    the inner loop; emits PlanViolations instead of replanning itself.
    """

    @abstractmethod
    def act(
        self,
        estimate: RocketEstimate,
        plan: Plan,
        predictor: BalloonPredictor,
        *,
        deadline: float,
    ) -> dict:
        """The env action dict for this step."""

    def drain_violations(self) -> list[PlanViolation]:
        """Violations detected since the last drain (attributed + logged by
        the agent, then handed to the replanner)."""
        return []


class InnerLoop(ABC):
    """Acceleration command -> actuator solution, against measured limits.

    Returns ``(action, achievable_accel, saturation_flags)``: the achieved
    acceleration is reported honestly (the 2D speed x AoA table bounds the
    lateral authority; unmodeled aero reaches 4.8 m/s^2 at 40-50 m/s), so the
    guidance above can re-aim instead of silently missing. Gimbal rate limit
    (60 deg/s, full sweep 0.5 s - the dominant actuator dynamic) and the
    sc3/4 first-order lag are compensated separately.
    """

    @abstractmethod
    def solve(
        self, accel_cmd: np.ndarray, estimate: RocketEstimate
    ) -> tuple[dict, np.ndarray, dict]: ...
