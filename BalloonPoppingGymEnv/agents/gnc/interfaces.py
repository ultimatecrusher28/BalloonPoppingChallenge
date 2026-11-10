"""Module contracts of the planner agent: the launch decision, the rocket state estimate, the navigation filter
(RocketNav) and the inner loop (InnerLoop)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LaunchDecision:
    """The one-shot launch command (the launch step co-issues inclination and heading; the env ignores its TVC
    command)."""

    t_launch: float  # simulation time to issue the launch step
    inclination_deg: float  # 90 = vertical
    heading_deg: float  # compass bearing of the lean
    max_rail_tilt_deg: float  # cap the solution was solved under


@dataclass
class RocketEstimate:
    """13-state estimate. ``quality`` holds expected 1-sigma error scales (attitude in rad, position in m, velocity in
    m/s); ``source`` names the filter that produced it."""

    t: float  # simulation time of validity
    position: np.ndarray  # (3,) launch frame
    velocity: np.ndarray  # (3,) launch frame
    quaternion: np.ndarray  # (4,) scalar-first body->world
    omega: np.ndarray  # (3,) body rates rad/s
    quality: dict[str, float] = field(
        default_factory=lambda: {"attitude": 0.0, "position": 0.0, "velocity": 0.0}
    )
    source: str = ""


class GncModule(ABC):
    """Base of the navigation filter."""

    def on_launch(self, decision: LaunchDecision, t_launch: float) -> None:
        """Called once when the launch step is issued. The launch attitude is the estimator's initial condition (sensors
        are nan before launch); default is a no-op."""


class RocketNav(GncModule):
    """Rocket state estimation. Consumes only ``observation`` content."""

    @abstractmethod
    def update(self, observation: dict, *, deadline: float) -> RocketEstimate:
        """Advance the estimate to the observation's time and return it."""


class InnerLoop(ABC):
    """Acceleration command -> actuator solution. Returns ``(action, achievable_accel, saturation_flags)``."""

    @abstractmethod
    def solve(
        self, accel_cmd: np.ndarray, estimate: RocketEstimate
    ) -> tuple[dict, np.ndarray, dict]: ...
