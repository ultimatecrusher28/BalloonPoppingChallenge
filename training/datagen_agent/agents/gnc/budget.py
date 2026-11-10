"""Shared wall-time budget with per-module accounting.

The env truncates the WHOLE episode when its wall clock exceeds
``max_wall_time`` (600 s). That limit is not in given_parameters, so it is
hardcoded here with a safety factor. Agent ``__init__`` and env ``reset`` run
before the env starts its clock (the free window used by ``prepare()``);
everything inside ``get_action`` is charged.

Every module call is wrapped in ``charge(name)`` so the spend per module lands
in ``report()`` (exported to the balloon-scope ``/meta`` record) and overruns
can be attributed instead of guessed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

# Hardcoded: the env's max_wall_time is 600 s and not part of given_parameters.
ENV_WALL_LIMIT_S = 600.0
SAFETY_FACTOR = 0.85


class Budget:
    """Monotonic-clock accounting against the episode wall-time limit."""

    def __init__(self, limit_s: float = ENV_WALL_LIMIT_S * SAFETY_FACTOR):
        self.limit_s = float(limit_s)
        self.spent = {}  # module name -> accumulated seconds
        self._t0 = time.perf_counter()

    # -- spending -----------------------------------------------------
    @contextmanager
    def charge(self, module: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.spent[module] = self.spent.get(module, 0.0) + (
                time.perf_counter() - start
            )

    # -- reading ------------------------------------------------------
    @property
    def elapsed(self) -> float:
        """Wall seconds since the budget was armed (first get_action)."""
        return time.perf_counter() - self._t0

    def arm(self) -> None:
        """Start the clock (call on the first get_action, not in __init__:
        construction happens in the free window)."""
        self._t0 = time.perf_counter()

    @property
    def remaining(self) -> float:
        return self.limit_s - self.elapsed

    def deadline(self, cap_s: float) -> float:
        """perf_counter deadline for one module call: at most ``cap_s``, never
        past the episode budget."""
        return time.perf_counter() + max(0.0, min(cap_s, self.remaining))

    def report(self) -> dict:
        """Per-module spend for the /meta record."""
        return {
            "limit_s": round(self.limit_s, 1),
            "elapsed_s": round(self.elapsed, 2),
            "per_module_s": {k: round(v, 2) for k, v in sorted(self.spent.items())},
        }
