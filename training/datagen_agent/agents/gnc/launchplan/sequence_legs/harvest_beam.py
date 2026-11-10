"""sequence-legs-harvest: keep the early tour LOW, so phase A exists at all.

A sibling of ``beam.py``, ``lts_beam.py`` and ``string_beam.py``; none of them
is modified. What changes is where the tour is allowed to be during the first
seconds of flight.

REPORT section 8.2 measured the gap this addresses. The best rival
(``taoyuan17``, 17 pops) scores its first five balloons at tb 2.78-11.34 and
z_agl 3.3-66 m, and all five were released AFTER its launch - it rides up
alongside freshly released balloons a few tens of metres from the pad. Our
``first_pop_tb`` is 7.65 s at roughly 150 m, and we score nothing in that
phase. ``research/limits/early_harvest.py`` measured the opportunity offline
against real balloon trajectories.

Why beam cannot get there by launching earlier (REPORT section 7.12 measured
-0.55 pops at 40 s and -1.00 at 44 s): ``_first_leg`` is a min-time climb to
the nearest ALREADY-RELEASED balloon, and min time to a balloon that is
itself climbing at 6.09 m/s spends nearly all of a T/W 1.21 vehicle's margin
going up. The tour therefore leaves the low layer within a couple of seconds
whatever time it launches, and the balloons that release in the next ten
seconds - the only pad-adjacent, low-altitude targets that ever exist - are
already far below it. Launch time alone moves the same climb-out earlier;
that is exactly what section 7.12 measured, and why this candidate changes
the tour shape and the launch time together.

REPORT section 7.15 is the reason this is a RANKING term rather than a bigger
candidate set: widening the re-scored candidate pool 8x was bit-identical, so
the search is not mis-ranking a good tour it already has - the good tours are
not being generated. Charging altitude in the beam's own ranking changes which
family of tours survives the beam, which is the generation step.

Two independent switchable terms, both defaulting to OFF so that with
``launch_time`` left at the baseline's value this candidate is bit-identical
to ``sequence-legs-beam``:

- ``z_penalty_s_per_m``: seconds charged per metre that a node sits above
  ``z_ref_m`` AGL, applied only while the node's pop time is inside
  ``harvest_s`` seconds of launch. The beam then prefers a tour that scores
  low and early over one that scores high and early at the same depth.
- ``low_branch``: add this many of the LOWEST released balloons to each
  node's branch candidate set, on top of the nearest ``branch``. Distance
  rank and altitude rank are different orders once the field spans 300 m of
  altitude, and the harvest targets are the low ones.
- ``trigger_z_agl_m``/``trigger_range_m``: hold the launch until a balloon
  that has ALREADY released is sitting low and near the pad, instead of
  launching on a fixed clock. ``early_harvest.py`` measured the earliest
  geometrically possible first pop (policy-free, one leg from rest inside the
  thrust ball) at 3.7-4.2 s mean for launches in 34-44 s but 1.9-5.5 s across
  seeds - the good moments exist and are observable, because a status flip is
  observable the step it happens. ``launch_deadline`` bounds the wait, so a
  seed where the trigger never fires launches on the old clock.

Used as the replanner's planner too (``replanner_kwargs: {planner:
sequence-legs-harvest}``), so the 1 Hz re-search that actually picks up a
newly released balloon inherits the same preference; ``replan_from`` is
inherited from beam unchanged and reads both terms through ``_rank``/``_near``.
"""

from __future__ import annotations

import numpy as np

from ...registry import register
from .beam import BeamSequenceLegs


@register("launchplan", "sequence-legs-harvest")
class HarvestSequenceLegs(BeamSequenceLegs):
    def __init__(
        self,
        given_parameters,
        model,
        *,
        z_penalty_s_per_m=0.0,
        z_ref_m=40.0,
        harvest_s=12.0,
        low_branch=0,
        trigger_z_agl_m=0.0,
        trigger_range_m=60.0,
        launch_deadline=48.5,
        **kwargs,
    ):
        super().__init__(given_parameters, model, **kwargs)
        # Seconds per metre of node altitude above ``z_ref_m`` AGL. 0.0 makes
        # ``_rank`` return exactly what beam's returns.
        self.z_penalty_s_per_m = float(z_penalty_s_per_m)
        # The altitude below which nothing is charged: the harvest layer. The
        # rival's five phase-A pops sit at 3.3-66 m AGL (REPORT section 8.2).
        self.z_ref_m = float(z_ref_m)
        # How long after launch the altitude term applies. Past this the tour
        # is in the cruise phase and altitude is not a cost.
        self.harvest_s = float(harvest_s)
        # Extra branch candidates taken by LOWEST altitude rather than by
        # nearest distance. 0 leaves ``_near`` as the baseline expression.
        self.low_branch = int(low_branch)
        # Release-triggered launch. 0.0 disables the whole gate, and ``plan``
        # is then literally beam's ``plan``.
        self.trigger_z_agl_m = float(trigger_z_agl_m)
        self.trigger_range_m = float(trigger_range_m)
        self.launch_deadline = float(launch_deadline)
        # Attribution: did the trigger fire, or did the deadline?
        self.launched_on_trigger = None

    # -- ranking -------------------------------------------------------
    def _rank(self, node):
        """Beam ranking cost in seconds, plus an altitude charge in the
        harvest window.

        With every term off this returns ``node.t_rel`` itself, so the sorts
        in ``_search`` are the exact baseline sorts and no float arithmetic
        is introduced.
        """
        base = super()._rank(node)
        if self.z_penalty_s_per_m == 0.0 or node.t_rel > self.harvest_s:
            return base
        excess = float(node.pos[2]) - (self.elevation + self.z_ref_m)
        if excess <= 0.0:
            return base
        return base + self.z_penalty_s_per_m * excess

    # -- candidate set -------------------------------------------------
    def _near(self, predictor, cand, pos, t_abs, k):
        """Nearest ``k`` candidates, plus the ``low_branch`` lowest ones.

        Nearest-by-distance and lowest-by-altitude are different orders here:
        the released field spans ~300 m of altitude by the time we launch, so
        a balloon 40 m away and 200 m up outranks one 90 m away and 5 m up on
        distance, while only the second is a harvest target. With
        ``low_branch=0`` this is literally the baseline slice, returned
        untouched.

        Altitude is read from the predictor's CURRENT snapshot, not from a
        prediction at ``t_abs``: this is the branch heuristic, and the leg
        solve re-predicts at the real pop time anyway (the same reasoning
        beam applies to the uncapped distance slice).
        """
        idx = super()._near(predictor, cand, pos, t_abs, k)
        if self.low_branch <= 0 or len(cand) == 0:
            return idx
        z = predictor.states[cand, 2]
        low = np.argsort(z)[: self.low_branch]
        return np.unique(np.concatenate([np.asarray(idx, int), np.asarray(low, int)]))

    # -- release-triggered launch --------------------------------------
    def trigger_ready(self, predictor) -> bool:
        """Is a freshly released balloon sitting low and near the pad?

        Reads only ``predictor.states``/``status``, which come from
        ``observation["balloon_states"]``/``["balloon_status"]`` - a status
        flip is visible on the step it happens, so "this balloon just
        released" is legal in-loop knowledge. What is NOT legal, and is not
        used here, is WHICH still-grounded balloon releases next: the env
        shuffles the release order in
        ``__reset_balloon_release_sequence``.
        """
        ids = predictor.released_ids()
        if len(ids) == 0:
            return False
        p = predictor.states[ids, :3]
        low = (p[:, 2] - self.elevation) <= self.trigger_z_agl_m
        near = np.linalg.norm(p[:, :2], axis=1) <= self.trigger_range_m
        return bool((low & near).any())

    def plan(self, observation, predictor, *, deadline: float):
        """Beam's plan, optionally held back until the trigger fires.

        With ``trigger_z_agl_m == 0.0`` this is a straight delegation and the
        candidate is bit-identical to beam.
        """
        if self.trigger_z_agl_m <= 0.0:
            return super().plan(observation, predictor, deadline=deadline)
        t_now = float(observation["simulation_time"])
        if t_now >= self.launch_time and t_now < self.launch_deadline:
            if not self.trigger_ready(predictor):
                return None
            if self.launched_on_trigger is None:
                self.launched_on_trigger = True
        elif self.launched_on_trigger is None and t_now >= self.launch_deadline:
            self.launched_on_trigger = False
        return super().plan(observation, predictor, deadline=deadline)
