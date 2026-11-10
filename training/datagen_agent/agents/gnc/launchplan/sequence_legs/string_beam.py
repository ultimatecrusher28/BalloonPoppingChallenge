"""sequence-legs-string: order the tour by STRING GEOMETRY, not by leg time.

A sibling of ``beam.py`` and ``lts_beam.py``; neither is modified. What
changes is the one thing REPORT section 7.9 identified as the remaining main
term -- the ORDER:

- ``beam.py`` expands to the nearest ``branch`` balloons and ranks nodes by
  ``(-depth, time)``. ``_leg`` reports MINIMUM TIME, and minimum time prefers
  targets that are far but straight ahead, so the search systematically picks
  long chords (measured 85.8 m at a 49.3 deg turn angle) even though short
  legs are the ones it accepts most often (section 7.9).
- Here the ORDER comes first and the legs are solved along it. Balloon
  positions from the predictor are a frozen rigid body in the cloud frame
  (a common translation fits the field to 3.7 m median over the whole 22 s
  cruise window, section 7.11), so a curvature-feasible polyline through them
  -- a string -- can be extracted once at launch and stays valid.
  ``string_geom.turn_chains`` is that extraction, and is exactly the code the
  offline measurement ran.
- Each leg then PINS its arrival velocity along the string's next direction
  (``lts.solve_leg`` with ``v_rel``), which is what makes a pass leave the
  vehicle already pointing down the string instead of having to turn into the
  next leg. Legs that cannot be solved pinned fall back to a free arrival.

Attribution rule: the string tours are ADDED to the candidate pool, and the
winner is chosen by the same swept-pop score ``beam.py`` already uses against
ALL released balloons. So the plan this candidate emits is never worse than
the baseline's BY THE PLAN'S OWN METRIC, and any bench difference is the
string ordering and nothing else. ``prefer_string`` > 0 requires the string
tour to beat the beam tour by that many pops before it is taken.

Only the INITIAL tour changes: ``replan_from`` delegates to the beam planner
(and the ``hybrid-insertion`` replanner builds its own beam instance anyway).
"""

from __future__ import annotations

import numpy as np

from ...interfaces import LaunchDecision, LaunchPlanner
from ...registry import register
from ...vehicle.limits import LimitTables
from . import string_geom
from .beam import BeamSequenceLegs
from .lts_beam import LtsSequenceLegs, _Node

_Z = np.array([0.0, 0.0, 1.0])


@register("launchplan", "sequence-legs-string")
class StringSequenceLegs(LaunchPlanner):
    def __init__(
        self,
        given_parameters,
        model,
        *,
        launch_time=48.5,
        max_rail_tilt_deg=30.0,
        v_cruise=22.7,
        c_lo=20.0,
        c_hi=70.0,
        n_chains=12,
        entry_offsets=3,
        pin_arrival=True,
        prefer_string=0,
        **kwargs,
    ):
        self.model = model
        self.tables = LimitTables()
        self.launch_time = float(launch_time)
        self.max_rail_tilt_deg = float(max_rail_tilt_deg)
        # The speed the turn radius is evaluated at. R = v^2 / a_lat(v) is the
        # only place the vehicle enters the string extraction; the measured
        # cruise speed (22.7 m/s, section 7.8) is the honest value.
        self.v_cruise = float(v_cruise)
        self.c_lo = float(c_lo)
        self.c_hi = float(c_hi)
        self.n_chains = int(n_chains)
        # A string's first member may be unreachable from the pad, so entering
        # it one or two balloons in is tried too.
        self.entry_offsets = int(entry_offsets)
        self.pin_arrival = bool(pin_arrival)
        self.prefer_string = int(prefer_string)
        self.elevation = float(given_parameters["environment"]["elevation"])
        kw = dict(kwargs)
        kw.update(launch_time=launch_time, max_rail_tilt_deg=max_rail_tilt_deg)
        self._lts = LtsSequenceLegs(given_parameters, model, **kw)
        self._beam = BeamSequenceLegs(given_parameters, model, **kw)
        self.last_sweep_score = 0
        # Attribution counters, read by the bench summary.
        self.string_wins = 0
        self.beam_wins = 0
        self.last_chain_n = 0
        # Both pools' best swept-pop score at the launch decision, so a tie
        # can be told apart from "no string tour was solvable at all".
        self.last_string_score = -1
        self.last_base_score = -1
        self.last_string_depth = 0

    def prepare(self, given_parameters: dict) -> None:
        self._lts.prepare(given_parameters)
        self._beam.prepare(given_parameters)

    # -- one tour along one string -------------------------------------
    def _tour(self, predictor, t_now, seq_ids, seq_pos):
        """Solve the legs of a fixed target order; ``None`` if the first leg
        cannot be flown at all. A short tour is kept: a string the vehicle can
        only get three balloons into is still a candidate."""
        pad = np.array([0.0, 0.0, self.elevation])
        node = _Node(t_rel=0.0, pos=pad, vel=np.zeros(3), axis=_Z)
        for idx, bid in enumerate(seq_ids):
            aim = None
            if self.pin_arrival and idx + 1 < len(seq_ids):
                d = seq_pos[idx + 1] - seq_pos[idx]
                n = np.linalg.norm(d)
                if n > 1e-6:
                    u = d / n

                    def aim(_bp, _bv, u=u):
                        return self.v_cruise * u

            from_pad = idx == 0
            got = self._lts._leg(
                predictor, int(bid), t_now, node, aim, from_pad=from_pad
            )
            if got is None and aim is not None:
                got = self._lts._leg(
                    predictor, int(bid), t_now, node, None, from_pad=from_pad
                )
            if got is None:
                break
            leg, p_end, v_end = got
            if from_pad:
                tilt = np.degrees(np.arccos(np.clip(leg.u_start[2], -1.0, 1.0)))
                if tilt > self.max_rail_tilt_deg:
                    return None
            node = _Node(
                t_rel=node.t_rel + leg.tau,
                pos=p_end,
                vel=v_end,
                axis=leg.u_end,
                targets=node.targets + (int(bid),),
                legs=node.legs + [leg],
            )
        return node if node.legs else None

    def _string_tours(self, predictor, t_now, ids):
        pos = predictor.states[ids, :3]
        R = self.v_cruise**2 / self.tables.a_lat_max(self.v_cruise)
        chains = string_geom.turn_chains(
            pos,
            R,
            c_lo=self.c_lo,
            c_hi=self.c_hi,
            top=self.n_chains,
            both_directions=True,
        )
        self.last_chain_n = max((c["n"] for c in chains), default=0)
        tours = []
        for ch in chains:
            members = ch["members"]
            for off in range(min(self.entry_offsets, max(1, len(members) - 1))):
                seq = members[off:]
                if len(seq) < 2:
                    break
                node = self._tour(
                    predictor, t_now, [int(ids[m]) for m in seq], pos[seq]
                )
                if node is not None:
                    tours.append(node)
        return tours

    # -- planning ------------------------------------------------------
    def plan(self, observation, predictor, *, deadline: float):
        t_now = float(observation["simulation_time"])
        if t_now < self.launch_time:
            return None
        ids = predictor.released_ids()
        if len(ids) == 0:
            return None
        base = self._beam.plan(observation, predictor, deadline=deadline)
        base_score = int(getattr(self._beam, "last_sweep_score", -1))

        self._lts._pred = predictor
        best, best_score = None, -1
        for node in self._string_tours(predictor, t_now, ids):
            ts, ps, _vs, _ac, _ax = self._lts._sample(node)
            score = self._lts._sweep_score(ts, ps, predictor, t_now)
            if score > best_score:
                best, best_score = node, score
        self.last_string_score = int(best_score)
        self.last_base_score = base_score
        self.last_string_depth = len(best.legs) if best is not None else 0
        if best is None or best_score <= base_score + self.prefer_string:
            self.beam_wins += 1
            self.last_sweep_score = base_score
            return base
        self.string_wins += 1

        u0 = best.legs[0].u_start
        tilt = min(
            np.degrees(np.arccos(np.clip(u0[2], -1.0, 1.0))), self.max_rail_tilt_deg
        )
        heading = (
            float(np.degrees(np.arctan2(u0[0], u0[1])) % 360.0)
            if np.hypot(u0[0], u0[1]) > 1e-9
            else 0.0
        )
        launch = LaunchDecision(
            t_launch=t_now,
            inclination_deg=90.0 - tilt,
            heading_deg=heading,
            max_rail_tilt_deg=self.max_rail_tilt_deg,
            first_target=best.legs[0].bid,
        )
        plan = self._lts._build([best], t_now, launch)
        self.last_sweep_score = int(getattr(self._lts, "last_sweep_score", 0))
        return plan if plan is not None else base

    def replan_from(self, position, velocity, axis, t_rel, launch, predictor):
        """In-flight re-search is the baseline's, untouched: this candidate
        changes the INITIAL tour and nothing else."""
        out = self._beam.replan_from(position, velocity, axis, t_rel, launch, predictor)
        self.last_sweep_score = int(getattr(self._beam, "last_sweep_score", 0))
        return out
