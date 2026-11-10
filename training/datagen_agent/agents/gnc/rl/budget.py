"""Time-budget-aware planning for PlannerAgent (agent kwarg ``budget``; None = off, the agent is unchanged).

Official limits per episode: the env truncates the episode once the wall time since the end of reset() reaches
max_wall_time (600 s, env stepping included), and PlannerAgent stops replanning once its own get_action time reaches
wall_budget_s (450 s). Plan cost depends on the machine, its load and the flight phase, so one fixed config either
leaves a fast machine idle or hits the 450 s gate on a slow one, after which no balloon is chosen any more.

PlanBudget chooses, at every decision step, the most expensive level of a ladder (planner knobs horizon, long_horizon,
depth, beam, branch and the replan interval) whose projected cost for the rest of the flight still fits. Inputs are
only the agent's own wall clock, the flight time since launch, the navigation estimate (altitude, vertical speed) and
given parameters (burn_time, max_time); no machine constant.

Cost model (seconds of agent time, all wall times multiplied by ``wall_scale``):
    plan at level l and flight time tau:   c = s * w_l * phi(tau; H_l)
s is the machine/load speed, learnt online as an EMA of c_obs / (w_l phi) over in-flight plans; w_l is the level's
per-plan cost relative to the base (1.0 unless the ladder entry gives a measured "w": a replan_dt change does not change
the per-plan cost, a horizon change does, and 1.0 is the safe guess). phi is the phase shape measured on art2
(research/sc40/budget/NOTES.md): 1 until burn_time - H (rollouts end inside the burn), then linear from k_h down to
k_burn at burnout (rollout steps past burnout take the cheap coast branch), then k_coast until the ground.
Projection for level l:
    P_l = s * w_l * nu_l * int_tau^tau_end phi(u; H_l) du,    nu_l = min(1 / dec_dt, 1 / replan_dt_l + lambda)
lambda = rate of target-popped/lost replans so far (prior 0.5/s). tau_end = burn_time + tail_s during the burn, the
drag-free ballistic time to the ground after it, both capped by max_time. The non-plan agent time (navigation, inner
loop, predictor feed) and the env's own stepping time are projected at their measured in-flight rates per second.
Level l fits if
    P_l <= min(target_s - A - o R, elapsed_s - E - (o + e) R)  (x (1 - hyst) when it is an upgrade)
and the next single plan, next_safety * s * w_l * phi(tau), fits in what is left now. A = agent time, E = time since
the agent's first get_action (= the env clock), o, e = non-plan agent / env rate, R = tau_end - tau.
When no level fits: plan only when the target is popped or lost ("event", at the last level's knobs) while one such
plan still fits; then "greedy": no rollouts, target = the balloon with the smallest rough reach time (as
BeamPlanner._cands), flown with the deep guidance variant. The official 450 s gate (real time, unscaled) still stops
plan() calls in every mode.
"""

from __future__ import annotations

import math
import time

G0 = 9.80665
CAST = dict(
    horizon=float, long_horizon=float, depth=int, beam=int, branch=int, replan_dt=float
)
# default ladder: the base config with the replan interval multiplied (the cheapest pop loss per second saved, NOTES.md)
DEFAULT_MULT = (1, 2, 3, 5, 7, 10, 15, 20)
EVENT, GREEDY = "event", "greedy"


class PlanBudget:
    def __init__(self, spec, base, *, burn_time, max_time, dec_dt):
        sp = dict(spec or {})
        self.target_s = float(sp.pop("target_s", 420.0))
        self.elapsed_s = float(sp.pop("elapsed_s", 560.0))
        self.scale = float(
            sp.pop("wall_scale", 1.0)
        )  # research only: emulate a machine this many times slower
        self.alpha = float(sp.pop("alpha", 0.15))
        self.hyst = float(sp.pop("hyst", 0.08))
        self.tail_s = float(sp.pop("tail_s", 10.0))
        # phase shape: k_h = value just after burn_time - H (1.0 = the shape tested in closed loop, Slurm 1209/1210;
        # the 10 s-horizon traces fit k_h 0.6, k_burn 0.25, k_coast 0.1, surrogate only, NOTES.md §5)
        self.k_h = float(sp.pop("k_h", 1.0))
        self.k_burn = float(sp.pop("k_burn", 0.5))
        self.k_coast = float(sp.pop("k_coast", 0.3))
        self.lost_prior = float(sp.pop("lost_prior", 0.5))
        self.next_safety = float(sp.pop("next_safety", 2.0))
        self.event = bool(sp.pop("event", True))
        self.greedy = bool(sp.pop("greedy", True))
        self.trace = (
            [] if sp.pop("trace", False) else None
        )  # research: (tau, level, plan ms) per plan in stats()
        lad = sp.pop("ladder", None)
        if sp:
            raise ValueError(f"budget: unknown keys {sorted(sp)}")
        self.burn, self.max_time, self.dec_dt = (
            float(burn_time),
            float(max_time),
            float(dec_dt),
        )
        base = {k: CAST[k](v) for k, v in base.items()}
        if lad is None:
            lad = [
                dict(replan_dt=round(m * base["replan_dt"], 6)) for m in DEFAULT_MULT
            ]
        self.levels = []
        for L in lad:
            L = dict(L or {})
            w = float(L.pop("w", 1.0))
            bad = set(L) - set(CAST)
            if bad:
                raise ValueError(f"budget ladder: unknown keys {sorted(bad)}")
            lv = dict(base)
            lv.update({k: CAST[k](v) for k, v in L.items()})
            lv["w"] = w
            self.levels.append(lv)
        if not self.levels:
            raise ValueError("budget ladder is empty")
        ev = dict(self.levels[-1])
        ev["replan_dt"] = math.inf
        self.ev_level = ev
        # episode state
        self.t_first = None
        self.launch = None  # (tau, A, E, t) at the first in-flight decision
        self.s = None
        self.plan_sum = 0.0  # scaled plan time since launch
        self.n_lost = 0
        self.cur = 0
        self.last_plan_t = -math.inf
        n = len(self.levels)
        self.n_plan = [0] * (n + 1)  # per level, last = event plans
        self.t_plan = [0.0] * (n + 1)  # real seconds
        self.dec = [0] * (n + 2)  # decisions per chosen level / event / greedy
        self.n_greedy = 0
        self.trans = []  # (tau, from, to)

    # -- clocks -------------------------------------------------------------------------
    def tick(self):
        if self.t_first is None:
            self.t_first = time.monotonic()

    def clocks(self, wall_used):
        return self.scale * wall_used, self.scale * (time.monotonic() - self.t_first)

    # -- cost shape ---------------------------------------------------------------------
    def phi(self, u, H):
        p = self.burn - H
        if u < p:
            return 1.0
        if u < self.burn:
            return self.k_h - (self.k_h - self.k_burn) * (u - p) / H
        return self.k_coast

    def phi_int(self, a, b, H):
        if b <= a:
            return 0.0
        p, q = self.burn - H, self.burn
        tot = max(0.0, min(b, p) - a)
        u1, u2 = max(a, p), min(b, q)
        if u2 > u1:  # linear part (its own formula: phi(q) itself is the coast value)
            f1 = self.k_h - (self.k_h - self.k_burn) * (u1 - p) / H
            f2 = self.k_h - (self.k_h - self.k_burn) * (u2 - p) / H
            tot += (u2 - u1) * 0.5 * (f1 + f2)
        return tot + self.k_coast * max(0.0, b - max(a, q))

    def tau_end(self, tau, t, alt, vz):
        if tau < self.burn:
            te = self.burn + self.tail_s
        else:
            h = max(float(alt), 0.0)
            te = tau + (vz + math.sqrt(vz * vz + 2.0 * G0 * h)) / G0
        return min(te, tau + max(0.0, self.max_time - t))

    # -- decision -----------------------------------------------------------------------
    def level(self, i):
        return self.ev_level if i == len(self.levels) else self.levels[i]

    def decide(self, tau, t, wall_used, alt, vz):
        """Mode for this decision step: (index, dt). index < len(levels): plan level (due every dt or on a lost
        target), len(levels): event (plan only on a lost target), len(levels) + 1: greedy (no plan())."""
        A, E = self.clocks(wall_used)
        if self.launch is None:
            self.launch = (tau, A, E, t)
        tau0, A0, E0, tl = self.launch
        span = tau - tau0
        if span > 0.5:
            o = max(0.0, A - A0 - self.plan_sum) / span
            e = max(0.0, (E - A) - (E0 - A0)) / span
        else:  # nothing measured in flight yet: the pre-launch env rate
            o, e = 0.0, max(0.0, E0 - A0) / max(tl, 1.0)
        R = max(0.0, self.tau_end(tau, t, alt, vz) - tau)
        room = min(self.target_s - A - o * R, self.elapsed_s - E - (o + e) * R)
        hard = min(self.target_s - A, self.elapsed_s - E)
        n = len(self.levels)
        if self.s is None:  # first in-flight plan: no cost measured yet
            i = 0 if hard > 0 else n + 1
        else:
            lam = (self.n_lost + 2.0 * self.lost_prior) / (max(span, 0.0) + 2.0)
            te = tau + R
            i = None
            for j, L in enumerate(self.levels):
                nu = min(1.0 / self.dec_dt, 1.0 / L["replan_dt"] + lam)
                c = self.s * L["w"]
                P = c * nu * self.phi_int(tau, te, L["horizon"])
                lim = room * (1.0 - self.hyst) if j < self.cur else room
                if (
                    P <= lim
                    and self.next_safety * c * self.phi(tau, L["horizon"]) <= hard
                ):
                    i = j
                    break
            if i is None:
                L = self.ev_level
                nxt = self.next_safety * self.s * L["w"] * self.phi(tau, L["horizon"])
                i = n if (self.event and nxt <= hard) else n + 1
        if i == n + 1 and not self.greedy:
            i = n  # greedy off: keep event plans (the official gate still applies)
        if i != self.cur:
            self.trans.append((round(tau, 2), self.cur, i))
            self.cur = i
        self.dec[i] += 1
        self.last = dict(A=A, E=E, room=room, R=R)
        dt = math.inf if i >= n else self.levels[i]["replan_dt"]
        return i, dt

    def apply(self, i, planner, agent):
        L = self.level(i)
        planner.horizon = L["horizon"]
        planner.long_horizon = L["long_horizon"]
        planner.depth = L["depth"]
        planner.beam = L["beam"]
        planner.branch = L["branch"]
        agent.replan_dt = L["replan_dt"]

    def record(self, i, cost, tau, t, lost_evt):
        L = self.level(i)
        c = self.scale * cost
        self.plan_sum += c
        x = c / (L["w"] * self.phi(tau, L["horizon"]))
        self.s = x if self.s is None else (1.0 - self.alpha) * self.s + self.alpha * x
        if lost_evt:
            self.n_lost += 1
        self.last_plan_t = t
        self.n_plan[i] += 1
        self.t_plan[i] += cost
        if self.trace is not None:
            self.trace.append((round(tau, 2), i, round(1e3 * cost, 1)))

    def stats(self):
        n = len(self.levels)
        names = [str(j) for j in range(n)] + [EVENT, GREEDY]
        return dict(
            wall_scale=self.scale,
            target_s=self.target_s,
            elapsed_s=self.elapsed_s,
            plans={names[j]: self.n_plan[j] for j in range(n + 1) if self.n_plan[j]},
            plan_wall={
                names[j]: round(self.t_plan[j], 1)
                for j in range(n + 1)
                if self.n_plan[j]
            },
            decisions={names[j]: self.dec[j] for j in range(n + 2) if self.dec[j]},
            greedy_picks=self.n_greedy,
            n_lost=self.n_lost,
            switches=len(self.trans),
            trans=self.trans[:200],
            s_end=None if self.s is None else round(self.s, 4),
            # scaled agent / elapsed time at the first in-flight decision (pre-launch env + launch plans)
            at_launch=None
            if self.launch is None
            else dict(A=round(self.launch[1], 1), E=round(self.launch[2], 1)),
            last=None
            if self.launch is None
            else {k: round(v, 1) for k, v in self.last.items()},
            levels=[{k: v for k, v in L.items()} for L in self.levels],
            **({} if self.trace is None else dict(trace=self.trace)),
        )
