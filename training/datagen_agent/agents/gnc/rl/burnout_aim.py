"""Burnout aim: in the burn's last ``window_s`` seconds, pick a fixed thrust direction whose
thrust-then-coast trajectory passes closest to an unpopped balloon, and fly it.

Shooting method: ~250 fixed directions (plus the policy's own command) are integrated in the
same rigid-body model the residual policy was trained on (research/rl/surrogate.py), through the
remaining burn and ``coast_s`` of coast. Balloons are extrapolated at their observed velocity.
Wind for the model's airspeed is estimated online from the observed horizontal drift of released
balloons near each altitude (no stored wind). Re-solved every decision (10 Hz), so model error
before burnout is corrected in closed loop; only the coast is open loop.

Feasibility (research/rl/RESULTS.md, 2026-09-25): from p5's official-simulator state at 28.1 s,
fixed directions pass within 1.4-1.8 m of four balloons (pop radius 1.5 m); surrogate coast
error from a real post-burnout state is 0.2 m at +3 s, 2.1 m at +5 s.

NOTE: imports the training surrogate from research/rl (vehicle tables + fitted aero). Move it
under /agents before any submission that relies on this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_RL = Path(__file__).resolve().parents[4] / "research" / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))
from surrogate import Params, RigidParams, Vehicle, step_rigid  # noqa: E402


class _ObsWindVehicle(Vehicle):
    """Vehicle whose wind(z) comes from balloon drift observations (set per call)."""

    def set_wind(self, b_pos, b_vel, live):
        z = b_pos[live, 2]
        if z.size < 2:
            self.wz, self.wu, self.wv = np.array([0.0, 1.0]), np.zeros(2), np.zeros(2)
            return
        grid = (
            np.linspace(z.min(), z.max(), 12)
            if z.max() > z.min() + 1
            else np.array([z.min(), z.min() + 1])
        )
        u, v = [], []
        for zg in grid:
            wgt = np.exp(-0.5 * ((z - zg) / 20.0) ** 2) + 1e-9
            u.append(np.sum(wgt * b_vel[live, 0]) / wgt.sum())
            v.append(np.sum(wgt * b_vel[live, 1]) / wgt.sum())
        self.wz, self.wu, self.wv = grid, np.array(u), np.array(v)


class BurnoutAim:
    def __init__(
        self,
        window_s=2.0,
        coast_s=6.0,
        accept_m=0.8,
        step_deg=7.5,
        dt=0.02,
        coast_pen=0.0,
    ):
        self.window_s, self.coast_s, self.accept_m, self.dt = (
            window_s,
            coast_s,
            accept_m,
            dt,
        )
        self.coast_pen = (
            coast_pen  # m per s of coast before arrival added to the miss when ranking
        )
        self.veh = _ObsWindVehicle(1)
        self.prm, self.rp = Params(), RigidParams()
        els = np.radians(np.arange(-60.0, 90.01, step_deg))
        azs = np.radians(np.arange(0.0, 360.0, step_deg))
        E, A = np.meshgrid(els, azs)
        E, A = E.ravel(), A.ravel()
        self.dirs = np.stack(
            [np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], 1
        )
        self.last = (
            None  # (balloon, miss_m, arrival_t_rel) of the last accepted solution
        )

    def active(self, t_rel):
        return self.veh.burn_time - self.window_s <= t_rel < self.veh.burn_time

    def solve(self, t_rel, pos, vel, quat, omega, b_pos, b_vel, status, policy_dir):
        """Returns a world thrust direction (3,) or None (keep the policy's command)."""
        live = status == 1
        if not live.any():
            return None
        self.veh.set_wind(b_pos, b_vel, status >= 1)
        dirs = np.vstack([self.dirs, policy_dir[None]])
        n = len(dirs)
        p = np.repeat(np.asarray(pos, float)[None], n, 0)
        v = np.repeat(np.asarray(vel, float)[None], n, 0)
        q = np.repeat(np.asarray(quat, float)[None], n, 0)
        w = np.repeat(np.asarray(omega, float)[None, :2], n, 0)
        g = np.zeros((n, 2))
        thr = np.full(n, 0.97)
        tr = np.full(n, float(t_rel))
        bp, bv = b_pos[live], b_vel[live]
        best = np.full((n, bp.shape[0]), np.inf)
        t_best = np.zeros((n, bp.shape[0]))
        steps = int((self.veh.burn_time + self.coast_s - t_rel) / self.dt)
        tau = 0.0
        for _ in range(steps):
            p, v, q, w, g, thr, _ = step_rigid(
                self.veh,
                self.prm,
                self.rp,
                p,
                v,
                q,
                w,
                g,
                thr,
                tr,
                dirs,
                np.ones(n),
                self.dt,
            )
            tr = tr + self.dt
            tau += self.dt
            b = bp + bv * tau
            d = np.linalg.norm(b[None] - p[:, None], axis=2)
            better = d < best
            best = np.where(better, d, best)
            t_best = np.where(better, tau, t_best)
            if (p[:, 2] < self.veh.elevation).all():
                break
        coast = np.maximum(t_rel + t_best - self.veh.burn_time, 0.0)
        score = np.where(best <= self.accept_m, best + self.coast_pen * coast, np.inf)
        if not np.isfinite(score).any():
            self.last = None
            return None
        i, j = np.unravel_index(np.argmin(score), score.shape)
        if best[i, j] > self.accept_m:
            self.last = None
            return None
        self.last = (
            int(np.flatnonzero(live)[j]),
            float(best[i, j]),
            float(t_rel + t_best[i, j]),
        )
        return dirs[i]
