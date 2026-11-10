"""Vectorized balloon-popping training env: real cached balloon tracks + surrogate rocket.

Decision interval: 0.5 s before launch, 0.1 s after (5 x 0.02 s substeps).
Action (N, 5): desired accel/10 in the yaw frame (features.action_to_command), throttle logit, launch logit (used only before launch).
Reward: +1 per pop, potential-based shaping on nearest released balloon
distance, -1 on ground impact during the burn.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import importlib.util as _ilu  # load the shared runtime module without the gym package
_spec = _ilu.spec_from_file_location(
    "rl_features", HERE / "agent_modules" / "features.py")  # training-time copy (training/README.md)
F = _ilu.module_from_spec(_spec); _spec.loader.exec_module(F)
from surrogate import Params, RigidParams, Vehicle, quat_from_nose, step_rigid  # noqa: E402
import os as _os
SURR = _os.environ.get("BPC_SURR", "fitted")
if SURR == "given":  # SPEC T2: given-parameters-only physics (surrogate_given.py)
    from surrogate_given import GivenVehicle as Vehicle, step_given as step_rigid  # noqa: E402,F811

DT_SUB, N_SUB = 0.01, 10
PRE_DT = 0.5
T_FORCE_LAUNCH = 75.0
MIN_INCL_DEG = 45.0
THR_LO = 0.3


def load_cache(pattern, limit=None, subdir="cache"):
    files = sorted(glob.glob(str(HERE / subdir / pattern)))[:limit]
    P, R = [], []
    for f in files:
        d = np.load(f)
        P.append(d["pos"]); R.append(d["release_step"])
    hz = float(np.load(files[0])["hz"])
    dt = float(np.load(files[0])["dt"])
    return np.stack(P), np.stack(R) * dt, hz, files  # pos (S,B,3,T) float32, release time (S,B)


class VecEnv:
    SNAP_KEYS = ("pos", "vel", "nose", "w", "thr", "q", "g", "t_rel", "t_abs", "launched", "status", "sidx")
    DR_FIELDS_P = ("ax0", "ax2", "n1", "n2")
    DR_FIELDS_R = ("cx", "cy", "ax", "ay", "dx", "dy")

    def __init__(self, n, cache, *, params=None, scenario=int(_os.environ.get("BPC_SC", 1)), seed=0, shaping=0.5, gamma=0.995, dr=0.0, dr_thrust=0.0):
        self.n = n
        self.P, self.R, self.hz, _ = cache
        self.S, self.B, _, self.T = self.P.shape
        self.veh = Vehicle(scenario)
        if SURR == "given":
            _wf = HERE / f"wind_obs_sc{scenario}.npz"
            if _wf.exists():
                _w = np.load(_wf); self.veh.set_wind(_w["z"] + self.veh.elevation * 0, _w["u"], _w["v"])
        self.prm = params or Params()
        self.rp = RigidParams()
        self.far = False  # append far-view features (features.far_view)
        self.string = None  # (T0, half_deg, discount) string ranking, features.build
        self.pop_time_bonus = 0.0  # train-only pace reward (ppo.py --pop-time-bonus)
        self.old_bonus = 0.0; self.old_age = 30.0  # train-only (ppo.py --old-bonus/--old-age)
        self.fresh_bonus = 0.0; self.fresh_age = 8.0  # train-only (ppo.py --fresh-bonus/--fresh-age)
        self.dr = float(dr); self.dr_thrust = float(dr_thrust); self.dr_launch = 0.0
        self.thrust_mult = np.ones(n)
        self.dr_mult = np.ones((n, len(self.DR_FIELDS_P) + len(self.DR_FIELDS_R)))
        self.rng = np.random.default_rng(seed)
        self.shaping, self.gamma = shaping, gamma
        self.pad = np.array([0.0, 0.0, self.veh.elevation])
        self.ep_ret = np.zeros(n); self.ep_pops = np.zeros(n, int)
        self.done_stats = []
        self.final_pops = np.full(n, -1)
        # optional per-env forced launch: time (s), inclination (deg), compass heading (deg)
        self.force_launch = None
        self.safety = "default"
        # mid-flight restarts (Song et al. 2023 init strategy): snapshots of own rollouts
        self.mid_p = 0.0; self.snaps = []; self.snap_max = 4000
        self.demo_p = 0.0; self.demo_snaps = []
        self._reset_idx(np.arange(n))

    # -- balloons ----------------------------------------------------------
    def _bpos(self, t):
        # time-major copy (S,T,B,3): contiguous (B,3) gathers (profiled: 18% of step time before)
        if getattr(self, "_Pt", None) is None or self._Pt.shape[0] != self.P.shape[0]:
            self._Pt = np.ascontiguousarray(self.P.transpose(0, 3, 1, 2))
        f = np.clip(t * self.hz, 0, self.T - 1.001)
        i = f.astype(int); w = (f - i)[:, None, None]
        s = self.sidx
        return (1 - w) * self._Pt[s, i] + w * self._Pt[s, i + 1]

    def _bpos_vel(self, t):
        f = np.clip(t * self.hz, 0, self.T - 1.001)
        i = f.astype(int); w = (f - i)[:, None, None]
        s = self.sidx
        a, b = self.P[s, :, :, i], self.P[s, :, :, i + 1]
        return (1 - w) * a + w * b, (b - a) * self.hz

    def _status_update(self):
        rel = (self.t_abs[:, None] >= self.R[self.sidx]) & (self.status == 0)
        self.status[rel] = 1

    # -- reset -------------------------------------------------------------
    def _reset_idx(self, idx):
        if not hasattr(self, "sidx"):
            n = self.n
            self.sidx = np.zeros(n, int); self.t_abs = np.zeros(n); self.t_rel = np.zeros(n)
            self.launched = np.zeros(n, bool); self.pos = np.zeros((n, 3)); self.vel = np.zeros((n, 3))
            self.nose = np.zeros((n, 3)); self.w = np.zeros((n, 2)); self.thr = np.ones(n)
            self.q = np.tile([1.0, 0, 0, 0], (n, 1)); self.g = np.zeros((n, 2))
            self.status = np.zeros((n, self.B), np.int8); self.phi = np.zeros(n)
        self.sidx[idx] = self.rng.integers(0, self.S, len(idx))
        if getattr(self, "dr_thrust", 0.0) > 0 and len(idx):
            self.thrust_mult[idx] = 1.0 + self.rng.uniform(-self.dr_thrust, self.dr_thrust, len(idx))
        if getattr(self, "dr", 0.0) > 0 and len(idx):
            self.dr_mult[idx] = 1.0 + self.rng.uniform(-self.dr, self.dr, (len(idx), self.dr_mult.shape[1]))
        self.t_abs[idx] = 0.0; self.t_rel[idx] = 0.0
        self.launched[idx] = False
        self.pos[idx] = self.pad; self.vel[idx] = 0.0
        self.nose[idx] = np.array([0.0, 0.0, 1.0]); self.w[idx] = 0.0; self.thr[idx] = 1.0
        self.q[idx] = np.array([1.0, 0, 0, 0]); self.g[idx] = 0.0
        self.status[idx] = 0
        self.phi[idx] = 0.0
        self.ep_ret[idx] = 0.0; self.ep_pops[idx] = 0
        if getattr(self, "demo_p", 0.0) > 0 and self.demo_snaps and len(idx):
            use = idx[self.rng.random(len(idx)) < self.demo_p]
            for i in use:
                snap = self.demo_snaps[self.rng.integers(len(self.demo_snaps))]
                for k, v in snap.items():
                    getattr(self, k)[i] = v
                self.ep_pops[i] = 0
            idx = np.setdiff1d(idx, use)
        if getattr(self, "mid_p", 0.0) > 0 and self.snaps and len(idx):
            use = idx[self.rng.random(len(idx)) < self.mid_p]
            for i in use:
                snap = self.snaps[self.rng.integers(len(self.snaps))]
                for k, v in snap.items():
                    getattr(self, k)[i] = v

    # -- observation ---------------------------------------------------------
    def obs(self):
        self._status_update()
        bp, bv = self._bpos_vel(self.t_abs)
        o, psi, _ = F.build(self.pos, self.vel, self.nose, self.launched, self.t_rel, self.t_abs,
                            bp, bv, self.status, elevation=self.veh.elevation,
                            burn_time=self.veh.burn_time, pad=self.pad, far=self.far, string=self.string)
        self.psi = psi
        return o

    def _potential(self, bp):
        rel = np.linalg.norm(bp - self.pos[:, None, :], axis=2)
        rel = np.where(self.status == 1, rel, np.inf)
        d = rel.min(axis=1)
        d = np.where(np.isfinite(d), np.minimum(d, 300.0), 300.0)
        return np.where(self.launched & (self.t_rel < self.veh.burn_time), -self.shaping * d / 100.0, 0.0)

    # -- step ----------------------------------------------------------------
    def step(self, act):
        n = self.n
        act = np.asarray(act, float)
        n_cmd = F.action_to_command(act, self.psi, self.pos[:, 2] - self.veh.elevation, self.vel[:, 2], safety=self.safety)
        thr_cmd = THR_LO + (1 - THR_LO) / (1 + np.exp(-act[:, 3]))
        reward = np.zeros(n)
        pops = np.zeros(n, int); young_pops = np.zeros(n, int); old_pops = np.zeros(n, int)

        pre = ~self.launched
        # --- pre-launch: wait or launch
        if pre.any() and self.force_launch is not None:
            fl_t, fl_i, fl_h = self.force_launch
            go = pre & (self.t_abs + 1e-9 >= fl_t)
            wait = pre & ~go
            self.t_abs[wait] = np.minimum(self.t_abs[wait] + PRE_DT, fl_t[wait])
            if go.any():
                ii, hh = np.radians(fl_i[go]), np.radians(fl_h[go])
                self.nose[go] = np.stack([np.cos(ii) * np.sin(hh), np.cos(ii) * np.cos(hh), np.sin(ii)], 1)
                self.q[go] = quat_from_nose(self.nose[go]); self.g[go] = 0.0
                self.launched[go] = True; self.t_rel[go] = 0.0
                self.pos[go] = self.pad; self.vel[go] = 0.0; self.w[go] = 0.0; self.thr[go] = 1.0
                self.t_abs[go] += 0.01
                self.phi[go] = self._potential(self._bpos(self.t_abs))[go]
        elif pre.any():
            go = pre & ((act[:, 4] > 0) | (self.t_abs + PRE_DT >= T_FORCE_LAUNCH))
            wait = pre & ~go
            self.t_abs[wait] += PRE_DT
            if go.any():
                nc = n_cmd[go].copy()
                el = np.arcsin(np.clip(nc[:, 2], -1, 1))
                el = np.maximum(el, np.radians(MIN_INCL_DEG))
                hz = np.arctan2(nc[:, 1], nc[:, 0])
                self.nose[go] = np.stack([np.cos(el) * np.cos(hz), np.cos(el) * np.sin(hz), np.sin(el)], 1)
                self.q[go] = quat_from_nose(self.nose[go]); self.g[go] = 0.0
                self.launched[go] = True; self.t_rel[go] = 0.0
                self.pos[go] = self.pad; self.vel[go] = 0.0; self.w[go] = 0.0; self.thr[go] = 1.0
                self.t_abs[go] += 0.01
                if self.dr_launch > 0:   # effective lift-off delay (real env launches a few steps late)
                    self.t_abs[go] += np.round(self.rng.uniform(0, self.dr_launch, go.sum()), 2)
                if getattr(self, "launch_delay", None) is not None:   # fixed per-env delay (sensitivity.py)
                    self.t_abs[go] += self.launch_delay[go]
                bp = self._bpos(self.t_abs)
                self.phi[go] = self._potential(bp)[go]

        # --- flight
        fl = self.launched & ~pre
        crashed = np.zeros(n, bool)
        if fl.any():
            idx = np.flatnonzero(fl)
            pos, vel, nose, w, thr = self.pos[idx], self.vel[idx], self.nose[idx], self.w[idx], self.thr[idx]
            q, g = self.q[idx], self.g[idx]
            t_rel, t_abs = self.t_rel[idx], self.t_abs[idx]
            nc, tc = n_cmd[idx], thr_cmd[idx]
            s_save = self.sidx
            prm, rp = self.prm, self.rp
            if self.dr_thrust > 0:
                from dataclasses import replace as _rep
                prm = _rep(prm, thrust_k=self.thrust_mult[idx])
            if self.dr > 0:
                from dataclasses import replace
                mult = self.dr_mult[idx]; nP = len(self.DR_FIELDS_P)
                prm = replace(prm, **{k: getattr(self.prm, k) * mult[:, i] for i, k in enumerate(self.DR_FIELDS_P)})
                rp = replace(self.rp, **{k: getattr(self.rp, k) * mult[:, nP + i] for i, k in enumerate(self.DR_FIELDS_R)})
            self.sidx = s_save[idx]
            b1 = self._bpos(t_abs)
            for _ in range(N_SUB):
                p0 = pos
                b0 = b1  # end-of-previous-substep positions (same t_abs)
                pos, vel, q, w, g, thr, nose = step_rigid(self.veh, prm, rp, pos, vel, q, w, g, thr,
                                                          t_rel, nc, tc, DT_SUB)
                t_rel = t_rel + DT_SUB; t_abs = t_abs + DT_SUB
                b1 = self._bpos(t_abs)
                # swept-segment pop test on released balloons
                st = self.status[idx]
                rel_rel = (t_abs[:, None] >= self.R[self.sidx]) & (st == 0)
                st[rel_rel] = 1
                a0 = p0[:, None, :] - b0; a1 = pos[:, None, :] - b1
                da = a1 - a0
                s = np.clip(-(a0 * da).sum(2) / np.maximum((da * da).sum(2), 1e-12), 0, 1)
                dmin2 = ((a0 + s[..., None] * da) ** 2).sum(2)
                hit = (st == 1) & (dmin2 <= self.veh.radius ** 2)
                st[hit] = 2
                self.status[idx] = st
                pops[idx] += hit.sum(1)
                if self.old_bonus:     # pops of balloons released > old_age s ago (high stack)
                    old_pops[idx] += (hit & (t_abs[:, None] - self.R[self.sidx] > self.old_age)).sum(1)
                if self.fresh_bonus:   # pops of balloons released < fresh_age s ago
                    young_pops[idx] += (hit & (t_abs[:, None] - self.R[self.sidx] < self.fresh_age)).sum(1)
            self.sidx = s_save
            self.pos[idx], self.vel[idx], self.nose[idx], self.w[idx], self.thr[idx] = pos, vel, nose, w, thr
            self.q[idx], self.g[idx] = q, g
            self.t_rel[idx], self.t_abs[idx] = t_rel, t_abs
            ground = (pos[:, 2] < self.veh.elevation) & (t_rel > 0.3)
            crashed[idx] = ground
        reward += pops
        if self.old_bonus:        # train-only: favour reaching the old, high balloons
            reward += old_pops * self.old_bonus
        if self.fresh_bonus:      # train-only: favour harvesting freshly released (low) balloons
            reward += young_pops * self.fresh_bonus
        if self.pop_time_bonus:   # early pops worth more (pace): 1 + k at launch, falling to 1 at burnout
            reward += pops * self.pop_time_bonus * np.clip(1.0 - self.t_rel / self.veh.burn_time, 0.0, 1.0)
        self.ep_pops += pops

        bp = self._bpos(self.t_abs)
        phi_new = self._potential(bp)
        reward += np.where(fl, self.gamma * phi_new - self.phi, 0.0)
        self.phi = phi_new
        burn_crash = crashed & (self.t_rel < self.veh.burn_time)
        reward -= burn_crash * 1.0
        timeout = self.t_abs >= 149.0
        if self.mid_p > 0:
            cand = np.flatnonzero(self.launched & (self.t_rel > 1.0) & (self.t_rel < self.veh.burn_time) & ~crashed)
            if len(cand):
                for i in self.rng.choice(cand, size=min(2, len(cand)), replace=False):
                    self.snaps.append({k: np.array(getattr(self, k)[i], copy=True) for k in self.SNAP_KEYS})
                if len(self.snaps) > self.snap_max:
                    self.snaps = self.snaps[-self.snap_max:]
        done = crashed | timeout
        self.ep_ret += reward
        if done.any():
            di = np.flatnonzero(done)
            self.final_pops[di] = self.ep_pops[di]
            for i in di:
                self.done_stats.append((self.ep_pops[i], self.ep_ret[i], self.t_abs[i] - self.t_rel[i], bool(burn_crash[i])))
            self._reset_idx(di)
        return self.obs(), reward, done
