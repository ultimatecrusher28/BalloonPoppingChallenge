"""Thrust-direction programs flown in GivenSim, for the whole-burn tour (tour.py) and the agent's flight of the locked
tour. Only given parameters (GivenSim) and observation-derived balloon predictions are used.

A program is K segments of constant thrust direction n_k over [t0, burn_time] at full throttle, each
n_k = normalise(d_k + a_k u_k + b_k w_k) around a start direction d_k (u_k, w_k orthonormal to it), then the velocity
direction from burn_time on (no thrust). It is flown closed loop in GivenSim from the full state (the agent's attitude
reference and actuators, as the planner's rows).

dirs_of(x, D0): angles (R, 2K) around the start directions D0 (K, 3) -> unit directions (R, K, 3).
program_cmd(dirs, t_rel, t_rel0, burn, vel=None): the segment direction at t_rel (the command a row or the agent flies;
  with vel the velocity direction from burn on).
fly_program(sim, full, pos, vel, t_rel0, dirs, dt, t_end_rel, t_prog0=None): programs flown from one full state.
active(dirs, t0, t_now, burn): the segments not finished at t_now and the start of the first of them (the warm start
  of the agent's re-shoot).
"""

from __future__ import annotations

import numpy as np

from ..follow.ff_ref import FF_DIM


def _basis(d):
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    a = np.where(
        np.abs(d[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    )
    u = np.cross(d, a)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    w = np.cross(d, u)
    return d, u, w


def dirs_of(x, D0):
    """Angles x (R, 2K) around start directions D0 (K, 3) -> unit directions (R, K, 3)."""
    d, u, w = _basis(D0)
    X = x.reshape(len(x), -1, 2)
    n = d[None] + X[..., 0:1] * u[None] + X[..., 1:2] * w[None]
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def program_cmd(dirs, t_rel, t_rel0, burn, vel=None):
    """The segment direction (R, 3) of programs dirs (R, K, 3) at t_rel (R,) for segments splitting [t_rel0, burn] (t_rel0
    a scalar or per row). With vel (R, 3): from burn on (t_rel >= burn - 1e-9) the velocity direction, the unpowered
    command of fly_program."""
    K = dirs.shape[1]
    k = np.clip(
        ((t_rel - t_rel0) / np.maximum(burn - t_rel0, 1e-9) * K).astype(int), 0, K - 1
    )
    n = dirs[np.arange(len(dirs)), k]
    if vel is not None:
        past = np.asarray(t_rel) >= burn - 1e-9
        if np.any(past):
            v = np.asarray(vel, float)[past]
            n = n.copy()
            n[past] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return n


def fly_program(sim, full, pos, vel, t_rel0, dirs, dt, t_end_rel, t_prog0=None):
    """Rows with programs dirs (R, K, 3) flown at full throttle from one full state at t_rel0 (full: q, w, g, thr and the
    feedforward state ff; sim: the planner's GivenSim, built with ff), then unpowered: positions (R, n, 3) (nan below the
    ground), t_rel (n,). The segments split [t_prog0, burn] (default t_prog0 = t_rel0; earlier for a program already
    partly flown)."""
    tp0 = t_rel0 if t_prog0 is None else float(t_prog0)
    R = len(dirs)
    p, v = (
        np.repeat(pos[None], R, 0).astype(float),
        np.repeat(vel[None], R, 0).astype(float),
    )
    q = np.repeat(np.asarray(full["q"], float)[None], R, 0)
    w = np.repeat(np.asarray(full["w"], float)[None], R, 0)
    g = np.repeat(np.asarray(full["g"], float)[None], R, 0)
    th = np.full(R, float(full["thr"]))
    F = np.repeat(np.asarray(full.get("ff", np.zeros(FF_DIM)), float)[None], R, 0)
    ffs = (np.ascontiguousarray(F[:, :3]), np.ascontiguousarray(F[:, 3:]))
    burn = float(sim.burn_time)
    n = int(round((t_end_rel - t_rel0) / dt))
    out = np.full((R, n + 1, 3), np.nan)
    out[:, 0] = p
    alive = np.ones(R, bool)
    t = t_rel0
    for k in range(n):
        nc = program_cmd(dirs, np.full(R, t), tp0, burn, vel=v)
        p, v, q, w, g, th, ffs = sim.step(
            p, v, q, w, g, th, np.full(R, t), nc, 1.0, dt, ffs=ffs, ff_normed=k > 0
        )
        t += dt
        alive &= p[:, 2] >= sim.elevation
        out[alive, k + 1] = p[alive]
    return out, t_rel0 + dt * np.arange(n + 1)


def active(dirs, t0, t_now, burn):
    """Warm start of a re-shoot: the segments of program dirs (K, 3) on [t0, burn] not finished at t_now, and the start of
    the first of them (same boundaries: the program from t_now is the program flown so far, so the re-shoot's first point
    is the program itself)."""
    dirs = np.asarray(dirs, float)
    K = len(dirs)
    L = (burn - t0) / K
    k0 = int(np.clip(np.floor((t_now - t0) / L + 1e-9), 0, K - 1)) if L > 0 else K - 1
    return dirs[k0:], t0 + k0 * L
