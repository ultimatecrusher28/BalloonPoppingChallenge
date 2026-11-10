"""Observation features of the planner's critic (BeamPlanner._value): the layout the critic was trained on.

Pure numpy, batched over a leading row axis. Inputs are only what the env's observation provides (rocket state estimate,
balloon states/status, time) plus given-parameter constants. Vectors are expressed in a yaw frame aligned with the
rocket's horizontal velocity (or nose heading when slow).
"""

from __future__ import annotations

import numpy as np

K_SLOTS = 16


def yaw_angle(vel, nose):
    """(N,) heading of the yaw frame: horizontal velocity if > 2 m/s, else nose."""
    vh = vel[:, :2]
    nh = nose[:, :2]
    use_v = np.linalg.norm(vh, axis=1) > 2.0
    ref = np.where(use_v[:, None], vh, nh)
    small = np.linalg.norm(ref, axis=1) < 1e-6
    ref = np.where(small[:, None], np.array([1.0, 0.0]), ref)
    return np.arctan2(ref[:, 1], ref[:, 0])


def _rot(vecs, psi):
    """World -> yaw frame for (N, ..., 3) vectors, psi (N,)."""
    c, s = np.cos(psi), np.sin(psi)
    shape = (-1,) + (1,) * (vecs.ndim - 2)
    c, s = c.reshape(shape), s.reshape(shape)
    x, y = vecs[..., 0], vecs[..., 1]
    return np.stack([c * x + s * y, -s * x + c * y, vecs[..., 2]], axis=-1)


def build(
    pos,
    vel,
    nose,
    launched,
    t_rel,
    t_abs,
    b_pos,
    b_vel,
    b_status,
    *,
    elevation,
    burn_time,
    pad,
):
    """Batched features.

    pos, vel, nose : (N, 3); launched : (N,) bool; t_rel, t_abs : (N,)
    b_pos, b_vel : (N, B, 3); b_status : (N, B) in {0 ground, 1 released, 2 popped}; pad : (3,) launch point
    Returns (obs (N, 14 + 10 K_SLOTS) float32: 14 rocket features, then 10 per slot of the K_SLOTS nearest balloons not
    popped, psi (N,), slot_idx (N, K) int, -1 = empty).
    """
    n = pos.shape[0]
    here = np.where(launched[:, None], pos, pad[None, :])
    psi = yaw_angle(vel, nose)
    rel = b_pos - here[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    key = np.where(b_status == 2, np.inf, dist)
    k = min(K_SLOTS, key.shape[1])
    idx = np.argpartition(key, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(key, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    valid = np.isfinite(np.take_along_axis(key, idx, axis=1))

    r_rel = _rot(np.take_along_axis(rel, idx[:, :, None], axis=1), psi)
    v_rel = _rot(
        np.take_along_axis(b_vel, idx[:, :, None], axis=1) - vel[:, None, :], psi
    )
    st = np.take_along_axis(b_status, idx, axis=1)
    d = np.take_along_axis(dist, idx, axis=1)
    slots = np.concatenate(
        [
            r_rel / 100.0,
            v_rel / 30.0,
            (st == 1)[..., None].astype(float),
            (d / 100.0)[..., None],
            np.clip(1.0 / (d + 1.0), 0, 1)[..., None] * 5.0,
            valid[..., None].astype(float),
        ],
        axis=2,
    )
    slots = np.where(valid[..., None], slots, 0.0)

    burn_left = np.clip(burn_time - t_rel, 0.0, None) * launched
    rocket = np.concatenate(
        [
            _rot(here - pad[None, :], psi)[:, :2] / 100.0,
            ((here[:, 2] - elevation) / 100.0)[:, None],
            _rot(vel, psi) / 30.0,
            _rot(nose, psi),
            (t_rel * launched / 30.0)[:, None],
            (burn_left / 30.0)[:, None],
            launched[:, None].astype(float),
            (t_abs / 60.0)[:, None],
            (np.linalg.norm(vel, axis=1) / 30.0)[:, None],
        ],
        axis=1,
    )
    parts = [rocket, slots.reshape(n, -1)]
    obs = np.concatenate(parts, axis=1).astype(np.float32)
    slot_idx = np.where(valid, idx, -1)
    return obs, psi, slot_idx
