"""Policy observation features, shared by surrogate training and the runtime agent.

Pure numpy, batched over a leading env axis. Inputs are only what the env's
observation provides (rocket state estimate, balloon states/status, time) plus
given-parameter constants. Vectors are expressed in a yaw frame aligned with the
rocket's horizontal velocity (or nose heading when slow), so the policy's
thrust command is heading-invariant; ``to_world`` maps actions back.
"""
from __future__ import annotations

import numpy as np

K_SLOTS = 16
ROCKET_DIM = 14
SLOT_DIM = 10
OBS_DIM = ROCKET_DIM + K_SLOTS * SLOT_DIM


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


def to_world(vecs, psi):
    """Yaw frame -> world for (N, 3)."""
    return _rot(vecs, -psi)


# Far view: balloons beyond the K nearest, binned by yaw-frame azimuth (8) x above/below (2).
# Added 2026-09-24 after profiling: the 17-pop rival's late targets ranked 30-70th nearest, outside
# the 16 slots. Per-sector situational features follow the "observe beyond the next gate" idea of
# Song et al. 2021, Autonomous Drone Racing with Deep RL (arXiv:2103.08624). Appended at the end, so
# slot offsets and base_action are unchanged; old 174-dim policies use far=False.
FAR_AZ, FAR_EL, FAR_F = 8, 2, 5
FAR_DIM = FAR_AZ * FAR_EL * FAR_F


def far_view(rel_w, b_status, near_idx, psi):
    """rel_w (N, B, 3) world rel positions; returns (N, FAR_DIM)."""
    n, nb, _ = rel_w.shape
    far = b_status != 2
    np.put_along_axis(far, np.clip(near_idx, 0, nb - 1), False, axis=1)
    r = _rot(rel_w, psi)
    d = np.linalg.norm(r, axis=2)
    az = ((np.arctan2(r[..., 1], r[..., 0]) + np.pi) / (2 * np.pi) * FAR_AZ).astype(int) % FAR_AZ
    b = az * FAR_EL + (r[..., 2] > 0)
    w = np.where(far, 1.0 / (1.0 + d / 100.0), 0.0)
    out = np.zeros((n, FAR_AZ * FAR_EL, FAR_F))
    for k in range(FAR_AZ * FAR_EL):
        wk = w * (b == k)
        sw = wk.sum(1)
        out[:, k, 0] = sw / 5.0
        out[:, k, 1:4] = (wk[..., None] * r).sum(1) / np.maximum(sw, 1e-6)[:, None] / 200.0
        dk = np.where(wk > 0, d, np.inf).min(1)
        out[:, k, 4] = np.where(np.isfinite(dk), dk, 600.0) / 300.0
    return out.reshape(n, -1)


def build(pos, vel, nose, launched, t_rel, t_abs, b_pos, b_vel, b_status,
          *, elevation, burn_time, pad=None, far=False, string=None):
    """Batched features.

    pos, vel, nose : (N, 3); launched : (N,) bool; t_rel, t_abs : (N,)
    b_pos, b_vel : (N, B, 3); b_status : (N, B) in {0 ground, 1 released, 2 popped}
    Returns (obs (N, OBS_DIM) float32, psi (N,), slot_idx (N, K) int, -1 = empty).
    """
    n = pos.shape[0]
    if pad is None:
        pad = np.array([0.0, 0.0, elevation])
    here = np.where(launched[:, None], pos, pad[None, :])
    psi = yaw_angle(vel, nose)
    rel = b_pos - here[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    key = np.where(b_status == 2, np.inf, dist)
    if string:   # (T0, half_deg, discount): from T0, rank up-and-downwind balloons as if nearer
        T0, half, disc = string
        rel_b = b_status == 1
        wsum = np.where(rel_b[..., None], b_vel[..., :2], 0.0).sum(1)
        wdir = wsum / np.maximum(np.linalg.norm(wsum, axis=1, keepdims=True), 1e-9)
        dh = np.linalg.norm(rel[..., :2], axis=2)
        cosang = (rel[..., :2] * wdir[:, None, :]).sum(2) / np.maximum(dh, 1e-9)
        cone = rel_b & (rel[..., 2] > 0) & (cosang >= np.cos(np.radians(half)))
        act = (launched & (t_rel >= T0))[:, None]
        key = np.where(act & cone, key * disc, key)
    k = min(K_SLOTS, key.shape[1])
    idx = np.argpartition(key, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(key, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    valid = np.isfinite(np.take_along_axis(key, idx, axis=1))

    r_rel = _rot(np.take_along_axis(rel, idx[:, :, None], axis=1), psi)
    v_rel = _rot(np.take_along_axis(b_vel, idx[:, :, None], axis=1) - vel[:, None, :], psi)
    st = np.take_along_axis(b_status, idx, axis=1)
    d = np.take_along_axis(dist, idx, axis=1)
    slots = np.concatenate([
        r_rel / 100.0,
        v_rel / 30.0,
        (st == 1)[..., None].astype(float),
        (d / 100.0)[..., None],
        np.clip(1.0 / (d + 1.0), 0, 1)[..., None] * 5.0,
        valid[..., None].astype(float),
    ], axis=2)
    slots = np.where(valid[..., None], slots, 0.0)

    burn_left = np.clip(burn_time - t_rel, 0.0, None) * launched
    rocket = np.concatenate([
        _rot(here - pad[None, :], psi)[:, :2] / 100.0,
        ((here[:, 2] - elevation) / 100.0)[:, None],
        _rot(vel, psi) / 30.0,
        _rot(nose, psi),
        (t_rel * launched / 30.0)[:, None],
        (burn_left / 30.0)[:, None],
        launched[:, None].astype(float),
        (t_abs / 60.0)[:, None],
        (np.linalg.norm(vel, axis=1) / 30.0)[:, None],
    ], axis=1)
    parts = [rocket, slots.reshape(n, -1)]
    if far:
        parts.append(far_view(rel, b_status, idx, psi))
    obs = np.concatenate(parts, axis=1).astype(np.float32)
    slot_idx = np.where(valid, idx, -1)
    return obs, psi, slot_idx


ACC_SCALE = 10.0      # policy action units -> m/s^2
G = 9.81


def action_to_command(act, psi, h_agl, vz, safety="default"):
    """Policy action (N, >=3) -> world unit thrust direction (N, 3).

    ``act[:, :3] * ACC_SCALE`` is a desired kinematic acceleration in the yaw
    frame; thrust points along it plus gravity compensation, so a zero action
    is vertical. Ground-safety filter (T/W ~1.2): near the ground or on a short
    time-to-impact the thrust elevation is floored.
    """
    a = np.asarray(act, float)[:, :3] * ACC_SCALE
    a[:, 2] += G
    n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    n = to_world(n, psi)
    t_imp = np.where(vz < -0.5, h_agl / np.maximum(-vz, 1e-3), np.inf)
    if safety == "loose":   # lets the policy linger low among freshly released balloons
        el_min = np.where((h_agl < 15.0) | (t_imp < 1.5), np.radians(60.0),
                          np.where((h_agl < 30.0) | (t_imp < 3.0), np.radians(35.0), -np.pi / 2))
    else:
        el_min = np.where((h_agl < 40.0) | (t_imp < 3.0), np.radians(65.0),
                          np.where((h_agl < 80.0) | (t_imp < 6.0), np.radians(45.0), -np.pi / 2))
    el = np.arcsin(np.clip(n[:, 2], -1, 1))
    fix = el < el_min
    if fix.any():
        hz = np.arctan2(n[fix, 1], n[fix, 0]); e = el_min[fix]
        n[fix] = np.stack([np.cos(e) * np.cos(hz), np.cos(e) * np.sin(hz), np.sin(e)], 1)
    return n


BASE_LAUNCH_T = 36.0


def low_open_slot(obs, slot, T, H, horiz=0):
    """Opening heuristic: for t_rel < T after launch, pursue the nearest slot whose balloon is
    below H m AGL (freshly released / still on the ground); otherwise keep ``slot``.
    horiz=1 ranks by horizontal distance (tested on z5: more churn, 10 vs 6 switches in 12 s).
    horiz=2 keeps the 3-D ranking but makes the band relative: balloon < rocket AGL + H, so a
    rising target does not leave a fixed band before it is reached (z5 churn cause).
    Uses only the feature vector (rocket AGL obs[:, 2], t_rel obs[:, 9], slot rel. position)."""
    n = obs.shape[0]
    sl = obs[:, ROCKET_DIM:OBS_DIM].reshape(n, K_SLOTS, SLOT_DIM)
    b_agl = obs[:, 2:3] * 100.0 + sl[:, :, 2] * 100.0
    ok = (sl[:, :, 9] > 0.5) & ((sl[:, :, 2] * 100.0 < H) if horiz == 2 else (b_agl < H))
    if horiz == 1:
        dh = np.where(ok, np.hypot(sl[:, :, 0], sl[:, :, 1]), np.inf)
        first = np.where(ok.any(1), dh.argmin(1), -1)
    else:
        first = np.where(ok.any(1), ok.argmax(1), -1)       # slots are sorted nearest-first
    active = (obs[:, 11] > 0.5) & (obs[:, 9] * 30.0 < T) & (first >= 0)
    return np.where(active, first, slot)


def late_up_slot(obs, slot, T0, T1, dz_min):
    """Late-phase heuristic: for T0 <= t_rel < T1, pursue the nearest slot whose balloon is at
    least dz_min m above the rocket (the old, high stack); otherwise keep ``slot``.
    Observation-only (slot rel. position obs, t_rel obs[:, 9])."""
    n = obs.shape[0]
    sl = obs[:, ROCKET_DIM:OBS_DIM].reshape(n, K_SLOTS, SLOT_DIM)
    ok = (sl[:, :, 9] > 0.5) & (sl[:, :, 6] > 0.5) & (sl[:, :, 2] * 100.0 >= dz_min)
    first = np.where(ok.any(1), ok.argmax(1), -1)
    tr = obs[:, 9] * 30.0
    active = (obs[:, 11] > 0.5) & (tr >= T0) & (tr < T1) & (first >= 0)
    return np.where(active, first, slot)


def string_slot(obs, slot, T0, half, disc=None):
    """Late 'string' heuristic: for T0 <= t_rel < burnout, pursue the first slot whose balloon is
    above the rocket and within +-half deg of the downwind direction (mean slot balloon velocity).
    Drifting balloons released from one spot form a line up and downwind (seed 304729: bearing
    94-110 deg, wind bearing 100 deg); older ones are further along it. Observation-only."""
    n = obs.shape[0]
    sl = obs[:, ROCKET_DIM:OBS_DIM].reshape(n, K_SLOTS, SLOT_DIM)
    ok_v = (sl[:, :, 9] > 0.5) & (sl[:, :, 6] > 0.5)
    v_abs = sl[:, :, 3:5] * 30.0 + obs[:, None, 3:5] * 30.0
    wsum = np.where(ok_v[..., None], v_abs, 0.0).sum(1)
    wdir = wsum / np.maximum(np.linalg.norm(wsum, axis=1, keepdims=True), 1e-9)
    r = sl[:, :, 0:3] * 100.0
    dh = np.linalg.norm(r[..., :2], axis=2)
    cosang = (r[..., :2] * wdir[:, None, :]).sum(2) / np.maximum(dh, 1e-9)
    ok = ok_v & (r[..., 2] > 0) & (cosang >= np.cos(np.radians(half)))
    first = np.where(ok.any(1), ok.argmax(1), -1)
    tr = obs[:, 9] * 30.0
    active = (obs[:, 11] > 0.5) & (tr >= T0) & (obs[:, 10] > 0) & (first >= 0)
    return np.where(active, first, slot)


def base_action(obs, slot=None, launch_ramp=False, pursue_ground=False, tgo_closing=False, low_open=None,
                late_up=None, string=None, boost=None):
    """Residual-policy base: lead pursuit of balloon ``slot`` (default 0, the
    nearest), launch at BASE_LAUNCH_T. Pure function of the feature vector,
    so training and runtime compute it identically. Returns (N, 5).
    ``low_open=(T, H)``: see low_open_slot."""
    n = obs.shape[0]
    slot = np.zeros(n, int) if slot is None else np.asarray(slot, int)
    if low_open:
        slot = low_open_slot(obs, slot, *low_open)
    if late_up:
        slot = late_up_slot(obs, slot, *late_up)
    if string:
        slot = string_slot(obs, slot, *string)
    sl = obs[:, ROCKET_DIM:OBS_DIM].reshape(n, K_SLOTS, SLOT_DIM)[np.arange(n), slot]
    r = sl[:, 0:3].astype(float) * 100.0
    vrel = sl[:, 3:6].astype(float) * 30.0
    released = sl[:, 6] > 0.5
    dist = np.linalg.norm(r, axis=1)
    if tgo_closing:   # range / closing-rate t_go (PN-style): a fast approach is not braked
        vc = -(r * vrel).sum(1) / np.maximum(dist, 1e-6)
        tgo = np.clip(dist / np.maximum(vc, 12.0), 0.8, 6.0)[:, None]
    else:
        tgo = np.clip(dist / 12.0, 0.8, 6.0)[:, None]
    a_des = 2.0 * (r + vrel * tgo) / tgo ** 2 / ACC_SCALE
    valid = sl[:, 9] > 0.5
    chase = (released | valid) if pursue_ground else released
    if boost:   # (T0, b): from T0 add b (action units, x10 m/s^2) along the line of sight -> accelerate, not just null ZEM
        T0, b = boost
        on = (obs[:, 9] * 30.0 >= T0) & (obs[:, 10] > 0)
        a_des = a_des + np.where(on[:, None], b * r / np.maximum(dist, 1e-6)[:, None], 0.0)
    a_des = np.clip(np.where(chase[:, None], a_des, 0.0), -2.0, 2.0)
    t_abs = obs[:, 12].astype(float) * 60.0
    if launch_ramp:   # smooth: a residual r shifts the launch time by ~2r seconds
        launch = np.clip((t_abs - BASE_LAUNCH_T + 0.25) * 0.5, -4.0, 4.0)
    else:
        launch = np.where(t_abs >= BASE_LAUNCH_T, 2.0, -2.0)
    thr = np.full((obs.shape[0], 1), 3.0)
    return np.concatenate([a_des, thr, launch[:, None]], 1)

OBS_DIM_FAR = OBS_DIM + FAR_DIM
