"""Batched swept-segment pop scoring — the planner's objective function.

A pop is decided by the *swept* geometry: the rocket's centre of mass traces a
segment over one timestep, each balloon traces another, and the balloon pops when
the minimum distance between those two segments is within ``balloon.radius``
(1.5 m). Scoring a candidate trajectory therefore means running the environment's
own test, and running it for a batch of candidates at once.

:func:`seg_seg_dist2` is a broadcasting mirror of
``BalloonPoppingEnv._segment_distance_squared_batch``: the minimum of a convex
quadratic over the unit square, evaluated at all five candidate points (the four
edges plus the interior stationary point) and taking the best. The interior
solution uses cross-product identities so the near-parallel denominator never
loses its significant digits — every score sits on the 1.5 m threshold, so a
1e-8 relative error there is a wrong answer, not a rounding difference.

Ordering does not matter to the popped *set*: the environment stops testing a
balloon once it is popped, but re-testing one only ever re-flags it. That makes
the whole sweep parallel over steps, so :func:`score_paths` culls with a cheap
point-distance bound over chunks of steps and runs the exact test only on what
survives. First-hit step is tracked as well, so a candidate can be ranked on how
early it finishes as well as on how many balloons it takes.
"""

from __future__ import annotations

import numpy as np

# Squared length below which a segment is a point (m^2), matching the env.
DEGENERATE_LENGTH_SQUARED = 1e-12


def _dot(u, v):
    return np.einsum("...i,...i->...", u, v)


def seg_seg_dist2(start_a, end_a, start_b, end_b):
    """Squared minimum distance between segment a and segment b.

    All four arguments are ``(..., 3)`` and broadcast against each other; the
    result has the broadcast leading shape. Mirrors the environment's
    ``_segment_distance_squared_batch`` exactly, including its degenerate cases.
    """
    start_a = np.asarray(start_a, dtype=float)
    end_a = np.asarray(end_a, dtype=float)
    start_b = np.asarray(start_b, dtype=float)
    end_b = np.asarray(end_b, dtype=float)

    start_a, start_b, dir_a, dir_b = np.broadcast_arrays(
        start_a, start_b, end_a - start_a, end_b - start_b
    )
    offset = start_a - start_b

    eps = DEGENERATE_LENGTH_SQUARED
    a = _dot(dir_a, dir_a)
    e = _dot(dir_b, dir_b)
    f = _dot(dir_b, offset)
    c = _dot(dir_a, offset)
    b = _dot(dir_a, dir_b)

    a_ok = a > eps
    e_ok = e > eps
    a_safe = np.where(a_ok, a, 1.0)
    e_safe = np.where(e_ok, e, 1.0)

    def d2(s, t):
        # Grouped the way the env groups it, so the two agree to the last bit on
        # pairs that straddle the radius.
        sep = (start_a + s[..., None] * dir_a) - (start_b + t[..., None] * dir_b)
        return _dot(sep, sep)

    zero = np.zeros_like(a)
    one = np.ones_like(a)
    edges = [
        (zero, np.clip(f / e_safe, 0.0, 1.0)),
        (one, np.clip((b + f) / e_safe, 0.0, 1.0)),
        (np.clip(-c / a_safe, 0.0, 1.0), zero),
        (np.clip((b - c) / a_safe, 0.0, 1.0), one),
    ]

    # ||u x v||^2 rather than a*e - b^2: the subtraction cancels near parallel.
    # Zero, not a tolerance -- exactly parallel puts the minimum on an edge,
    # merely near parallel does not.
    normal = np.cross(dir_a, dir_b)
    denom = _dot(normal, normal)
    solvable = denom > 0.0
    denom_safe = np.where(solvable, denom, 1.0)
    # (p x q) . (r x s) = (p.r)(q.s) - (p.s)(q.r), so these are b*f - c*e and
    # a*f - b*c without the cancelling subtraction.
    s_int = np.where(
        solvable,
        np.clip(_dot(np.cross(dir_b, offset), normal) / denom_safe, 0.0, 1.0),
        edges[0][0],
    )
    t_int = np.where(
        solvable,
        np.clip(_dot(np.cross(dir_a, offset), normal) / denom_safe, 0.0, 1.0),
        edges[0][1],
    )

    regular = d2(s_int, t_int)
    for s, t in edges:
        regular = np.minimum(regular, d2(s, t))

    # Degenerate rocket sweep: the segment is a point, project it onto b.
    point_a = d2(zero, np.where(e_ok, np.clip(f / e_safe, 0.0, 1.0), 0.0))
    # Degenerate balloon sweep: project it onto a.
    point_b = d2(np.clip(-c / a_safe, 0.0, 1.0), zero)

    return np.where(a_ok, np.where(e_ok, regular, point_b), point_a)


def _poppable_mask(poppable, n_balloons, n_steps):
    """Normalize ``poppable`` to a ``(M, N)`` per-step boolean mask."""
    if poppable is None:
        return np.ones((n_balloons, n_steps), dtype=bool)
    mask = np.asarray(poppable, dtype=bool)
    if mask.ndim == 1:
        return np.broadcast_to(mask[:, None], (n_balloons, n_steps))
    return np.broadcast_to(mask[:, :n_steps], (n_balloons, n_steps))


def poppable_from_release(release_steps, n_balloons, n_steps):
    """``(M, N)`` mask: balloon m is poppable on step k once it has been released.

    ``release_steps`` is the per-balloon release step index (the environment
    releases one balloon every ``balloon.release_interval``).
    """
    k = np.arange(n_steps)[None, :]
    return k >= np.asarray(release_steps, dtype=int).reshape(n_balloons, 1)


def score_paths(
    rocket_paths,
    balloon_paths,
    radius,
    poppable=None,
    *,
    chunk=64,
):
    """Score a batch of rocket paths against a set of balloon paths.

    Parameters
    ----------
    rocket_paths : (B, N+1, 3) or (N+1, 3)
        Candidate rocket positions, sampled on the same time grid as the balloons.
    balloon_paths : (M, N+1, 3)
    radius : float
        Pop radius (``balloon.radius``).
    poppable : (M,) or (M, N) bool, optional
        Which balloons are poppable, optionally per step (see
        :func:`poppable_from_release`). Default: all, always.
    chunk : int
        Steps per broad-phase block. Trades peak memory for Python overhead.

    Returns
    -------
    counts : (B,) int
    popped : (B, M) bool
    d_min : (B, M) float
        Closest approach (m) over poppable steps. Exact wherever the broad phase
        found a candidate; a point-distance upper bound elsewhere.
    first_step : (B, M) int
        Step index of the first hit, or ``-1``. Candidate ``b`` finishes its tour
        at ``first_step[b, popped[b]].max()``.
    """
    rp = np.asarray(rocket_paths, dtype=float)
    single = rp.ndim == 2
    if single:
        rp = rp[None]
    bp = np.asarray(balloon_paths, dtype=float)

    n_batch = rp.shape[0]
    n_balloons = bp.shape[0]
    n_steps = min(rp.shape[1], bp.shape[1]) - 1
    if n_steps < 1:
        raise ValueError("paths must contain at least two samples")
    rp = rp[:, : n_steps + 1]
    bp = bp[:, : n_steps + 1]

    mask = _poppable_mask(poppable, n_balloons, n_steps)
    r2 = float(radius) ** 2

    popped = np.zeros((n_batch, n_balloons), dtype=bool)
    d_min = np.full((n_batch, n_balloons), np.inf)
    first_hit = np.full((n_batch, n_balloons), n_steps, dtype=int)

    for k0 in range(0, n_steps, chunk):
        k1 = min(k0 + chunk, n_steps)
        r_start = rp[:, k0:k1]  # (B, K, 3)
        r_end = rp[:, k0 + 1 : k1 + 1]
        b_start = bp[:, k0:k1]  # (M, K, 3)
        b_end = bp[:, k0 + 1 : k1 + 1]

        # Broad phase: the swept segments cannot come closer than the distance
        # between their start points less the two sweep lengths.
        gap = np.linalg.norm(r_start[:, None] - b_start[None], axis=-1)  # (B, M, K)
        reach = radius + (
            np.linalg.norm(r_end - r_start, axis=-1)[:, None, :]
            + np.linalg.norm(b_end - b_start, axis=-1)[None, :, :]
        )
        live = mask[None, :, k0:k1] & (gap <= reach)
        np.minimum(
            d_min, np.where(mask[None, :, k0:k1], gap, np.inf).min(axis=2), out=d_min
        )
        if not live.any():
            continue

        bi, mi, ki = np.nonzero(live)
        exact = np.sqrt(
            seg_seg_dist2(
                r_start[bi, ki], r_end[bi, ki], b_start[mi, ki], b_end[mi, ki]
            )
        )
        np.minimum.at(d_min, (bi, mi), exact)

        hit = exact * exact <= r2
        if not hit.any():
            continue
        hb, hm, hk = bi[hit], mi[hit], ki[hit] + k0
        popped[hb, hm] = True
        np.minimum.at(first_hit, (hb, hm), hk)

    first_step = np.where(popped, first_hit, -1)
    counts = popped.sum(axis=1).astype(int)
    if single:
        return int(counts[0]), popped[0], d_min[0], first_step[0]
    return counts, popped, d_min, first_step


def shaping(d_min, radius, popped=None, *, softness=3.0):
    """Smooth near-miss credit in ``[0, 1)`` per balloon, summed over the batch.

    A pop count is piecewise constant, so an optimizer gets no signal from a
    candidate that misses every balloon by 2 m versus one that misses by 20.
    ``sigmoid((radius - d) / softness)`` supplies that gradient without
    displacing the true count, which stays the dominant term.
    """
    d = np.asarray(d_min, dtype=float)
    z = np.clip((float(radius) - d) / float(softness), -60.0, 60.0)
    credit = 1.0 / (1.0 + np.exp(-z))
    if popped is not None:
        credit = np.where(popped, 0.0, credit)
    return credit.sum(axis=-1)
