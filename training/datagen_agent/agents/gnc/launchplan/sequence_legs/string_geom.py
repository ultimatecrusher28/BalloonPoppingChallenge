"""Near-collinear balloon strings in the cloud frame: pure geometry, no state.

Measured context (research/limits/REPORT.md sections 7.9, 7.11). The velocity
turn radius ``R = v^2 / a_lat(v)`` is 75-120 m while the balloon field's
nearest-neighbour spacing is 30 m, so no vehicle can hop freely between
adjacent balloons: only a near-straight sequence -- a STRING -- can be flown
through the field. The balloon field is also nearly a rigid body over a whole
episode (a common translation fits it to 3.7 m median over the 22 s cruise
window), so a string extracted once at launch stays a string.

Two definitions live here, and they answer different questions:

- :func:`line_residual` measures how close a set of points is to one straight
  line. A tube of the 1.5 m pop radius is the STRICT corridor test.
- :func:`turn_chains` finds the longest polyline through the points whose
  every vertex turn is inside what the turn radius allows. The turn model is

      theta_i <= (c_in + c_out) / (2 R)

  because the turn rate is ``a_lat / v`` rad/s and the turn at vertex ``i``
  may be spread over half of each adjacent leg, giving it
  ``(c_in + c_out) / (2 v)`` seconds. This is the small-angle form of the
  ``c = 2 R sin(theta/2)`` relation that fits both our runs and TeamDJ's.

The search is an exact layered DP over ``(last edge, visited set)``: the turn
constraint prunes so hard that the state space stays tiny (measured: 0.05 s
and far under ``max_states`` for 100 balloons at ``c_hi = 70``), so
``max_states`` is a guard, not a tuning knob. Chains are simple paths by
construction (the visited set is part of the state).
"""

from __future__ import annotations

import numpy as np


def line_residual(points) -> float:
    """RMS distance of ``points`` from their own best-fit line, in metres.

    Zero is perfectly collinear. A value comparable to the point spacing
    means the set is a scribble, not a string.
    """
    p = np.asarray(points, float)
    if p.shape[0] < 3:
        return 0.0
    w = p - p.mean(axis=0)
    u = np.linalg.svd(w, full_matrices=False)[2][0]
    perp = w - (w @ u)[:, None] * u
    return float(np.sqrt((perp**2).sum(axis=1).mean()))


def chain_metrics(points, members) -> dict:
    """Chords, vertex turn angles (deg) and off-line RMS of one chain."""
    pts = np.asarray(points, float)[list(members)]
    ch = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    out = {
        "members": [int(m) for m in members],
        "n": len(members),
        "chords": [float(x) for x in ch],
        "angles": [],
        "offline_rms": line_residual(pts),
    }
    if ch.size >= 2:
        d = np.diff(pts, axis=0) / ch[:, None]
        ang = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1.0, 1.0)))
        out["angles"] = [float(x) for x in ang]
    return out


def _edges(P, c_lo, c_hi):
    V = P[None, :, :] - P[:, None, :]  # V[i, j] = P[j] - P[i]
    L = np.linalg.norm(V, axis=2)
    ok = (L >= c_lo) & (L <= c_hi)
    np.fill_diagonal(ok, False)
    edges = [(int(i), int(j)) for i in range(P.shape[0]) for j in np.flatnonzero(ok[i])]
    return edges, V, L


def _successors(edges, V, L, R, n):
    out_of = [[] for _ in range(n)]
    for k, (i, _j) in enumerate(edges):
        out_of[i].append(k)
    succ = [[] for _ in edges]
    for k, (i, j) in enumerate(edges):
        a = V[i, j] / L[i, j]
        budget_in = L[i, j]
        for m in out_of[j]:
            q = edges[m][1]
            if q == i:
                continue
            b = V[j, q] / L[j, q]
            theta = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
            if theta <= (budget_in + L[j, q]) / (2.0 * R):
                succ[k].append(m)
    return succ


def _reconstruct(history, key, depth, edges):
    seq = [edges[key[0]][1]]
    cur, d = key, depth
    while d > 0:
        seq.append(edges[cur[0]][0])
        cur = history[d][cur]
        d -= 1
    seq.append(edges[cur[0]][0])
    return list(reversed(seq))


def turn_chains(
    points,
    R,
    *,
    c_lo=20.0,
    c_hi=70.0,
    max_depth=18,
    max_states=40000,
    top=8,
    both_directions=False,
):
    """Longest curvature-feasible polylines through ``points``.

    ``R`` is the velocity turn radius in metres; only chords inside
    ``[c_lo, c_hi]`` are legs. Returns up to ``top`` chains as
    :func:`chain_metrics` dicts, deepest first. Chains are deduplicated by
    member SET, so each string appears once; ``both_directions`` adds the
    reversed traversal of each, which matters to a planner because the entry
    end decides whether the string is reachable from the pad at all.
    """
    P = np.asarray(points, float)
    n = P.shape[0]
    if n < 2:
        return []
    edges, V, L = _edges(P, c_lo, c_hi)
    if not edges:
        return []
    succ = _successors(edges, V, L, float(R), n)

    history = [{(k, (1 << i) | (1 << j)): None for k, (i, j) in enumerate(edges)}]
    while len(history) - 1 < int(max_depth):
        nxt = {}
        for key in history[-1]:
            k, mask = key
            for m in succ[k]:
                q = edges[m][1]
                if mask >> q & 1:
                    continue
                nxt.setdefault((m, mask | (1 << q)), key)
            if len(nxt) >= max_states:
                break
        if not nxt:
            break
        history.append(nxt)

    chains, seen = [], set()
    for depth in range(len(history) - 1, -1, -1):
        for key in history[depth]:
            if key[1] in seen:
                continue
            seen.add(key[1])
            seq = _reconstruct(history, key, depth, edges)
            chains.append(chain_metrics(P, seq))
            if both_directions:
                chains.append(chain_metrics(P, list(reversed(seq))))
            if len(chains) >= top:
                return chains
        if len(chains) >= top:
            break
    return chains
