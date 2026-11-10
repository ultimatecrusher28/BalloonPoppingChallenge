"""Line-of-sight fill of the planner's guidance law (gnc/rl/planner.py guidance(los_fill) / los_fill_dir; lean rows).
Pure numpy checks of the law, plus the row-table plumbing of PlannerAgent(los_variants)."""

from importlib.util import find_spec

import numpy as np
import pytest

from BalloonPoppingGymEnv.agents.gnc.rl.planner import G, guidance, los_fill_dir

_STACK = find_spec("rocketpy") is not None


def _case(n=64, seed=0):
    """Random rocket / balloon geometries (high above the ground floor: h_agl 200 m, climbing)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 50.0, (n, 3))
    pos[:, 2] = 300.0 + rng.uniform(0.0, 100.0, n)
    vel = rng.normal(0.0, 30.0, (n, 3))
    vel[:, 2] = np.abs(vel[:, 2]) + 5.0
    tp = pos + rng.normal(0.0, 120.0, (n, 3))
    tv = rng.normal(0.0, 3.0, (n, 3))
    h = np.full(n, 200.0)
    return pos, vel, tp, tv, h


def _a_cmd(pos, vel, tp, tv, nav, tgo_min, amax=20.0):
    """The law's clipped lead-pursuit acceleration (guidance, no aim offset, no boost)."""
    r = tp - pos
    vrel = tv - vel
    dist = np.linalg.norm(r, axis=1)
    vc = -(r * vrel).sum(1) / np.maximum(dist, 1e-6)
    tgo = np.clip(dist / np.maximum(vc, 12.0), tgo_min, 6.0)[:, None]
    return np.clip(nav * (r + vrel * tgo) / tgo**2, -amax, amax), r / dist[:, None]


def test_fill_zero_is_the_old_law_exactly():
    pos, vel, tp, tv, h = _case()
    n_old = guidance(pos, vel, tp, tv, h, nav=3.0, tgo_min=0.3)
    A = np.full(len(pos), 30.0)
    # scalar 0, a per-row zero array, with or without a_av: bit-identical
    for kw in (
        dict(los_fill=0.0),
        dict(los_fill=np.zeros(len(pos))),
        dict(los_fill=0.0, a_av=A),
        dict(los_fill=np.zeros(len(pos)), a_av=A),
    ):
        assert np.array_equal(
            guidance(pos, vel, tp, tv, h, nav=3.0, tgo_min=0.3, **kw), n_old
        )
    # mixed rows: the f = 0 rows of a batch with fill rows are unchanged too
    f = np.where(np.arange(len(pos)) % 2 == 0, 0.0, 1.0)
    n_mix = guidance(pos, vel, tp, tv, h, nav=3.0, tgo_min=0.3, los_fill=f, a_av=A)
    assert np.array_equal(n_mix[f == 0], n_old[f == 0])
    assert not np.allclose(n_mix[f > 0], n_old[f > 0])
    # also through the viable-intercept path (vi rows): the vi = 0 rows there are unchanged
    nose = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    vi = np.where(np.arange(len(pos)) % 3 == 0, 0.9, 0.0)
    kw = dict(nav=3.0, tgo_min=0.3, vi=vi, nose=nose, a_av=A)
    assert np.array_equal(
        guidance(pos, vel, tp, tv, h, los_fill=np.zeros(len(pos)), **kw),
        guidance(pos, vel, tp, tv, h, **kw),
    )


def test_fill_keeps_the_normal_acceleration_and_pushes_along_the_los():
    pos, vel, tp, tv, h = _case(256, seed=1)
    A = 45.0  # above |a_cmd + G z| <= 41 m/s^2 (components clipped at 20), so nearly every row is feasible
    a_cmd, u = _a_cmd(pos, vel, tp, tv, nav=3.0, tgo_min=0.3)
    w = a_cmd.copy()
    w[:, 2] += G
    wp = w - (w * u).sum(1, keepdims=True) * u
    feas = np.linalg.norm(wp, axis=1) < 0.9 * A
    assert feas.sum() > 50
    n = guidance(
        pos,
        vel,
        tp,
        tv,
        h,
        nav=3.0,
        tgo_min=0.3,
        los_fill=1.0,
        a_av=np.full(len(pos), A),
    )
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    acc = A * n
    acc[:, 2] -= G  # net acceleration of full thrust along n (gravity included)
    perp = lambda x: x - (x * u).sum(1, keepdims=True) * u  # noqa: E731
    assert np.allclose(perp(acc)[feas], perp(a_cmd)[feas], atol=1e-9)
    along = (acc * u).sum(1)
    assert (along[feas] > 0).all()
    # the law alone has (nearly) no thrust left along the LOS on these rows compared with the fill
    n0 = guidance(pos, vel, tp, tv, h, nav=3.0, tgo_min=0.3)
    acc0 = A * n0
    acc0[:, 2] -= G
    assert np.median(along[feas] - (acc0 * u).sum(1)[feas]) > 0


def test_fill_dir_clips_infeasible_normal_commands():
    rng = np.random.default_rng(2)
    u = rng.normal(size=(32, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = rng.normal(0.0, 100.0, (32, 3))  # far beyond A
    A = np.full(32, 20.0)
    n_reg = w / np.linalg.norm(w, axis=1, keepdims=True)
    n = los_fill_dir(n_reg, w, u, np.ones(32), A)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    np_ = n - (n * u).sum(1, keepdims=True) * u
    assert np.allclose(
        np.linalg.norm(np_, axis=1), 0.98, atol=1e-9
    )  # |w_perp / A| capped at 0.98
    assert ((n * u).sum(1) > 0).all()
    # partial fill: unit((1 - f) n_reg + f n_los)
    nh = los_fill_dir(n_reg, w, u, np.full(32, 0.5), A)
    ref = 0.5 * n_reg + 0.5 * n
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert np.allclose(nh, ref, atol=1e-12)


def test_no_usable_thrust_keeps_the_regular_direction():
    pos, vel, tp, tv, h = _case(16, seed=3)
    n_old = guidance(pos, vel, tp, tv, h, nav=2.0, tgo_min=0.8)
    for a_av in (
        None,
        0.0,
        np.zeros(16),
        np.nan,
        np.full(16, np.nan),
        np.full(16, 1e-4),
    ):
        n = guidance(pos, vel, tp, tv, h, nav=2.0, tgo_min=0.8, los_fill=1.0, a_av=a_av)
        assert np.array_equal(n, n_old), a_av
    # per row: only the rows with a usable A are filled
    A = np.array([30.0, 0.0, np.nan, 30.0] * 4)
    n = guidance(pos, vel, tp, tv, h, nav=2.0, tgo_min=0.8, los_fill=1.0, a_av=A)
    bad = ~(np.isfinite(A) & (A > 1e-3))
    assert np.array_equal(n[bad], n_old[bad])
    assert not np.allclose(n[~bad], n_old[~bad])


@pytest.mark.skipif(not _STACK, reason="simulation stack not installed")
def test_planner_agent_appends_los_rows_before_the_x_rows():
    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import (
        VI_VARIANTS,
        PlannerAgent,
    )
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from tests.agents.finals_cfg import agent_kwargs

    _, gp = load_scenario_parameters(40)
    var = [
        [2.0, 0.8, 0.0, 0.0, 0.0],
        [3.0, 0.3, 0.0, 0.0, 0.0],
        [3.0, 0.3, 10.0, 0.0, 0.0],
    ]
    base = PlannerAgent(gp, **agent_kwargs(variants=var, los_variants=[]))
    lean = PlannerAgent(
        gp,
        **agent_kwargs(
            variants=var, los_variants=[[2.0, 0.8, 0.0, 1.0], [4.0, 0.3, 0.0, 0.5]]
        ),
    )
    nx = len(base.planner.xr["vars"]) + 1  # the x variants and the keep row
    V0, V1 = base.planner.V, lean.planner.V
    assert V0.shape == (len(var) + len(VI_VARIANTS) + nx, 12) and not V0[:, 11].any()
    n_reg = len(var) + len(VI_VARIANTS)
    assert lean.planner.nV0 == n_reg + 2 and V1.shape == (n_reg + 2 + nx, 12)
    assert np.array_equal(V1[:n_reg], V0[:n_reg]) and np.array_equal(
        V1[n_reg + 2 :], V0[n_reg:]
    )
    assert np.array_equal(V1[n_reg], [2.0, 0.8, 0.0] + [0.0] * 8 + [1.0])
    assert np.array_equal(V1[n_reg + 1], [4.0, 0.3, 0.0] + [0.0] * 8 + [0.5])
    assert list(lean.planner.xcode) == [-1] * (n_reg + 2) + list(range(nx))
