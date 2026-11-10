"""The locked tour program and its re-shoot schedule: coast_shoot.program_cmd picks the segment of [t0, burn_time] and the
velocity direction from burn on; coast_shoot.active keeps the segments not finished yet (same boundaries); the agent's
_shoot_step re-shoots every shoot_reshoot_dt s from the lock and, once at most t_left s of the burn is left
(shoot_reshoot_late), at once and then every dt s with its iteration count, while more than tmin s of the burn is left; a
new lock restarts the clock; from burnout on the command is the velocity direction."""

from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

_STACK = find_spec("rocketpy") is not None
T_LAUNCH = 44.0
_D = np.array([[0.2, 0.0, 1.0], [0.3, 0.1, 1.0], [0.1, 0.3, 1.0], [0.0, 0.2, 1.0]])
_D = _D / np.linalg.norm(_D, axis=1, keepdims=True)


def test_program_cmd_segments_and_after_burn():
    from BalloonPoppingGymEnv.agents.gnc.rl.coast_shoot import program_cmd

    dirs = np.repeat(_D[None], 6, 0)
    t = np.array([26.0, 26.99, 27.0, 29.99, 30.0, 31.0])
    k = [0, 0, 1, 3, 3, 3]
    assert np.array_equal(program_cmd(dirs, t, 26.0, 30.0), _D[k])
    assert np.array_equal(
        program_cmd(dirs, t, np.full(6, 26.0), 30.0), _D[k]
    )  # t0 per row
    vel = np.tile([3.0, 0.0, 4.0], (6, 1))
    n = program_cmd(dirs, t, 26.0, 30.0, vel=vel)
    assert np.array_equal(n[:4], _D[k[:4]]) and np.allclose(n[4:], [0.6, 0.0, 0.8])


def test_active_keeps_the_segments_not_finished():
    from BalloonPoppingGymEnv.agents.gnc.rl.coast_shoot import active

    burn = 30.0
    D0, ts = active(_D, 26.0, 26.5, burn)
    assert np.array_equal(D0, _D) and ts == 26.0
    d3, ts3 = active(_D, 26.0, 27.2, burn)
    assert np.array_equal(d3, _D[1:]) and np.isclose(ts3, 26.0 + (burn - 26.0) / 4)


@pytest.mark.skipif(not _STACK, reason="simulation stack not installed")
def test_the_reshoot_schedule():
    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import PlannerAgent
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from tests.agents.finals_cfg import agent_kwargs

    a = PlannerAgent(load_scenario_parameters(40)[1], **agent_kwargs())
    a.t_launch = T_LAUNCH
    burn = a.sim.burn_time
    assert (a.shoot_reshoot_dt, a.shoot_reshoot_tmin, a.shoot_reshoot_late) == (
        0.5,
        0.3,
        (4.0, 0.1, 0.05, 8.0),
    )
    calls = []

    def fake(sc, obs, full, pos, vel, t_rel, D0, tseg, bpos, burn_, iters=None):
        calls.append((round(t_rel, 3), iters))
        return dict(
            dirs=np.asarray(D0).tolist(),
            miss=[0.1, 0.2],
            tpass=[T_LAUNCH + 31.0, T_LAUNCH + 32.0],
            it=1,
        )

    a._tour_reshoot = fake
    vel = np.array([3.0, 0.0, 40.0])
    est = SimpleNamespace(
        position=np.array([0.0, 0.0, 400.0]),
        velocity=vel,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.zeros(3),
    )
    sc = dict(dirs=_D.tolist(), t0=20.0, dt=0.01, tour=dict())

    def at(t_rel):
        return a._shoot_step(est, None, T_LAUNCH + t_rel, t_rel)

    a.planner.shoot_cur = sc
    n = at(20.0)  # the lock: no re-shoot yet, segment 0
    assert (
        np.array_equal(n, _D[0])
        and calls == []
        and np.isclose(a._shoot_next, T_LAUNCH + 20.5)
    )
    at(20.3)
    at(20.5)
    at(20.7)
    assert (
        calls == [(20.5, None)]
        and a.shoot_stats["n"] == 1
        and a.log[-1][0] == "shoot_re"
    )
    assert a.planner.shoot_cur is not sc and a.planner.shoot_cur["n_re"] == 1
    # a new lock 25.8 s after launch: the 0.5 s clock would re-shoot at 26.3, but at 26.0 only 4 s of the burn is left: the
    # late schedule starts at once, then every 0.1 s with 8 iterations
    a.planner.shoot_cur = dict(sc, t0=25.8)
    calls.clear()
    at(25.8)
    assert calls == [] and np.isclose(a._shoot_next, T_LAUNCH + 26.3)
    at(26.0)
    at(26.05)
    at(26.1)
    assert calls == [(26.0, 8), (26.1, 8)]
    calls.clear()
    at(29.9)  # 0.1 s left > tmin 0.05: re-shoot
    at(29.96)  # 0.04 s left: none
    assert calls == [(29.9, 8)]
    assert np.allclose(
        at(burn), vel / np.linalg.norm(vel)
    )  # from burnout on: the velocity direction
