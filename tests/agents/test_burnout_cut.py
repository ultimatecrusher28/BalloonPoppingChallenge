"""GivenSim's burnout cut: the thrust of a step is T0 times the step's share before the env's burn out (burn_time + ENV_DT
after the agent's launch time, as the env's motor: thrust up to burn out, 0 after), so on the env's 0.01 s grid it is the
plain step and a longer step across the burn out gets the pro-rata thrust; the rollout tables (step_tables) carry the same
T0 as step's own time_tables; the agent only runs with the cut."""

from importlib.util import find_spec

import numpy as np
import pytest

_STACK = find_spec("rocketpy") is not None
pytestmark = pytest.mark.skipif(not _STACK, reason="simulation stack not installed")


@pytest.fixture(scope="module")
def gp():
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

    return load_scenario_parameters(40)[1]


@pytest.fixture(scope="module")
def sim(gp):
    from BalloonPoppingGymEnv.agents.gnc.rl.given_sim import GivenSim

    return GivenSim(gp, ff=True)


def _full_thrust(s, t):
    return s.thrust_t[int(min(t, s.burn_time) / s.dt_tab)]


def test_the_cut_on_the_env_grid(sim):
    from BalloonPoppingGymEnv.agents.gnc.rl.given_sim import ENV_DT

    b = sim.burn_time
    assert sim.burn_end == b + ENV_DT
    T0 = [
        float(sim.time_tables(np.array([t]), 0.01)[0][0])
        for t in (b - 0.01, b, b + 0.01)
    ]
    assert (
        T0[0] == _full_thrust(sim, b - 0.01) > 0.0
    )  # the env's thrust is on in the step 30.00 -> 30.01 s, off after
    assert T0[1] == _full_thrust(sim, b) and T0[2] == 0.0


def test_a_long_step_from_burn_time_gets_its_share(sim):
    from BalloonPoppingGymEnv.agents.gnc.rl.given_sim import ENV_DT

    b = sim.burn_time
    for t, dt in ((b, 0.04), (b - 0.02, 0.05), (b + 0.005, 0.01)):
        share = (b + ENV_DT - t) / dt  # 0.25, 0.6, 0.5
        T0, burn, _ = sim.time_tables(np.array([t]), dt)
        assert (
            np.isclose(T0[0], _full_thrust(sim, t) * share, rtol=1e-12, atol=0.0)
            and burn[0]
        ), (t, dt)


def test_rollout_tables_carry_the_cut(sim):
    b = sim.burn_time
    dt = 0.04
    trk = b + np.array([[-0.10, -0.03], [-0.03, 0.0], [0.0, 0.005], [0.02, 0.04]])
    X1, B1 = sim.step_tables(trk, True, dt)
    T1, burn1, _ = sim.time_tables(trk, dt)
    assert np.array_equal(X1[0], T1) and np.array_equal(B1, burn1)
    assert (T1[0] > 0).all() and (T1[2] > 0).all() and (T1[3, 1] == 0.0)
    f, tic = sim._cut(trk, dt)
    ti = (np.maximum(trk, 0) / sim.dt_tab).astype(int)
    exp = sim._step_fields(sim.thrust_t[tic] * f, sim._tab3[ti])
    assert np.allclose(np.moveaxis(X1, 0, -1), exp, equal_nan=True)


def test_the_agent_requires_the_cut(gp):
    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import PlannerAgent
    from tests.agents.finals_cfg import agent_kwargs

    with pytest.raises(ValueError):
        PlannerAgent(gp, **agent_kwargs(sim_burnout_cut=False))
