"""WindProfilePredictor.wind_table: the GivenSim hand-off holds only [z_lo, highest balloon] (wind-estimation, #4)."""

import numpy as np

from BalloonPoppingGymEnv.agents.gnc.rl.predict import WindProfilePredictor


def _fed(top=150.0):
    p = WindProfilePredictor(z_min=20.0)
    rng = np.random.default_rng(0)
    for k in range(60):
        t = 0.1 * k
        z = rng.uniform(21.0, top, 40)
        pos = np.stack([np.zeros(40), np.zeros(40), z], 1)
        vel = np.stack([0.01 * z, -0.005 * z, np.full(40, 6.0)], 1)
        p.observe(t, pos, vel, np.ones(40, bool))
    p.refit(6.0, pos, vel, np.ones(40, bool))
    return p


def test_none_without_profile():
    assert WindProfilePredictor(z_min=20.0).wind_table(30.0) is None


def test_range_is_zlo_to_highest_data():
    p = _fed(top=150.0)
    z, u, v = p.wind_table(30.0)
    assert z[0] >= 30.0 and z[0] < 31.0
    assert 150.0 <= z[-1] <= 151.0 + 1e-9
    # inside the range the table is the profile itself
    np.testing.assert_allclose(u, p.U[(p.zg >= 30.0) & (p.zg <= z[-1]), 0])
    # outside, np.interp holds the end values: no linear continuation above the data
    assert np.interp(500.0, z, u) == u[-1] and np.interp(20.0, z, v) == v[0]
    assert abs(u[-1] - 0.01 * z[-1]) < 0.05


def test_whole_grid_when_no_zlo():
    p = _fed(top=150.0)
    z, _, _ = p.wind_table(None)
    assert z[0] == p.zg[0]
