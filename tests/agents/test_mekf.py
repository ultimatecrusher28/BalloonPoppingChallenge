"""MekfNav: model-aided 19-state filter.

Synthetic sensor stream only (no environment). The reference values were produced on the same stream by the
implementation validated on sc40 in open and closed loop (branch chwang-imu-gnss-noise-filter@ddef0eb0, config
nav_rate_model). They agree bit for bit on the machine that produced them; BLAS kernels differ between CPUs in the
last bits, so the check allows a relative 1e-9, far below any change to the filter.
"""

import copy
import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.agents.gnc.interfaces import LaunchDecision
    from BalloonPoppingGymEnv.agents.gnc.rocketnav.mekf import MekfNav
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

REF = {
    "position": [-0.009115945875158581, 0.040291873711485335, 37.39564455075354],
    "velocity": [-0.05885022168617821, -0.030996380808415224, 5.647975938031315],
    "quaternion": [
        0.978248346300016,
        0.008948543780621843,
        -0.007400650320246429,
        -0.20711187049594165,
    ],
    "omega": [0.0614065487935129, -0.037571527381145234, 0.11955963653221945],
}


def _stream(n=600, seed=11, spike_at=None):
    """Early boost at 100 Hz from launch at t = 36 s: sc40 noise levels, pad at z = 20 m, 0.9 m/s^2 net climb."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        t = 36.02 + 0.01 * k
        gyro = np.array([0.05, -0.03, 0.01]) + rng.normal(0, 0.2, 3)
        acc = np.array([0.0, 0.0, 10.7]) + rng.normal(0, 0.01, 3)
        if spike_at is not None and k == spike_at:
            acc = np.array(
                [0.8, 0.2, 1086.6]
            )  # the env's numerical spike at motor cutoff
        pos = np.array([0.0, 0.0, 20.0]) + rng.normal(0, 3.0, 3)
        vel = np.array([0.0, 0.0, 0.24 + 0.9 * 0.01 * k]) + rng.normal(0, 0.1, 3)
        out.append((t, np.concatenate([gyro, acc, pos, vel])))
    return out


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestMekf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _params, cls.given = load_scenario_parameters(40)

    def _nav(self, inclination=90.0, heading=0.0):
        nav = MekfNav(self.given, None)
        nav.on_launch(
            LaunchDecision(
                t_launch=36.0,
                inclination_deg=inclination,
                heading_deg=heading,
                max_rail_tilt_deg=90.0,
            ),
            36.0,
        )
        return nav

    def _run(self, stream, actuation=None):
        nav = self._nav()
        ests = []
        for k, (t, s) in enumerate(stream):
            g, thr = actuation(k) if actuation else (np.zeros(2), 1.0)
            nav.set_actuation(g, thr)
            ests.append(
                nav.update(
                    {"simulation_time": t, "rocket_sensors": s}, deadline=float("inf")
                )
            )
        return nav, ests

    def test_matches_validated_filter(self):
        nav = self._nav(inclination=85.0, heading=30.0)
        rng = np.random.default_rng(2)
        for k, (t, s) in enumerate(_stream()):
            nav.set_actuation(rng.uniform(-5, 5, 2), 1.0 - 0.3 * (k % 50 == 0))
            est = nav.update(
                {"simulation_time": t, "rocket_sensors": s}, deadline=float("inf")
            )
        for name, ref in REF.items():
            np.testing.assert_allclose(
                getattr(est, name), ref, rtol=1e-9, atol=1e-12, err_msg=name
            )

    def test_noise_and_launch_state_from_given(self):
        nav = self._nav()
        s = self.given["rocket"]["sensors"]
        rate = s["sampling_rate"] ** 0.5
        self.assertAlmostEqual(
            nav.sig_g, s["gyro_noise_density"] * rate
        )  # 0.2 rad/s on sc40
        self.assertAlmostEqual(nav.sig_a, s["accelerometer_noise_density"] * rate)
        self.assertAlmostEqual(nav.q_bg, s["gyro_random_walk_density"] ** 2)
        self.assertAlmostEqual(nav.q_ba, s["accelerometer_random_walk_density"] ** 2)
        self.assertAlmostEqual(nav.r_pos, s["gnss_position_accuracy"] ** 2)
        self.assertAlmostEqual(nav.r_alt, s["gnss_altitude_accuracy"] ** 2)
        self.assertAlmostEqual(nav.r_vel, s["gnss_velocity_accuracy"] ** 2)
        np.testing.assert_array_equal(
            nav.p, [0.0, 0.0, self.given["environment"]["elevation"]]
        )
        self.assertTrue(nav.needs_actuation)

    def test_first_sample_initialises_velocity_from_gnss(self):
        st = _stream()
        _, ests = self._run(st)
        np.testing.assert_array_equal(ests[0].velocity, st[0][1][9:12])
        np.testing.assert_array_equal(ests[0].position, [0.0, 0.0, 20.0])

    def test_gate_removes_the_cutoff_spike(self):
        _, clean = self._run(_stream())
        _, spiked = self._run(_stream(spike_at=300))
        np.testing.assert_array_equal(spiked[299].velocity, clean[299].velocity)
        self.assertLess(
            abs(spiked[310].velocity[2] - spiked[299].velocity[2]), 0.5
        )  # no 10 m/s jump
        self.assertLess(np.linalg.norm(spiked[-1].velocity - clean[-1].velocity), 0.1)

    def test_trapezoid_position_update_with_local_gravity(self):
        nav = self._nav()
        t, s = _stream()[0]
        nav.set_actuation(np.zeros(2), 1.0)
        nav.update(
            {"simulation_time": t, "rocket_sensors": s}, deadline=float("inf")
        )  # initialises the state
        nav.v = np.array([0.0, 0.0, 1.0])
        p0 = nav.p.copy()
        nav._propagate(np.zeros(3), np.array([0.0, 0.0, 10.0]), 0.01, 5.0)
        a_z = 10.0 - nav.gravity(p0[2])
        self.assertAlmostEqual(nav.p[2] - p0[2], 0.01 + 0.5 * a_z * 1e-4, places=12)
        self.assertAlmostEqual(
            nav.gravity(20.0), 9.78762, places=5
        )  # WGS84 at the sc40 pad, not 9.80665

    def test_rate_is_smoothed(self):
        nav, ests = self._run(_stream())
        w = np.array([e.omega for e in ests])
        raw = np.array([s[0:3] for _, s in _stream()])
        self.assertTrue(np.isfinite(w).all())
        self.assertEqual(nav.P.shape, (19, 19))
        self.assertLess(np.diff(w[:, 0]).std(), 0.1 * np.diff(raw[:, 0]).std())

    def test_needs_gyro_noise(self):
        given = copy.deepcopy(self.given)
        given["rocket"]["sensors"]["gyro_noise_density"] = 0.0
        with self.assertRaises(ValueError):
            MekfNav(given, None)


if __name__ == "__main__":
    unittest.main()
