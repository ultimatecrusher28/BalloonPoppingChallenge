"""Model-aided rate pieces of MekfNav: given moment model, actuators, local gravity, disturbance bank.

Synthetic data only (no environment). References: the official ActiveRocketPy actuator class and RocketPy's Environment
gravity.
"""

import unittest
import warnings
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.agents.gnc.rocketnav.rate_model import (
        GivenActuator,
        GmBank,
        RateModel,
        local_gravity,
    )
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestRateModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _params, cls.given = load_scenario_parameters(40)
        cls.rm = RateModel(cls.given)

    def test_local_gravity_matches_rocketpy_environment(self):
        from rocketpy import Environment

        e = self.given["environment"]
        ref = Environment(
            latitude=e["latitude"], longitude=e["longitude"], elevation=e["elevation"]
        )
        g = local_gravity(self.given)
        for h in (e["elevation"], 500.0, 2500.0):
            self.assertAlmostEqual(
                g(h), float(ref.gravity(h)), places=7
            )  # RocketPy tabulates the formula (6e-9)
        self.assertAlmostEqual(g(20.0), 9.78762, places=5)

    def test_thrust_off_after_burn(self):
        i, burning = self.rm.index(self.rm.burn)
        self.assertFalse(burning)
        i, burning = self.rm.index(self.rm.burn - self.rm.dt)
        self.assertTrue(burning)

    def test_actuators_match_activerocketpy(self):
        from rocketpy.rocket.actuator.actuator import Actuator

        class _Act(
            Actuator
        ):  # the official base class is abstract only in its info methods
            def info(self):
                pass

            def all_info(self):
                pass

        c = self.given["rocket"]["control"]
        dt = float(self.given["simulation"]["time_step"])
        rng = np.random.default_rng(5)
        for tau, rate, lo, hi, init in (
            (
                c["gimbal_time_constant"],
                c["gimbal_rate_limit"],
                -c["max_gimbal_angle"],
                c["max_gimbal_angle"],
                0.0,
            ),
            (
                c["throttle_time_constant"],
                c["throttle_rate_limit"],
                c["throttle_range"][0],
                c["throttle_range"][1],
                1.0,
            ),
        ):
            ref = _Act(
                "test",
                demand_rate=1 / dt,
                actuator_range=(lo, hi),
                actuator_rate_limit=rate,
                clamp=True,
                actuator_initial_output=init,
                actuator_time_constant=tau,
            )
            mine = GivenActuator(tau, rate, lo, hi, init, dt, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for u in rng.uniform(lo - 2, hi + 2, 400):
                    ref.actuator_output = float(u)
                    self.assertEqual(
                        float(mine.apply([u])[0]), float(ref.actuator_output)
                    )

    def test_f_max_separates_spike(self):
        i, _ = self.rm.index(
            self.rm.burn - self.rm.dt
        )  # just before cutoff: lowest mass, largest normal samples
        fmax = self.rm.f_max(16.0, i, 300.0, 1.0)
        self.assertGreater(
            fmax, 13.3
        )  # largest normal accelerometer sample measured on sc40 (13.28 m/s^2)
        self.assertLess(fmax, 1086.0 / 10)  # the cutoff spike is far above
        i, _ = self.rm.index(self.rm.burn + 3.0)
        self.assertLess(self.rm.f_max(60.0, i, 300.0, 1.0), 1086.0 / 10)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestGmBank(unittest.TestCase):
    def _run(self, q, tau, seconds=30.0, seed=0):
        dt, sig_g, rw = 0.01, 0.2, 1e-3
        bank = GmBank(dt, sig_g, rw, 30.0, 10.0)
        rng = np.random.default_rng(seed)
        phi = np.exp(-dt / tau)
        s2 = q / (2 * tau)
        d = rng.normal(0, np.sqrt(s2), 2)
        w = np.zeros(2)
        b = np.zeros(2)
        for _ in range(int(seconds / dt)):
            w = w + dt * d
            d = phi * d + rng.normal(0, np.sqrt(s2 * (1 - phi**2)), 2)
            b = b + rng.normal(0, rw * np.sqrt(dt), 2)
            bank.step(np.zeros(2), w + b + rng.normal(0, sig_g, 2))
        return bank.estimate()

    def test_grid_spans_given_limits(self):
        bank = GmBank(0.01, 0.2, 1e-3, 30.0, 10.0)
        q = np.exp(bank.lq).reshape(bank.shape)
        t = np.exp(bank.lt).reshape(bank.shape)
        self.assertAlmostEqual(q.min(), 1e-6)
        self.assertAlmostEqual(q.max(), 4.0)
        self.assertAlmostEqual(t.min(), 0.01)
        self.assertAlmostEqual(t.max(), 30.0)
        self.assertLessEqual(q[1, 0] / q[0, 0], 2.0 + 1e-12)
        self.assertLessEqual(t[0, 1] / t[0, 0], 2.0 + 1e-12)

    def test_q_recovered_within_tolerance(self):
        for q in (1e-3, 1e-2, 1e-1):
            q_hat, _, _ = self._run(q, 0.1)
            self.assertLess(
                abs(np.log(q_hat / q)), np.log(4.0), msg=f"q {q} q_hat {q_hat}"
            )

    def test_order_preserved_near_grid_edges(self):
        lo, _, _ = self._run(2e-6, 0.1)
        mid, _, _ = self._run(1e-3, 0.1)
        hi, _, _ = self._run(2.0, 0.1)
        self.assertLess(lo, mid)
        self.assertLess(mid, hi)


if __name__ == "__main__":
    unittest.main()
