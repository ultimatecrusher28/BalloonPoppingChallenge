"""The agent-side rocket model must match the live environment's curves.

``gnc/vehicle/rocket_model.py`` rebuilds mass(t) and thrust(t) from
given_parameters alone; ``tests/agents/data/rocket_constants_s0.json`` holds
the curves extracted from a live scenario-0 environment. A model that
drifts from the simulator plans legs the simulator will not fly.
"""

import json
import os
import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

_CONSTANTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "rocket_constants_s0.json"
)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
@unittest.skipUnless(os.path.exists(_CONSTANTS), "extracted constants not present")
class TestRocketModelMatchesEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from BalloonPoppingGymEnv.agents.gnc.vehicle.rocket_model import RocketModel
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        with open(_CONSTANTS) as f:
            cls.env_curves = json.load(f)
        scenario = int(cls.env_curves.get("scenario", 0))
        _params, given = load_scenario_parameters(scenario if scenario else 1)
        cls.model = RocketModel(given)

    def test_mass_curve(self):
        t = np.asarray(self.env_curves["t"], float)
        mass_env = np.asarray(self.env_curves["mass"], float)
        mass_model = self.model.mass(t)
        rel = np.abs(mass_model - mass_env) / mass_env
        self.assertLess(
            float(rel.max()), 2e-3, f"mass curve drifts up to {rel.max():.2e} relative"
        )

    def test_thrust_curve(self):
        t = np.asarray(self.env_curves["t"], float)
        thrust_env = np.asarray(self.env_curves["thrust_full"], float)
        burn_out = float(self.env_curves["burn_out"])
        burning = (t > 0.05) & (t < burn_out - 0.05)
        thrust_model = self.model.thrust(t)
        np.testing.assert_allclose(
            thrust_model[burning], thrust_env[burning], rtol=1e-3
        )
        after = t > burn_out + 0.05
        if after.any():
            self.assertTrue(
                np.all(thrust_model[after] == 0.0),
                "model must produce no thrust after burnout",
            )

    def test_burn_time_matches(self):
        self.assertAlmostEqual(
            self.model.burn_time,
            float(self.env_curves["burn_out"]) - float(self.env_curves["burn_start"]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
