"""RLAgent: residual learned guidance on top of the given-mode inner loop.

Decisions mirror the training env (research/rl/env_vec.py): every PRE_DT
before launch (launch when the launch logit > 0), every DECISION_DT after.
The policy (numpy MLP, weights from research/rl/runs/*/best.npz) outputs a
residual on ``features.base_action``; ``features.action_to_command`` maps it
to a thrust direction; ``thrust_dir`` + ``CompensatedInnerLoop`` fly it at
100 Hz. Reads only observation + given_parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ....envs.balloon_world import check_action
from ...base_agent import BaseAgent
from .. import rocketnav as _rocketnav  # noqa: F401 (register)
from .. import registry
from ..follow.inner_loop import CompensatedInnerLoop, quat_to_matrix
from ..interfaces import LaunchDecision
from ..vehicle.given_limits import GivenLimits
from ..vehicle.rocket_model import RocketModel
from . import features as F
from .thrust_dir import accel_for_thrust_dir

PRE_DT = 0.5
DECISION_DT = 0.1
T_FORCE_LAUNCH = 75.0
MIN_INCL_DEG = 45.0
THR_LO = 0.3


def _idle(launch=False, attitude=(90.0, 0.0)):
    return {
        "launch": bool(launch),
        "launch_inclination_heading": np.array(attitude, float),
        "tvc": np.zeros(2),
        "throttle": 1.0,
        "roll": 0.0,
    }


class MLPPolicy:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.mean, self.var = d["obs_mean"], d["obs_var"]
        self.layers = []
        k = 0
        while f"pi_W{k}" in d.files:
            self.layers.append((d[f"pi_W{k}"], d[f"pi_b{k}"]))
            k += 1
        self.meta = json.loads(str(d["meta"]))

    def __call__(self, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x


class RLAgent(BaseAgent):
    def __init__(
        self,
        given_parameters,
        *,
        policy_path,
        base_only=False,
        force_launch_t=None,
        burnout_aim=None,
        rocketnav="passthrough",
        **_kwargs,
    ):
        super().__init__(given_parameters)
        gp = given_parameters
        self.model = RocketModel(gp)
        self.tables = GivenLimits(gp, self.model, turn_model="slew")
        self.loop = CompensatedInnerLoop(gp, self.model, tables=self.tables)
        self.nav = registry.build(
            "rocketnav", rocketnav, gp, self.model
        )  # "mekf" for noisy sc2/3
        path = Path(policy_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        self.policy = MLPPolicy(path)
        self.base_only = bool(base_only)
        self.force_launch_t = None if force_launch_t is None else float(force_launch_t)
        self.elev = float(self.model.elevation)
        self.pad = np.array([0.0, 0.0, self.elev])
        self.t_launch = None
        self.next_pre = 0.0
        self.next_dec = None
        self.n_cmd = np.array([0.0, 0.0, 1.0])
        self.thr_cmd = 1.0
        self.pops_seen = 0
        self._last_good = None
        self.aim = None
        if (
            burnout_aim
        ):  # dict of BurnoutAim kwargs, or True for defaults (off by default)
            from .burnout_aim import BurnoutAim

            self.aim = BurnoutAim(
                **(burnout_aim if isinstance(burnout_aim, dict) else {})
            )
        self.aim_log = []

    # -- policy --------------------------------------------------------------
    def _features(self, observation, pos, vel, nose, launched, t_rel, t):
        bs = np.asarray(observation["balloon_states"], float)
        st = np.asarray(observation["balloon_status"]).reshape(1, -1)
        o, psi, _ = F.build(
            pos[None],
            vel[None],
            nose[None],
            np.array([launched]),
            np.array([t_rel]),
            np.array([t]),
            bs[None, :, :3],
            bs[None, :, 3:6],
            st,
            elevation=self.elev,
            burn_time=self.model.burn_time,
            pad=self.pad,
            far=bool(self.policy.meta.get("far", False)),
            string=self.policy.meta.get("string"),
        )
        return o, psi

    def _act(self, o):
        ramp = bool(self.policy.meta.get("launch_ramp", False))
        pg = bool(self.policy.meta.get("pursue_ground", False))
        tc = bool(self.policy.meta.get("tgo_closing", False))
        lo = self.policy.meta.get("low_open")
        lu = self.policy.meta.get("late_up")
        sg = self.policy.meta.get("string")
        bo = self.policy.meta.get("boost")
        if self.base_only:
            return F.base_action(o, launch_ramp=ramp, pursue_ground=pg, tgo_closing=tc)
        out = self.policy(o)
        slot = out[:, 5:].argmax(1) if self.policy.meta.get("slot") else None
        return (
            F.base_action(
                o,
                slot,
                launch_ramp=ramp,
                pursue_ground=pg,
                tgo_closing=tc,
                low_open=lo,
                late_up=lu,
                string=sg,
                boost=bo,
            )
            + out[:, :5]
        )

    def get_action(self, observation):
        t = float(observation["simulation_time"])
        if self.t_launch is None:
            if self.force_launch_t is not None:
                if t + 1e-9 < self.force_launch_t:
                    return _idle()
            elif t + 1e-9 < self.next_pre:
                return _idle()
            self.next_pre = t + PRE_DT
            nose0 = np.array([0.0, 0.0, 1.0])
            o, psi = self._features(
                observation, self.pad, np.zeros(3), nose0, False, 0.0, t
            )
            a = self._act(o)
            if (
                self.force_launch_t is None
                and a[0, 4] <= 0
                and t + PRE_DT < T_FORCE_LAUNCH
            ):
                return _idle()
            n = F.action_to_command(a, psi, np.array([0.0]), np.array([0.0]))[0]
            el = max(np.degrees(np.arcsin(np.clip(n[2], -1, 1))), MIN_INCL_DEG)
            head = float(
                np.degrees(np.arctan2(n[0], n[1])) % 360.0
            )  # compass: 0 = +Y (north)
            self.t_launch = t
            self.next_dec = t + DECISION_DT
            dec = LaunchDecision(
                t_launch=t,
                inclination_deg=float(el),
                heading_deg=head,
                max_rail_tilt_deg=90.0 - MIN_INCL_DEG,
            )
            self.nav.on_launch(dec, t)
            self.loop.on_launch(t)
            self.n_cmd = np.array(
                [
                    np.cos(np.radians(el)) * np.sin(np.radians(head)),
                    np.cos(np.radians(el)) * np.cos(np.radians(head)),
                    np.sin(np.radians(el)),
                ]
            )
            return _idle(launch=True, attitude=(float(el), head))

        est = self.nav.update(observation, deadline=float("inf"))
        t_rel = t - self.t_launch
        if t + 1e-9 >= self.next_dec:
            self.next_dec += DECISION_DT
            nose = quat_to_matrix(est.quaternion)[:, 2]
            o, psi = self._features(
                observation,
                np.asarray(est.position, float),
                np.asarray(est.velocity, float),
                nose,
                True,
                t_rel,
                t,
            )
            a = self._act(o)
            self.n_cmd = F.action_to_command(
                a,
                psi,
                np.array([est.position[2] - self.elev]),
                np.array([est.velocity[2]]),
                safety=self.policy.meta.get("safety", "default"),
            )[0]
            self.thr_cmd = float(THR_LO + (1 - THR_LO) / (1 + np.exp(-a[0, 3])))
            if self.aim is not None and self.aim.active(t_rel):
                bs = np.asarray(observation["balloon_states"], float)
                st = np.asarray(observation["balloon_status"]).reshape(-1)
                d = self.aim.solve(
                    t_rel,
                    est.position,
                    est.velocity,
                    est.quaternion,
                    est.omega,
                    bs[:, :3],
                    bs[:, 3:6],
                    st,
                    self.n_cmd,
                )
                if d is not None:
                    self.n_cmd, self.thr_cmd = d, 1.0
                self.aim_log.append((round(t_rel, 2), self.aim.last))
        acc = accel_for_thrust_dir(
            self.n_cmd, self.thr_cmd, est, self.model, self.tables, t_rel
        )
        action, _, _ = self.loop.solve(acc, est)
        if check_action(action):
            return self._last_good if self._last_good is not None else _idle()
        self._last_good = action
        return action
