"""The finals agent never reads the clock (PR #154, the 600 s wall limit; the team's decision for the finals).

1. Static: no module under BalloonPoppingGymEnv/agents imports time or datetime, and none names a clock function
   (perf_counter, monotonic, process_time, thread_time, time_ns, clock_gettime, os.times).
2. One whole sc40 flight with the real clock and one with the time module's clocks jumping by a large random amount at
   every call made inside the agent's get_action (the forked tour workers inherit it) give the same action at every step,
   the same step SHA-1 (every step's full-precision rocket state, balloon status and action), the same pop times and the
   same tour decision records. Slow (two whole flights with the reset, about 15 min on 24 CPUs): skipped unless
   BPC_CLOCK_TEST=SEED:HOUR[,...] is set; BPC_CLOCK_CFG chooses the config (default: the finals config).
"""

import ast
import contextlib
import copy
import hashlib
import io
import os
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "BalloonPoppingGymEnv" / "agents"
CLOCKS = {
    "perf_counter",
    "perf_counter_ns",
    "monotonic",
    "monotonic_ns",
    "process_time",
    "process_time_ns",
    "thread_time",
    "thread_time_ns",
    "time_ns",
    "clock_gettime",
    "clock_gettime_ns",
    "times",
}
PAIRS = [p for p in os.environ.get("BPC_CLOCK_TEST", "").split(",") if p.strip()]
CFG = os.environ.get("BPC_CLOCK_CFG", str(ROOT / "finals" / "sc40_finals_lm10.yaml"))


def _clock_uses(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [
                a.name
                for a in node.names
                if a.name.split(".")[0] in ("time", "datetime")
            ]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in ("time", "datetime"):
                out.append(node.module)
            out += [a.name for a in node.names if a.name in CLOCKS]
        elif isinstance(node, ast.Attribute) and node.attr in CLOCKS:
            if node.attr != "times" or (
                isinstance(node.value, ast.Name) and node.value.id == "os"
            ):  # os.times()
                out.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in CLOCKS - {"times"}:
            out.append(node.id)
    return out


def test_no_agent_module_reads_the_clock():
    files = sorted(AGENTS.rglob("*.py"))
    assert len(files) > 10
    bad = {str(f.relative_to(ROOT)): u for f in files if (u := _clock_uses(f))}
    assert bad == {}


class _JumpClock:
    """time.time / perf_counter / monotonic / process_time stand-ins: the real value plus an offset that grows by a random
    1-1e5 s at every call."""

    NAMES = (
        "time",
        "perf_counter",
        "monotonic",
        "process_time",
        "time_ns",
        "perf_counter_ns",
        "monotonic_ns",
    )

    def __init__(self, seed=0):
        self.real = {k: getattr(time, k) for k in self.NAMES}
        self._rng = np.random.default_rng(seed)
        self._off = 0.0
        self.n = 0

    def _fake(self, k):
        f = self.real[k]

        def g():
            self.n += 1
            self._off += float(self._rng.uniform(1.0, 1e5))
            v = f()
            return v + (int(self._off * 1e9) if k.endswith("_ns") else self._off)

        return g

    @contextlib.contextmanager
    def on(self):
        for k in self.NAMES:
            setattr(time, k, self._fake(k))
        try:
            yield
        finally:
            for k, f in self.real.items():
                setattr(time, k, f)


def _strip(x):
    """the record without the timing-dependent queue bookkeeping (the workers' finish order, tasks finished after the
    lock, which CPUs the allocation gave)"""
    if isinstance(x, dict):
        return {
            k: _strip(v)
            for k, v in x.items()
            if k not in ("queue", "extra", "pin_cpus")
        }
    if isinstance(x, (list, tuple)):
        return [_strip(v) for v in x]
    return x


def _flight(cfg, seed, hour, clock=None):
    import BalloonPoppingGymEnv.evaluation.evaluate as ev
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

    sp, gp = ev.load_scenario_parameters(cfg["scenario_number"])
    sp = copy.deepcopy(sp)
    sp["scenario"]["random_seed"] = int(seed)
    d = list(sp["environment"]["date"])
    d[3] = int(hour)
    sp["environment"]["date"] = d
    sp["simulation"]["max_wall_time"] = float(
        "inf"
    )  # the env's own limit (its clock is never faked here)
    Cls = ev._load_agent_class(cfg["agent_module_path"], cfg["agent_class_name"])
    agent = Cls(gp, **dict(cfg.get("agent_kwargs") or {}))
    env = BalloonPoppingEnv(render_mode=None, parameters=sp)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(seed=int(seed))
    keys = ("launch", "launch_inclination_heading", "tvc", "roll", "throttle")
    h = hashlib.sha1()
    act, pops, pop_t, t, done = [], 0, [], 0.0, False
    dt = float(sp["simulation"]["time_step"])
    while not done:
        if clock is None:
            a = agent.get_action(obs)
        else:
            with clock.on():
                a = agent.get_action(obs)
        act.append(b"".join(np.asarray(a[k], np.float64).tobytes() for k in keys))
        obs, r, term, trunc, info = env.step(a)
        t += dt
        h.update(np.asarray(info["rocket_states"], float).tobytes())
        h.update(np.asarray(obs["balloon_status"]).tobytes())
        h.update(
            np.asarray(a["tvc"], float).tobytes()
            + np.asarray(a["throttle"], float).tobytes()
        )
        if info["popped_count"] > pops:
            pop_t += [round(t, 2)] * (info["popped_count"] - pops)
            pops = info["popped_count"]
        done = term or trunc
    return dict(
        act=act,
        sha=h.hexdigest(),
        pops=pops,
        pop_t=pop_t,
        tour=agent.tour_rec,
        prev=agent.tour_prev,
        log=agent.log,
    )


@pytest.mark.skipif(
    not PAIRS, reason="slow (two whole flights per state): set BPC_CLOCK_TEST=SEED:HOUR"
)
@pytest.mark.parametrize("pair", PAIRS)
def test_actions_do_not_depend_on_the_clock(pair):
    import yaml

    s, hr = map(int, pair.split(":"))
    cfg = yaml.safe_load(open(CFG, encoding="utf-8-sig"))
    real = _flight(cfg, s, hr)
    clock = _JumpClock(seed=s)
    fake = _flight(cfg, s, hr, clock=clock)
    print(
        f"{pair}: pops {real['pops']} / {fake['pops']}, sha {real['sha']} / {fake['sha']}, clock calls in get_action "
        f"{clock.n}, tour locked {(real['tour'] or {}).get('locked')} {(real['tour'] or {}).get('cand')}"
    )
    assert real["tour"] is not None  # the tour decision ran
    assert len(fake["act"]) == len(real["act"]) and fake["act"] == real["act"]
    assert fake["sha"] == real["sha"] and fake["pop_t"] == real["pop_t"]
    assert _strip(fake["tour"]) == _strip(real["tour"]) and _strip(
        fake["prev"]
    ) == _strip(real["prev"])
    assert repr(fake["log"]) == repr(
        real["log"]
    )  # repr: the re-shoot misses hold nan (popped targets)
