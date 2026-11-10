"""The agent that generated the critic's training data, frozen (BalloonPoppingGymEnv/agents of commit eeaf0573, the same
agent code as 17191343 where the deep-plan targets ran): use() makes the package name BalloonPoppingGymEnv.agents resolve
to training/datagen_agent/agents, while BalloonPoppingGymEnv.envs and .evaluation stay the official package. Call it
before anything imports BalloonPoppingGymEnv.agents. The finals agent (BalloonPoppingGymEnv/agents) never imports this."""

import sys
import types
from pathlib import Path

AGENTS = Path(__file__).resolve().parent / "agents"


def use():
    import BalloonPoppingGymEnv

    loaded = [m for m in sys.modules if m.startswith("BalloonPoppingGymEnv.agents")]
    if loaded:
        raise RuntimeError(f"BalloonPoppingGymEnv.agents already imported: {loaded[:3]}")
    m = types.ModuleType("BalloonPoppingGymEnv.agents")
    m.__path__ = [str(AGENTS)]
    m.__package__ = "BalloonPoppingGymEnv.agents"
    sys.modules["BalloonPoppingGymEnv.agents"] = m
    BalloonPoppingGymEnv.agents = m
    return m
