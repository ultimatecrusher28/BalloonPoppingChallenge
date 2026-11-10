"""The finals config for the agent tests: agent_kwargs of finals/sc40_finals_lm10.yaml, optionally overridden."""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "finals" / "sc40_finals_lm10.yaml"


def agent_kwargs(**over):
    kw = copy.deepcopy(yaml.safe_load(open(CFG, encoding="utf-8-sig"))["agent_kwargs"])
    kw.update(over)
    return kw
