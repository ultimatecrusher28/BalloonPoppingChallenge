"""Static compliance checks of the finals agent (the competition rules and the team's packaging rules):
- every import in BalloonPoppingGymEnv/agents resolves inside the agent package, the env package, numpy / scipy / rocketpy
  or the standard library, and none names research or training (the training scripts are not part of the agent);
- no code string (docstrings excluded) names a research/ or training/ path;
- the config names the shipped entry point and the shipped critic file."""

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "BalloonPoppingGymEnv" / "agents"
CFG = ROOT / "finals" / "sc40_finals_lm10.yaml"
ALLOWED = {"numpy", "scipy", "rocketpy", "BalloonPoppingGymEnv", "__future__"} | set(
    sys.stdlib_module_names
)


def _docstrings(tree):
    out = set()
    for node in ast.walk(tree):
        if (
            isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                out.add(id(first.value))
    return out


def _check(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
    docs = _docstrings(tree)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in ALLOWED or top in ("research", "training"):
                    bad.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                top = (node.module or "").split(".")[0]
                if top not in ALLOWED or top in ("research", "training"):
                    bad.append(f"from {node.module} import")
            else:  # a relative import must stay inside BalloonPoppingGymEnv
                depth = len(path.relative_to(ROOT / "BalloonPoppingGymEnv").parts) - 1
                if node.level > depth + 1:
                    bad.append(f"relative import level {node.level} leaves the package")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docs
        ):
            v = node.value
            if "research" in v or "training/" in v or "training\\" in v:
                bad.append(f"string {v[:60]!r}")
    return bad


def test_agent_imports_and_paths():
    files = sorted(AGENTS.rglob("*.py"))
    assert len(files) > 10
    bad = {str(f.relative_to(ROOT)): b for f in files if (b := _check(f))}
    assert bad == {}


def test_config_names_the_shipped_files():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8-sig"))
    assert (ROOT / cfg["agent_module_path"]).is_file() and cfg[
        "agent_class_name"
    ] == "PlannerAgent"
    crit = AGENTS / "gnc" / "rl" / "weights" / cfg["agent_kwargs"]["critic_path"]
    assert crit.is_file()
    assert (
        cfg["scenario_number"] == 40
        and cfg["leaderboard_submission"] is False
        and cfg["team_secret"] == "TBD"
    )
    assert sorted(p.name for p in (AGENTS / "gnc" / "rl" / "weights").iterdir()) == [
        crit.name
    ]
