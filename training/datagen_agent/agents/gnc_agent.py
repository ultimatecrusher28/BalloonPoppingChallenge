"""Loader-friendly entry point for the five-module GNC agent.

The evaluator/harness load an agent class from a *file path* via
``importlib.util.spec_from_file_location``, which does not establish package
context for relative imports. This thin module re-exports ``GncAgent`` using
an absolute import so it can be pointed at directly:

    agent_module_path: ./BalloonPoppingGymEnv/agents/gnc_agent.py
    agent_class_name:  GncAgent
"""

from BalloonPoppingGymEnv.agents.gnc.agent import GncAgent

__all__ = ["GncAgent"]
