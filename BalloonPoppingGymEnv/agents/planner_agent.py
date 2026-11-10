"""Loader-friendly entry point for the on-board rollout planner agent.

agent_module_path: ./BalloonPoppingGymEnv/agents/planner_agent.py
agent_class_name:  PlannerAgent
"""

from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import PlannerAgent

__all__ = ["PlannerAgent"]
