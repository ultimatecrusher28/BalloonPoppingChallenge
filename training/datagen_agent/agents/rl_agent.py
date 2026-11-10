"""Loader-friendly entry point for the residual RL agent (see gnc_agent.py for why).

agent_module_path: ./BalloonPoppingGymEnv/agents/rl_agent.py
agent_class_name:  RLAgent
"""

from BalloonPoppingGymEnv.agents.gnc.rl.agent import RLAgent

__all__ = ["RLAgent"]
