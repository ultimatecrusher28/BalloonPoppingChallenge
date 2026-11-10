"""GNC package of the planner agent (agents/planner_agent.py -> rl/planner_agent.py).

Subpackages: rl (beam planner, given-parameter simulator, balloon predictor, whole-burn tour), rocketnav (navigation
filter), follow (inner loop and its feedforward reference), vehicle (rocket and lateral-force models from given
parameters). Contracts live in ``interfaces.py``; the navigation filter is chosen by name through ``registry.py``.

Only ``observation`` and ``given_parameters`` are consumed (competition rule).
"""
