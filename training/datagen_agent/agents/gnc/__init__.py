"""Five-module GNC agent package.

Modules: launchplan (launch decision + initial tour), replan (new-balloon
replanning), rocketnav (rocket state estimation), balloonnav (balloon
prediction), follow (path following + inner loop). Contracts live in
``interfaces.py``; candidates register by readable name in ``registry.py``
and are selected from the eval config's ``agent_kwargs``.

Only ``observation`` and ``given_parameters`` are consumed (competition rule).
"""
