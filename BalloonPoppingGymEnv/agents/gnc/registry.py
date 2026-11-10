"""Candidate registry: readable name -> factory, per module kind. The agent builds its navigation filter by the name in
``agent_kwargs["rocketnav"]``; factories are called as ``factory(given_parameters, model, **kwargs)`` where ``model`` is
the shared ``RocketModel``."""

from __future__ import annotations

KINDS = ("rocketnav",)
_REGISTRY: dict[str, dict[str, object]] = {kind: {} for kind in KINDS}


def register(kind: str, name: str):
    """Class/function decorator: register a candidate factory under a
    human-readable name (letter-number-only codes are banned by convention)."""
    if kind not in _REGISTRY:
        raise KeyError(f"unknown module kind {kind!r} (kinds: {KINDS})")

    def wrap(factory):
        _REGISTRY[kind][name] = factory
        return factory

    return wrap


def build(kind: str, name: str, given_parameters: dict, model, **kwargs):
    """Instantiate the named candidate for a module kind."""
    try:
        factory = _REGISTRY[kind][name]
    except KeyError:
        known = sorted(_REGISTRY.get(kind, {}))
        raise KeyError(f"no {kind} candidate named {name!r}; known: {known}") from None
    return factory(given_parameters, model, **kwargs)
