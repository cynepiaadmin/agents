"""Provider registry — the seam that lets a provider self-register.

A provider module calls ``register_provider("<name>", builder)`` at import
time; ``build_provider`` (factory.py) looks the name up here. This module
imports ONLY the ``LLMProvider`` ABC, so provider modules can import it freely
without creating an import cycle with the factory.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from browser_agent.runtime.providers.base import LLMProvider

# name -> callable(spec_dict) -> LLMProvider
ProviderBuilder = Callable[[dict], LLMProvider]
_REGISTRY: Dict[str, ProviderBuilder] = {}


def register_provider(name: str, builder: ProviderBuilder) -> ProviderBuilder:
    """Register (or replace) the builder for a provider name."""
    _REGISTRY[name] = builder
    return builder


def get_provider_builder(name: str) -> Optional[ProviderBuilder]:
    """Return the builder for ``name``, or None if it is not registered."""
    return _REGISTRY.get(name)


def registered_providers() -> List[str]:
    """Names currently registered (sorted) — handy for errors/tests."""
    return sorted(_REGISTRY)


__all__ = [
    "ProviderBuilder",
    "register_provider",
    "get_provider_builder",
    "registered_providers",
]