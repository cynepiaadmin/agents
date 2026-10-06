"""Provider factory — resolve an LLMProvider from a config spec.

Named providers register in ``registry.py`` (the stub does so here). Any OTHER
provider_name is treated as an OpenAI-compatible endpoint — the documented
catch-all for openrouter / vllm / ollama / lm-studio, which all speak the
OpenAI chat-completions shape.
"""

from __future__ import annotations

from browser_agent.runtime.providers.base import LLMProvider
from browser_agent.runtime.providers.openai_compat import _build_openai_compat
from browser_agent.runtime.providers.registry import get_provider_builder

# Import for their registration side effect (register_provider runs at import).
from browser_agent.runtime.providers import openai_compat as _openai_compat  # noqa: F401
from browser_agent.runtime.providers import stub as _stub  # noqa: F401


def build_provider(spec):
    """Resolve an LLMProvider from a config dict or pass one through.

    ``{"provider_name": "stub"}`` is offline; any name registered via
    ``register_provider`` is built by its builder; everything else is an
    OpenAI-compatible HTTP provider. api_credentials may use ``${ENV_VAR}`` —
    that interpolation is done by the config loader, not here.
    """
    if isinstance(spec, LLMProvider):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"build_provider expects dict|LLMProvider, got {type(spec)}")
    name = spec.get("provider_name")
    if name is None:
        raise ValueError("provider spec missing 'provider_name'")

    builder = get_provider_builder(name)
    if builder is not None:
        return builder(spec)
    return _build_openai_compat(spec)


__all__ = ["build_provider"]