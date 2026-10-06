"""Provider package — one module per provider, re-exported for compatibility.

Layout:
  base.py           message/response types + the LLMProvider ABC
  registry.py       name -> builder registry (self-registration seam)
  stub.py           StubProvider (offline / tests)
  openai_compat.py  OpenAICompatProvider (any OpenAI-compatible endpoint)
  factory.py        build_provider(spec)

Every name previously importable from ``browser_agent.runtime.providers`` is
re-exported here, so existing imports keep working unchanged. Add a provider by
dropping a module that calls ``register_provider("<name>", builder)`` and
importing it from ``factory.py`` (or here) for its registration side effect.
"""

from browser_agent.runtime.providers.base import (
    ChatCompletionResponse,
    ConversationTurn,
    LLMProvider,
    ToolCall,
    ToolResult,
)
from browser_agent.runtime.providers.factory import build_provider
from browser_agent.runtime.providers.openai_compat import OpenAICompatProvider
from browser_agent.runtime.providers.registry import (
    register_provider,
    registered_providers,
)
from browser_agent.runtime.providers.stub import StubProvider

__all__ = [
    "ChatCompletionResponse", "ConversationTurn", "LLMProvider",
    "ToolCall", "ToolResult", "StubProvider", "OpenAICompatProvider",
    "build_provider", "register_provider", "registered_providers",
]