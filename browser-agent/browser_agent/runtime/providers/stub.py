"""Scripted provider for offline tests — no network, no host."""

from __future__ import annotations

from typing import Any, List, Optional

from browser_agent.runtime.providers.base import LLMProvider
from browser_agent.runtime.providers.registry import register_provider


class StubProvider(LLMProvider):
    """Scripted provider for offline tests.

    ``script`` entries, consumed one per call: a str (final answer), a list of
    ToolCall, a ChatCompletionResponse, or a callable(system, messages).
    Once exhausted, ``default_response`` (or the last user message) is returned.
    """

    provider_name = "stub"

    def __init__(self, script: Optional[List[Any]] = None,
                 default_response: Optional[str] = None):
        self._script = list(script or [])
        self._default = default_response
        self._i = 0

    async def chat_completion(self, system, messages, tools=None,
                              tool_choice="auto", model=None,
                              temperature=None, max_tokens=None):
        if self._i < len(self._script):
            resp = self._script[self._i]
            self._i += 1
        else:
            last_user = next((m.content for m in reversed(messages)
                              if m.role == "user"), "")
            resp = self._default if self._default is not None else f"echo: {last_user}"
        if callable(resp):
            resp = resp(system, messages)
        return resp


def _build_stub(spec: dict) -> StubProvider:
    return StubProvider(script=spec.get("script"),
                        default_response=spec.get("default"))


register_provider("stub", _build_stub)


__all__ = ["StubProvider"]