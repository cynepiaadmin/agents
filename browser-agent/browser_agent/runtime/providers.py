"""Provider-independent LLM types + the provider contract.

Self-contained (no dependency on the host repo) so the agent runs and is
testable offline. Mirrors the shape xa-agent uses: the loop depends ONLY on
``LLMProvider``.

Message types:
  ConversationTurn(role, content, tool_calls, tool_results)
  ToolCall(tool_name, arguments, call_id)
  ToolResult(status, data, error, summary, tool_call_id)
  ChatCompletionResponse(content, tool_calls, usage)
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger("browser_agent.runtime.providers")


# ── message / response types ───────────────────────────────────────────────
class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None


class ToolResult(BaseModel):
    status: str = "success"
    data: Any = None
    error: Optional[str] = None
    summary: Optional[str] = None
    tool_call_id: Optional[str] = None


class ConversationTurn(BaseModel):
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None


class ChatCompletionResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Dict[str, Any]] = None


# ── provider protocol ──────────────────────────────────────────────────────
class LLMProvider(abc.ABC):
    provider_name: str = "abstract"

    @abc.abstractmethod
    async def chat_completion(
        self,
        system: ConversationTurn,
        messages: List[ConversationTurn],
        tools: Optional[list] = None,
        tool_choice: str = "auto",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[str, List[ToolCall], ChatCompletionResponse]:
        ...


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


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible chat-completions endpoint, via stdlib urllib only.

    No third-party SDK, so the agent imports cleanly regardless of what is
    installed. The blocking HTTP call runs in a thread so the async loop is safe.
    """

    def __init__(self, provider_name: str = "openai", api_credentials: str = "",
                 chat_model: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 120.0, **_ignored: Any):
        self.provider_name = provider_name
        self._api_key = api_credentials or ""
        self._model = chat_model or "gpt-4o-mini"
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._base_url = base if base.endswith("/v1") else base + "/v1"
        self._timeout = timeout

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json
        import urllib.error
        import urllib.request

        url = f"{self._base_url}/chat/completions"
        req = urllib.request.Request(
            url, data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        return _json.loads(body)

    async def chat_completion(self, system, messages, tools=None,
                              tool_choice="auto", model=None,
                              temperature=None, max_tokens=None):
        import asyncio

        def _run():
            payload: Dict[str, Any] = {
                "model": model or self._model,
                "messages": self._build(system, messages),
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            return self._post(payload)

        resp = await asyncio.to_thread(_run)
        return self._normalize(resp)

    @staticmethod
    def _build(system, messages) -> List[Dict[str, Any]]:
        """Serialise BOTH sides of a tool round-trip, with matching call ids.

        Dropping the tool result messages (or emitting them without
        tool_call_id) breaks tool-use against strict OpenAI-compatible servers.
        """
        import json as _json
        import uuid as _uuid

        out = [{"role": system.role, "content": system.content}]
        for m in messages:
            results = getattr(m, "tool_results", None)
            if results:
                for r in results:
                    out.append({
                        "role": "tool",
                        "tool_call_id": getattr(r, "tool_call_id", None) or "",
                        "content": (r.data if isinstance(r.data, str)
                                    else _json.dumps(r.data or {}, default=str)),
                    })
                continue
            entry: Dict[str, Any] = {"role": m.role, "content": m.content}
            calls = getattr(m, "tool_calls", None)
            if calls:
                entry["tool_calls"] = [{
                    "id": c.call_id or f"call_{_uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": c.tool_name,
                        "arguments": (c.arguments if isinstance(c.arguments, str)
                                      else _json.dumps(c.arguments or {},
                                                       default=str)),
                    },
                } for c in calls]
            out.append(entry)
        return out

    @staticmethod
    def _normalize(resp: Dict[str, Any]):
        import json
        usage = resp.get("usage")
        msg = ((resp.get("choices") or [{}])[0]).get("message", {})
        raw_calls = msg.get("tool_calls")
        if raw_calls:
            calls = []
            for c in raw_calls:
                fn = c.get("function", {})
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append(ToolCall(tool_name=fn.get("name", "unknown"),
                                      arguments=args or {}, call_id=c.get("id")))
            return ChatCompletionResponse(content=msg.get("content"),
                                          tool_calls=calls, usage=usage)
        return ChatCompletionResponse(content=msg.get("content"), usage=usage)


def build_provider(spec):
    """Resolve an LLMProvider from a config dict or pass one through.

    Only ``{"provider_name": "stub"}`` is offline; everything else is
    OpenAI-compatible HTTP. api_credentials may use ``${ENV_VAR}`` — that
    interpolation is done by the config loader, not here.
    """
    if isinstance(spec, LLMProvider):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"build_provider expects dict|LLMProvider, got {type(spec)}")
    name = spec.get("provider_name")
    if name is None:
        raise ValueError("provider spec missing 'provider_name'")
    if name == "stub":
        return StubProvider(script=spec.get("script"),
                            default_response=spec.get("default"))
    return OpenAICompatProvider(
        provider_name=name,
        api_credentials=spec.get("api_credentials") or spec.get("api_key") or "",
        chat_model=spec.get("chat_model"),
        base_url=spec.get("base_url") or None,
        timeout=float(spec.get("timeout", 120.0)),
    )


__all__ = [
    "ChatCompletionResponse", "ConversationTurn", "LLMProvider",
    "ToolCall", "ToolResult", "StubProvider", "OpenAICompatProvider",
    "build_provider",
]