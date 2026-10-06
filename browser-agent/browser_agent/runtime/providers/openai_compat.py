"""OpenAI-compatible chat-completions provider — stdlib urllib only."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from browser_agent.runtime.providers.base import (
    ChatCompletionResponse,
    LLMProvider,
    ToolCall,
)
from browser_agent.runtime.providers.registry import register_provider


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


def _build_openai_compat(spec: dict) -> OpenAICompatProvider:
    """Builder for any OpenAI-compatible endpoint (the documented catch-all)."""
    return OpenAICompatProvider(
        provider_name=spec.get("provider_name") or "openai",
        api_credentials=spec.get("api_credentials") or spec.get("api_key") or "",
        chat_model=spec.get("chat_model"),
        base_url=spec.get("base_url") or None,
        timeout=float(spec.get("timeout", 120.0)),
    )


register_provider("openai", _build_openai_compat)


__all__ = ["OpenAICompatProvider"]