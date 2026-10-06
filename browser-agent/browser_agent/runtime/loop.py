"""Provider-independent REACT loop for the browser agent.

Same contract as the harness loop: LLM → parse tool calls → run them through an
injected ``executor`` → feed results back → repeat until a final answer or
``max_steps``. It depends ONLY on the ``LLMProvider`` protocol and the injected
executor, so a StubProvider in tests and a real provider in production drive
exactly the same code.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from browser_agent.runtime.providers import (
    ChatCompletionResponse,
    ConversationTurn,
    LLMProvider,
    ToolCall,
    ToolResult,
)

StepCallback = Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]
Executor = Callable[[str, Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]

DEFAULT_SYSTEM_PROMPT = (
    "You are a browser-use agent. You complete web tasks by calling the "
    "browser tools one step at a time."
)


def normalize_response(resp: Any):
    """Return (content, tool_calls) for whatever the provider returned."""
    if isinstance(resp, str):
        return resp, []
    if isinstance(resp, ChatCompletionResponse):
        return resp.content, list(resp.tool_calls or [])
    if isinstance(resp, list):
        return None, [c if isinstance(c, ToolCall) else ToolCall(**dict(c))
                      for c in resp]
    content = getattr(resp, "content", None)
    calls = getattr(resp, "tool_calls", None)
    if content is None and calls is None:
        return None, []
    return content, [c if isinstance(c, ToolCall) else ToolCall(**dict(c))
                     for c in (calls or [])]


async def _exec(executor: Executor, tc: ToolCall) -> ToolResult:
    raw = executor(tc.tool_name, tc.arguments or {})
    if inspect.isawaitable(raw):
        raw = await raw
    if isinstance(raw, ToolResult):
        return raw
    if isinstance(raw, dict):
        # A dict executor result: prefer its own status/error keys, falling
        # back to the {"status","data","error"} harness shape.
        status = raw.get("status")
        error = raw.get("error")
        if status is None:
            status = "error" if error else "success"
        return ToolResult(status=status, data=raw.get("data", raw),
                          error=error, summary=raw.get("summary"),
                          tool_call_id=tc.call_id)
    return ToolResult(status="success", data=raw, tool_call_id=tc.call_id)


async def _emit(cb: Optional[StepCallback], step: Dict[str, Any]) -> None:
    if cb is None:
        return
    out = cb(step)
    if inspect.isawaitable(out):
        await out


async def run_agent_turn(
    provider: LLMProvider,
    user_message: str,
    *,
    executor: Executor,
    system_prompt: str = "",
    tools: Optional[list] = None,
    max_steps: int = 12,
    on_step: Optional[StepCallback] = None,
    model: Optional[str] = None,
    history: Optional[List[ConversationTurn]] = None,
) -> Dict[str, Any]:
    """Run one user request through the REACT loop.

    ``history`` seeds prior turns BEFORE the current user message, so
    multi-turn conversations carry across turns (the caller owns persistence).

    Returns ``{"answer", "steps", "history", "truncated"}`` where each step is
    ``{"kind": "tool_call"|"answer", "step", "reasoning", "tool_calls",
    "tool_results"}``.
    """
    system = ConversationTurn(role="system",
                              content=system_prompt or DEFAULT_SYSTEM_PROMPT)
    working: List[ConversationTurn] = [ConversationTurn(role="user",
                                                        content=user_message)]
    if history:
        working = list(history) + working

    steps: List[Dict[str, Any]] = []
    last_content: Optional[str] = None
    active_tools: List[Any] = list(tools) if tools else []

    for step in range(max_steps):
        resp = await provider.chat_completion(
            system=system, messages=working, tools=active_tools,
            tool_choice="auto", model=model)
        content, tool_calls = normalize_response(resp)
        last_content = content

        working.append(ConversationTurn(role="assistant", content=content or "",
                                        tool_calls=tool_calls or None))

        if tool_calls:
            results: List[ToolResult] = []
            for tc in tool_calls:
                tr = await _exec(executor, tc)
                results.append(tr)
                working.append(ConversationTurn(
                    role="tool",
                    content=_safe_json(tr.data) if tr.status == "success"
                    else (tr.error or "tool error"),
                    tool_results=[tr],
                ))
            rec = {
                "kind": "tool_call", "step": step,
                "reasoning": content or None,
                "tool_calls": [{"name": tc.tool_name, "arguments": tc.arguments}
                               for tc in tool_calls],
                "tool_results": [{"status": tr.status, "data": tr.data,
                                  "error": tr.error} for tr in results],
            }
            steps.append(rec)
            await _emit(on_step, rec)
            continue

        rec = {"kind": "answer", "step": step,
               "reasoning": content or None, "content": content}
        steps.append(rec)
        await _emit(on_step, rec)
        return {"answer": content, "steps": steps, "history": working,
                "truncated": False}

    return {"answer": last_content, "steps": steps, "history": working,
            "truncated": True}


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return str(data)


__all__ = ["run_agent_turn", "normalize_response", "StepCallback", "Executor",
           "DEFAULT_SYSTEM_PROMPT"]