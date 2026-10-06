"""Provider-independent LLM message/response types + the provider contract.

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


__all__ = [
    "ChatCompletionResponse",
    "ConversationTurn",
    "LLMProvider",
    "ToolCall",
    "ToolResult",
]