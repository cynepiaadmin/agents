"""browser_agent — agent side of the A2E browser environment.

Public surface:

    BrowserAgentRuntime   drives a browser-env host over A2E + owns the REACT loop
    run_agent_turn        provider-independent REACT loop
    build_browser_tools   OpenAI-style tool schemas for the browser action space
    build_provider        stub (offline) / OpenAI-compatible (production)
"""

from browser_agent.runtime.orchestrator import BrowserAgentRuntime
from browser_agent.runtime.loop import run_agent_turn
from browser_agent.runtime.browser_tools import build_browser_tools, tool_names
from browser_agent.runtime.providers import (
    ChatCompletionResponse,
    ConversationTurn,
    LLMProvider,
    StubProvider,
    ToolCall,
    ToolResult,
    build_provider,
)

__all__ = [
    "BrowserAgentRuntime",
    "run_agent_turn",
    "build_browser_tools",
    "tool_names",
    "build_provider",
    "ChatCompletionResponse",
    "ConversationTurn",
    "LLMProvider",
    "StubProvider",
    "ToolCall",
    "ToolResult",
]