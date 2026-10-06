"""Loop + tool-schema tests — offline, no host, no network.

These pin the two things the agent depends on that are easy to break silently:
the REACT loop's contract, and the tool→env-action mapping (a rename in either
repo must fail here rather than at runtime).
"""

from __future__ import annotations

import asyncio

import pytest

from browser_agent.runtime.loop import normalize_response, run_agent_turn
from browser_agent.runtime.providers import (
    ChatCompletionResponse,
    ConversationTurn,
    StubProvider,
    ToolCall,
    build_provider,
)
from browser_agent.runtime.browser_tools import (
    action_for_tool,
    build_browser_tools,
    payload_for_tool,
    tool_names,
)


# ── provider contract ──────────────────────────────────────────────────────
def test_build_provider_stub_is_offline():
    p = build_provider({"provider_name": "stub"})
    assert isinstance(p, StubProvider)


def test_build_provider_missing_name_fails():
    with pytest.raises(ValueError):
        build_provider({"chat_model": "x"})


def test_build_provider_rejects_non_dict():
    with pytest.raises(TypeError):
        build_provider("gpt-4")


def test_openai_compat_normalizes_tool_calls():
    from browser_agent.runtime.providers import OpenAICompatProvider
    resp = OpenAICompatProvider._normalize({
        "choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "browser_click",
                                                      "arguments": "{\"index\":1}"}}],
        }}],
        "usage": {"total_tokens": 10},
    })
    assert resp.tool_calls and resp.tool_calls[0].tool_name == "browser_click"
    assert resp.tool_calls[0].arguments == {"index": 1}
    assert resp.tool_calls[0].call_id == "c1"


def test_build_messages_keeps_tool_results_with_call_ids():
    """Dropping (or untagging) tool messages breaks strict OpenAI servers."""
    from browser_agent.runtime.providers import OpenAICompatProvider, ToolResult
    sys_t = ConversationTurn(role="system", content="s")
    turns = [
        ConversationTurn(role="assistant", content="",
                         tool_calls=[ToolCall(tool_name="browser_text",
                                              arguments={}, call_id="c1")]),
        ConversationTurn(role="tool", content="{...}",
                         tool_results=[ToolResult(status="success",
                                                  tool_call_id="c1")]),
    ]
    msgs = OpenAICompatProvider._build(sys_t, turns)
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["tool_call_id"] == "c1"
    assert msgs[1]["tool_calls"][0]["id"] == "c1"


# ── normalize_response ─────────────────────────────────────────────────────
def test_normalize_str_and_list_and_response():
    assert normalize_response("hi") == ("hi", [])
    content, calls = normalize_response([{"tool_name": "x", "arguments": {}}])
    assert content is None and calls[0].tool_name == "x"
    content, calls = normalize_response(
        ChatCompletionResponse(content="a", tool_calls=[]))
    assert content == "a" and calls == []


# ── REACT loop ─────────────────────────────────────────────────────────────
def test_loop_single_final_answer():
    provider = StubProvider(script=["all done"])
    res = asyncio.run(run_agent_turn(
        provider, "hi", executor=lambda n, a: {"status": "success", "data": {}}))
    assert res["answer"] == "all done"
    assert res["truncated"] is False
    assert [s["kind"] for s in res["steps"]] == ["answer"]


def test_loop_feeds_tool_results_back_to_the_llm():
    seen = []

    def executor(name, args):
        seen.append((name, args))
        return {"status": "success", "data": {"url": "https://shop.test/"}}

    provider = StubProvider(script=[
        [ToolCall(tool_name="browser_text", arguments={}, call_id="c1")],
        "the page says hello",
    ])
    res = asyncio.run(run_agent_turn(provider, "read the page", executor=executor))
    assert seen == [("browser_text", {})]
    assert res["answer"] == "the page says hello"
    # the turn order must be: user → assistant(tool_calls) → tool(result) →
    # assistant(final answer). The tool result is what the model reads next.
    roles = [t.role for t in res["history"]]
    assert roles == ["user", "assistant", "tool", "assistant"], roles
    tool_turn = res["history"][2]
    assert tool_turn.tool_results[0].status == "success"
    assert res["history"][-1].role == "assistant"


def test_loop_executor_error_becomes_error_result_not_crash():
    def executor(name, args):
        return {"status": "error", "error": "boom"}

    provider = StubProvider(script=[
        [ToolCall(tool_name="browser_navigate", arguments={"url": "x"}, call_id="c1")],
        "ok",
    ])
    res = asyncio.run(run_agent_turn(provider, "go", executor=executor))
    step = res["steps"][0]
    assert step["tool_results"][0]["status"] == "error"
    assert "boom" in step["tool_results"][0]["error"]
    assert res["answer"] == "ok"


def test_loop_async_executor_is_awaited():
    async def executor(name, args):
        await asyncio.sleep(0)
        return {"status": "success", "data": "async-ok"}

    provider = StubProvider(script=[
        [ToolCall(tool_name="env_start", arguments={})],
        "done",
    ])
    res = asyncio.run(run_agent_turn(provider, "start", executor=executor))
    assert res["answer"] == "done"
    assert res["steps"][0]["tool_results"][0]["data"] == "async-ok"


def test_loop_respects_max_steps_and_reports_truncated():
    def executor(name, args):
        return {"status": "success", "data": {}}

    def forever(system, messages):
        return [ToolCall(tool_name="browser_text", arguments={})]

    provider = StubProvider(script=[forever] * 5)
    res = asyncio.run(run_agent_turn(provider, "loop", executor=executor,
                                     max_steps=3))
    assert res["truncated"] is True
    assert len(res["steps"]) == 3
    assert all(s["kind"] == "tool_call" for s in res["steps"])


def test_loop_seeds_prior_history_before_current_message():
    provider = StubProvider(script=["final"])
    prior = [ConversationTurn(role="user", content="earlier question"),
             ConversationTurn(role="assistant", content="earlier answer")]
    res = asyncio.run(run_agent_turn(provider, "now", executor=lambda n, a: {},
                                     history=prior))
    roles = [t.role for t in res["history"]]
    assert roles[:3] == ["user", "assistant", "user"]


def test_on_step_callback_receives_tool_calls():
    observed = []

    def executor(name, args):
        return {"status": "success", "data": {}}

    provider = StubProvider(script=[
        [ToolCall(tool_name="browser_back", arguments={})],
        "done",
    ])
    asyncio.run(run_agent_turn(provider, "x", executor=executor,
                               on_step=observed.append))
    assert observed[0]["kind"] == "tool_call"
    assert observed[0]["tool_calls"][0]["name"] == "browser_back"
    assert observed[1]["kind"] == "answer"


# ── tool schemas + mapping ─────────────────────────────────────────────────
def test_tool_schemas_are_well_formed():
    tools = build_browser_tools()
    names = [t["function"]["name"] for t in tools]
    assert len(names) == len(set(names)), "duplicate tool names"
    assert {"env_start", "env_finish", "env_observe",
            "browser_navigate", "browser_click", "browser_fill",
            "browser_text", "browser_tasks", "browser_score"} <= set(names)
    for t in tools:
        assert t["type"] == "function"
        params = t["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert isinstance(params.get("required", []), list)
        # every required arg must be declared
        for r in params.get("required", []):
            assert r in params["properties"]


def test_env_start_requires_at_least_a_task_or_url_in_description():
    """env_start has no hard-required field (task XOR start_url), so the
    description must carry the contract."""
    tools = {t["function"]["name"]: t for t in build_browser_tools()}
    desc = tools["env_start"]["function"]["description"]
    assert "FIRST" in desc


def test_every_step_tool_maps_to_a_known_env_action():
    from browser_agent.runtime.browser_tools import is_non_step_tool
    for name in tool_names():
        if is_non_step_tool(name):
            continue
        assert action_for_tool(name) is not None, f"{name} has no env action"


def test_payload_for_tool_renames_only_declared_keys():
    """click/fill pass args through unchanged; navigate renames url (already
    same name); screenshot renames full_page (already same). The point of the
    test is that UNDECLARED keys are never dropped."""
    p = payload_for_tool("browser_click", {"index": 3, "selector": "#a"})
    assert p == {"index": 3, "selector": "#a"}
    p = payload_for_tool("browser_navigate", {"url": "https://x"})
    assert p == {"url": "https://x"}
    p = payload_for_tool("browser_screenshot", {"full_page": True})
    assert p == {"full_page": True}


def test_browser_text_maps_to_read_only_not_a_step():
    from browser_agent.runtime.browser_tools import READ_ONLY_TOOLS
    assert "browser_text" in READ_ONLY_TOOLS
    assert action_for_tool("browser_text") is None


def test_action_space_covers_the_documented_contract():
    from browser_agent.runtime.browser_tools import SUPPORTED_ACTIONS
    for expected in ("navigate", "click", "fill", "press", "scroll",
                     "back", "forward", "reload", "extract", "screenshot",
                     "wait_for", "dom", "record", "console"):
        assert expected in SUPPORTED_ACTIONS, f"{expected} missing"