"""Agent ↔ host integration over REAL HTTP.

Spawns the sibling browser-env repo as a subprocess (the test boundary), then
drives it with BrowserAgentRuntime exactly the way production does: over TCP,
through the session handshake, with the agent's own REACT loop.

No LLM is contacted — the loop is driven by StubProvider scripts.
"""

from __future__ import annotations

import asyncio

import pytest

from browser_agent.runtime.browser_tools import action_for_tool, payload_for_tool
from browser_agent.runtime.orchestrator import BrowserAgentRuntime
from browser_agent.runtime.providers import StubProvider, ToolCall


@pytest.fixture
def runtime(env_host):
    rt = BrowserAgentRuntime(env_host.base_url,
                             system_prompt="You are a browser agent.",
                             max_steps=10)
    rt.connect()
    yield rt
    rt.disconnect()


# ── handshake / capability negotiation ─────────────────────────────────────
def test_connect_negotiates_env_and_tools(runtime):
    assert "env" in runtime.caps
    assert "tools" in runtime.caps


# ── task catalogue ─────────────────────────────────────────────────────────
def test_list_tasks_over_http(runtime):
    tasks = runtime.list_tasks()
    names = {t["name"] for t in tasks}
    assert {"add-widget-to-cart", "search-for-gadget",
            "read-widget-price", "browse-and-back"} <= names
    widget = next(t for t in tasks if t["name"] == "add-widget-to-cart")
    assert widget["start_url"] == "https://shop.test/"


# ── episode control ────────────────────────────────────────────────────────
def test_env_start_and_observe(runtime):
    start = runtime.env_start(task="add-widget-to-cart")
    assert start["episode_id"]
    assert start["state"]["url"] == "https://shop.test/"
    assert start["state"]["max_steps"] == 12

    obs = runtime.env_observe()
    assert obs["url"] == "https://shop.test/"
    assert obs["done"] is False
    assert any(e["text"] == "Widget" for e in obs["elements"])


def test_env_start_requires_task_or_start_url(runtime):
    with pytest.raises(Exception) as ei:
        runtime.env_start()
    assert "start_url" in str(ei.value)


def test_step_actually_moves_the_browser(runtime):
    runtime.env_start(task="add-widget-to-cart")
    obs = runtime.env_step("click", {"index": 1})
    assert obs["url"] == "https://shop.test/product/1"
    assert obs["step_num"] == 1
    assert obs["done"] is False
    assert obs["last_error"] == ""


def test_full_episode_scores_one(runtime):
    """The whole episode through the agent: 4 steps → reward 1.0."""
    runtime.env_start(task="add-widget-to-cart")
    runtime.env_step("click", {"index": 1})
    runtime.env_step("click", {"index": 0})
    runtime.env_step("navigate", {"url": "https://shop.test/cart"})
    final = runtime.env_finish("Widget added")

    assert final["done"] is True
    assert final["reward"] == 1.0
    assert final["passed"] is True
    assert all(final["criteria"].values()), final["criteria"]
    assert final["terminal_reason"] == "finish"

    score = runtime.score()
    assert score["score"] == 1.0 and score["passed"] is True


def test_max_steps_terminates_episode(runtime):
    runtime.env_start(task="add-widget-to-cart", max_steps=3)
    state = runtime.env_observe()
    for _ in range(3):
        state = runtime.env_step("scroll", {"direction": "down"})
    assert state["done"] is True
    assert state["terminal_reason"] == "max_steps"
    assert 0.0 <= state["reward"] < 1.0


def test_unknown_action_surfaces_error_but_keeps_episode_alive(runtime):
    runtime.env_start(task="add-widget-to-cart")
    bad = runtime.env_step("teleport", {})
    assert "action_error" in bad
    assert bad["done"] is False
    # still usable
    good = runtime.env_step("back", {})
    assert "action_error" not in good


def test_acting_without_episode_is_a_clear_error(runtime):
    with pytest.raises(RuntimeError) as ei:
        runtime.env_step("click", {"index": 1})
    assert "env_start" in str(ei.value)


def test_close_ends_the_episode(runtime):
    runtime.env_start(task="add-widget-to-cart")
    assert runtime.env_close()["closed"] is True
    assert runtime.episode_id is None
    with pytest.raises(RuntimeError):
        runtime.env_observe()


# ── tool routing (the REACT executor) ──────────────────────────────────────
async def _exec(rt, name, args):
    return await rt.execute_tool(name, args)


def test_execute_tool_routes_browser_actions_through_env_step(runtime):
    """browser_click must become one env/step so the episode is scored."""
    asyncio.run(_exec(runtime, "env_start", {"task": "add-widget-to-cart"}))
    step0 = runtime.env_observe()["step_num"]

    res = asyncio.run(_exec(runtime, "browser_click", {"index": 1}))
    assert res["status"] == "success", res.get("error")
    assert res["data"]["url"] == "https://shop.test/product/1"
    # step counter advanced => it went through env/step, not a raw tool call
    assert res["data"]["step_num"] == step0 + 1


def test_execute_tool_browser_text_is_read_only(runtime):
    asyncio.run(_exec(runtime, "env_start", {"task": "add-widget-to-cart"}))
    before = runtime.env_observe()["step_num"]
    res = asyncio.run(_exec(runtime, "browser_text", {}))
    assert res["status"] == "success"
    assert res["data"]["url"] == "https://shop.test/"
    assert "elements" in res["data"]
    # reading does not consume a step
    assert runtime.env_observe()["step_num"] == before


def test_execute_tool_without_episode_returns_error(runtime):
    res = asyncio.run(_exec(runtime, "browser_click", {"index": 1}))
    assert res["status"] == "error"
    assert "env_start" in res["error"]


def test_execute_tool_unknown_name_lists_available(runtime):
    res = asyncio.run(_exec(runtime, "browser_fly", {}))
    assert res["status"] == "error"
    assert "browser_navigate" in res["error"]


def test_execute_tool_env_finish_and_score(runtime):
    asyncio.run(_exec(runtime, "env_start", {"task": "add-widget-to-cart"}))
    asyncio.run(_exec(runtime, "browser_click", {"index": 1}))
    asyncio.run(_exec(runtime, "browser_click", {"index": 0}))
    asyncio.run(_exec(runtime, "browser_navigate",
                      {"url": "https://shop.test/cart"}))
    fin = asyncio.run(_exec(runtime, "env_finish", {"answer": "Widget"}))
    assert fin["status"] == "success"
    assert fin["data"]["done"] is True
    assert fin["data"]["reward"] == 1.0

    sc = asyncio.run(_exec(runtime, "browser_score", {}))
    assert sc["status"] == "success"
    assert sc["data"]["passed"] is True


def test_execute_tool_browser_tasks(runtime):
    res = asyncio.run(_exec(runtime, "browser_tasks", {}))
    assert res["status"] == "success"
    assert len(res["data"]["tasks"]) == 4


# ── direct tool calls (tools/* capability, unwrapping) ─────────────────────
def test_direct_tool_call_unwraps_nested_tool_result(runtime):
    payload = runtime.call_tool("browser_navigate",
                                {"url": "https://shop.test/product/2"})
    assert payload["success"] is True, payload.get("error")
    assert payload["data"]["url"] == "https://shop.test/product/2"


def test_direct_tool_failure_is_returned_not_raised(runtime):
    payload = runtime.call_tool("browser_click", {"selector": "#does-not-exist"})
    assert payload["success"] is False
    assert payload["error"]


def test_tool_and_env_share_one_browser_state(runtime):
    runtime.env_start(task="add-widget-to-cart")
    runtime.call_tool("browser_click", {"index": 1})
    # the tools surface moved the browser; env/observe must see it
    assert runtime.env_observe()["url"] == "https://shop.test/product/1"


# ── full REACT turn with a scripted LLM ────────────────────────────────────
def test_react_turn_completes_task_with_scripted_llm(env_host):
    """End to end, no network: a scripted LLM emits the tool calls, the REACT
    loop routes them through env/step, the host scores the episode, and the
    turn returns an answer. Proves loop + routing + protocol + scoring together.
    """
    provider = StubProvider(script=[
        [ToolCall(tool_name="env_start",
                  arguments={"task": "add-widget-to-cart"}, call_id="c1")],
        [ToolCall(tool_name="browser_click", arguments={"index": 1},
                  call_id="c2")],
        [ToolCall(tool_name="browser_click", arguments={"index": 0},
                  call_id="c3")],
        [ToolCall(tool_name="browser_navigate",
                  arguments={"url": "https://shop.test/cart"}, call_id="c4")],
        [ToolCall(tool_name="env_finish", arguments={"answer": "Widget"},
                  call_id="c5")],
        "Added the Widget to the cart and checked out.",
    ])
    rt = BrowserAgentRuntime(env_host.base_url, llm=provider,
                             system_prompt="You are a browser agent.",
                             max_steps=10)
    rt.connect()
    try:
        res = rt.run_turn_sync("Add the Widget to the cart.")
        assert res["truncated"] is False
        assert res["answer"] == "Added the Widget to the cart and checked out."
        assert [s["kind"] for s in res["steps"]] == ["tool_call"] * 5 + ["answer"]
        # the final answer step must carry no tool calls, the tool steps must
        # have succeeded (a failed step would be status=error in the record)
        for s in res["steps"][:-1]:
            assert all(r["status"] == "success" for r in s["tool_results"]), s
        score = rt.score()
        assert score["passed"] is True and score["score"] == 1.0
        assert rt.episode_id is not None
    finally:
        rt.disconnect()


def test_react_turn_surfaces_host_error_and_stops_cleanly(env_host):
    """A tool the host rejects must come back as an error RESULT — the loop
    must keep running and still return an answer."""
    provider = StubProvider(script=[
        [ToolCall(tool_name="browser_click", arguments={"index": 1},
                  call_id="c1")],           # no episode open -> host error
        "I could not act because no episode was open.",
    ])
    rt = BrowserAgentRuntime(env_host.base_url, llm=provider, max_steps=6)
    rt.connect()
    try:
        res = rt.run_turn_sync("click something")
        assert res["answer"] == "I could not act because no episode was open."
        step = res["steps"][0]
        assert step["tool_results"][0]["status"] == "error"
        assert "env_start" in step["tool_results"][0]["error"]
    finally:
        rt.disconnect()


def test_react_turn_observe_route_does_not_burn_steps(env_host):
    """browser_text is read-only: the loop must be able to re-orient repeatedly
    without advancing the episode."""
    provider = StubProvider(script=[
        [ToolCall(tool_name="env_start",
                  arguments={"task": "add-widget-to-cart"}, call_id="c1")],
        [ToolCall(tool_name="browser_text", arguments={}, call_id="c2")],
        [ToolCall(tool_name="browser_text", arguments={}, call_id="c3")],
        [ToolCall(tool_name="env_finish", arguments={}, call_id="c4")],
        "done reading",
    ])
    rt = BrowserAgentRuntime(env_host.base_url, llm=provider, max_steps=8)
    rt.connect()
    try:
        res = rt.run_turn_sync("read the page twice")
        assert res["answer"] == "done reading"
        texts = [s for s in res["steps"]
                 if s.get("tool_calls")
                 and s["tool_calls"][0]["name"] == "browser_text"]
        assert len(texts) == 2
        # both observes reported step 0 (no steps consumed by reading)
        for s in texts:
            assert s["tool_results"][0]["data"]["step_num"] == 0
    finally:
        rt.disconnect()