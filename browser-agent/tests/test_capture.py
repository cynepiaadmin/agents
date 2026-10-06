"""Capture surfaces over the agent↔host wire: DOM, console, screenshot, record.

The agent gets no special casing — these are ordinary tools routed through
env/step, exactly like any other browser action. What is asserted here is the
shape the LLM actually receives, since that is what it reasons over.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from browser_agent.runtime.orchestrator import BrowserAgentRuntime


@pytest.fixture
def rt(env_host):
    r = BrowserAgentRuntime(env_host.base_url, max_steps=10)
    r.connect()
    yield r
    r.disconnect()


def _start(rt, task="add-widget-to-cart"):
    asyncio.run(rt.execute_tool("env_start", {"task": task}))


def _exec(rt, name, args=None):
    return asyncio.run(rt.execute_tool(name, args or {}))


# ── DOM capture ────────────────────────────────────────────────────────────
def test_agent_dom_capture_returns_diffable_payload(rt):
    _start(rt)
    res = _exec(rt, "browser_dom")
    assert res["status"] == "success", res.get("error")
    dom = res["data"]["dom"]
    assert dom["html"].startswith("<!DOCTYPE html>")
    assert 'style="' in dom["html"], "computed styles missing"
    assert dom["checksum"].startswith("sha256:")
    assert dom["stylesheet_count"] >= 1
    assert any("font-size" in s["text"] for s in dom["stylesheets"])
    assert dom["interactive"]
    # it consumed a step — DOM capture is a step like any other action
    assert res["data"]["step_num"] == 1


def test_agent_dom_diff_detects_a_change(rt):
    _start(rt)
    _exec(rt, "browser_dom")                     # baseline
    _exec(rt, "browser_click", {"index": 1})     # navigate to product page
    res = _exec(rt, "browser_dom", {"mode": "diff"})
    assert res["status"] == "success", res.get("error")
    diff = res["data"]["dom"]["diff"]
    assert diff["changed"] is True
    assert diff["checksum_equal"] is False
    assert any(k in diff["summary"] for k in ("element", "stylesheets",
                                              "changed"))
    assert diff["unified"]


def test_agent_dom_diff_without_baseline_is_an_error_result(rt):
    _start(rt)
    res = _exec(rt, "browser_dom", {"mode": "diff"})
    assert res["status"] == "error"
    assert "no previous capture" in res["error"]


# ── console output ─────────────────────────────────────────────────────────
def test_agent_reads_console_output(rt):
    _start(rt)
    res = _exec(rt, "browser_console")
    assert res["status"] == "success", res.get("error")
    console = res["data"]["console"]
    assert console["count"] >= 1
    assert any("page loaded" in e["text"] for e in console["entries"])


def test_agent_console_surfaces_page_warnings(rt):
    _start(rt)
    _exec(rt, "browser_navigate", {"url": "https://shop.test/cart"})
    res = _exec(rt, "browser_console")
    assert res["status"] == "success", res.get("error")
    entries = res["data"]["console"]["entries"]
    assert any("localStorage" in e["text"] for e in entries), entries
    assert any(e["type"] == "warn" for e in entries)


def test_console_delta_rides_along_with_the_step(rt):
    """The agent should see new console output WITHOUT asking for it."""
    _start(rt)
    _exec(rt, "browser_click", {"index": 1})
    res = _exec(rt, "browser_click", {"index": 0})   # add to cart
    assert res["status"] == "success"
    console = res["data"].get("console")
    assert console and console.get("new"), res["data"].keys()
    assert any("cart: added Widget" in e["text"] for e in console["new"])


# ── screenshot ─────────────────────────────────────────────────────────────
def test_agent_screenshot_returns_png_bytes(rt):
    _start(rt)
    res = _exec(rt, "browser_screenshot")
    assert res["status"] == "success", res.get("error")
    shot = res["data"]["screenshot"]
    assert shot["mime"] == "image/png"
    assert shot["data"], "screenshot omitted (over byte cap?)"


# ── recording ──────────────────────────────────────────────────────────────
def test_agent_records_episode_to_jsonl(rt, env_host):
    _start(rt)
    res = _exec(rt, "browser_record", {"mode": "start"})
    assert res["status"] == "success", res.get("error")
    assert res["data"]["record"]["recording"] is True
    path = res["data"]["record"]["path"]

    _exec(rt, "browser_click", {"index": 1})
    _exec(rt, "browser_click", {"index": 0})
    _exec(rt, "browser_navigate", {"url": "https://shop.test/cart"})
    done = _exec(rt, "env_finish", {"answer": "Widget"})

    # the terminal step reports the finalized artifact
    assert done["status"] == "success"
    assert done["data"]["record"]["recording"] is False
    assert done["data"]["record"]["entries"] >= 5

    # the file lives on the HOST, so read it from the host's fixture tree
    assert os.path.exists(path), path
    with open(path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    assert len(rows) >= 5
    assert rows[-1]["done"] is True
    assert rows[-1]["reward"] == 1.0
    # the recording carries console output when it happened on that step
    assert any("action" in r for r in rows)


def test_agent_record_status_before_start(rt):
    _start(rt)
    res = _exec(rt, "browser_record", {"mode": "status"})
    assert res["status"] == "success"
    assert res["data"]["record"]["recording"] is False


def test_agent_record_rejects_unknown_mode(rt):
    _start(rt)
    res = _exec(rt, "browser_record", {"mode": "rewind"})
    assert res["status"] == "error"
    assert "start|stop|status" in res["error"]


# ── all capture tools appear in the advertised schema ──────────────────────
def test_capture_tools_are_advertised_to_the_llm():
    from browser_agent.runtime.browser_tools import build_browser_tools
    names = [t["function"]["name"] for t in build_browser_tools()]
    assert {"browser_dom", "browser_console", "browser_record",
            "browser_screenshot"} <= set(names)
    # every advertised tool must be routable (no dead schema entries)
    from browser_agent.runtime.browser_tools import action_for_tool, is_non_step_tool
    for n in names:
        if is_non_step_tool(n):
            continue
        assert action_for_tool(n) is not None, f"{n} advertised but not routable"


def test_react_turn_can_use_the_capture_tools(env_host):
    """End-to-end: the loop calls browser_dom then browser_console and the
    host answers both — proving the new tools are wired all the way through."""
    from browser_agent.runtime.providers import StubProvider, ToolCall
    provider = StubProvider(script=[
        [ToolCall(tool_name="env_start",
                  arguments={"task": "add-widget-to-cart"}, call_id="c1")],
        [ToolCall(tool_name="browser_dom", arguments={}, call_id="c2")],
        [ToolCall(tool_name="browser_console", arguments={}, call_id="c3")],
        [ToolCall(tool_name="env_finish", arguments={}, call_id="c4")],
        "done capturing",
    ])
    r = BrowserAgentRuntime(env_host.base_url, llm=provider, max_steps=8)
    r.connect()
    try:
        res = r.run_turn_sync("capture the page")
        assert res["answer"] == "done capturing"
        dom_step = res["steps"][1]
        assert dom_step["tool_results"][0]["status"] == "success"
        assert "dom" in dom_step["tool_results"][0]["data"]
        console_step = res["steps"][2]
        assert console_step["tool_results"][0]["status"] == "success"
        assert "console" in console_step["tool_results"][0]["data"]
    finally:
        r.disconnect()