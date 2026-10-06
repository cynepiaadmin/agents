"""Command line for the browser agent.

    browser-agent run   --task add-widget-to-cart     # scripted episode, no LLM
    browser-agent ask   --task add-widget-to-cart "..." # REACT turn with an LLM
    browser-agent tasks                              # list env tasks
    browser-agent tools                              # print the tool schemas

`run` is deliberately LLM-free: it drives a fixed policy through the env
capability so you can verify the host, the episode and the scoring end to end
before wiring a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List

from browser_agent.runtime.bootstrap import build_runtime


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _policy_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic no-LLM policy for `run`.

    Greedy: if a link's href points at a URL containing any task keyword that we
    have not visited, click it; otherwise finish. Enough to exercise the whole
    reset → step → score → done path.
    """
    goal = (state.get("goal") or "").lower()
    url = (state.get("url") or "").lower()
    elements = state.get("elements") or []

    for el in elements:
        href = (el.get("href") or "").lower()
        text = (el.get("text") or "").lower()
        if not href or href == url:
            continue
        if goal and any(k in goal for k in ("cart", "checkout")) and "cart" in text:
            return {"action": "browser_click", "args": {"index": el["index"]}}
        if goal and any(k in goal for k in ("search", "gadget", "find")) and el.get("tag") == "input":
            return {"action": "browser_fill", "args": {"index": el["index"],
                                                       "text": "gadget"}}
        if goal and any(k in goal for k in ("back", "history")) and "back" in text:
            return {"action": "browser_click", "args": {"index": el["index"]}}

    # fallback: try to reach the cart page if the goal mentions it
    if "cart" in goal and "/cart" not in url:
        for el in elements:
            if (el.get("href") or "").endswith("/cart"):
                return {"action": "browser_click", "args": {"index": el["index"]}}
    return {"action": "env_finish", "args": {"answer": "goal attempted"}}


def cmd_run(args) -> int:
    rt = build_runtime(args.runtime_config, orchestrate_host=args.spawn_host,
                       base_url=args.base_url)
    rt.connect()
    try:
        tasks = rt.list_tasks()
        if args.task:
            rt.env_start(task=args.task)
        else:
            rt.env_start(start_url=args.start_url)
        print(f"episode {rt.episode_id} started")

        for _ in range(args.max_steps):
            state = rt.env_observe()
            if state.get("done"):
                break
            decision = _policy_action(state)
            print(f"  step {state.get('step_num')}: {decision['action']} {decision['args']}")
            if decision["action"] == "env_finish":
                state = rt.env_finish(decision["args"].get("answer", ""))
                break
            from browser_agent.runtime.browser_tools import action_for_tool, payload_for_tool
            at = action_for_tool(decision["action"])
            state = rt.env_step(at, payload_for_tool(decision["action"],
                                                     decision["args"]))
            if state.get("done"):
                break

        final = state
        score = rt.score()
        print(_dump({"episode_id": rt.episode_id,
                     "url": final.get("url"),
                     "step_num": final.get("step_num"),
                     "done": final.get("done"),
                     "reward": final.get("reward"),
                     "score": score}))
        return 0 if score.get("passed") else 1
    finally:
        rt.disconnect()


def cmd_ask(args) -> int:
    rt = build_runtime(args.runtime_config, orchestrate_host=args.spawn_host,
                       base_url=args.base_url)
    rt.connect()
    try:
        if args.task:
            rt.env_start(task=args.task)
        elif args.start_url:
            rt.env_start(start_url=args.start_url)
        print(f"episode {rt.episode_id}")
        res = rt.run_turn_sync(args.message, max_steps=args.max_steps)
        print(_dump({"answer": res.get("answer"),
                     "truncated": res.get("truncated"),
                     "steps": len(res.get("steps", [])),
                     "score": rt.score()}))
        return 0 if (not res.get("truncated") and rt.score().get("passed")) else 1
    finally:
        rt.disconnect()


def cmd_tasks(args) -> int:
    # The catalogue is a LOCAL dataset (browser_tasks/) — listing it needs no
    # host connection and must work with nothing listening.
    from browser_agent.tasking import load_tasks
    print(_dump(list(load_tasks().values())))
    return 0


def cmd_tools(args) -> int:
    from browser_agent.runtime.browser_tools import build_browser_tools
    print(_dump(build_browser_tools()))
    return 0


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="browser-agent",
                                 description="A2E browser agent")
    ap.add_argument("--runtime-config", default=None,
                    help="path to runtime_config.yaml (default: the shipped one)")
    ap.add_argument("--base-url", default=None,
                    help="override the env host URL")
    ap.add_argument("--spawn-host", action="store_true",
                    help="spawn the browser-env container if nothing is listening")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="drive an episode with a fixed policy (no LLM)")
    p_run.add_argument("--task", default="")
    p_run.add_argument("--start-url", default="")
    p_run.add_argument("--max-steps", type=int, default=10)
    p_run.set_defaults(fn=cmd_run)

    p_ask = sub.add_parser("ask", help="one REACT turn with an LLM")
    p_ask.add_argument("message")
    p_ask.add_argument("--task", default="")
    p_ask.add_argument("--start-url", default="")
    p_ask.add_argument("--max-steps", type=int, default=12)
    p_ask.set_defaults(fn=cmd_ask)

    p_tasks = sub.add_parser("tasks", help="list the environment's tasks")
    p_tasks.set_defaults(fn=cmd_tasks)

    p_tools = sub.add_parser("tools", help="print the tool schemas")
    p_tools.set_defaults(fn=cmd_tools)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())