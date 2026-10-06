"""OpenAI-style tool schemas + the name→env-action mapping for the browser agent.

The agent advertises browser capabilities as ordinary function tools. Each
browser_* tool maps onto ONE ``env/step`` action so the episode's step counter,
reward, and done condition stay authoritative — the agent never acts outside an
episode while one is open, and every action is scored.

Two families:
  browser_*  — act on / read the page (browser actions)
  env_*      — episode control (start / finish / observe / score / tasks)

``browser_text`` maps to ``env/observe`` (read-only) so re-reading the page
does not consume an episode step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── browser_* tool name -> (env action_type, payload key mapping) ──────────
# Each value is (action_type, {arg_key: payload_key}) — i.e. how the tool's
# argument object becomes the env action payload.
_ACTION_MAP: Dict[str, tuple] = {
    "browser_navigate": ("navigate", {"url": "url"}),
    "browser_click":    ("click",    {}),      # selector/index pass through
    "browser_fill":     ("fill",     {}),      # selector/index/text pass through
    "browser_press":    ("press",    {}),      # key pass through
    "browser_scroll":   ("scroll",   {}),      # direction/amount pass through
    "browser_back":     ("back",     {}),
    "browser_forward":  ("forward",  {}),
    "browser_reload":   ("reload",   {}),
    "browser_extract":  ("extract",  {}),      # selector/index pass through
    "browser_screenshot": ("screenshot", {"full_page": "full_page"}),
    # style-flattened DOM capture (mode=capture|diff) — the diff payload is
    # what an agent uses to answer "did my action change the page?"
    "browser_dom":      ("dom",     {}),
    # console output for the episode (max_entries / clear pass through)
    "browser_console":  ("console", {}),
    # episode recording control (mode=start|stop|status)
    "browser_record":   ("record",  {}),
    "browser_wait":     ("wait_for", {"timeout_ms": "timeout_ms"}),
}

#: Read-only tools that map to observe instead of a step.
READ_ONLY_TOOLS = frozenset({"browser_text"})

#: Meta tools served by env/* messages (task catalogue, scoring) rather than
#: an action. They are never routed through env/step.
META_TOOLS = frozenset({"browser_tasks", "browser_score"})

#: Tools that are NOT env/step actions: env_* (episode control), read-only
#: (observe), and meta. This is the single source of truth for the router and
#: for tests — a tool added here must be handled explicitly by the executor.
def is_non_step_tool(name: str) -> bool:
    return name.startswith("env_") or name in READ_ONLY_TOOLS or name in META_TOOLS


def _p(name: str, type_: str, desc: str, required: bool = False,
       enum: Optional[List[str]] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"name": name, "type": type_, "description": desc,
                         "required": required}
    if enum:
        d["enum"] = enum
    return d


def build_browser_tools() -> List[Dict[str, Any]]:
    """OpenAI function-calling schemas for the browser action surface."""
    def fn(name: str, desc: str, params: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {p["name"]: {
                        "type": p["type"],
                        "description": p["description"],
                        **({"enum": p["enum"]} if p.get("enum") else {}),
                    } for p in params},
                    "required": [p["name"] for p in params if p["required"]],
                },
            },
        }

    target = [
        _p("selector", "string", "CSS selector for the target element."),
        _p("index", "integer",
           "Element index from the latest snapshot. Preferred over selector — "
           "indexes are stable and reported back by browser_text."),
    ]

    return [
        fn("env_start", "Begin an episode. Call this FIRST, before any "
           "browser_* tool. A task name is resolved against the LOCAL "
           "browser_tasks dataset and its full spec is sent to the host; "
           "otherwise start from a URL.",
           [_p("task", "string", "Task name from browser_tasks (the local "
                                  "dataset), or an inline task spec."),
            _p("start_url", "string", "Absolute URL to open (when no task name)."),
            _p("goal", "string", "Free-form goal for an ad-hoc start_url episode."),
            _p("max_steps", "integer", "Step budget for the episode.")]),

        fn("env_finish", "End the episode and score it against the task's "
           "success criteria. Call when the goal is met or you cannot proceed.",
           [_p("answer", "string", "Your final answer / claim, if any.")]),

        fn("env_observe", "Re-read the current page without consuming a step.",
           []),

        fn("browser_text", "Return the page's URL, title, visible text and the "
           "indexed interactive-element list. Use this to orient yourself. "
           "Does not consume an episode step.", []),

        fn("browser_navigate", "Open an absolute URL.",
           [_p("url", "string", "Absolute URL to open.", True)]),

        fn("browser_click", "Click an element. Target by index (from "
           "browser_text) or selector.",
           target),

        fn("browser_fill", "Type text into an input or textarea.",
           target + [_p("text", "string", "Text to enter.", True)]),

        fn("browser_press", "Press a keyboard key (e.g. Enter to submit).",
           [_p("key", "string", "Key name, e.g. Enter.", True)] + target),

        fn("browser_scroll", "Scroll the page.",
           [_p("direction", "string", "Scroll direction.",
               enum=["down", "up", "top", "bottom"]),
            _p("amount", "integer", "Pixels to scroll (default 500).")]),

        fn("browser_back", "Go back in browser history.", []),
        fn("browser_forward", "Go forward in browser history.", []),
        fn("browser_reload", "Reload the current page.", []),

        fn("browser_extract", "Extract an element's visible text (or the whole "
           "body when no target is given).", target),

        fn("browser_screenshot", "Capture a PNG screenshot (base64).",
           [_p("full_page", "boolean", "Capture the full scrollable page.")]),

        fn("browser_dom", "Capture the style-flattened DOM: markup with each "
           "element's COMPUTED style inline, plus every stylesheet. Pass "
           "mode='diff' to instead compare against the previous capture and "
           "report what changed (checksum + unified diff) — use this to check "
           "whether an action actually had an effect.",
           [_p("mode", "string", "capture (default) | diff",
               enum=["capture", "diff"]),
            _p("max_bytes", "integer", "Cap on captured markup bytes.")]),

        fn("browser_console", "Read the page's console output for this episode: "
           "log/warn/error lines and uncaught JS errors. Call this when an "
           "action did nothing — the cause is usually a page error here.",
           [_p("max_entries", "integer", "Max lines to return."),
            _p("clear", "boolean", "Clear the buffer after reading.")]),

        fn("browser_record", "Start/stop a JSONL recording of this episode on "
           "the host (step, action, url, reward, console, optional screenshot "
           "frames). Recording is finalized automatically when the episode ends.",
           [_p("mode", "string", "Operation.", enum=["start", "stop", "status"]),
            _p("include_screenshots", "boolean",
               "Embed a base64 screenshot frame per step (start only).")]),

        fn("browser_wait", "Wait until an element is present.",
           target + [_p("timeout_ms", "integer", "Max wait in milliseconds.")]),

        fn("browser_tasks", "List the known tasks from the local browser_tasks "
           "dataset (the host keeps no catalogue).", []),

        fn("browser_score", "Score the current episode against its criteria.", []),
    ]


def tool_names() -> List[str]:
    return [t["function"]["name"] for t in build_browser_tools()]


def action_for_tool(tool_name: str) -> Optional[str]:
    """Map a browser_* tool name to its env action_type (None if not a step)."""
    pair = _ACTION_MAP.get(tool_name)
    return pair[0] if pair else None


def payload_for_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a tool's argument object into an env action payload.

    ``_ACTION_MAP`` entries list which arguments are RENAMED on the way in;
    everything else (selector, index, text, key, ...) passes through unchanged,
    because the env's action payload uses the same names.
    """
    _, renaming = _ACTION_MAP.get(tool_name, (None, {}))
    payload: Dict[str, Any] = {}
    for k, v in (args or {}).items():
        payload[renaming.get(k, k)] = v
    return payload


#: The env actions each browser tool is allowed to drive (asserted in tests so
#: a rename in either repo cannot drift silently).
SUPPORTED_ACTIONS = sorted({v[0] for v in _ACTION_MAP.values()} | {"extract"})

__all__ = [
    "build_browser_tools", "tool_names", "action_for_tool",
    "payload_for_tool", "READ_ONLY_TOOLS", "META_TOOLS",
    "is_non_step_tool", "SUPPORTED_ACTIONS",
]