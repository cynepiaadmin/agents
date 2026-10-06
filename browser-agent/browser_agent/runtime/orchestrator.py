"""BrowserAgentRuntime — the agent side of the browser environment.

Drives a browser-env host over A2E (HTTP) and owns the REACT loop. It imports
ONLY the a2e SDK — never ``browser_env`` — so the two repos stay separate
packages that share just the protocol.

Responsibilities:
  * handshake + TYPE_MAP registration (the #1 silent-hang cause)
  * episode control: start / step / observe / finish / score / list tasks
  * tool routing for the REACT loop (browser_* -> env/step)
  * provider resolution (stub for tests, OpenAI-compatible for production)

Action routing rule: while an episode is open, EVERY browser_* tool becomes one
``env/step``, so the host's step counter, reward, and done condition stay
authoritative. Acting outside an episode raises a clear error instead of
silently mutating the page unscored.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from a2e.core.client.client import A2EClient, A2EClientError
from a2e.core.transports.http import HTTPTransport

from a2e.caps.env.protocol import (
    ENV_TYPE_MAP,
    EnvCloseRequest,
    EnvResetRequest,
    EnvStepRequest,
)
from a2e.caps.tools.protocol import TOOL_TYPE_MAP, ToolCallRequest

from browser_agent.runtime.browser_tools import (
    READ_ONLY_TOOLS,
    action_for_tool,
    build_browser_tools,
    is_non_step_tool,
    payload_for_tool,
    tool_names,
)
from browser_agent.runtime.loop import run_agent_turn
from browser_agent.runtime.providers import LLMProvider, build_provider

logger = logging.getLogger("browser_agent.runtime")

# Normalizer: TYPE_MAP keys may be MessageType enum members, not strings.
def _norm(msg_types: dict) -> dict:
    return {getattr(k, "value", k): v for k, v in msg_types.items()}


class BrowserAgentRuntime:
    """Agent-side runtime driving a browser-env host over A2E."""

    def __init__(self, base_url: str, *, auth_token: str = "",
                 agent_caps: Optional[List[str]] = None,
                 llm: Any = None,
                 system_prompt: str = "",
                 max_steps: int = 12,
                 observation_chars: int = 4000,
                 observation_elements: int = 40):
        self._base_url = base_url
        self._agent_caps = agent_caps or ["env", "tools"]
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._obs_chars = observation_chars
        self._obs_elements = observation_elements

        # llm is either an LLMProvider instance or a build_provider() spec
        # dict; resolved lazily so the runtime imports with no provider set.
        self._llm_spec = llm
        self._llm_provider: Optional[LLMProvider] = (
            llm if isinstance(llm, LLMProvider) else None)

        logger = logging.getLogger("browser_agent.runtime")
        self._transport = HTTPTransport(base_url, logger, session_id=None)
        self._client = A2EClient(transport=self._transport, logger=logger,
                                 agent_id="browser-agent",
                                 auth_token=auth_token,
                                 agent_caps=self._agent_caps)
        # A2EClient seeds only A2E_BASE_TYPE_MAP. Register the SDK namespaces
        # we consume (env, tools) or responses decode to a bare message and
        # rpc() hangs until timeout with no error anywhere. NO custom wire
        # types: everything else rides the SDK's env/* + tool/* messages.
        self._client.update_msg_types(_norm(ENV_TYPE_MAP))
        self._client.update_msg_types(_norm(TOOL_TYPE_MAP))

        self.caps: List[str] = []
        self._episode_id: Optional[str] = None
        self._last_obs: Dict[str, Any] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    def connect(self) -> List[str]:
        self._client.connect()
        self.caps = [getattr(c.capability, "value", str(c.capability))
                     for c in self._client.capabilities()]
        logger.info("connected to %s caps=%s", self._base_url, self.caps)
        return self.caps

    def disconnect(self) -> None:
        try:
            self._client.disconnect()
        except Exception:
            logger.exception("disconnect failed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()
        return False

    # ── provider ───────────────────────────────────────────────────────────
    @property
    def provider(self) -> LLMProvider:
        if self._llm_provider is None:
            if self._llm_spec is None:
                raise RuntimeError(
                    "No LLM configured. Pass llm= (an LLMProvider or a "
                    "build_provider() spec dict) to BrowserAgentRuntime.")
            self._llm_provider = build_provider(self._llm_spec)
        return self._llm_provider

    @property
    def episode_id(self) -> Optional[str]:
        return self._episode_id

    # ── episode control (env/* namespace) ──────────────────────────────────
    def env_start(self, *, task: str = "", start_url: str = "",
                  goal: str = "", max_steps: int = 0,
                  options: Optional[dict] = None, timeout: int = 30) -> dict:
        opts = dict(options or {})
        if task:
            # Tasks live outside the environment: resolve the NAME against the
            # LOCAL browser_tasks/ dataset and send the FULL spec object. The
            # host keeps no catalogue and never sees a bare name.
            from browser_agent.tasking import resolve_task
            opts["task"] = resolve_task(task)
        elif start_url:
            opts["start_url"] = start_url
        if goal:
            opts["goal"] = goal
        if max_steps:
            opts["max_steps"] = int(max_steps)
        resp = self._client.rpc(EnvResetRequest(env_name="browser", options=opts),
                                timeout=timeout)
        state = resp.obs.state.model_dump()
        self._episode_id = resp.obs.episode_id
        self._last_obs = state
        return {"episode_id": self._episode_id, "state": state,
                "render": resp.obs.metadata.get("render_text", "")}

    def env_step(self, action_type: str, payload: Optional[dict] = None,
                 timeout: int = 60) -> dict:
        if not self._episode_id:
            raise RuntimeError(
                "No open episode. Call env_start(task=...) or "
                "env_start(start_url=...) first.")
        resp = self._client.rpc(
            EnvStepRequest(episode_id=self._episode_id,
                           action={"action_type": action_type,
                                   "payload": payload or {}}),
            timeout=timeout)
        return self._obs_result(resp.obs)

    def env_observe(self, timeout: int = 30) -> dict:
        if not self._episode_id:
            raise RuntimeError("No open episode. Call env_start(...) first.")
        from a2e.caps.env.protocol import EnvObserveRequest
        resp = self._client.rpc(EnvObserveRequest(episode_id=self._episode_id),
                                timeout=timeout)
        return self._obs_result(resp.obs)

    def env_finish(self, answer: str = "", timeout: int = 60) -> dict:
        return self.env_step("finish", {"answer": answer} if answer else {},
                             timeout=timeout)

    def env_close(self, timeout: int = 30) -> dict:
        if not self._episode_id:
            return {"closed": False, "detail": "no open episode"}
        resp = self._client.rpc(EnvCloseRequest(episode_id=self._episode_id),
                                timeout=timeout)
        self._episode_id = None
        self._last_obs = {}
        return {"closed": resp.closed}

    def list_tasks(self) -> List[dict]:
        """The task catalogue is a LOCAL dataset (browser_tasks/), not a host
        capability — the host has no catalogue to ask."""
        from browser_agent.tasking import load_tasks
        return list(load_tasks().values())

    def score(self, timeout: int = 30) -> dict:
        """Score the open episode via the SDK tool surface (tool/call) — the
        host publishes no custom score message; the terminal observation
        carries the reward, and this tool re-derives it on demand."""
        if not self._episode_id:
            return {"score": 0.0, "passed": False, "error": "no open episode"}
        payload = self.call_tool("browser_score", {}, timeout=timeout)
        if not payload.get("success"):
            return {"score": 0.0, "passed": False,
                    "error": payload.get("error") or "score failed"}
        data = payload.get("data") or {}
        return {"episode_id": self._episode_id,
                "score": float(data.get("score", 0.0)),
                "passed": bool(data.get("passed", False)),
                "criteria": data.get("criteria", {}),
                "error": ""}

    # ── direct tool calls (tools/* capability) ─────────────────────────────
    def call_tool(self, name: str, arguments: Optional[dict] = None,
                  timeout: int = 60) -> dict:
        """Call a host tool directly via tool/call.

        The wire nests: ToolCallResponse.data is a ToolResult, and the tool's
        own payload dict lives at ``ToolResult.data``. Unwrap both so callers
        see one result, not two.
        """
        resp = self._client.rpc(
            ToolCallRequest(tool_name=name, arguments=arguments or {}),
            timeout=timeout)
        payload = getattr(resp.data, "data", None)
        if isinstance(payload, dict):
            return payload
        return {"success": bool(getattr(resp.data, "success", False)),
                "error": getattr(resp.data, "error", None),
                "data": payload}

    # ── REACT executor ─────────────────────────────────────────────────────
    async def execute_tool(self, name: str, args: Dict[str, Any]) -> dict:
        """Route one LLM tool call. Returns {"status", "data"|"error"}."""
        try:
            if name == "env_start":
                data = self.env_start(
                    task=args.get("task", ""),
                    start_url=args.get("start_url", ""),
                    goal=args.get("goal", ""),
                    max_steps=int(args.get("max_steps", 0) or 0))
                return {"status": "success", "data": data}
            if name == "env_observe":
                return {"status": "success", "data": self.env_observe()}
            if name == "env_finish":
                return {"status": "success",
                        "data": self.env_finish(args.get("answer", ""))}
            if name == "browser_tasks":
                return {"status": "success", "data": {"tasks": self.list_tasks()}}
            if name == "browser_score":
                return {"status": "success", "data": self.score()}

            # Read-only browser tools (browser_text): observe the page WITHOUT
            # consuming an episode step. Kept explicit so the intent is visible
            # at the branch, not hidden behind a name lookup.
            if name in READ_ONLY_TOOLS:
                return {"status": "success", "data": self.env_observe()}

            # Everything else is a browser action driven through env/step.
            action_type = action_for_tool(name)
            if action_type is None:
                if is_non_step_tool(name):
                    return {"status": "error",
                            "error": f"{name!r} has no route in execute_tool; "
                                     f"it is a non-step tool and must be "
                                     f"handled explicitly"}
                return {"status": "error",
                        "error": f"unknown tool {name!r}; available: "
                                 f"{sorted(tool_names())}"}
            data = self.env_step(action_type, payload_for_tool(name, args))
            # The host signals a failed action INSIDE the observation
            # (action_ok=False + last_error/action_error), never by raising —
            # so the wrapper status must be derived from it, otherwise the LLM
            # would see status=success for an action that did nothing. The real
            # reason lives in state.last_error (cleared at the start of each
            # step, so it always describes THIS action).
            failure = (data.get("action_error")
                       or data.get("last_error")
                       or ("action rejected by the environment"
                           if data.get("action_ok") is False else ""))
            if failure:
                return {"status": "error", "error": str(failure), "data": data}
            return {"status": "success", "data": data}
        except A2EClientError as exc:
            return {"status": "error", "error": f"[{exc.code}] {exc.message}"}
        except Exception as exc:  # never raise into the loop
            logger.warning("tool %s failed: %s", name, exc, exc_info=True)
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    async def run_turn(self, user_message: str, *, on_step=None,
                       max_steps: Optional[int] = None,
                       model: Optional[str] = None,
                       history: Optional[list] = None) -> dict:
        """One REACT turn: the LLM drives the browser via execute_tool."""
        return await run_agent_turn(
            self.provider, user_message,
            executor=self.execute_tool,
            system_prompt=self._system_prompt,
            tools=build_browser_tools(),
            max_steps=max_steps or self._max_steps,
            on_step=on_step, model=model, history=history)

    def run_turn_sync(self, user_message: str, **kw) -> dict:
        """Sync wrapper (CLI / gateway paths are not on an event loop)."""
        return asyncio.run(self.run_turn(user_message, **kw))

    # ── observation shaping ────────────────────────────────────────────────
    def _obs_result(self, obs) -> dict:
        """Shape an EnvObservation into the compact dict the LLM sees."""
        state = obs.state.model_dump()
        md = obs.metadata or {}
        body = {
            "url": state.get("url", ""),
            "title": state.get("title", ""),
            "step_num": state.get("step_num"),
            "max_steps": state.get("max_steps"),
            "done": bool(obs.done),
            "reward": obs.reward,
            "last_error": state.get("last_error", ""),
            "goal": state.get("goal", ""),
            "visited": state.get("visited", []),
            "page_text": (state.get("page_text") or "")[: self._obs_chars],
            "elements": (state.get("elements") or [])[: self._obs_elements],
        }
        if obs.done:
            body["score"] = md.get("score")
            body["passed"] = md.get("passed")
            body["criteria"] = md.get("criteria")
            body["terminal_reason"] = md.get("terminal_reason")
        if md.get("extract") is not None:
            body["extracted"] = md["extract"]
        if md.get("value") is not None:
            body["value"] = md["value"]
        # Capture surfaces: pass through in full — the agent explicitly asked
        # for the page / the recording state / the console.
        if md.get("dom"):
            body["dom"] = md["dom"]
        if md.get("record"):
            body["record"] = md["record"]
        if md.get("console"):
            body["console"] = md["console"]
        if md.get("render"):
            body["screenshot"] = md["render"]
        if md.get("action_error"):
            body["action_error"] = md["action_error"]
        if md.get("action_ok") is False:
            body["action_ok"] = False
        self._last_obs = body
        return body


__all__ = ["BrowserAgentRuntime"]