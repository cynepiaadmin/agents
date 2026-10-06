"""Composition root for the agent side: config → runtime → connected client.

Responsibilities kept here (and nowhere else):
  * load runtime_config.yaml and interpolate ``${ENV_VAR}`` from the process env
  * resolve the env host's base_url (and optionally spawn the container)
  * build a connected BrowserAgentRuntime

The host is a SEPARATE container. This module never imports ``browser_env`` —
it reaches the host over HTTP, spawning a container via Docker when nothing is
listening (``orchestrate_host=True``).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from browser_agent.runtime.orchestrator import BrowserAgentRuntime

logger = logging.getLogger("browser_agent.runtime.bootstrap")

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "runtime_config.yaml"
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(value: Any, *, strict: bool = False) -> Any:
    """Resolve ``${ENV_VAR}`` references.

    ``strict=True`` turns a missing variable into a hard error — used for
    credentials, where silently substituting an empty string would produce a
    confusing auth failure later instead of a clear startup error.
    """
    if isinstance(value, dict):
        return {k: _interpolate(v, strict=strict) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, strict=strict) for v in value]
    if not isinstance(value, str):
        return value

    def _sub(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name not in os.environ:
            if strict:
                raise ValueError(
                    f"environment variable {name!r} referenced in the runtime "
                    f"config is not set; export it (e.g. `export {name}=...`) "
                    f"before starting the agent")
            return ""
        return os.environ[name]

    return _VAR_RE.sub(_sub, value)


def load_runtime_config(path: Optional[str] = None,
                        *, strict_env: bool = False) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"runtime config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {cfg_path}")
    return _interpolate(raw, strict=strict_env)


def resolve_base_url(cfg: Dict[str, Any]) -> str:
    """base_url precedence: XA_ENV_BASE_URL env var > host.base_url > default."""
    host = cfg.get("host", {}) or {}
    return (os.environ.get("XA_ENV_BASE_URL")
            or host.get("base_url")
            or "http://localhost:8770")


def build_runtime(config_path: Optional[str] = None, *, orchestrate_host: bool = False,
                  strict_env: bool = False,
                  base_url: Optional[str] = None) -> BrowserAgentRuntime:
    """Load config, (optionally) ensure the host, and build a runtime.

    The returned runtime is NOT yet connected — call ``connect()`` (or use it
    as a context manager).
    """
    cfg = load_runtime_config(config_path, strict_env=strict_env)
    url = base_url or resolve_base_url(cfg)

    if orchestrate_host:
        from browser_agent.runtime.docker import ensure_env_host
        host_cfg = cfg.get("host", {}) or {}
        ensure_env_host(
            url,
            health_path=host_cfg.get("health_path", "/health"),
            docker_image=host_cfg.get("docker_image"),
            config_path=host_cfg.get("config_path") or None,
        )

    obs = cfg.get("observation", {}) or {}
    return BrowserAgentRuntime(
        url,
        auth_token=cfg.get("auth_token", "") or "",
        agent_caps=cfg.get("agent_caps") or ["env", "tools"],
        llm=cfg.get("llm"),
        system_prompt=cfg.get("system_prompt", "") or "",
        max_steps=int(cfg.get("max_steps", 12)),
        observation_chars=int(obs.get("chars", 4000)),
        observation_elements=int(obs.get("elements", 40)),
    )


__all__ = ["load_runtime_config", "resolve_base_url", "build_runtime"]