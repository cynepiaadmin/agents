# syntax=docker/dockerfile:1
# browser-agent — agent side of the A2E browser environment.
#
# Drives a browser-env container over HTTP (env/* episodes + tools/* calls) and
# owns the REACT loop. Contains NO browser and NO browser_env code: the agent
# and the environment are separate packages sharing only the a2e SDK.
#
# Build from the PARENT directory (holds browser-agent/, browser-env/ and
# a2eprotocol/):
#   docker build -f browser-agent/Dockerfile -t browser-agent:latest .
# or from this directory:
#   docker build -f Dockerfile -t browser-agent:latest ..
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── a2e SDK: vendored from the LOCAL checkout, same revision as the host ──
# pyproject.toml's [tool.uv.sources] points at ../a2eprotocol/python-sdk; from
# /app that escapes the base dir, so COPY the SDK to the exact path it resolves
# to and rewrite the reference below.
COPY a2eprotocol/python-sdk /a2eprotocol/python-sdk

# ── project metadata + source ─────────────────────────────────────────────
COPY browser-agent/pyproject.toml ./
COPY browser-agent/browser_agent ./browser_agent

# ── install ───────────────────────────────────────────────────────────────
RUN sed -i 's#\.\./a2eprotocol/python-sdk#/a2eprotocol/python-sdk#g' pyproject.toml \
    && python -m pip install --upgrade pip \
    && python -m pip install "/a2eprotocol/python-sdk" \
    && python -m pip install .

# ── build-time contract gates ─────────────────────────────────────────────
# 1) The SDK surface the agent consumes must exist (a stale pin fails here,
#    not as a silent RPC hang at runtime).
RUN python -c "\
from a2e.core.client.client import A2EClient; \
from a2e.core.transports.http import HTTPTransport; \
from a2e.caps.env.protocol import ENV_TYPE_MAP, EnvResetRequest, EnvStepRequest; \
from a2e.caps.tools.protocol import TOOL_TYPE_MAP, ToolCallRequest; \
assert 'env/reset/req' in [getattr(k,'value',k) for k in ENV_TYPE_MAP], 'env reset missing'; \
assert 'tool/call/req' in [getattr(k,'value',k) for k in TOOL_TYPE_MAP], 'tool call missing'; \
print('a2e client+env+tools contract OK')"

# 2) The agent package must import and the config must parse.
RUN python -c "\
import yaml, pathlib; \
from browser_agent.runtime.orchestrator import BrowserAgentRuntime; \
from browser_agent.runtime.bootstrap import load_runtime_config, build_runtime; \
cfg = load_runtime_config('/app/browser_agent/runtime/runtime_config.yaml'); \
assert cfg.get('agent_caps'), 'agent_caps missing'; \
assert cfg.get('system_prompt'), 'system_prompt missing'; \
print('browser_agent OK; caps =', cfg['agent_caps'])"

# 3) The tool schemas must build (they are the agent's whole action surface).
RUN python -c "\
from browser_agent.runtime.browser_tools import build_browser_tools, tool_names; \
t = build_browser_tools(); assert len(t) >= 12, f'only {len(t)} tools'; \
print('tool schemas OK:', len(t), 'tools')"

# No browser here by design: state it so nobody adds playwright by mistake.
RUN python -c "\
import pathlib, sys; \
bad = list(pathlib.Path('/app').rglob('*playwright*')); \
assert not bad, f'playwright must not be in the agent image: {bad}'; \
print('no browser in agent image (by design)')"

EXPOSE 8080
ENTRYPOINT ["python", "-m", "browser_agent.cli"]
CMD ["--help"]