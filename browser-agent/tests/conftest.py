"""Test fixtures: the browser-env host as a REAL subprocess over HTTP.

This is the test boundary. Production code in this repo must never import
``browser_env`` — it only ever talks to the host over HTTP. The fixture below
launches the sibling repo as a subprocess (`python -m browser_env.app`) so the
suite exercises the true wire: TCP, sessions, SSE, and encoding.

Pitfalls encoded here:
  * free port per test (no cross-test collisions)
  * absolute config path (the app validates ROOT exists; no task file — tasks
    are caller-supplied on env/reset, never host config)
  * start_new_session=True + os.killpg: a plain terminate() leaves uvicorn's
    SSE reader alive and the suite hangs after the last test
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]           # browser-agent/
BROWSER_ENV_ROOT = REPO_ROOT.parent / "browser-env"       # sibling host repo


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _healthy(base_url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health",
                                    timeout=timeout) as resp:
            return getattr(resp, "status", 200) < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _host_config(tmp_path: Path, port: int) -> Path:
    """Write an explicit host config (absolute paths, scripted backend)."""
    root = (tmp_path / "state").resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "host_config.yaml"
    cfg.write_text(f"""host_id: "browser-env-itest"
transport:
  type: http
  config:
    base_url: "http://127.0.0.1:{port}"
    send_path: "/send"
    stream_path: "/stream"
server:
  host: "127.0.0.1"
  port: {port}
auth_token: null
audit:
  enabled: false
  path: null
  session_id_source: host_id
snapshot_store:
  type: file
  config:
    root: "{root}"
plugins:
  - name: browser_env
    type: env
    cls: browser_env.plugin.BrowserEnvPlugin
    metadata:
      enabled: true
      priority: 0
      exclusive: false
      BACKEND: "scripted"
      ROOT: "{root}"
      DEFAULT_MAX_STEPS: 8
      MAX_TEXT_CHARS: 4000
      MAX_ELEMENTS: 60
  - name: browser_tools
    type: tools
    cls: browser_env.browser_tool.BrowserToolPlugin
    metadata:
      enabled: true
      priority: 5
      exclusive: false
      ENV_PLUGIN: "browser_env"
      DEFAULT_URL: ""
      MAX_TEXT_CHARS: 4000
""")
    return cfg


class HostHandle:
    def __init__(self, base_url: str, proc: subprocess.Popen):
        self.base_url = base_url
        self.proc = proc

    def close(self) -> None:
        if self.proc.poll() is None:
            # Kill the whole process group: uvicorn keeps reader threads that a
            # plain terminate() does not reap, hanging the suite at teardown.
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.proc.kill()


@pytest.fixture
def env_host(tmp_path) -> HostHandle:
    if not (BROWSER_ENV_ROOT / "browser_env" / "app.py").exists():
        pytest.skip(f"browser-env host repo not found at {BROWSER_ENV_ROOT}")
    if not shutil.which(sys.executable):  # pragma: no cover
        pytest.skip("no python interpreter")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    cfg = _host_config(tmp_path, port)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BROWSER_ENV_ROOT)] + ([env["PYTHONPATH"]]
                                   if env.get("PYTHONPATH") else []))
    env["A2E_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "browser_env.app", "--config", str(cfg)],
        cwd=str(BROWSER_ENV_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )

    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"browser-env host exited early (rc={proc.returncode}):\n{out}")
        if _healthy(base_url):
            break
        time.sleep(0.25)
    else:
        handle = HostHandle(base_url, proc)
        handle.close()
        pytest.fail(f"browser-env host never became healthy at {base_url}")

    handle = HostHandle(base_url, proc)
    yield handle
    handle.close()


__all__ = ["HostHandle", "env_host", "BROWSER_ENV_ROOT", "REPO_ROOT"]