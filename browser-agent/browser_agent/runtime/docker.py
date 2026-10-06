"""A2E browser environment host driver (agent side).

`ensure_env_host` guarantees a browser-env host is REACHABLE, spawning it as a
separate Docker container if it is not already running. Coupling is Docker-only:
the agent never imports `browser_env`, never runs the host as a subprocess in
production, and talks to it exclusively over HTTP.

Dependency direction: browser_agent --(HTTP)--> browser-env.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger("browser_agent.runtime.docker")


@dataclass
class EnvHandle:
    """A live (or already-running) env host. ``spawned=False`` means we
    attached, so teardown is a no-op."""

    base_url: str
    teardown: Callable[[], None] = lambda: None
    spawned: bool = False
    container_id: Optional[str] = None
    process: Optional["subprocess.Popen"] = field(default=None, repr=False)


def _probe(base_url: str, health_path: str = "/health", timeout: float = 2.0) -> bool:
    try:
        with urlopen(Request(base_url.rstrip("/") + health_path, method="GET"),
                     timeout=timeout) as resp:
            return getattr(resp, "status", 200) < 500
    except (URLError, HTTPError, OSError, ValueError):
        return False


def _wait_healthy(base_url: str, health_path: str, timeout: float,
                  poll_interval: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _probe(base_url, health_path):
            logger.info("env host healthy at %s", base_url)
            return
        time.sleep(poll_interval)
    raise RuntimeError(f"env host did not become healthy at {base_url} "
                       f"within {timeout}s")


def ensure_env_host(base_url: str, *, health_path: str = "/health",
                    docker_image: Optional[str] = None,
                    config_path: Optional[str] = None,
                    health_timeout: float = 45.0,
                    poll_interval: float = 0.5,
                    env_extra: Optional[dict] = None) -> EnvHandle:
    """Ensure a browser-env host answers at ``base_url``/``health_path``.

    1. Already up → attach (spawned=False, teardown is a no-op).
    2. Otherwise spawn a container from ``docker_image`` (docker required) and
       poll until healthy.
    """
    if _probe(base_url, health_path):
        logger.info("env host already up at %s", base_url)
        return EnvHandle(base_url=base_url, spawned=False)

    if not (docker_image and shutil.which("docker")):
        raise RuntimeError(
            f"env host not reachable at {base_url} and cannot be spawned: "
            f"set host.docker_image in runtime_config.yaml and build the image "
            f"(make -C browser-env build). docker_image={docker_image!r} "
            f"docker_available={bool(shutil.which('docker'))}")

    logger.info("spawning env host (image=%s config=%s)", docker_image, config_path)
    port = urlparse(base_url).port or 8770
    cmd = ["docker", "run", "-d", "--rm", "-p", f"{port}:{port}"]
    if env_extra:
        for k, v in env_extra.items():
            cmd += ["-e", f"{k}={v}"]
    if config_path and os.path.exists(config_path):
        # config_path MUST be absolute: docker -v rejects relative host paths.
        config_path = os.path.abspath(config_path)
        cmd += ["-v", f"{config_path}:/app/browser_env/host_config.yaml:ro"]
        cmd += [docker_image, "--config", "/app/browser_env/host_config.yaml"]
    else:
        cmd += [docker_image]

    try:
        # shell=False: pass the argv list verbatim (no shlex quoting — that
        # would pass literal quote characters as part of each argument).
        cid = subprocess.check_output(cmd, text=True,
                                      stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        logger.error("docker run failed (rc=%s): %s", exc.returncode, exc.output)
        raise
    logger.info("env host container started: %s", cid)

    handle = EnvHandle(base_url=base_url, spawned=True, container_id=cid)

    def _teardown():
        try:
            subprocess.run(["docker", "stop", cid], check=False,
                           capture_output=True)
        except Exception as exc:
            logger.warning("docker stop failed: %s", exc)

    handle.teardown = _teardown
    _wait_healthy(base_url, health_path, health_timeout, poll_interval)
    return handle


__all__ = ["EnvHandle", "ensure_env_host", "_probe"]