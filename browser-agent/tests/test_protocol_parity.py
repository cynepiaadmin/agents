"""Guardrails: SDK-only wire, shared protocol not code, separated configs.

Four properties are enforced here, each of which has broken a real system
before:

1. SDK-only wire — neither repo may invent A2E message types. Everything the
   two sides exchange must be a message the a2eprotocol/python-sdk defines
   (env/*, tool/*). A custom `.../req` type string or an A2EMessage subclass
   defined locally is exactly the over-engineering that silently forks the
   protocol: it decodes to a bare message on the other side and the RPC hangs
   with no exception anywhere. Score, tasks and captures all ride the SDK's
   env/* + tool/* surfaces.

2. No reverse coupling — browser_agent must not import browser_env outside the
   test boundary. The host is a container the agent talks to over HTTP, not a
   library it links against.

3. Config separation — the host config (A2EHostConfig) and the agent config
   must stay separate files; agent keys inside the host config raise a
   pydantic ValidationError at host startup.

4. Tasks outside the environment — the task dataset (browser_tasks/) is a
   sibling of each package, never host configuration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_REPO = REPO_ROOT.parent / "browser-env"
AGENT_DIR = REPO_ROOT / "browser_agent"

#: Production trees (tests excluded — the test boundary may reference anything).
_PRODUCTION = (
    (REPO_ROOT, AGENT_DIR),
    (HOST_REPO, HOST_REPO / "browser_env"),
)


def _py_files() -> list[Path]:
    files = []
    for root, tree in _PRODUCTION:
        if tree.exists():
            files.extend(p for p in tree.rglob("*.py")
                         if "tests" not in p.parts and "scripts" not in p.parts)
    return files


# ── 1. SDK-only wire (textual, no import) ──────────────────────────────────
def test_no_custom_message_type_strings():
    """No repo-defined `namespace/verb/req|resp` literals outside the SDK."""
    offenders = []
    pattern = re.compile(r'"([a-z][a-z0-9_]*/[a-z0-9_/]+/(?:req|resp))"')
    for py in _py_files():
        for match in pattern.finditer(py.read_text()):
            offenders.append(f"{py.name}: {match.group(1)}")
    assert not offenders, (
        f"custom wire types found: {offenders}. Only a2eprotocol/python-sdk "
        f"may define message types; use env/* or tool/* for new behavior.")


def test_no_local_a2esubclass_message_definitions():
    """Neither repo may subclass A2EMessage — message schemas come from the SDK."""
    offenders = []
    pattern = re.compile(r"^class\s+\w+\s*\(\s*A2EMessage\s*\)",
                         flags=re.MULTILINE)
    for py in _py_files():
        if pattern.search(py.read_text()):
            offenders.append(str(py))
    assert not offenders, (
        f"A2EMessage subclasses defined outside the SDK: {offenders}")


# ── 2. no reverse coupling ─────────────────────────────────────────────────
def test_agent_code_never_imports_browser_env():
    """The only allowed reference to browser_env is the test boundary."""
    offenders = []
    for py in AGENT_DIR.rglob("*.py"):
        if "tests" in py.parts:          # test boundary: spawns the host
            continue
        text = py.read_text()
        if re.search(r'^\s*(from\s+browser_env|import\s+browser_env)', text,
                     flags=re.MULTILINE):
            offenders.append(str(py.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"browser_env imported from production agent code: {offenders}. The "
        f"host is a container reached over HTTP, never a linked library.")


def test_agent_cli_does_not_reference_browser_env_package():
    cli = (AGENT_DIR / "cli.py").read_text()
    assert "import browser_env" not in cli
    assert "from browser_env" not in cli


# ── 3. config separation ───────────────────────────────────────────────────
def test_host_config_has_no_agent_keys():
    """A2EHostConfig forbids extra fields — an agent key here is a startup
    ValidationError, so catch it in the repo, not in the container."""
    from a2e.schema import A2EHostConfig
    import yaml
    cfg_path = HOST_REPO / "browser_env" / "host_config.yaml"
    if not cfg_path.exists():
        pytest.skip("browser-env sibling repo not present")
    raw = yaml.safe_load(cfg_path.read_text())
    for banned in ("llm", "agent_caps", "system_prompt", "provider_name",
                   "api_credentials", "max_steps"):
        assert banned not in raw, f"agent-side key {banned!r} in host config"
    cfg = A2EHostConfig(**raw)   # must validate
    assert cfg.server.port == 8770


def test_agent_config_has_no_host_plugin_keys():
    import yaml
    path = AGENT_DIR / "runtime" / "runtime_config.yaml"
    raw = yaml.safe_load(path.read_text())
    assert "plugins" not in raw, "plugin list belongs to the HOST config"
    assert "transport" not in raw, "transport belongs to the HOST config"
    assert raw.get("agent_caps") == ["env", "tools"]
    assert raw.get("system_prompt")


def test_both_configs_exist_in_their_own_repo():
    assert (HOST_REPO / "browser_env" / "host_config.yaml").exists()
    assert (AGENT_DIR / "runtime" / "runtime_config.yaml").exists()


# ── 4. tasks outside the environment ───────────────────────────────────────
def test_no_task_file_in_host_config_or_package():
    """The host ships no task catalogue: not in config, not in the package."""
    cfg_path = HOST_REPO / "browser_env" / "host_config.yaml"
    if not cfg_path.exists():
        pytest.skip("browser-env sibling repo not present")
    assert "TASKS_FILE" not in cfg_path.read_text()
    assert not (HOST_REPO / "browser_env" / "tasks.json").exists()


def test_task_dataset_sits_beside_the_packages():
    """browser_tasks/ is a dataset at the same level as the packages."""
    assert (REPO_ROOT / "browser_tasks" / "tasks.json").exists()
    if (HOST_REPO / "browser_env").exists():
        assert (HOST_REPO / "browser_tasks" / "tasks.json").exists()
