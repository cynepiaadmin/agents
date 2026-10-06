"""Local task dataset loader — tasks live OUTSIDE the environment.

``browser_tasks/`` sits beside the ``browser_agent`` package (the same shape as
``browser-env/browser_tasks/``): a dataset, not part of either package's code.
The runtime resolves a task NAME to its full spec here and passes the whole
spec object in ``env/reset`` options. The host never sees a bare name — it
keeps no catalogue to resolve one against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

#: The dataset: a sibling directory of this package, never inside it.
DATASET_DIR = Path(__file__).resolve().parents[1] / "browser_tasks"
DATASET_FILE = DATASET_DIR / "tasks.json"

REQUIRED_FIELDS = ("name", "start_url", "goal")


class TaskError(ValueError):
    """Raised for a missing or malformed task definition."""


def load_tasks(path: Path = DATASET_FILE) -> Dict[str, dict]:
    """Load the local dataset: a JSON object with a ``tasks`` list.

    A malformed entry fails loudly here (caller side), not at reset time on
    the host.
    """
    if not path.exists():
        raise TaskError(f"task dataset not found: {path}")
    raw = json.loads(path.read_text())
    tasks = raw.get("tasks", raw) if isinstance(raw, dict) else raw
    out: Dict[str, dict] = {}
    for spec in tasks:
        _validate(spec)
        out[spec["name"]] = spec
    return out


def resolve_task(raw: Union[str, dict]) -> dict:
    """Turn a task name (or an inline spec) into the full spec object."""
    if isinstance(raw, dict):
        _validate(raw)
        return raw
    tasks = load_tasks()
    spec = tasks.get(str(raw))
    if spec is None:
        raise TaskError(
            f"unknown task {raw!r}; available: {sorted(tasks)} "
            f"(local dataset: {DATASET_FILE})"
        )
    return spec


def _validate(spec) -> None:
    if not isinstance(spec, dict):
        raise TaskError(f"task must be an object, got {type(spec).__name__}")
    missing = [k for k in REQUIRED_FIELDS if k not in spec]
    if missing:
        raise TaskError(f"task missing required field(s): {missing}")


def task_names() -> List[str]:
    return sorted(load_tasks())


__all__ = [
    "DATASET_DIR",
    "DATASET_FILE",
    "TaskError",
    "load_tasks",
    "resolve_task",
    "task_names",
]
