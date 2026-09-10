"""Deployment state.

The record of what the operator has provisioned: pod, droplet, firewall. It
is written only by the deploy commands and read by everything that needs to
reach those resources. Kept out of ``.env`` on purpose: ``.env`` is operator
input (secrets, settings), this file is program output, and the two must not
share a writer.

Stored as JSON under ``.devbuilder/`` in the repo root (gitignored).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from devbuilder.paths import paths

logger = logging.getLogger(__name__)

STATE_DIR = ".devbuilder"
STATE_FILE = "state.json"
STATE_VERSION = 1


class StateError(Exception):
    """The state file exists but cannot be read or parsed."""


@dataclass(frozen=True)
class DeployState:
    """
    Provisioned resources. Every field is None until the matching ``up``
    command has run, and returns to None after ``terminate``/``destroy``.
    """

    pod_id: str | None = None
    vllm_endpoint: str | None = None
    droplet_id: int | None = None
    droplet_ip: str | None = None
    firewall_id: str | None = None

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


def state_path(base_dir: Path | None = None) -> Path:
    """Location of the state file for a repo root (defaults to ``paths.base_dir``)."""
    root = base_dir if base_dir is not None else paths.base_dir
    return root / STATE_DIR / STATE_FILE


def load_state(path: Path | None = None) -> DeployState:
    """
    Read the state file. A missing file is an empty state, not an error.

    Raises:
        StateError: If the file exists but is not valid state.
    """
    path = path or state_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DeployState()
    except OSError as exc:
        raise StateError(f"Could not read {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise StateError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        raise StateError(
            f"{path} has an unsupported format. Preserve the file and inspect it; "
            "it may contain resource IDs needed for recovery."
        )

    known = set(DeployState.field_names())
    unknown = set(data) - known - {"version"}
    if unknown:
        raise StateError(f"{path} has unknown fields: {sorted(unknown)}")

    return DeployState(**{k: v for k, v in data.items() if k in known})


def save_state(state: DeployState, path: Path | None = None) -> Path:
    """
    Write the state file atomically (temp file + rename) so a crash mid-write
    never leaves a half-written record.
    """
    path = path or state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": STATE_VERSION, **asdict(state)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    logger.debug("State written to %s", path)
    return path


def update_state(path: Path | None = None, **changes: object) -> DeployState:
    """Load, apply ``changes``, save, and return the new state."""
    unknown = set(changes) - set(DeployState.field_names())
    if unknown:
        raise ValueError(f"Unknown state fields: {sorted(unknown)}")
    state = replace(load_state(path), **changes)
    save_state(state, path)
    logger.info("State updated: %s", ", ".join(f"{k}={v}" for k, v in changes.items()))
    return state


def clear_state(*names: str, path: Path | None = None) -> DeployState:
    """Reset the named fields to None."""
    return update_state(path, **dict.fromkeys(names))
