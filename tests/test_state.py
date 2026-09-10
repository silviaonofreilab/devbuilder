"""Deployment state file: round trip, defaults, atomicity, validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devbuilder.deploy.state import (
    STATE_VERSION,
    DeployState,
    StateError,
    clear_state,
    load_state,
    save_state,
    state_path,
    update_state,
)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return state_path(tmp_path)


def test_missing_file_is_empty_state(path: Path) -> None:
    assert load_state(path) == DeployState()


def test_round_trip(path: Path) -> None:
    state = DeployState(pod_id="abc", vllm_endpoint="https://x", droplet_id=7)
    save_state(state, path)
    assert load_state(path) == state
    assert json.loads(path.read_text())["version"] == STATE_VERSION


def test_save_creates_directory_and_leaves_no_temp_file(path: Path) -> None:
    save_state(DeployState(), path)
    assert path.is_file()
    assert list(path.parent.iterdir()) == [path]


def test_update_merges_and_clear_resets(path: Path) -> None:
    update_state(path, pod_id="p1", vllm_endpoint="https://p1")
    update_state(path, droplet_ip="203.0.113.10")
    assert load_state(path) == DeployState(
        pod_id="p1", vllm_endpoint="https://p1", droplet_ip="203.0.113.10"
    )

    clear_state("pod_id", "vllm_endpoint", path=path)
    assert load_state(path) == DeployState(droplet_ip="203.0.113.10")


def test_update_rejects_unknown_field(path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown state fields"):
        update_state(path, bogus=1)


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps({"version": 99}),
        json.dumps({"version": STATE_VERSION, "surprise": 1}),
        json.dumps([1, 2]),
    ],
)
def test_invalid_file_raises(path: Path, body: str) -> None:
    path.parent.mkdir()
    path.write_text(body)
    with pytest.raises(StateError):
        load_state(path)
