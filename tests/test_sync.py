"""The droplet .env rendered by devbuilder.deploy.droplet.sync."""

from __future__ import annotations

from pathlib import Path

import pytest

from devbuilder.deploy.droplet.sync import SyncError, render_env
from devbuilder.deploy.state import DeployState

STATE = DeployState(vllm_endpoint="https://pod.example.com")


@pytest.fixture
def env_file(tmp_path: Path):
    def write(body: str) -> Path:
        path = tmp_path / ".env"
        path.write_text(body)
        return path

    return write


def test_laptop_only_keys_are_removed(env_file) -> None:
    body = "\n".join(
        [
            "DIGITALOCEAN_TOKEN=do-secret",
            "DO_SSH_KEY_FINGERPRINT=aa:bb",
            "SSH_ALLOW_CIDR=auto",
            "RUNPOD_API_KEY=rp-secret",
            "GHCR_PAT=ghcr-secret",
            "GHCR_USER=someone",
            "VPS_HOST=203.0.113.10",
            "VLLM_ENDPOINT=https://stale.example.com",
        ]
    )
    result = render_env(env_file(body), STATE)

    for secret in ("do-secret", "rp-secret", "ghcr-secret"):
        assert secret not in result
    assert "VPS_HOST" not in result
    assert "GHCR_USER" not in result
    assert "DO_SSH_KEY_FINGERPRINT" not in result
    assert "SSH_ALLOW_CIDR" not in result
    assert "stale.example.com" not in result


def test_endpoint_comes_from_state_not_env(env_file) -> None:
    result = render_env(env_file("QDRANT_PORT=6333\n"), STATE)
    assert "VLLM_ENDPOINT=https://pod.example.com" in result

    assert "VLLM_ENDPOINT=\n" in render_env(env_file(""), DeployState())


def test_vps_api_token_is_renamed_and_local_token_dropped(env_file) -> None:
    result = render_env(env_file("API_TOKEN=local-only\nVPS_API_TOKEN=remote-secret\n"), STATE)

    assert "API_TOKEN=remote-secret" in result
    assert "local-only" not in result
    assert "VPS_API_TOKEN" not in result


def test_values_containing_equals_are_preserved(env_file) -> None:
    result = render_env(env_file("VPS_API_TOKEN=a=b=c\nTUNNEL_TOKEN=x=y\n"), STATE)

    assert "API_TOKEN=a=b=c" in result
    assert "TUNNEL_TOKEN=x=y" in result


def test_exported_assignments_are_recognised(env_file) -> None:
    body = "export DIGITALOCEAN_TOKEN=do-secret\nexport VPS_API_TOKEN=remote-secret\n"
    result = render_env(env_file(body), STATE)

    assert "do-secret" not in result
    assert "API_TOKEN=remote-secret" in result


def test_comments_and_blank_lines_survive(env_file) -> None:
    body = "# deployment\n\nCLOUDFLARE_TAG=2026.5.0\n"
    assert render_env(env_file(body), STATE).startswith(body)


def test_missing_env_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SyncError):
        render_env(tmp_path / "nonexistent", STATE)


def test_unlisted_keys_are_not_forwarded(env_file) -> None:
    # Allowlist, not denylist: a key nobody has vetted stays on the laptop.
    result = render_env(env_file("SOME_NEW_PROVIDER_TOKEN=secret\nTUNNEL_TOKEN=t\n"), STATE)
    assert "SOME_NEW_PROVIDER_TOKEN" not in result
    assert "secret" not in result
    assert "TUNNEL_TOKEN=t" in result
