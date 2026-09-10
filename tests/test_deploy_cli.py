"""devbuilder-deploy: per-command input validation, providers mocked."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from devbuilder.deploy import cli
from devbuilder.deploy.droplet import lifecycle


def _settings(fingerprint: str = "") -> dict:
    return {"vps": {"digitalocean": {"name": "demo", "ssh_key_fingerprint": fingerprint}}}


@pytest.fixture
def no_creation_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """No cloud-init file on disk, no .env loading, settings supplied per test."""
    monkeypatch.setattr(cli.paths, "base_dir", tmp_path)  # configs/ does not exist here
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)


def _run(settings: dict, *argv: str) -> int:
    with (
        patch.object(cli.cfg, "load_config", lambda _p: settings),
        patch("sys.argv", ["devbuilder-deploy", "droplet", *argv]),
    ):
        return cli.main()


def test_destroy_does_not_need_creation_inputs(no_creation_inputs) -> None:
    with patch.object(lifecycle, "destroy") as destroy:
        assert _run(_settings(), "destroy") == 0
    destroy.assert_called_once()


def test_status_does_not_need_creation_inputs(no_creation_inputs) -> None:
    with patch.object(lifecycle, "status") as status:
        assert _run(_settings(), "status") == 0
    status.assert_called_once()


def test_up_requires_fingerprint(no_creation_inputs) -> None:
    with patch.object(lifecycle, "up") as up:
        assert _run(_settings(), "up") == cli.EXIT_CONFIG
    up.assert_not_called()


def test_up_requires_cloud_init_file(no_creation_inputs) -> None:
    with patch.object(lifecycle, "up") as up:
        assert _run(_settings(fingerprint="aa:bb"), "up") == cli.EXIT_CONFIG
    up.assert_not_called()
