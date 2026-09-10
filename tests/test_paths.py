from __future__ import annotations

import pytest

from devbuilder.paths import PathManager


def test_explicit_base_dir_wins(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVBUILDER_ROOT", "/nonexistent")
    assert PathManager(tmp_path).base_dir == tmp_path.resolve()


def test_base_dir_from_environment(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVBUILDER_ROOT", str(tmp_path))
    assert PathManager().base_dir == tmp_path.resolve()


def test_base_dir_defaults_to_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEVBUILDER_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert PathManager().base_dir == tmp_path.resolve()


def test_base_dir_is_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVBUILDER_ROOT", "relative/root")
    assert PathManager().base_dir.is_absolute()


@pytest.mark.parametrize(
    ("attribute", "relative"),
    [
        ("configs", "configs"),
        ("data", "data"),
        ("raw", "data/raw"),
        ("processed", "data/processed"),
        ("runs", "runs"),
        ("artifacts", "artifacts"),
        ("models", "artifacts/models"),
    ],
)
def test_derived_paths(tmp_path, attribute: str, relative: str) -> None:
    paths = PathManager(tmp_path)
    assert getattr(paths, attribute) == tmp_path.resolve() / relative


def test_ensure_dirs_creates_directories(tmp_path) -> None:
    paths = PathManager(tmp_path)
    paths.ensure_dirs()

    for directory in (
        paths.configs,
        paths.raw,
        paths.processed,
        paths.models,
        paths.runs,
    ):
        assert directory.is_dir()


def test_ensure_dirs_is_idempotent(tmp_path) -> None:
    paths = PathManager(tmp_path)
    paths.ensure_dirs()
    paths.ensure_dirs()
    assert paths.models.is_dir()
