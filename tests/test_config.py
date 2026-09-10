from pathlib import Path
from typing import Any

import pytest

from devbuilder.config import _expand, load_config
from devbuilder.paths import paths


@pytest.fixture
def project_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("VLLM_ENDPOINT", "http://localhost:8000")

    return load_config(paths.configs / "settings.yaml")


def test_project_config(project_config: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "data",
        "encoder",
        "reranker",
        "vectorstore",
        "decoder",
    }
    assert required_sections <= project_config.keys()

    assert "name" in project_config["project"]
    assert "raw_file" in project_config["data"]
    assert "chunk_size" in project_config["data"]
    assert "model_id" in project_config["encoder"]
    assert "model_id" in project_config["reranker"]
    assert "top_k_default" in project_config["vectorstore"]
    assert "model_id" in project_config["decoder"]
    assert "max_prompt_chars" in project_config["decoder"]


def test_raw_data_file_exists(project_config: dict[str, Any]) -> None:
    raw_file = paths.raw / project_config["data"]["raw_file"]
    assert raw_file.is_file()


def test_environment_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_HOST", "localhost")
    monkeypatch.setenv("TEST_PORT", "7000")
    monkeypatch.setenv("TEST_ID", "010")
    monkeypatch.setenv("EMPTY_VALUE", "")

    expanded = _expand(
        {
            "url": "http://${TEST_HOST}:8080",
            "values": [
                "${TEST_PORT:-6333}",
                "${TEST_ID}",
                "${EMPTY_VALUE:-fallback}",
            ],
        }
    )

    assert expanded == {
        "url": "http://localhost:8080",
        "values": [7000, "010", "fallback"],
    }


def test_required_environment_variable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REQUIRED_VALUE", raising=False)

    with pytest.raises(ValueError, match="unset or empty: REQUIRED_VALUE"):
        _expand("${REQUIRED_VALUE}")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("${TEST_VALUE:default}", "Invalid environment placeholder"),
        ("${TEST_VALUE", "Unclosed environment placeholder"),
    ],
)
def test_malformed_placeholder_raises(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _expand(value)


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config(tmp_path / "missing.yaml")


def test_non_mapping_config_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config root must be a mapping"):
        load_config(config_path)
