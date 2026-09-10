"""vLLM client settings validation. No network: the pod itself is smoke-tested by `pod up`."""

from __future__ import annotations

import pytest
from openai import OpenAIError

import devbuilder.config as cfg
from devbuilder.decoder import (
    VLLMConfigError,
    VLLMContextLengthError,
    VLLMDecoder,
    VLLMGenerationError,
)
from devbuilder.paths import paths


@pytest.fixture(scope="module")
def decoder() -> dict:
    return cfg.load_config(paths.configs / "settings.yaml")["decoder"]


def test_from_settings_raises_when_base_url_empty(decoder: dict) -> None:
    with pytest.raises(VLLMConfigError):
        VLLMDecoder.from_settings(
            {**decoder, "base_url": ""},
            system_prompt="unused",
        )


def test_from_settings_raises_on_missing_key(decoder: dict) -> None:
    incomplete = {k: v for k, v in decoder.items() if k != "timeout_s"}
    with pytest.raises(VLLMConfigError, match="timeout_s"):
        VLLMDecoder.from_settings(incomplete, system_prompt="unused")


def _decoder(decoder: dict, monkeypatch, error: Exception) -> VLLMDecoder:
    d = VLLMDecoder.from_settings({**decoder, "base_url": "http://unused"}, system_prompt="s")

    def create(**kwargs):
        raise error

    monkeypatch.setattr(d.client.chat.completions, "create", create)
    return d


def test_context_length_rejection_is_a_distinct_error(decoder: dict, monkeypatch) -> None:
    err = OpenAIError("This model's maximum context length is 2048 tokens; requested 2300")
    with pytest.raises(VLLMContextLengthError):
        _decoder(decoder, monkeypatch, err).generate("p")


def test_other_backend_errors_stay_generic(decoder: dict, monkeypatch) -> None:
    with pytest.raises(VLLMGenerationError) as exc:
        _decoder(decoder, monkeypatch, OpenAIError("boom")).generate("p")
    assert not isinstance(exc.value, VLLMContextLengthError)
