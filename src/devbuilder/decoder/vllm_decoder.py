from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)


class VLLMError(Exception):
    """
    Base for all vLLM client errors.
    """


class VLLMConfigError(VLLMError):
    """
    Decoder configuration is missing or invalid.
    """


class VLLMGenerationError(VLLMError):
    """
    Generation request failed.
    """


class VLLMContextLengthError(VLLMGenerationError):
    """
    The server rejected the request because prompt + max_tokens exceed the
    model's context. The backend's count is authoritative; the API's
    character budget is only a heuristic in front of it.
    """


class VLLMDecoder:
    """
    Decoder served by vLLM: a thin client of its OpenAI-compatible HTTP API.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        system_prompt: str,
        timeout: float = 30.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
        api_key_env: str = "VLLM_API_KEY",
    ) -> None:
        base_url = base_url.rstrip("/")

        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        api_key = os.getenv(api_key_env)
        if not api_key:
            logger.warning("%s not set; sending unauthenticated requests", api_key_env)
            api_key = "not-needed"

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        logger.info("VLLMDecoder initialized (base_url=%s, model=%s)", base_url, model_id)

    @classmethod
    def from_settings(
        cls,
        decoder: dict[str, Any],
        *,
        system_prompt: str,
        api_key_env: str = "VLLM_API_KEY",
    ) -> VLLMDecoder:
        """
        Build a client from the decoder settings.
        """
        try:
            base_url = decoder["base_url"]
            model_id = decoder["model_id"]
            timeout = decoder["timeout_s"]
            max_tokens = decoder["max_tokens"]
            temperature = decoder["temperature"]
        except KeyError as exc:
            raise VLLMConfigError(f"missing decoder setting: {exc.args[0]}") from exc

        if not base_url:
            raise VLLMConfigError("decoder.base_url empty; run `make pod-up` first")

        return cls(
            base_url=base_url,
            model_id=model_id,
            system_prompt=system_prompt,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key_env=api_key_env,
        )

    def is_ready(self) -> bool:
        """
        Check that the server is reachable and serving the expected model.
        """
        try:
            models = self.client.models.list()
        except OpenAIError as exc:
            logger.warning("Readiness check failed: %s", exc)
            return False

        served = {m.id for m in models.data}
        if self.model not in served:
            logger.warning("Model %s not served; available: %s", self.model, sorted(served))
            return False
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float = 1.0,
        stop: Sequence[str] | None = None,
    ) -> str:
        """
        Run a single chat completion and return the assistant's reply.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system or self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                top_p=top_p,
                stop=list(stop) if stop else None,
            )
        except OpenAIError as exc:
            message = str(exc)
            if "context length" in message.lower() or "maximum context" in message.lower():
                logger.warning("vLLM rejected the prompt for length: %s", message)
                raise VLLMContextLengthError(message) from exc
            logger.error("vLLM generation failed: %s", message)
            raise VLLMGenerationError(message) from exc

        if not response.choices:
            raise VLLMGenerationError("vLLM returned no choices")

        content = response.choices[0].message.content
        if content is None:
            raise VLLMGenerationError("vLLM returned no content")

        return content.strip()
