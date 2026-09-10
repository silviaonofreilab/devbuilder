from __future__ import annotations

from typing import Protocol


class DecoderBackend(Protocol):
    """Interface implemented by decoders: generate an answer from a prompt."""

    model: str
    """Identifier of the served model, for logs and readiness checks."""

    def is_ready(self) -> bool:
        """True if the backend is reachable and serving the expected model."""
        ...

    def generate(self, prompt: str) -> str:
        """Return the model's reply to ``prompt``."""
        ...
