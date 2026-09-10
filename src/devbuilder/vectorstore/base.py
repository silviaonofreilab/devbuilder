from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    chunk: dict[str, Any]
    score: float


class IndexIntegrityError(RuntimeError):
    """The collection is absent, incomplete, or was built by a different encoder."""
