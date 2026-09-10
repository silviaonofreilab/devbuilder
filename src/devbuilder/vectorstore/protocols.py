from __future__ import annotations

from typing import Any, Protocol


class VectorStoreBackend(Protocol):
    """Interface implemented by vector stores: nearest chunks for a query."""

    def retrieve(self, query: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        """Return (payload, score) pairs for the top-k matches."""
        ...
