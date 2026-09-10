from __future__ import annotations

from typing import Protocol


class RerankerBackend(Protocol):
    """Interface implemented by rerankers: score passages against a query."""

    @property
    def max_length(self) -> int:
        """Token budget for one (query, passage) pair."""
        ...

    def query_tokens(self, query: str) -> int:
        """Tokens the query occupies in a pair, per this model's tokenizer."""
        ...

    def rank(
        self,
        query: str,
        passages: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Return passage indices and scores, best first."""
        ...
