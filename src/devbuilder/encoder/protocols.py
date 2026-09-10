from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """Interface implemented by text-embedding backends."""

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of each embedding vector."""
        ...

    @property
    def embedding_fingerprint(self) -> str:
        """
        Identity of the complete embedding transformation: graph, tokenizer,
        pooling, normalization, and input length. Stored with an index so a
        different transformation cannot silently query it.
        """
        ...

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts into a matrix with shape
        ``(len(texts), embedding_dim)``.
        """
        ...
