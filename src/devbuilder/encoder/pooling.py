"""
Pooling and normalization utilities.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Pooling = Literal["mean", "max", "cls"]


def pool(token_embeddings: np.ndarray, attention_mask: np.ndarray, strategy: Pooling) -> np.ndarray:
    """
    Reduce token-level embeddings to one vector per sequence.

    Args:
        token_embeddings: (batch, seq, hidden)
        float array; the transformer's last_hidden_state.
        attention_mask: (batch, seq)
        int/float array; 1 for real tokens, 0 for padding. Broadcasted to (batch, seq, 1) internally.
        strategy: "mean" (masked average), "max" (masked max), or "cls" (the first token).

    Returns:
        (batch, hidden) float32 array.

    Raises:
        ValueError: If strategy is not one of the supported options.
    """
    mask = attention_mask[..., None].astype(np.float32)
    tokens = token_embeddings.astype(np.float32, copy=False)

    if strategy == "mean":
        summed = (tokens * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts
    if strategy == "max":
        # Push padding to -inf so it never wins the max.
        masked = tokens + (1.0 - mask) * np.finfo(np.float32).min
        return masked.max(axis=1)
    if strategy == "cls":
        return tokens[:, 0, :]
    raise ValueError(f"Unsupported pooling strategy: {strategy!r}. Use 'mean', 'max', or 'cls'.")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of x (vector-wise unit norm).

    Args:
        x: (batch, hidden) float array.

    Returns:
        (batch, hidden) float32 array with each row of unit norm. Zero rows
        are returned unchanged (norm clipped to a small epsilon to avoid NaNs).
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norm, 1e-9, None)).astype(np.float32)
