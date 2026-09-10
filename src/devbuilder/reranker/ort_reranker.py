"""
ONNX Runtime reranker.

Loads an exported cross-encoder artifact through
:class:`onnxruntime.InferenceSession`, tokenizes query-passage pairs with the
fast HuggingFace tokenizer in NumPy mode, and returns one relevance score per
passage. The classification head emits the score directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime import SessionOptions
from transformers import AutoTokenizer

from devbuilder.manifest import (
    TOKENIZER_SUBDIR,
    RerankerManifest,
    read_manifest,
    verify_manifest,
)

logger = logging.getLogger(__name__)


class OrtReranker:
    """
    Score query-passage pairs using ONNX Runtime.

    The reranker is fully described by its manifest. The constructor reads the
    manifest, verifies it against the artifact on disk, and configures the
    inference session.
    """

    def __init__(
        self,
        models_dir: Path,
        model_name: str,
        *,
        batch_size: int,
        provider: str | None = None,
        session_options: SessionOptions | None = None,
    ) -> None:
        """
        Initialize the reranker from an exported artifact directory.

        Args:
            models_dir: Root directory containing exported models.
            model_name: Subdirectory under models_dir that holds the artifact.
            batch_size: Number of query-passage pairs per inference call.
            provider: ONNX Runtime execution provider. Defaults to
                ``"CPUExecutionProvider"``.
            session_options: Optional preconfigured ONNX Runtime session
                options. When omitted, intra-operation parallelism defaults
                to one thread.

        Raises:
            ValueError: If batch_size is less than one.
            RuntimeError: If the graph lacks a 'logits' output or declares a
                multi-label head.
            ManifestMissingError: If manifest.json is absent.
            IncompatibleArtifactError: If the manifest doesn't match the artifact.
            CorruptArtifactError: If the artifact's sha256 doesn't match the manifest.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.models_dir = models_dir
        self.model_name = model_name
        self.batch_size = batch_size

        artifact_dir = models_dir / model_name
        manifest = read_manifest(artifact_dir, RerankerManifest)
        verify_manifest(manifest, model_dir=artifact_dir)

        self._max_length: int = manifest.max_length

        tokenizer_dir = artifact_dir / TOKENIZER_SUBDIR
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
        )

        if session_options is None:
            session_options = SessionOptions()
            session_options.intra_op_num_threads = 1

        onnx_path = artifact_dir / manifest.artifact_filename
        providers = [provider or "CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers,
        )

        self._input_names = {item.name for item in self._session.get_inputs()}

        logits_output = next(
            (output for output in self._session.get_outputs() if output.name == "logits"),
            None,
        )
        if logits_output is None:
            raise RuntimeError("Reranker graph does not expose the required 'logits' output.")

        declared_shape = logits_output.shape
        if declared_shape is None or len(declared_shape) != 2 or declared_shape[-1] != 1:
            raise RuntimeError(
                f"Reranker requires logits with shape (batch, 1); graph declares {declared_shape}."
            )

        logger.info(
            "OrtReranker ready (model=%s, file=%s, provider=%s)",
            model_name,
            manifest.artifact_filename,
            providers[0],
        )

    def query_tokens(self, query: str) -> int:
        """
        Count query tokens plus the special tokens required for a query-passage pair.

        ``max_length - query_tokens(query)`` is the token budget available
        for passage text.
        """
        query_ids = self._tokenizer(query, add_special_tokens=False)["input_ids"]
        pair_special_tokens = self._tokenizer.num_special_tokens_to_add(pair=True)
        return len(query_ids) + pair_special_tokens

    @property
    def max_length(self) -> int:
        """Pair-token budget the artifact was exported with."""
        return self._max_length

    def score(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        """
        Return one raw relevance logit per passage.

        Pairs each passage with the query, feeds only inputs declared by the
        ONNX graph, and chunks by batch_size to bound memory. Scores are raw
        logits rather than probabilities; their ordering is what matters.

        Raises:
            RuntimeError: If the output shape does not match the passage count.
        """
        if not passages:
            return np.empty((0,), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            chunks.append(self._score_batch(query, batch))

        scores = np.concatenate(chunks, axis=0)
        expected_shape = (len(passages),)

        if scores.shape != expected_shape:
            raise RuntimeError(
                f"Reranker returned shape {scores.shape}; expected {expected_shape}."
            )

        return scores

    def _score_batch(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        """Score one batch of at most batch_size passages."""
        encoded = self._tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation="only_second",
            max_length=self._max_length,
            return_tensors="np",
        )

        missing_inputs = self._input_names - encoded.keys()
        if missing_inputs:
            raise RuntimeError(
                f"Tokenizer did not produce required inputs: {sorted(missing_inputs)}."
            )

        feed = {name: encoded[name] for name in self._input_names}

        logits = self._session.run(["logits"], feed)[0]
        return logits.reshape(-1).astype(np.float32, copy=False)

    def rank(
        self,
        query: str,
        passages: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Return the top-k passage indices and scores, best first."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")

        scores = self.score(query, passages)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(int(index), float(scores[index])) for index in order]
