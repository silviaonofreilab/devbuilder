"""
ONNX Runtime encoder.

Loads an exported encoder artifact through
:class:`onnxruntime.InferenceSession`, tokenizes with the fast HuggingFace
tokenizer in NumPy mode, and applies pooling plus optional L2 normalization
in pure NumPy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime import SessionOptions
from transformers import AutoTokenizer

from devbuilder.encoder.pooling import Pooling, l2_normalize, pool
from devbuilder.manifest import (
    TOKENIZER_SUBDIR,
    EncoderManifest,
    embedding_fingerprint,
    read_manifest,
    verify_manifest,
)

logger = logging.getLogger(__name__)


class OrtEncoder:
    """
    Encode text to embeddings using ONNX Runtime.

    The encoder is fully described by its manifest.
    The constructor reads the manifest, verifies it against the artifact on disk,
    and configures the inference session.
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
        Initialize the encoder from an exported artifact directory.

        Args:
            models_dir: Root directory containing exported encoders.
            model_name: Subdirectory under models_dir that holds the artifact.
            batch_size: Number of texts per inference call.
            provider: ONNX Runtime execution provider. Defaults to "CPUExecutionProvider".
            session_options: Optional preconfigured ONNX Runtime session options.
                When omitted, intra-operation parallelism defaults to one thread.

        Raises:
            ValueError: If batch_size is less than one.
            RuntimeError: If the graph lacks the required output.
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
        manifest = read_manifest(artifact_dir, EncoderManifest)
        verify_manifest(manifest, model_dir=artifact_dir)

        self._pooling: Pooling = manifest.pooling
        self._normalize: bool = manifest.normalize
        self._max_length: int = manifest.max_length
        self._embedding_dim: int = manifest.embedding_dim
        self._embedding_fingerprint: str = embedding_fingerprint(manifest)

        tokenizer_dir = artifact_dir / TOKENIZER_SUBDIR
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

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
        self._input_names = {i.name for i in self._session.get_inputs()}
        output_names = {item.name for item in self._session.get_outputs()}
        if "last_hidden_state" not in output_names:
            raise RuntimeError(
                "Encoder graph does not expose the required 'last_hidden_state' output."
            )

        logger.info(
            "OrtEncoder ready (model=%s, file=%s, pooling=%s, normalize=%s, provider=%s)",
            model_name,
            manifest.artifact_filename,
            self._pooling,
            self._normalize,
            providers[0],
        )

    @property
    def embedding_dim(self) -> int:
        """
        Dimensionality of the produced embeddings, as declared by the manifest.
        """
        return self._embedding_dim

    @property
    def embedding_fingerprint(self) -> str:
        """
        Identity of the whole embedding transformation (see
        :func:`devbuilder.manifest.embedding_fingerprint`).
        """
        return self._embedding_fingerprint

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of strings to a (n, embedding_dim) float32 matrix.

        Tokenizes in NumPy mode, feeds only the inputs the ONNX graph declares
        (so models that omit token_type_ids are handled), pools, and L2-
        normalizes if the manifest says so. Internally chunks by batch_size
        to bound memory.
        """
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunks.append(self._encode_batch(texts[start : start + self.batch_size]))

        embeddings = np.concatenate(chunks, axis=0)
        expected_shape = (len(texts), self._embedding_dim)

        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Encoder returned shape {embeddings.shape}; expected {expected_shape}."
            )
        return embeddings

    def _encode_batch(self, batch: list[str]) -> np.ndarray:
        """Encode one batch sized at most batch_size."""
        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )

        missing_inputs = self._input_names - encoded.keys()
        if missing_inputs:
            raise RuntimeError(
                f"Tokenizer did not produce required inputs: {sorted(missing_inputs)}."
            )

        feed = {name: encoded[name] for name in self._input_names}
        last_hidden_state = self._session.run(
            ["last_hidden_state"],
            feed,
        )[0]

        pooled = pool(
            last_hidden_state,
            encoded["attention_mask"],
            self._pooling,
        )
        if self._normalize:
            pooled = l2_normalize(pooled)

        return pooled.astype(np.float32, copy=False)
