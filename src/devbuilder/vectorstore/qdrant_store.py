from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from devbuilder.encoder import EmbeddingBackend
from devbuilder.pipelines.preprocessing import Chunk
from devbuilder.vectorstore.base import IndexIntegrityError, RetrievalResult

logger = logging.getLogger(__name__)

_POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "devbuilder.vectorstore")

# One reserved point per collection records how the index was built. It is
# written last, so its presence means indexing completed; its payload says
# which embedding transformation and corpus produced the vectors. Retrieval
# filters it out.
_META_POINT_ID = str(uuid.uuid5(_POINT_ID_NAMESPACE, "index-meta"))
_META_KIND = "index_meta"
_NOT_META = Filter(must_not=[FieldCondition(key="kind", match=MatchValue(value=_META_KIND))])


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """SHA-256 over the chunk identities and texts, in order."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.chapter}:{c.chunk_index}:{c.text}\n".encode())
    return h.hexdigest()


class QdrantVectorStore:
    """
    Qdrant wrapper for indexing and retrieving Chunk objects.

    Connects to a running Qdrant instance (local or remote).
    Collection is created on build_index() if it does not exist.
    """

    def __init__(
        self,
        embeddings: EmbeddingBackend,
        store_name: str,
        host: str,
        port: int,
        client: QdrantClient | None = None,
    ):
        # The vector size is a property of the encoder artifact (its manifest),
        # not a separate setting. An existing collection is checked against the
        # encoder (identity, not just dimension) before it is used.
        self.embeddings = embeddings
        self.store_name = store_name
        self.vector_size = embeddings.embedding_dim

        if client is None:
            self._client = QdrantClient(host=host, port=port)
            logger.info(
                "QdrantVectorStore ready (collection=%s, host=%s:%s)",
                self.store_name,
                host,
                port,
            )
        else:
            self._client = client
            logger.info(
                "QdrantVectorStore ready (collection=%s, injected_client=%s)",
                self.store_name,
                type(client).__name__,
            )

    def _read_meta(self) -> dict[str, Any] | None:
        """The index-meta payload, or None if the collection or the marker is absent."""
        if not self._client.collection_exists(self.store_name):
            return None
        points = self._client.retrieve(self.store_name, ids=[_META_POINT_ID], with_payload=True)
        return points[0].payload if points else None

    def _check_index(self) -> dict[str, Any]:
        """
        Raise unless the collection exists, indexing completed, and the index
        was built by the embedding transformation this store is using.
        Returns the meta payload for further checks.
        """
        meta = self._read_meta()
        if meta is None:
            raise IndexIntegrityError(
                f"Collection {self.store_name!r} is missing or was never fully indexed; "
                "run the indexer."
            )
        if meta.get("embedding_fingerprint") != self.embeddings.embedding_fingerprint:
            raise IndexIntegrityError(
                f"Collection {self.store_name!r} was built by a different encoder "
                f"(index={str(meta.get('embedding_fingerprint'))[:12]}, "
                f"current={self.embeddings.embedding_fingerprint[:12]}); re-index with "
                "--force-recreate."
            )
        self._validate_collection_vector_size()
        return meta

    def is_ready(self) -> bool:
        try:
            self._check_index()
            return True
        except Exception as exc:
            logger.warning(
                "Qdrant readiness check failed (collection=%s): %s", self.store_name, exc
            )
            return False

    def _validate_collection_vector_size(self) -> None:
        collection = self._client.get_collection(self.store_name)
        vectors_config = collection.config.params.vectors

        if not isinstance(vectors_config, VectorParams):
            raise ValueError(
                f"Collection {self.store_name!r} does not use a single unnamed vector."
            )

        if vectors_config.size != self.vector_size:
            raise ValueError(
                f"Collection {self.store_name!r} has vector size "
                f"{vectors_config.size}; expected {self.vector_size}."
            )

    def build_index(self, chunks: Sequence[Chunk], *, force_recreate: bool = False) -> None:
        """
        Build the Qdrant collection from chunks.

        Idempotent by default: a complete collection built by this encoder is
        left as is; an incomplete or foreign one raises. Pass
        ``force_recreate=True`` to rebuild. Embeddings are computed before
        anything is deleted, so an encoder failure never costs the old index.
        The meta point is written last: its presence marks completion.

        Raises:
            ValueError: If ``chunks`` is empty or the encoder returns a bad shape.
            IndexIntegrityError: If an existing collection is incomplete or was
                built by a different encoder and ``force_recreate`` is False.
        """
        if not chunks:
            raise ValueError("No chunks provided; cannot build index.")

        if self._client.collection_exists(self.store_name) and not force_recreate:
            meta = self._check_index()
            if meta.get("corpus_sha256") != corpus_fingerprint(chunks):
                raise IndexIntegrityError(
                    f"Collection {self.store_name!r} was built from a different corpus; "
                    "re-index with --force-recreate."
                )
            logger.info("Collection %s already exists and is complete; skipping", self.store_name)
            return

        texts = [c.text for c in chunks]
        vectors = self.embeddings.embed_texts(texts)

        expected_shape = (len(chunks), self.vector_size)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"Embedding backend returned shape {vectors.shape}; expected {expected_shape}."
            )

        if self._client.collection_exists(self.store_name):
            logger.info("Recreating collection %s (force_recreate=True)", self.store_name)
            self._client.delete_collection(self.store_name)

        self._client.create_collection(
            collection_name=self.store_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{c.chapter}:{c.chunk_index}")),
                vector=vector.tolist(),
                payload={"chapter": c.chapter, "chunk_index": c.chunk_index, "text": c.text},
            )
            for c, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.store_name, points=points)

        meta_vector = [0.0] * self.vector_size
        meta_vector[0] = 1.0  # any valid unit vector; never returned (see _NOT_META)
        self._client.upsert(
            collection_name=self.store_name,
            points=[
                PointStruct(
                    id=_META_POINT_ID,
                    vector=meta_vector,
                    payload={
                        "kind": _META_KIND,
                        "embedding_fingerprint": self.embeddings.embedding_fingerprint,
                        "corpus_sha256": corpus_fingerprint(chunks),
                        "chunk_count": len(chunks),
                    },
                )
            ],
        )
        logger.info("Indexed %d chunks into %s", len(points), self.store_name)

    def retrieve(self, query: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        # Checked on every call, not cached: the indexer may replace the
        # collection while this process keeps running. One point lookup by id;
        # negligible next to embedding the query.
        self._check_index()

        vector = self.embeddings.embed_texts([query])[0].tolist()
        hits = self._client.query_points(
            collection_name=self.store_name,
            query=vector,
            query_filter=_NOT_META,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[tuple[dict[str, Any], float]] = []
        for hit in hits:
            if hit.payload is None:
                raise RuntimeError(f"Qdrant point {hit.id} returned without a payload.")
            results.append((hit.payload, float(hit.score)))

        return results

    def retrieve_chunks(
        self, query: str, chunks: Sequence[Chunk], top_k: int
    ) -> list[RetrievalResult]:
        """
        Return original chunk dicts and score, matched by (chapter, chunk_index).
        """
        hits = self.retrieve(query, top_k=top_k)

        lookup: dict[tuple[int | None, int], dict[str, Any]] = {
            (c.chapter, c.chunk_index): asdict(c) for c in chunks
        }

        results: list[RetrievalResult] = []
        for payload, score in hits:
            chapter = payload.get("chapter")
            chunk_index = payload.get("chunk_index")
            key = (
                int(chapter) if chapter is not None else None,
                int(chunk_index) if chunk_index is not None else -1,
            )

            chunk = lookup.get(key)
            if chunk is None:
                logger.warning(
                    "Retrieved Qdrant point has no matching source chunk "
                    "(collection=%s, chapter=%s, chunk_index=%s)",
                    self.store_name,
                    key[0],
                    key[1],
                )
                chunk = {
                    "chapter": key[0],
                    "chunk_index": key[1],
                    "text": payload.get("text", ""),
                }

            results.append(RetrievalResult(chunk=chunk, score=float(score)))

        return results
