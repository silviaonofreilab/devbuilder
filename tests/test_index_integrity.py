"""
Index integrity: completion marker, encoder identity, and non-destructive
rebuilds. Hermetic: a fake encoder and an in-memory Qdrant.
"""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from devbuilder.pipelines.preprocessing import Chunk
from devbuilder.vectorstore import IndexIntegrityError, QdrantVectorStore

DIM = 4
CHUNKS = [
    Chunk(chapter=1, chunk_index=0, text="Alice fell down the rabbit hole."),
    Chunk(chapter=1, chunk_index=1, text="The Cheshire Cat grinned."),
    Chunk(chapter=2, chunk_index=0, text="The Hatter poured tea."),
]


class FakeEncoder:
    """Deterministic embeddings; identity is whatever fingerprint the test chooses."""

    def __init__(self, sha: str, fail: bool = False) -> None:
        self.embedding_fingerprint = sha
        self.embedding_dim = DIM
        self.fail = fail

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self.fail:
            raise RuntimeError("encoder exploded")
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % DIM] = 1.0
        return out


def make_store(client: QdrantClient, sha: str = "sha-A", fail: bool = False) -> QdrantVectorStore:
    return QdrantVectorStore(
        embeddings=FakeEncoder(sha, fail=fail),
        store_name="integrity",
        host="unused",
        port=0,
        client=client,
    )


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_ready_only_after_a_complete_build(client: QdrantClient) -> None:
    store = make_store(client)
    assert store.is_ready() is False

    store.build_index(CHUNKS)
    assert store.is_ready() is True


def test_interrupted_upload_is_not_ready(client: QdrantClient, monkeypatch) -> None:
    store = make_store(client)
    real_upsert = client.upsert
    calls = {"n": 0}

    def upsert_then_fail(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # chunk upload succeeds, the completion marker never lands
            return real_upsert(*args, **kwargs)
        raise RuntimeError("connection lost")

    monkeypatch.setattr(client, "upsert", upsert_then_fail)
    with pytest.raises(RuntimeError):
        store.build_index(CHUNKS)

    assert client.collection_exists("integrity")  # points are there...
    assert store.is_ready() is False  # ...but the index is not complete
    with pytest.raises(IndexIntegrityError):
        store.retrieve("tea", top_k=1)


def test_failed_encoding_keeps_the_old_index(client: QdrantClient) -> None:
    make_store(client).build_index(CHUNKS)

    broken = make_store(client, fail=True)
    with pytest.raises(RuntimeError):
        broken.build_index(CHUNKS, force_recreate=True)

    survivor = make_store(client)
    assert survivor.is_ready() is True
    assert survivor.retrieve("tea", top_k=1)


def test_different_encoder_is_refused_until_rebuilt(client: QdrantClient) -> None:
    make_store(client, sha="sha-A").build_index(CHUNKS)

    other = make_store(client, sha="sha-B")  # same dimension, different model
    assert other.is_ready() is False
    with pytest.raises(IndexIntegrityError, match="different encoder"):
        other.retrieve("tea", top_k=1)
    with pytest.raises(IndexIntegrityError):
        other.build_index(CHUNKS)  # idempotent path must not accept a foreign index

    other.build_index(CHUNKS, force_recreate=True)
    assert other.is_ready() is True


def test_meta_point_is_never_retrieved(client: QdrantClient) -> None:
    store = make_store(client)
    store.build_index(CHUNKS)

    hits = store.retrieve("anything", top_k=50)
    assert len(hits) == len(CHUNKS)
    assert all("text" in payload for payload, _ in hits)


def test_reindex_by_another_encoder_is_noticed_by_a_live_store(client: QdrantClient) -> None:
    live = make_store(client, sha="sha-A")
    live.build_index(CHUNKS)
    assert live.retrieve("tea", top_k=1)  # store is warm and serving

    make_store(client, sha="sha-B").build_index(CHUNKS, force_recreate=True)  # the indexer ran

    assert live.is_ready() is False
    with pytest.raises(IndexIntegrityError, match="different encoder"):
        live.retrieve("tea", top_k=1)  # no stale cache lets this through


def test_changed_corpus_is_refused_on_the_default_path(client: QdrantClient) -> None:
    store = make_store(client)
    store.build_index(CHUNKS)

    edited = CHUNKS[:-1] + [Chunk(chapter=2, chunk_index=0, text="The Hatter poured coffee.")]
    with pytest.raises(IndexIntegrityError, match="different corpus"):
        store.build_index(edited)  # same encoder, different text: must not silently skip

    store.build_index(edited, force_recreate=True)
    store.build_index(edited)  # now idempotent again
