"""Qdrant indexing and retrieval over a small in-memory store."""

import pytest
from qdrant_client import QdrantClient

import devbuilder.config as cfg
from devbuilder.encoder import OrtEncoder
from devbuilder.paths import paths
from devbuilder.pipelines.preprocessing import Chunk
from devbuilder.vectorstore import QdrantVectorStore

pytestmark = pytest.mark.artifact("encoder")

SAMPLE_CHUNKS = [
    Chunk(
        chapter=1,
        chunk_index=0,
        text="Alice fell down the rabbit hole and found herself in Wonderland.",
    ),
    Chunk(
        chapter=1,
        chunk_index=1,
        text="The Cheshire Cat grinned and slowly disappeared into the tree.",
    ),
    Chunk(
        chapter=2,
        chunk_index=0,
        text="The Mad Hatter poured tea and spoke in riddles at the endless party.",
    ),
]


@pytest.fixture(scope="module")
def settings():
    return cfg.load_config(paths.configs / "settings.yaml")


@pytest.fixture(scope="module")
def encoder(settings) -> OrtEncoder:
    enc_cfg = settings["encoder"]
    return OrtEncoder(
        models_dir=paths.models,
        model_name=enc_cfg["model_name"],
        batch_size=2,
    )


@pytest.fixture(scope="module")
def store(settings, encoder: OrtEncoder) -> QdrantVectorStore:
    s = QdrantVectorStore(
        embeddings=encoder,
        store_name="test",
        host="unused",
        port=0,
        client=QdrantClient(":memory:"),
    )
    s.build_index(SAMPLE_CHUNKS)
    return s


def test_top_result_matches_query(store):
    hits = store.retrieve("Cheshire Cat", top_k=1)
    assert "Cheshire" in hits[0][0]["text"]


def test_scores_are_cosine_range(store):
    hits = store.retrieve("tea party", top_k=3)
    assert all(-1.0 <= h[1] <= 1.0 for h in hits)


def test_retrieval_is_deterministic(store):
    hits_1 = store.retrieve("Cheshire Cat", top_k=2)
    hits_2 = store.retrieve("Cheshire Cat", top_k=2)

    ids_1 = [(h[0]["chapter"], h[0]["chunk_index"]) for h in hits_1]
    ids_2 = [(h[0]["chapter"], h[0]["chunk_index"]) for h in hits_2]

    assert ids_1 == ids_2


def test_retrieve_chunks_returns_original_chunks(store):
    results = store.retrieve_chunks("Cheshire Cat", chunks=SAMPLE_CHUNKS, top_k=2)
    assert len(results) == 2
    assert all("text" in r.chunk for r in results)
    assert isinstance(results[0].score, float)


def test_build_index_smoke(store):
    hits = store.retrieve("rabbit hole", top_k=1)
    assert len(hits) == 1
    assert isinstance(hits[0][1], float)
    assert {"chapter", "chunk_index", "text"} <= hits[0][0].keys()
