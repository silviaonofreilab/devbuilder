"""Unit tests for RAG. Mocks the vectorstore—no live Qdrant required."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from devbuilder.rag import RAG

TEMPLATE = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def _fake_store(payloads_with_scores: list[tuple[dict, float]]) -> SimpleNamespace:
    """
    Build a stand-in for QdrantVectorStore that records its call args.
    """
    state: dict = {}

    def retrieve(query: str, top_k: int):
        state["query"] = query
        state["top_k"] = top_k
        return payloads_with_scores[:top_k]

    return SimpleNamespace(retrieve=retrieve, _state=state)


def test_retrieve_passes_query_and_topk_to_vectorstore():
    """
    RAG.retrieve forwards query and top_k unchanged.
    """
    store = _fake_store(
        [
            ({"text": "doc1"}, 0.9),
            ({"text": "doc2"}, 0.8),
            ({"text": "doc3"}, 0.7),
        ]
    )
    rag = RAG(vectorstore=store, template=TEMPLATE, top_k=2)
    rag.retrieve("my question")

    assert store._state["query"] == "my question"
    assert store._state["top_k"] == 2


def test_retrieve_extracts_text_from_payload():
    """
    RAG.retrieve returns chunk texts, dropping scores and other metadata.
    """
    store = _fake_store(
        [
            ({"text": "doc1", "chapter": 1}, 0.9),
            ({"text": "doc2", "chapter": 2}, 0.8),
        ]
    )
    rag = RAG(vectorstore=store, template=TEMPLATE, top_k=2)
    assert rag.retrieve("q") == ["doc1", "doc2"]


def test_retrieve_filters_unusable_text():
    """
    Payloads without nonblank string text are excluded.
    """
    store = _fake_store(
        [
            ({"chapter": 1}, 0.9),
            ({"text": ""}, 0.8),
            ({"text": "   "}, 0.7),
            ({"text": 123}, 0.6),
            ({"text": "doc1"}, 0.5),
        ]
    )
    rag = RAG(vectorstore=store, template=TEMPLATE, top_k=5)

    assert rag.retrieve("q") == ["doc1"]


def test_enhance_user_prompt_injects_context_and_query():
    """
    The template's {context} and {query} placeholders are populated.
    """
    store = _fake_store(
        [
            ({"text": "Alice met the hatter."}, 0.9),
            ({"text": "Tea time was endless."}, 0.8),
        ]
    )
    rag = RAG(vectorstore=store, template=TEMPLATE, top_k=2)
    prompt = rag.enhance_user_prompt("Where is the hatter?")

    assert "Alice met the hatter." in prompt
    assert "Tea time was endless." in prompt
    assert "Where is the hatter?" in prompt
    assert prompt.startswith("Context:")


def test_enhance_user_prompt_reuses_contexts():
    """
    Supplied contexts are rendered without performing retrieval.
    """
    store = _fake_store([])
    rag = RAG(vectorstore=store, template=TEMPLATE, top_k=2)

    prompt = rag.enhance_user_prompt(
        "Where is the hatter?",
        contexts=[
            "Alice met the hatter.",
            "Tea time was endless.",
        ],
    )

    assert "Alice met the hatter." in prompt
    assert "Tea time was endless." in prompt
    assert "Where is the hatter?" in prompt
    assert prompt.startswith("Context:")
    assert store._state == {}


def _fake_reranker(order: list[tuple[int, float]]) -> SimpleNamespace:
    """
    Stand-in reranker returning a fixed ranking, recording call args.
    """
    state: dict = {}

    def rank(query: str, passages: list[str], top_k: int):
        state["query"] = query
        state["passages"] = passages
        state["top_k"] = top_k
        return order[:top_k]

    return SimpleNamespace(rank=rank, _state=state)


def test_retrieve_with_reranker_fetches_fetch_k_and_reorders():
    """
    RAG fetches fetch_k candidates and returns reranker scores and order.
    """
    store = _fake_store(
        [
            ({"text": "doc1"}, 0.9),
            ({"text": "doc2"}, 0.8),
            ({"text": "doc3"}, 0.7),
        ]
    )
    reranker = _fake_reranker(order=[(2, 0.9), (0, 0.5)])
    rag = RAG(
        vectorstore=store,
        template=TEMPLATE,
        top_k=2,
        reranker=reranker,
        fetch_k=3,
    )

    results = rag.retrieve_with_scores("q")

    assert results == [
        ({"text": "doc3"}, 0.9),
        ({"text": "doc1"}, 0.5),
    ]
    assert store._state["top_k"] == 3
    assert reranker._state["top_k"] == 2
    assert reranker._state["passages"] == ["doc1", "doc2", "doc3"]


@pytest.mark.parametrize(
    ("top_k", "reranker", "fetch_k", "message"),
    [
        (0, None, None, "top_k must be at least 1"),
        (2, SimpleNamespace(), None, "fetch_k is required"),
        (2, SimpleNamespace(), 1, "fetch_k must be greater than or equal to top_k"),
    ],
)
def test_constructor_validation(
    top_k: int,
    reranker: SimpleNamespace | None,
    fetch_k: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RAG(
            vectorstore=_fake_store([]),
            template=TEMPLATE,
            top_k=top_k,
            reranker=reranker,
            fetch_k=fetch_k,
        )
