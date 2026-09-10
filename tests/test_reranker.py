"""ONNX reranker scoring and ranking behavior."""

from __future__ import annotations

import numpy as np
import pytest

import devbuilder.config as cfg
from devbuilder.paths import paths
from devbuilder.reranker import OrtReranker

pytestmark = pytest.mark.artifact("reranker")


@pytest.fixture(scope="module")
def settings():
    return cfg.load_config(paths.configs / "settings.yaml")


@pytest.fixture(scope="module")
def reranker(settings) -> OrtReranker:
    rr_cfg = settings["reranker"]
    return OrtReranker(
        models_dir=paths.models,
        model_name=rr_cfg["model_name"],
        batch_size=2,
    )


def test_score_shape_and_finite(reranker: OrtReranker):
    scores = reranker.score(
        "who is alice",
        ["Alice fell down the hole.", "Tea was served."],
    )

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_score_relevance_ordering(reranker: OrtReranker):
    scores = reranker.score(
        "who fell down the rabbit hole",
        ["Alice fell down the rabbit hole.", "The weather in Paris is mild."],
    )

    assert scores[0] > scores[1]


def test_score_empty_passages(reranker: OrtReranker):
    scores = reranker.score("query", [])

    assert scores.shape == (0,)


def test_rank_orders_and_slices(reranker: OrtReranker):
    passages = [
        "The weather in Paris is mild.",
        "Alice fell down the rabbit hole.",
        "Stock markets closed higher.",
    ]

    ranked = reranker.rank(
        "who fell down the rabbit hole",
        passages,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0][0] == 1

    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_topk_exceeds_passages(reranker: OrtReranker):
    ranked = reranker.rank("q", ["a", "b"], top_k=5)

    assert sorted(index for index, _ in ranked) == [0, 1]


def test_query_tokens_counts_with_the_model_tokenizer(reranker: OrtReranker):
    short = reranker.query_tokens("Who is the Hatter?")
    longer = reranker.query_tokens("Who is the Hatter and why does he keep a watch?")
    assert 3 < short < longer <= reranker.max_length
