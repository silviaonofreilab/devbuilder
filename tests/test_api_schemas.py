"""Request schema bounds for the API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from devbuilder.api import (
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    MIN_PASSAGE_TOKENS,
    AskRequest,
    RetrieveRequest,
    _check_query_fits_reranker,
)


def test_query_at_limit_accepted() -> None:
    assert len(AskRequest(query="x" * MAX_QUERY_CHARS).query) == MAX_QUERY_CHARS


@pytest.mark.parametrize("model", [AskRequest, RetrieveRequest])
def test_query_over_limit_rejected(model) -> None:
    with pytest.raises(ValidationError):
        model(query="x" * (MAX_QUERY_CHARS + 1))


@pytest.mark.parametrize("model", [AskRequest, RetrieveRequest])
def test_empty_query_rejected(model) -> None:
    with pytest.raises(ValidationError):
        model(query="")


def test_top_k_bounds() -> None:
    assert RetrieveRequest(query="q", top_k=MAX_TOP_K).top_k == MAX_TOP_K
    for bad in (0, MAX_TOP_K + 1):
        with pytest.raises(ValidationError):
            RetrieveRequest(query="q", top_k=bad)


class _Reranker:
    """Counts one token per word; budget 512 like the deployed cross-encoder."""

    max_length = 512

    def query_tokens(self, query: str) -> int:
        return len(query.split())


def test_query_that_leaves_room_for_a_passage_passes() -> None:
    state = SimpleNamespace(reranker=_Reranker())
    _check_query_fits_reranker(state, "word " * (512 - MIN_PASSAGE_TOKENS))


def test_query_that_starves_the_passage_is_rejected() -> None:
    state = SimpleNamespace(reranker=_Reranker())
    with pytest.raises(HTTPException) as exc:
        _check_query_fits_reranker(state, "word " * (512 - MIN_PASSAGE_TOKENS + 1))
    assert exc.value.status_code == 422
