from __future__ import annotations

import logging
from typing import Any

from devbuilder.reranker import RerankerBackend
from devbuilder.vectorstore import VectorStoreBackend

logger = logging.getLogger(__name__)


class RAG:
    """
    Retrieves chunks and assembles RAG-enhanced user prompts.
    """

    def __init__(
        self,
        vectorstore: VectorStoreBackend,
        template: str,
        top_k: int,
        reranker: RerankerBackend | None = None,
        fetch_k: int | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if reranker is not None:
            if fetch_k is None:
                raise ValueError("fetch_k is required when a reranker is provided")
            if fetch_k < top_k:
                raise ValueError("fetch_k must be greater than or equal to top_k")

        self.vectorstore = vectorstore
        self.template = template
        self.top_k = top_k
        self.reranker = reranker
        # Without a reranker there is no candidate pool to narrow: fetch exactly top_k.
        self.fetch_k = fetch_k if reranker is not None else top_k

    @staticmethod
    def _filter_text_results(
        results: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Remove results whose payload has no usable text.
        """
        return [
            (payload, score)
            for payload, score in results
            if isinstance(payload.get("text"), str) and payload["text"].strip()
        ]

    def retrieve_with_scores(self, query: str) -> list[tuple[dict[str, Any], float]]:
        """
        Return up to top-k (payload, score) pairs.

        Without reranking, scores are Qdrant similarity scores. With reranking,
        scores are cross-encoder relevance logits. Scores from the two retrieval
        modes are not directly comparable.
        """
        results = self.vectorstore.retrieve(query, top_k=self.fetch_k)
        usable_results = self._filter_text_results(results)

        dropped = len(results) - len(usable_results)
        if dropped:
            logger.warning(
                "Dropped %d retrieved payloads without usable text",
                dropped,
            )

        if self.reranker is None or not usable_results:
            final_results = usable_results
        else:
            texts = [payload["text"] for payload, _ in usable_results]
            ranked = self.reranker.rank(query, texts, top_k=self.top_k)
            final_results = [(usable_results[index][0], score) for index, score in ranked]

        logger.debug(
            "RAG retrieval: retrieved=%d usable=%d returned=%d reranked=%s",
            len(results),
            len(usable_results),
            len(final_results),
            self.reranker is not None,
        )

        return final_results

    def retrieve(self, query: str) -> list[str]:
        """
        Return the top-k usable chunk texts for `query`.
        """
        return [payload["text"] for payload, _ in self.retrieve_with_scores(query)]

    def enhance_user_prompt(self, query: str, contexts: list[str] | None = None) -> str:
        """
        Build the model's user message from retrieved context and the query.

        Pass `contexts` to reuse previously retrieved texts and avoid performing
        retrieval twice.
        """
        if contexts is None:
            contexts = self.retrieve(query)
        context = "\n\n".join(contexts)
        return self.template.format(context=context, query=query)
