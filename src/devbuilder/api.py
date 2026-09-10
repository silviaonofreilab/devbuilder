"""FastAPI orchestrator for DevBuilder RAG.

Wires retrieval (Qdrant + encoder and reranker) and generation (vLLM on RunPod)
behind HTTP endpoints. Run locally with `make api-up`.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import devbuilder.config as cfg
from devbuilder.decoder import DecoderBackend, VLLMConfigError, VLLMContextLengthError, VLLMDecoder
from devbuilder.encoder import OrtEncoder
from devbuilder.observability import configure_logging, install_request_context
from devbuilder.paths import paths
from devbuilder.rag import RAG
from devbuilder.reranker import OrtReranker
from devbuilder.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


# Schemas

# Request bounds. A query is one question, not a document. Characters are a
# first, cheap gate; the reranker then counts the query with its own
# tokenizer and rejects it if too little of the 512-token pair budget would
# be left for a passage (reranker truncation drops passage text, never the
# query). top_k caps a single Qdrant read.
MAX_QUERY_CHARS = 1000
MAX_TOP_K = 50

# Minimum share of the reranker pair budget that must remain for the passage.
MIN_PASSAGE_TOKENS = 128

# Prompt budget. The decoder's tokenizer is not in this image, so the
# assembled prompt is bounded in characters (decoder.max_prompt_chars). This
# is a heuristic (~4 chars per token; punctuation-heavy text tokenizes worse)
# and it excludes the system message and chat-template overhead. vLLM's own
# context check is authoritative; when it fires, the API reports it as a 422
# rather than a 500. runpod.max_model_len is sized with margin over
# max_prompt_chars / 4 + decoder.max_tokens so that the heuristic usually
# fires first, not so that it always does.


class Source(BaseModel):
    chapter: int | None = None
    chunk_index: int | None = None
    score: float
    text: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class RetrieveResponse(BaseModel):
    sources: list[Source]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    qdrant_ok: bool
    llm_ok: bool
    ready: bool


# Auth

_bearer = HTTPBearer(auto_error=False)


async def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """
    Reject requests without a valid bearer token.
    """
    expected = request.app.state.api_token
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), expected.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Lifespan builds


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Construct all objects once; tear down on shutdown.
    """
    token = os.environ.get("API_TOKEN")
    if not token:
        raise RuntimeError("API_TOKEN is not set")
    app.state.api_token = token

    settings = cfg.load_config(paths.configs / "settings.yaml")

    enc_cfg = settings["encoder"]
    encoder = OrtEncoder(
        models_dir=paths.models,
        model_name=enc_cfg["model_name"],
        batch_size=enc_cfg["batch_size"],
    )

    rr_cfg = settings["reranker"]
    reranker = OrtReranker(
        models_dir=paths.models,
        model_name=rr_cfg["model_name"],
        batch_size=rr_cfg["batch_size"],
    )

    vs_cfg = settings["vectorstore"]
    store = QdrantVectorStore(
        embeddings=encoder,
        store_name=vs_cfg["name"],
        host=vs_cfg["host"],
        port=vs_cfg["port"],
    )

    rag = RAG(
        vectorstore=store,
        template=settings["prompts"]["rag_template"],
        top_k=vs_cfg["top_k_default"],
        reranker=reranker,
        fetch_k=rr_cfg["fetch_k"],
    )

    # The decoder is optional at startup: retrieval must work without a GPU
    # pod. /ready reports llm_ok=false and /ask returns 503 until it is set.
    llm: DecoderBackend | None
    try:
        llm = VLLMDecoder.from_settings(
            settings["decoder"],
            system_prompt=settings["prompts"]["system"],
        )
    except VLLMConfigError as exc:
        logger.warning("Decoder disabled: %s", exc)
        llm = None

    app.state.store = store
    app.state.rag = rag
    app.state.reranker = reranker
    app.state.llm = llm
    app.state.max_prompt_chars = int(settings["decoder"]["max_prompt_chars"])

    logger.info(
        "API ready (collection=%s, model=%s)",
        store.store_name,
        llm.model if llm is not None else "none",
    )
    yield
    logger.info("API shutting down")


configure_logging()
app = FastAPI(title="DevBuilder RAG", lifespan=lifespan)
install_request_context(app)


# Helpers


def _to_sources(results: list[tuple[dict[str, Any], float]]) -> list[Source]:
    """Convert (payload, score) tuples to Source models."""
    return [
        Source(
            chapter=p.get("chapter"),
            chunk_index=p.get("chunk_index"),
            score=float(s),
            text=p.get("text", ""),
        )
        for p, s in results
    ]


def _check_query_fits_reranker(state: Any, query: str) -> None:
    """
    422 if the query leaves less than MIN_PASSAGE_TOKENS of the reranker's pair
    budget. Counted with the reranker's tokenizer, so it is exact for it.
    """
    used = state.reranker.query_tokens(query)
    room = state.reranker.max_length - used
    if room < MIN_PASSAGE_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Query is {used} reranker tokens; at most "
                f"{state.reranker.max_length - MIN_PASSAGE_TOKENS} are allowed so a passage "
                "still fits. Shorten the question."
            ),
        )


# Endpoints


@app.get("/health")
async def health() -> HealthResponse:
    """
    Liveness probe: API process is up.
    """
    return HealthResponse(status="ok")


@app.get("/ready", dependencies=[Depends(require_token)])
async def ready(request: Request) -> ReadyResponse:
    """
    Readiness probe: dependencies (Qdrant, vLLM) are reachable.
    """
    state = request.app.state
    qdrant_ok = await run_in_threadpool(state.store.is_ready)
    llm_ok = state.llm is not None and await run_in_threadpool(state.llm.is_ready)
    return ReadyResponse(qdrant_ok=qdrant_ok, llm_ok=llm_ok, ready=qdrant_ok and llm_ok)


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_token)])
async def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
    """
    Retrieve top-k chunks for a query. No generation.
    """
    state = request.app.state
    top_k = req.top_k or state.rag.top_k
    try:
        results = await run_in_threadpool(state.store.retrieve, req.query, top_k)
    except Exception:
        logger.exception("Retrieval failed for query=%r", req.query[:100])
        raise HTTPException(status_code=500, detail="Retrieval failed") from None
    return RetrieveResponse(sources=_to_sources(results))


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_token)])
async def ask(req: AskRequest, request: Request) -> AskResponse:
    """
    Full RAG: retrieve candidates, rerank, then generate an answer. The number
    of contexts is fixed by config, since the reranker's candidate pool
    (fetch_k) is sized for it.
    """
    state = request.app.state
    if state.llm is None:
        raise HTTPException(status_code=503, detail="Generation backend unavailable")
    _check_query_fits_reranker(state, req.query)
    try:
        results = await run_in_threadpool(state.rag.retrieve_with_scores, req.query)
        contexts = [p.get("text", "") for p, _ in results]
        prompt = state.rag.enhance_user_prompt(req.query, contexts=contexts)
    except Exception:
        logger.exception("Retrieval failed for query=%r", req.query[:100])
        raise HTTPException(status_code=500, detail="Ask failed") from None

    if len(prompt) > state.max_prompt_chars:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Assembled prompt is {len(prompt)} characters; the decoder budget is "
                f"{state.max_prompt_chars}. Shorten the question."
            ),
        )

    try:
        answer = await run_in_threadpool(state.llm.generate, prompt)
    except VLLMContextLengthError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt exceeds the decoder's context window; shorten the question. ({exc})",
        ) from None
    except Exception:
        logger.exception("Generation failed for query=%r", req.query[:100])
        raise HTTPException(status_code=500, detail="Ask failed") from None

    return AskResponse(answer=answer, sources=_to_sources(results))
