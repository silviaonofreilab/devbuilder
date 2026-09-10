# syntax=docker/dockerfile:1.7
# Multi-stage build for the DevBuilder FastAPI orchestrator.
#
# Stage 1 (builder): install uv and resolve dependencies into a venv.
# Stage 2 (runtime): copy the venv + source into a slim base, run as non-root.
#
# Build and run from the repository root:
#   make api-up

# Base images are pinned by multi-arch index digest so a rebuild months later
# produces the same layers on arm64 (laptop) and amd64 (CI, droplet). To bump:
#   docker buildx imagetools inspect python:3.11-slim-bookworm | grep Digest
#   docker buildx imagetools inspect ghcr.io/astral-sh/uv:<ver> | grep Digest
ARG PYTHON_VERSION=3.11
ARG PYTHON_DIGEST=sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
ARG UV_VERSION=0.12.3
ARG UV_DIGEST=sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc

#### Stage 1: builder ####

FROM ghcr.io/astral-sh/uv:${UV_VERSION}@${UV_DIGEST} AS uv
FROM python:${PYTHON_VERSION}-slim-bookworm@${PYTHON_DIGEST} AS builder
COPY --from=uv /uv /uvx /usr/local/bin/

# Avoid writing .pyc files and buffer logs during the build.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Resolve and install dependencies first for better layer caching.
# Mounts the lockfile and project metadata.
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra api

# Add the project source and install the package itself.
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra api

#### Stage 2: runtime ####

FROM python:${PYTHON_VERSION}-slim-bookworm@${PYTHON_DIGEST} AS runtime

ARG GHCR_OWNER
ARG REPO_NAME=devbuilder

LABEL org.opencontainers.image.source="https://github.com/${GHCR_OWNER}/${REPO_NAME}"
LABEL org.opencontainers.image.description="DevBuilder RAG API — FastAPI + ONNX Runtime + Qdrant client"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.title="${REPO_NAME}-api"
LABEL org.opencontainers.image.documentation="https://github.com/${GHCR_OWNER}/${REPO_NAME}#readme"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Create an unprivileged user. Using a fixed UID/GID makes bind-mounted
# host directories predictable to chown if needed.
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Copy the resolved venv and the source tree from the builder.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --chown=app:app configs ./configs

# Directories that will be bind-mounted at runtime; create them now so the
# container can start even when a mount is read-only.
RUN mkdir -p artifacts/models data/raw data/processed runs \
 && chown -R app:app /app

USER app

EXPOSE 8080

# Liveness probe matches the FastAPI /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status == 200 else 1)"

# Single worker: the encoder and Qdrant client are loaded once in the lifespan
# and shared across requests. Scale horizontally by running more containers.
# uvicorn's access log stays off: devbuilder.observability writes one line per
# request with request id, status, and duration instead.
CMD ["uvicorn", "devbuilder.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--no-access-log"]
