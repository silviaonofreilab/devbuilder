# syntax=docker/dockerfile:1.7
# Multi-stage build for the DevBuilder Gradio UI.
#
# Replaceable client that calls the FastAPI /ask endpoint.
#
# Build and run from the repository root:
#   make ui-up

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

# Install only the `ui` dependency group: the UI never imports the ML runtime,
# so the project itself and its core dependencies stay out of this image.
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group ui

#### Stage 2: runtime ####

FROM python:${PYTHON_VERSION}-slim-bookworm@${PYTHON_DIGEST} AS runtime

ARG GHCR_OWNER
ARG REPO_NAME=devbuilder

LABEL org.opencontainers.image.source="https://github.com/${GHCR_OWNER}/${REPO_NAME}"
LABEL org.opencontainers.image.description="DevBuilder Gradio UI — lightweight client for the RAG API"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.title="${REPO_NAME}-ui"
LABEL org.opencontainers.image.documentation="https://github.com/${GHCR_OWNER}/${REPO_NAME}#readme"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Create an unprivileged user. Using a fixed UID/GID makes bind-mounted
# host directories predictable to chown if needed.
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Copy the resolved venv and the UI module. The package is not installed
# (see builder stage), so the source tree is put on PYTHONPATH directly.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/devbuilder/__init__.py /app/src/devbuilder/__init__.py
COPY --chown=app:app src/devbuilder/ui /app/src/devbuilder/ui
ENV PYTHONPATH=/app/src

USER app

EXPOSE 7860

CMD ["python", "-m", "devbuilder.ui.gradio_app"]
