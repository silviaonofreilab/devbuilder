"""
Request-scoped logging for the API.

Every request gets an id (taken from ``X-Request-ID`` if the caller sent one,
generated otherwise), carried in a context variable so any log line emitted
while handling the request is tagged with it, echoed back in the response,
and summarised in one access line with status and duration. This replaces
uvicorn's access log, which has no id and no timing.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request

logger = logging.getLogger("devbuilder.access")

REQUEST_ID_HEADER = "X-Request-ID"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"

# Paths polled by health checks; logged at DEBUG so they do not drown real traffic.
QUIET_PATHS = frozenset({"/health"})

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def current_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """
    Root logging with the request id in the format. Idempotent: uvicorn and
    tests may import the app more than once.
    """
    root = logging.getLogger()
    if not any(isinstance(f, RequestIdFilter) for h in root.handlers for f in h.filters):
        logging.basicConfig(level=level, format=LOG_FORMAT)
        for handler in root.handlers:
            handler.addFilter(RequestIdFilter())
    root.setLevel(level)
    # One line per outbound HTTP call is noise next to the access line.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def install_request_context(app: FastAPI) -> None:
    """Register the request-id + access-log middleware on ``app``."""

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        token = _request_id.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            level = logging.DEBUG if request.url.path in QUIET_PATHS else logging.INFO
            logger.log(
                level,
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            return response
        finally:
            _request_id.reset(token)
