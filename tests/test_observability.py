"""Request ids and access lines from devbuilder.observability."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from devbuilder.observability import (
    REQUEST_ID_HEADER,
    RequestIdFilter,
    current_request_id,
    install_request_context,
)


@pytest.fixture
def client(caplog) -> TestClient:
    caplog.handler.addFilter(RequestIdFilter())
    app = FastAPI()
    install_request_context(app)

    @app.get("/echo")
    async def echo() -> dict[str, str]:
        logging.getLogger("devbuilder.test").info("inside handler")
        return {"request_id": current_request_id()}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


def test_generated_id_is_echoed_and_visible_to_handler(client: TestClient) -> None:
    resp = client.get("/echo")
    request_id = resp.headers[REQUEST_ID_HEADER]
    assert len(request_id) == 12
    assert resp.json()["request_id"] == request_id


def test_caller_supplied_id_is_kept(client: TestClient) -> None:
    resp = client.get("/echo", headers={REQUEST_ID_HEADER: "trace-42"})
    assert resp.headers[REQUEST_ID_HEADER] == "trace-42"
    assert resp.json()["request_id"] == "trace-42"


def test_id_does_not_leak_between_requests(client: TestClient) -> None:
    client.get("/echo", headers={REQUEST_ID_HEADER: "first"})
    assert client.get("/echo").json()["request_id"] != "first"
    assert current_request_id() == "-"


def test_access_line_has_method_path_status_duration(client: TestClient, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="devbuilder.access"):
        client.get("/echo", headers={REQUEST_ID_HEADER: "trace-42"})
    record = next(r for r in caplog.records if r.name == "devbuilder.access")
    assert record.getMessage().startswith("GET /echo -> 200 (")
    assert record.getMessage().endswith(" ms)")
    assert record.request_id == "trace-42"


def test_health_is_logged_at_debug_only(client: TestClient, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="devbuilder.access"):
        client.get("/health")
    assert not [r for r in caplog.records if r.name == "devbuilder.access"]
