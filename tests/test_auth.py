"""Tests for bearer-token auth."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from devbuilder.api import app as real_app
from devbuilder.api import require_token

TOKEN = "test-token"
PUBLIC_PATHS = {"/health", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.state.api_token = TOKEN

    @app.get("/protected", dependencies=[Depends(require_token)])
    async def protected() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_valid_token_accepted(client: TestClient) -> None:
    resp = client.get("/protected", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200


def test_missing_token_rejected(client: TestClient) -> None:
    resp = client.get("/protected")
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_wrong_token_rejected(client: TestClient) -> None:
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_non_ascii_token_rejected(client: TestClient) -> None:
    # Bytes: httpx rejects non-ASCII header strings before they reach the server.
    resp = client.get("/protected", headers={"Authorization": b"Bearer token"})
    assert resp.status_code == 401


@pytest.mark.parametrize("header", ["Basic abc", TOKEN, "Bearer", "Bearer "])
def test_malformed_header_rejected(client: TestClient, header: str) -> None:
    resp = client.get("/protected", headers={"Authorization": header})
    assert resp.status_code == 401


def test_scheme_is_case_insensitive(client: TestClient) -> None:
    resp = client.get("/protected", headers={"Authorization": f"bearer {TOKEN}"})
    assert resp.status_code == 200


def test_health_is_public() -> None:
    resp = TestClient(real_app).get("/health")
    assert resp.status_code == 200


def test_all_routes_require_token() -> None:
    for route in real_app.routes:
        if not isinstance(route, APIRoute) or route.path in PUBLIC_PATHS:
            continue
        calls = [dep.call for dep in route.dependant.dependencies]
        assert require_token in calls, f"{route.path} is unauthenticated"


def test_missing_api_token_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="API_TOKEN"), TestClient(real_app):
        pass
