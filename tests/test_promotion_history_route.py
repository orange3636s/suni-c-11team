"""Route-ordering regression for A-4: GET /api/models/promotion-history
must not be shadowed by GET /api/models/{model_id}.

FastAPI/Starlette match routes in registration order, so this can only be
caught by actually resolving a request through the router -- calling
`get_promotion_history()` directly (bypassing routing) would pass even
with the routes in the wrong order. Uses the same raw-ASGI-request
convention as test_model_deletion.py's `_asgi_request`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.main import app
from api.routes import data as data_routes


def _asgi_request(method: str, path: str) -> tuple[int, bytes]:
    messages: list[dict[str, object]] = []
    request_sent = False

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    status_code = next(
        int(message["status"]) for message in messages if message["type"] == "http.response.start"
    )
    body = b"".join(bytes(message.get("body", b"")) for message in messages if message["type"] == "http.response.body")
    return status_code, body


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime_db = tmp_path / "runtime" / f"dashboard_{uuid4().hex}.db"
    artifact_root = tmp_path / "runtime"
    test_settings = SimpleNamespace(runtime_db_path=runtime_db, runtime_artifact_dir=artifact_root)
    monkeypatch.setattr(data_routes, "settings", test_settings)
    return test_settings


def test_promotion_history_route_is_not_shadowed_by_model_detail(isolated_settings: SimpleNamespace) -> None:
    status_code, body = _asgi_request("GET", "/api/models/promotion-history")
    assert status_code == 200, body
    assert b'"items"' in body
