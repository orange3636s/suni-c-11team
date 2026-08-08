"""D-3: POST /api/notify/slack/test must be able to test an already
-connected channel without the caller re-supplying the webhook URL --
the settings summary only ever exposes a masked value
(`webhook_masked`), so the frontend has no way to resend the real one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.routes import notify as notify_routes
from api.schemas.notify import SlackTestRequest
from src.notifications import settings_store


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime_db = tmp_path / "runtime" / f"dashboard_{uuid4().hex}.db"
    artifact_root = tmp_path / "runtime"
    test_settings = SimpleNamespace(runtime_db_path=runtime_db, runtime_artifact_dir=artifact_root)
    monkeypatch.setattr(notify_routes, "settings", test_settings)
    return test_settings


def test_test_slack_without_webhook_uses_connected_channel(
    isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = notify_routes._store()
    settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#eng-yield")

    sent_urls = []
    monkeypatch.setattr(
        notify_routes.senders, "send_slack_test", lambda url: (sent_urls.append(url), (True, None))[1]
    )

    result = notify_routes.test_slack(SlackTestRequest(webhook_url=None))

    assert result == {"ok": True, "error": None}
    assert sent_urls == ["https://hooks.slack.com/services/FAKE/FAKE/FAKE"]


def test_test_slack_without_webhook_and_not_connected_fails_cleanly(isolated_settings: SimpleNamespace) -> None:
    result = notify_routes.test_slack(SlackTestRequest(webhook_url=None))
    assert result["ok"] is False
    assert "연결된" in result["error"]


def test_test_slack_still_accepts_explicit_webhook_for_preconnect_flow(
    isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disconnected form's "test before connecting" flow must keep working."""
    sent_urls = []
    monkeypatch.setattr(
        notify_routes.senders, "send_slack_test", lambda url: (sent_urls.append(url), (True, None))[1]
    )
    result = notify_routes.test_slack(SlackTestRequest(webhook_url="https://hooks.slack.com/services/NEW/NEW/NEW"))
    assert result == {"ok": True, "error": None}
    assert sent_urls == ["https://hooks.slack.com/services/NEW/NEW/NEW"]
