"""Tests for src/notifications/settings_store.py -- persistence (never
localStorage, spec §D-1), masking (spec §D-2), and the settings summary
shape consumed by both GET /api/state/latest and the settings panel.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.notifications import settings_store
from src.runtime.store import RuntimeStore


def _store() -> tuple[RuntimeStore, Path]:
    root = Path(__file__).parent / ".tmp_runtime"
    root.mkdir(exist_ok=True)
    path = root / f"dashboard_{uuid4().hex}.db"
    return RuntimeStore(path), path


def _cleanup(path: Path) -> None:
    path.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def test_mask_slack_webhook_hides_middle_and_most_of_last_segment():
    # Deliberately NOT shaped like a real Slack webhook (T{8-10}/B{8-10}/{24
    # alnum}) -- an earlier version of this test used realistic-length dummy
    # segments and GitHub's secret scanner flagged it as a plausible Slack
    # webhook on push, even though every character was a placeholder. The
    # masking logic under test only looks at the last path segment, so the
    # exact shape of the other segments doesn't matter for coverage.
    masked = settings_store.mask_slack_webhook("https://hooks.slack.com/services/FAKE-TEAM-ID/FAKE-BOT-ID/xxxTESTPLACEHOLDERVALUE")
    assert masked == "hooks.slack.com/…/xxx****"
    assert "FAKE-TEAM-ID" not in masked
    assert "FAKE-BOT-ID" not in masked


def test_mask_chat_id_keeps_only_first_four_digits():
    assert settings_store.mask_chat_id("123456789") == "1234****"


def test_is_valid_slack_webhook_url_checks_domain():
    assert settings_store.is_valid_slack_webhook_url("https://hooks.slack.com/services/abc")
    assert not settings_store.is_valid_slack_webhook_url("https://evil.example.com/services/abc")
    assert not settings_store.is_valid_slack_webhook_url("http://hooks.slack.com/services/abc")  # not https


def test_slack_roundtrip_and_summary():
    store, path = _store()
    try:
        assert settings_store.get_settings_summary(store)["slack"]["connected"] is False
        settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE-TEAM/FAKE-BOT/xxxsecretplaceholder", channel="#eng-yield")
        summary = settings_store.get_settings_summary(store)
        assert summary["slack"]["connected"] is True
        assert summary["slack"]["target"] == "#eng-yield"
        assert "secret" not in summary["slack"]["webhook_masked"]
        assert summary["slack"]["webhook_masked"].startswith("hooks.slack.com/")
    finally:
        _cleanup(path)


def test_gmail_stays_pending_until_verified():
    store, path = _store()
    try:
        token = settings_store.start_gmail_verification(store, email="user@example.com")
        summary = settings_store.get_settings_summary(store)
        assert summary["gmail"]["connected"] is False
        assert summary["gmail"]["pending"] is True

        ok = settings_store.verify_gmail(store, token=token)
        assert ok is True
        summary = settings_store.get_settings_summary(store)
        assert summary["gmail"]["connected"] is True
        assert summary["gmail"]["pending"] is False
    finally:
        _cleanup(path)


def test_gmail_verify_rejects_wrong_token():
    store, path = _store()
    try:
        settings_store.start_gmail_verification(store, email="user@example.com")
        assert settings_store.verify_gmail(store, token="wrong-token") is False
        assert settings_store.get_settings_summary(store)["gmail"]["connected"] is False
    finally:
        _cleanup(path)


def test_gmail_verify_is_single_use():
    store, path = _store()
    try:
        token = settings_store.start_gmail_verification(store, email="user@example.com")
        assert settings_store.verify_gmail(store, token=token) is True
        # A second verify with the same (already-consumed) token must fail.
        assert settings_store.verify_gmail(store, token=token) is False
    finally:
        _cleanup(path)


def test_conditions_default_is_severe_and_danger_only():
    store, path = _store()
    try:
        conditions = settings_store.get_conditions(store)
        assert set(conditions["grades"]) == {"심각", "위험"}
        assert "주의" not in conditions["grades"]
        assert conditions["timing"] == settings_store.TIMING_ON_ANALYSIS
    finally:
        _cleanup(path)


def test_disconnect_removes_channel():
    store, path = _store()
    try:
        settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE-TEAM/FAKE-BOT/FAKE-TOKEN", channel="#x")
        assert settings_store.get_settings_summary(store)["slack"]["connected"] is True
        settings_store.disconnect(store, "slack")
        assert settings_store.get_settings_summary(store)["slack"]["connected"] is False
    finally:
        _cleanup(path)
