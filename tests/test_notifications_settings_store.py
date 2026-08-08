"""Tests for src/notifications/settings_store.py -- persistence (never
localStorage, spec §D-1), masking (spec §D-2), and the settings summary
shape consumed by both GET /api/state/latest and the settings panel.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def test_gmail_pending_expires_after_ttl():
    # 지시서 W: 5분이 지난 인증 대기 레코드는 조회 시점에 삭제되고
    # 미연결 상태로 복귀해야 한다.
    store, path = _store()
    try:
        stale_requested_at = (
            datetime.now(timezone.utc) - timedelta(seconds=settings_store.PENDING_TTL_SECONDS + 1)
        ).isoformat()
        store.set_app_state(
            settings_store.GMAIL_PENDING_STATE_KEY,
            {"email": "user@example.com", "token": "tok", "requested_at": stale_requested_at},
        )
        assert settings_store.get_gmail_pending(store) is None
        summary = settings_store.get_settings_summary(store)
        assert summary["gmail"] == {"connected": False, "pending": False, "email": None, "verified_at": None}
    finally:
        _cleanup(path)


def test_gmail_pending_within_ttl_is_not_expired():
    store, path = _store()
    try:
        token = settings_store.start_gmail_verification(store, email="user@example.com")
        # 방금 발송했으므로 만료되지 않아야 한다.
        record = settings_store.get_gmail_pending(store)
        assert record is not None
        assert record["token"] == token
        assert settings_store.get_settings_summary(store)["gmail"]["pending"] is True
    finally:
        _cleanup(path)


def test_gmail_verify_fails_after_ttl_expired():
    store, path = _store()
    try:
        stale_requested_at = (
            datetime.now(timezone.utc) - timedelta(seconds=settings_store.PENDING_TTL_SECONDS + 1)
        ).isoformat()
        store.set_app_state(
            settings_store.GMAIL_PENDING_STATE_KEY,
            {"email": "user@example.com", "token": "tok", "requested_at": stale_requested_at},
        )
        # 만료된 뒤에는 정확한 토큰이어도 인증되면 안 된다.
        assert settings_store.verify_gmail(store, token="tok") is False
    finally:
        _cleanup(path)


def test_gmail_verified_record_never_expires():
    # 연결 완료된 레코드는 requested_at이 아무리 오래돼도 만료되지 않는다
    # -- 서버 재시작·재접속 후에도 계속 연결 상태를 유지해야 한다.
    store, path = _store()
    try:
        old_requested_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        store.set_app_state(
            settings_store.STATE_KEYS["gmail"],
            {"email": "user@example.com", "verified": True, "verified_at": old_requested_at},
        )
        record = settings_store.get_gmail(store)
        assert record is not None
        assert record["verified"] is True
        assert settings_store.get_settings_summary(store)["gmail"]["connected"] is True
    finally:
        _cleanup(path)


def test_reconnect_attempt_timing_out_does_not_destroy_existing_connection():
    """A-8 회귀: 기존에 연결 완료된(verified) 이메일이 있는 상태에서 새
    이메일로 재인증을 시도했다가 5분 안에 끝내지 못해도, 기존 연결은
    그대로 살아 있어야 한다. 이전에는 pending이 연결 완료 레코드와 같은
    키를 덮어써서, pending이 만료되며 함께 지워졌다."""
    store, path = _store()
    try:
        first_token = settings_store.start_gmail_verification(store, email="old@example.com")
        assert settings_store.verify_gmail(store, token=first_token) is True
        assert settings_store.get_gmail(store)["email"] == "old@example.com"

        settings_store.start_gmail_verification(store, email="new@example.com")
        stale_requested_at = (
            datetime.now(timezone.utc) - timedelta(seconds=settings_store.PENDING_TTL_SECONDS + 1)
        ).isoformat()
        pending = store.get_app_state(settings_store.GMAIL_PENDING_STATE_KEY)
        store.set_app_state(settings_store.GMAIL_PENDING_STATE_KEY, {**pending, "requested_at": stale_requested_at})

        assert settings_store.get_gmail_pending(store) is None  # 재인증 시도는 만료됨

        record = settings_store.get_gmail(store)
        assert record is not None
        assert record["email"] == "old@example.com"
        summary = settings_store.get_settings_summary(store)
        assert summary["gmail"]["connected"] is True
        assert summary["gmail"]["email"] == "old@example.com"
    finally:
        _cleanup(path)


def test_conditions_default_is_severe_only():
    store, path = _store()
    try:
        conditions = settings_store.get_conditions(store)
        assert set(conditions["grades"]) == {"심각"}
        assert "위험" not in conditions["grades"]
        assert "주의" not in conditions["grades"]
        assert conditions["timing"] == settings_store.TIMING_ON_ANALYSIS
    finally:
        _cleanup(path)


def test_conditions_migrates_legacy_daily_8am_timing():
    store, path = _store()
    try:
        # 지시서 N-2: 예전에 "daily_8am"으로 저장된 설정을 읽으면 새 값
        # "daily_9am"으로 변환되어야 한다 -- 깨진 값으로 남으면 안 된다.
        store.set_app_state(settings_store.STATE_KEYS["conditions"], {"grades": ["심각"], "timing": "daily_8am"})
        conditions = settings_store.get_conditions(store)
        assert conditions["timing"] == settings_store.TIMING_DAILY_9AM
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
