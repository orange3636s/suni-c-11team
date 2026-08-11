"""Tests for src/notifications/yield_update_dispatch.py (VE-1) -- 억제
규칙(시간당 예산/수동 최소 간격) 두 가지만 유지하는지 검증한다. 24시간/
신규분 dedupe는 없다 -- 분석이 일어날 때마다 보낸다(같은 내용이 반복돼도
스킵하지 않는다). RuntimeStore는 test_notify_dispatch.py와 같은 임시
sqlite 파일 패턴을 쓴다."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.notifications import settings_store, yield_update_dispatch as yud
from src.notifications.yield_update_senders import YieldUpdatePayload, YieldUpdateTop10Item
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


def _connect_slack(store: RuntimeStore) -> None:
    settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#eng-yield")


def _payload(top10_y: float = 85.0) -> YieldUpdatePayload:
    return YieldUpdatePayload(
        dataset_label="test.csv",
        timestamp_label="14:20",
        top10=(YieldUpdateTop10Item(lot_wafer_id="L001W25", y=top10_y, reliability_count=3),),
        target_blocks=(),
    )


def test_no_connected_channel_skips():
    store, path = _store()
    try:
        result = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_REFRESH)
        assert result["skipped"] is True
        assert "채널" in result["reason"]
    finally:
        _cleanup(path)


def test_connected_channel_sends(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        sent = []

        def _fake_send(url, body):
            sent.append(body)
            return True, None

        monkeypatch.setattr(yud.senders, "send_slack_webhook", _fake_send)

        result = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_REFRESH)
        assert result["skipped"] is False
        assert result["results"]["slack"]["ok"] is True
        assert len(sent) == 1
    finally:
        _cleanup(path)


def test_identical_content_sends_again_no_dedupe(monkeypatch):
    """QD-1: 24시간/신규분 dedupe는 제거됐다 -- 같은 내용이 반복돼도
    매 분석마다 발송한다."""
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(yud.senders, "send_slack_webhook", lambda *a, **k: (True, None))

        first = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_REFRESH)
        assert first["skipped"] is False

        second = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_REFRESH)
        assert second["skipped"] is False
    finally:
        _cleanup(path)


def test_changed_content_sends_again(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(yud.senders, "send_slack_webhook", lambda *a, **k: (True, None))

        first = yud.dispatch_yield_update(store, _payload(85.0), trigger=yud.TRIGGER_REFRESH)
        assert first["skipped"] is False

        second = yud.dispatch_yield_update(store, _payload(86.5), trigger=yud.TRIGGER_REFRESH)
        assert second["skipped"] is False
    finally:
        _cleanup(path)


def test_hourly_budget_exceeded_skips_further_sends(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(yud.senders, "send_slack_webhook", lambda *a, **k: (True, None))
        monkeypatch.setattr(yud, "HOURLY_SEND_BUDGET", 2)

        results = [
            yud.dispatch_yield_update(store, _payload(80.0 + i), trigger=yud.TRIGGER_REFRESH) for i in range(3)
        ]
        assert [r["skipped"] for r in results] == [False, False, True]
        assert "예산" in results[2]["reason"]
    finally:
        _cleanup(path)


def test_manual_trigger_requires_on_analysis_timing():
    store, path = _store()
    try:
        _connect_slack(store)
        settings_store.save_conditions(store, grades=["심각"], timing=[settings_store.TIMING_DAILY_9AM])

        result = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_MANUAL)
        assert result["skipped"] is True
        assert "분석 실행 직후" in result["reason"]
    finally:
        _cleanup(path)


def test_manual_trigger_respects_minimum_interval(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(yud.senders, "send_slack_webhook", lambda *a, **k: (True, None))
        settings_store.save_conditions(store, grades=["심각"], timing=[settings_store.TIMING_ON_ANALYSIS])

        first = yud.dispatch_yield_update(store, _payload(80.0), trigger=yud.TRIGGER_MANUAL)
        assert first["skipped"] is False

        # 내용이 바뀌어도(신규분 있음) 10분 최소 간격이 아직 지나지 않았으면 막힌다.
        second = yud.dispatch_yield_update(store, _payload(90.0), trigger=yud.TRIGGER_MANUAL)
        assert second["skipped"] is True
        assert "10분" in second["reason"]
    finally:
        _cleanup(path)


def test_refresh_trigger_ignores_timing_settings(monkeypatch):
    """VE-1: 자동 갱신은 timing 설정과 무관하게 발송 후보다."""
    store, path = _store()
    try:
        _connect_slack(store)
        settings_store.save_conditions(store, grades=["심각"], timing=[])  # 전부 해제
        monkeypatch.setattr(yud.senders, "send_slack_webhook", lambda *a, **k: (True, None))

        result = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_REFRESH)
        assert result["skipped"] is False
    finally:
        _cleanup(path)
