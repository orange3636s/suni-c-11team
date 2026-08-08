"""Tests for src/notifications/dispatch.py -- the three behaviors spec calls
out as 핵심: (17) low-reliability datasets never send, (19) duplicate
(dataset, wafer, grade) within 24h is skipped, (20) an escalated grade is
sent even within that window.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.notifications import dispatch, settings_store
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
    settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE-TEAM/FAKE-BOT/FAKE-TOKEN", channel="#eng-yield")


def _alarm(wafer="L001W01", grade="심각", pct=0.3) -> dict:
    return {"lot_wafer_id": wafer, "risk_percentile": pct, "grade": grade, "reason": "Step1_D1 = 14.0 (경고선 12.0 초과)"}


def test_low_reliability_skips_send(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        called = {"sent": False}
        monkeypatch.setattr(dispatch.senders, "send_slack_alarm", lambda *a, **k: called.update(sent=True) or (True, None))

        result = dispatch.dispatch_alarm_notifications(
            store,
            trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train",
            dataset_label="train.CSV",
            alarms=[_alarm()],
            reliability_grade="낮음",
            reliability_score=10,
        )
        assert result["skipped"] is True
        assert "신뢰도" in result["reason"]
        assert called["sent"] is False
    finally:
        _cleanup(path)


def test_high_reliability_sends_to_connected_channel(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        sent_payloads = []

        def _fake_send(url, payload):
            sent_payloads.append(payload)
            return True, None

        monkeypatch.setattr(dispatch.senders, "send_slack_alarm", _fake_send)

        result = dispatch.dispatch_alarm_notifications(
            store,
            trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train",
            dataset_label="train.CSV",
            alarms=[_alarm()],
            reliability_grade="높음",
            reliability_score=85,
        )
        assert result["skipped"] is False
        assert result["sent_count"] == 1
        assert len(sent_payloads) == 1
    finally:
        _cleanup(path)


def test_duplicate_within_24h_is_skipped(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(dispatch.senders, "send_slack_alarm", lambda *a, **k: (True, None))

        first = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm()],
            reliability_grade="높음", reliability_score=85,
        )
        assert first["skipped"] is False

        second = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm()],
            reliability_grade="높음", reliability_score=85,
        )
        assert second["skipped"] is True
        assert "이미 발송" in second["reason"]
    finally:
        _cleanup(path)


def test_escalated_grade_resends_within_24h(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(dispatch.senders, "send_slack_alarm", lambda *a, **k: (True, None))
        # 지시서 N-3: 발송 대상 등급 기본값이 "심각"만으로 바뀌었다 -- 이
        # 테스트는 등급 승격(위험 -> 심각) 시 24시간 내에도 재발송되는지를
        # 보는 것이 목적이므로, 기본값에 기대지 않고 두 등급 모두 대상으로
        # 명시적으로 설정한다.
        settings_store.save_conditions(store, grades=["위험", "심각"], timing=settings_store.TIMING_ON_ANALYSIS)

        first = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm(grade="위험")],
            reliability_grade="높음", reliability_score=85,
        )
        assert first["skipped"] is False

        escalated = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm(grade="심각")],
            reliability_grade="높음", reliability_score=85,
        )
        assert escalated["skipped"] is False
        assert escalated["sent_count"] == 1
    finally:
        _cleanup(path)


def test_no_connected_channel_skips():
    store, path = _store()
    try:
        result = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm()],
            reliability_grade="높음", reliability_score=85,
        )
        assert result["skipped"] is True
        assert "채널" in result["reason"]
    finally:
        _cleanup(path)


def test_alarm_outside_configured_grades_is_not_sent(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        settings_store.save_conditions(store, grades=["심각"], timing=settings_store.TIMING_ON_ANALYSIS)
        monkeypatch.setattr(dispatch.senders, "send_slack_alarm", lambda *a, **k: (True, None))

        result = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm(grade="주의")],
            reliability_grade="높음", reliability_score=85,
        )
        assert result["skipped"] is True
    finally:
        _cleanup(path)


def test_on_analysis_trigger_is_skipped_when_daily_only_is_selected(monkeypatch):
    """A-6 회귀: "매일 9시만"(daily_9am)을 선택했으면 분석 실행 직후
    (on_analysis) 트리거로는 발송되지 않아야 한다 -- 이전에는 timing이
    저장만 되고 어느 발송 경로도 읽지 않아 항상 발송됐다."""
    store, path = _store()
    try:
        _connect_slack(store)
        settings_store.save_conditions(store, grades=["심각"], timing=settings_store.TIMING_DAILY_9AM)
        called = {"sent": False}
        monkeypatch.setattr(dispatch.senders, "send_slack_alarm", lambda *a, **k: called.update(sent=True) or (True, None))

        result = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_ON_ANALYSIS,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm()],
            reliability_grade="높음", reliability_score=85,
        )
        assert result["skipped"] is True
        assert called["sent"] is False
    finally:
        _cleanup(path)


def test_daily_trigger_is_skipped_when_on_analysis_only_is_selected(monkeypatch):
    """A-6 반대 방향: 기본값(on_analysis)만 선택돼 있으면 매일 09:00
    스케줄러 잡(daily_9am 트리거)으로는 발송되지 않아야 한다."""
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(dispatch.senders, "send_slack_alarm", lambda *a, **k: (True, None))

        result = dispatch.dispatch_alarm_notifications(
            store, trigger=settings_store.TIMING_DAILY_9AM,
            dataset_id="train", dataset_label="train.CSV", alarms=[_alarm()],
            reliability_grade="높음", reliability_score=85,
        )
        assert result["skipped"] is True
        assert result["reason"] == "발송 시점 설정과 일치하지 않음"
    finally:
        _cleanup(path)
