"""J-5: 갱신 시 신규 알람 자동 발송 -- 차단 조건(게이트 미달/발송 시점
미설정/수동 업로드 10분 간격)과 "이전 스냅샷 대비 신규" 판정, 시간당
예산을 검증한다. 실제 채널 발송은 연결된 채널이 없으므로 항상 no-op이다.

EB그룹: 폴백(SQL 미연결 데모)·수동 업로드 모드는 더 이상 차단되지
않는다 -- 대신 메시지에 출처가 붙고(수동은 추가로 10분 최소 간격).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from src.automation import refresh_dispatch
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


def _item(wafer: str, grade: str) -> dict:
    return {"lot_wafer_id": wafer, "lot_id": "L001", "grade": grade, "risk_percentile": 1.0, "reason": ""}


def test_blocked_when_gate_not_passed() -> None:
    store, path = _store()
    try:
        refresh_dispatch.dispatch_new_alarms(
            store, mode="sql", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=False, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert log[0]["blocked_reason"] == "게이트 미달"
    finally:
        _cleanup(path)


def test_fallback_mode_is_no_longer_blocked() -> None:
    """EB-2: 폴백(SQL 미연결 데모) 모드는 더 이상 차단하지 않는다 --
    연결된 채널이 없어 결국 "연결된 채널 없음"으로 스킵되지만, 그것은
    이전의 전용 "폴백 모드" 차단과는 다른 사유다."""
    store, path = _store()
    try:
        refresh_dispatch.dispatch_new_alarms(
            store, mode="fallback", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert "폴백" not in (log[0]["blocked_reason"] or "")
        assert log[0]["new_alarm_count"] == 1
        assert log[0]["source"] == "fallback"
    finally:
        _cleanup(path)


def test_fallback_mode_message_has_demo_source_note(monkeypatch) -> None:
    """EB-3: 폴백 모드로 실제 발송될 때 본문 첫 줄에 [데모] 출처가
    붙는다."""
    store, path = _store()
    try:
        settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#x")
        captured = {}

        def _fake_send(url, payload):
            captured["payload"] = payload
            return True, None

        monkeypatch.setattr(refresh_dispatch.dispatch.senders, "send_slack_alarm", _fake_send)
        refresh_dispatch.dispatch_new_alarms(
            store, mode="fallback", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        assert captured["payload"].source_note == "[데모] 내장 데이터 기준 — 실제 공정 데이터가 아닙니다"
    finally:
        _cleanup(path)


def _fake_registry(filename: str):
    class _FakeRegistry:
        def get_summary(self, dataset_id):
            return {"original_filename": filename}

    return _FakeRegistry()


def test_manual_mode_is_no_longer_blocked(monkeypatch) -> None:
    """EB-1: 수동 업로드 모드도 더 이상 차단하지 않는다."""
    store, path = _store()
    try:
        monkeypatch.setattr("api.routes.datasets.get_dataset_registry", lambda: _fake_registry("uploaded_0809.csv"))
        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert log[0]["new_alarm_count"] == 1
        assert log[0]["source"] == "manual"
    finally:
        _cleanup(path)


def test_manual_mode_message_has_manual_source_note_with_filename(monkeypatch) -> None:
    """EB-3: 수동 업로드로 발송될 때 본문 첫 줄에 [수동] 원본 파일명이
    붙는다."""
    store, path = _store()
    try:
        settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#x")
        monkeypatch.setattr("api.routes.datasets.get_dataset_registry", lambda: _fake_registry("uploaded_0809.csv"))
        captured = {}

        def _fake_send(url, payload):
            captured["payload"] = payload
            return True, None

        monkeypatch.setattr(refresh_dispatch.dispatch.senders, "send_slack_alarm", _fake_send)
        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        assert captured["payload"].source_note == "[수동] uploaded_0809.csv 업로드 결과"
    finally:
        _cleanup(path)


def test_manual_mode_throttled_within_10_minutes(monkeypatch) -> None:
    """EB-4: 직전 수동 발송(실제로 시도된 발송)으로부터 10분이 지나지
    않았으면 새 업로드는 억제되고, 사유·다음 발송 가능 시각이 기록된다."""
    store, path = _store()
    try:
        settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#x")
        monkeypatch.setattr("api.routes.datasets.get_dataset_registry", lambda: _fake_registry("first.csv"))
        monkeypatch.setattr(refresh_dispatch.dispatch.senders, "send_slack_alarm", lambda *a, **k: (True, None))

        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        log_after_first = store.list_refresh_dispatch_log()
        assert log_after_first[0]["blocked_reason"] is None

        # 다른 파일을 곧바로 올려도(등급 악화로 "신규"는 성립) 10분 안이면
        # 억제된다 -- 억제는 소스 조회/reliability 계산 이전에 일어나므로
        # eval_dataset_id는 실존 여부와 무관하다.
        monkeypatch.setattr("api.routes.datasets.get_dataset_registry", lambda: _fake_registry("second.csv"))
        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="ds-2",
            alarm_items=[_item("W2", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:05:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert "10분" in log[0]["blocked_reason"]
        assert "다음 발송 가능" in log[0]["blocked_reason"]
        # 억제는 24시간 dedupe 로그(notify_sent_log)에 남지 않는다 --
        # W2/심각에 대해 아직 아무것도 기록되지 않았어야 한다.
        assert store.recent_notifications("ds-2", "2000-01-01T00:00:00+00:00", channel="slack") == []
    finally:
        _cleanup(path)


def test_manual_mode_dispatches_again_after_throttle_window(monkeypatch) -> None:
    """EB-4: 10분이 지나면 억제 없이 정상 발송되고, dedupe에 막히지
    않는다(억제 시 아무것도 기록하지 않았으므로)."""
    store, path = _store()
    try:
        settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#x")
        monkeypatch.setattr("api.routes.datasets.get_dataset_registry", lambda: _fake_registry("first.csv"))
        monkeypatch.setattr(refresh_dispatch.dispatch.senders, "send_slack_alarm", lambda *a, **k: (True, None))

        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        # 직전 발송 시각을 11분 전으로 되돌려 간격이 지난 것처럼 만든다.
        past = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
        store.set_app_state(refresh_dispatch._MANUAL_DISPATCH_THROTTLE_STATE_KEY, {"last_sent_at": past})

        monkeypatch.setattr("api.routes.datasets.get_dataset_registry", lambda: _fake_registry("second.csv"))
        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W2", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:05:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert log[0]["blocked_reason"] is None
        assert log[0]["new_alarm_count"] == 1
    finally:
        _cleanup(path)


def test_manual_mode_not_throttled_when_no_new_alarms(monkeypatch) -> None:
    """같은 파일을 두 번 올리면 두 번째는 신규 0건이라 자연히 막히고,
    이때는 10분 타이머가 시작되지 않는다(실제로 아무것도 보내지
    않았으므로)."""
    store, path = _store()
    try:
        store.save_refresh_snapshot(
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "alarms": {"items_top": [{"lot_wafer_id": "W1", "grade": "심각"}]},
            }
        )
        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="ds-1",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:01:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert log[0]["new_alarm_count"] == 0
        assert refresh_dispatch._manual_dispatch_next_allowed_at(store) is None
    finally:
        _cleanup(path)


def test_blocked_when_timing_not_on_analysis() -> None:
    store, path = _store()
    try:
        settings_store.save_conditions(store, grades=["심각"], timing=[settings_store.TIMING_DAILY_9AM])
        refresh_dispatch.dispatch_new_alarms(
            store, mode="sql", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert "on_analysis" in log[0]["blocked_reason"]
    finally:
        _cleanup(path)


def test_no_dispatch_recorded_when_no_new_alarms() -> None:
    """이전 스냅샷에 이미 있던 (wafer, grade) 조합은 "신규"가 아니다."""
    store, path = _store()
    try:
        store.save_refresh_snapshot(
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "alarms": {"items_top": [{"lot_wafer_id": "W1", "grade": "심각"}]},
            }
        )
        refresh_dispatch.dispatch_new_alarms(
            store, mode="sql", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-02T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert log[0]["new_alarm_count"] == 0
        assert log[0]["blocked_reason"] is None
    finally:
        _cleanup(path)


def test_grade_escalation_counts_as_new() -> None:
    """같은 wafer라도 등급이 악화되면(주의 -> 심각) 신규로 본다."""
    store, path = _store()
    try:
        store.save_refresh_snapshot(
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "alarms": {"items_top": [{"lot_wafer_id": "W1", "grade": "주의"}]},
            }
        )
        refresh_dispatch.dispatch_new_alarms(
            store, mode="sql", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-02T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        # 연결된 채널이 없어 "연결된 채널 없음"으로 스킵되지만, new_alarm_count는
        # 1이어야 한다(악화를 신규로 인정했다는 뜻).
        assert log[0]["new_alarm_count"] == 1
    finally:
        _cleanup(path)


def test_hourly_budget_also_applies_to_manual_mode(monkeypatch) -> None:
    """EB-4: 차단을 풀었다고 기존 시간당 예산까지 느슨해지지 않는다 --
    수동 업로드도 이 예산을 소비한다."""
    store, path = _store()
    try:
        monkeypatch.setattr("api.routes.datasets.get_dataset_registry", lambda: _fake_registry("uploaded.csv"))
        for i in range(refresh_dispatch.HOURLY_SEND_BUDGET):
            store.record_notifications_sent("ds-1", [(f"W{i}", "심각")], channel="slack")

        refresh_dispatch.dispatch_new_alarms(
            store, mode="manual", train_dataset_id="train", eval_dataset_id="ds-1",
            alarm_items=[_item("W99", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert "예산" in log[0]["blocked_reason"]
    finally:
        _cleanup(path)


def test_hourly_budget_blocks_and_records() -> None:
    store, path = _store()
    try:
        now = datetime.now(timezone.utc)
        # 예산(6건)을 이미 채운 것처럼 최근 발송 로그를 심는다.
        for i in range(refresh_dispatch.HOURLY_SEND_BUDGET):
            store.record_notifications_sent("test", [(f"W{i}", "심각")], channel="slack")

        refresh_dispatch.dispatch_new_alarms(
            store, mode="sql", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W99", "심각")], gate_passed=True, snapshot_created_at=now.isoformat(),
        )
        log = store.list_refresh_dispatch_log()
        assert "예산" in log[0]["blocked_reason"]
    finally:
        _cleanup(path)
