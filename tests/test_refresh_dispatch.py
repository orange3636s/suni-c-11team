"""J-5: 갱신 시 신규 알람 자동 발송 -- 차단 조건(게이트 미달/폴백 모드/
발송 시점 미설정)과 "이전 스냅샷 대비 신규" 판정, 시간당 예산을
검증한다. 실제 채널 발송은 연결된 채널이 없으므로 항상 no-op이다.
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


def test_blocked_in_fallback_mode() -> None:
    store, path = _store()
    try:
        refresh_dispatch.dispatch_new_alarms(
            store, mode="fallback", train_dataset_id="train", eval_dataset_id="test",
            alarm_items=[_item("W1", "심각")], gate_passed=True, snapshot_created_at="2026-01-01T00:00:00+00:00",
        )
        log = store.list_refresh_dispatch_log()
        assert "폴백 모드" in log[0]["blocked_reason"]
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
