"""SE그룹: 알림 기록 -- dispatch_yield_update의 모든 종료 경로(발송/
건너뜀)가 notify_history에 기록을 남기고, 발송 시 메시지 전문을 그대로
보관하는지 검증한다."""

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


def _payload() -> YieldUpdatePayload:
    return YieldUpdatePayload(
        dataset_label="lot_2026w32.csv",
        timestamp_label="09:00",
        top10=(YieldUpdateTop10Item(lot_wafer_id="L001W25", y=85.0, reliability_count=3),),
        target_blocks=(),
        model_label="LGBM_20260811_0912",
    )


def test_skip_without_connected_channel_records_history():
    store, path = _store()
    try:
        result = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_REFRESH)
        assert result["skipped"] is True

        items = store.list_notify_history()
        assert len(items) == 1
        assert items[0]["status"] == "skipped"
        assert items[0]["skip_reason"] == "연결된 채널 없음"
        assert items[0]["message_text"] is None
        assert items[0]["channels"] == []
    finally:
        _cleanup(path)


def test_successful_send_records_full_message_text(monkeypatch):
    store, path = _store()
    try:
        _connect_slack(store)
        monkeypatch.setattr(yud.senders, "send_slack_webhook", lambda *a, **k: (True, None))

        result = yud.dispatch_yield_update(store, _payload(), trigger=yud.TRIGGER_REFRESH)
        assert result["skipped"] is False

        items = store.list_notify_history()
        assert len(items) == 1
        entry = items[0]
        assert entry["status"] == "sent"
        assert entry["channels"] == ["slack"]
        assert entry["sent_count"] == 1
        assert entry["dataset_label"] == "lot_2026w32.csv"
        assert entry["model_version"] == "LGBM_20260811_0912"
        # SE-3: 발송 당시의 메시지 원문을 그대로 보관한다(재계산하지 않는다).
        assert "L001W25" in entry["message_text"]
        assert "예측 수율이 낮은 WF TOP 10" in entry["message_text"]
    finally:
        _cleanup(path)


def test_retention_keeps_only_recent_100_rows():
    store, path = _store()
    try:
        for i in range(105):
            store.record_notify_history(
                trigger="refresh",
                channels=[],
                dataset_label=None,
                model_version=None,
                status="skipped",
                skip_reason=f"case {i}",
            )
        items = store.list_notify_history(limit=200)
        assert len(items) == 100
    finally:
        _cleanup(path)
