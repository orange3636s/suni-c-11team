from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

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


def test_purge_old_notification_log_removes_only_stale_rows() -> None:
    """H-3②: notify_sent_log는 24시간 재발송 방지 조회에만 쓰이므로,
    보관 기간보다 오래된 행만 지우고 최근 행은 남겨야 한다."""
    store, path = _store()
    try:
        old_cutoff = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        with store._connect() as connection:
            connection.execute(
                "INSERT INTO notify_sent_log (dataset_id, wafer_id, grade, sent_at, channel) VALUES (?,?,?,?,?)",
                ("train", "old-wafer", "심각", old_cutoff, "slack"),
            )
        store.record_notifications_sent("train", [("new-wafer", "심각")], channel="slack")

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        deleted = store.purge_old_notification_log(older_than_iso=cutoff)

        assert deleted == 1
        remaining = store.recent_notifications("train", "1970-01-01T00:00:00+00:00", channel="slack")
        assert [row["wafer_id"] for row in remaining] == ["new-wafer"]
    finally:
        _cleanup(path)
