"""J-3: 자동 갱신 스냅샷 저장/복원 -- 원자적 저장(단일 UPSERT)과 스키마
버전 검사(다르면 복원하지 않음)를 확인한다."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.runtime.store import REFRESH_SNAPSHOT_SCHEMA_VERSION, RuntimeStore


def _store() -> tuple[RuntimeStore, Path]:
    root = Path(__file__).parent / ".tmp_runtime"
    root.mkdir(exist_ok=True)
    path = root / f"dashboard_{uuid4().hex}.db"
    return RuntimeStore(path), path


def _cleanup(path: Path) -> None:
    path.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def test_save_and_restore_snapshot_round_trip() -> None:
    store, path = _store()
    try:
        assert store.get_refresh_snapshot_status() == {"snapshot": None, "stale_version": False}
        assert store.get_refresh_snapshot_meta() is None

        snapshot = {
            "created_at": "2026-08-08T14:20:00+09:00",
            "source": {"mode": "fallback", "train_dataset": "train", "eval_dataset": "test", "row_count": 1000},
            "model": {"champion_version": "v14"},
            "analysis": {"paretoByTarget": {}},
            "alarms": {"counts": {"심각": 3}},
            "monitoring": {"predicted_yield": None},
            "errors": [],
        }
        store.save_refresh_snapshot(snapshot)

        status = store.get_refresh_snapshot_status()
        assert status["stale_version"] is False
        assert status["snapshot"]["source"]["mode"] == "fallback"
        assert status["snapshot"]["schema_version"] == REFRESH_SNAPSHOT_SCHEMA_VERSION

        meta = store.get_refresh_snapshot_meta()
        assert meta == {"created_at": "2026-08-08T14:20:00+09:00"}
    finally:
        _cleanup(path)


def test_stale_schema_version_is_not_restored() -> None:
    """스키마가 바뀐 뒤 남아 있는 옛 버전 스냅샷은 복원하지 않는다 --
    조용히 빈 화면이 되는 대신 stale_version=True로 알린다."""
    store, path = _store()
    try:
        store.save_refresh_snapshot({"created_at": "2026-01-01T00:00:00+00:00", "source": {}})
        # 스키마 버전이 실제로 바뀐 것처럼 직접 손상시킨다.
        with store._connect() as connection:
            connection.execute(
                "UPDATE app_state SET value_json = json_set(value_json, '$.schema_version', 1) WHERE state_key = ?",
                ("automation:refresh_snapshot",),
            )

        status = store.get_refresh_snapshot_status()
        assert status["snapshot"] is None
        assert status["stale_version"] is True
        assert store.get_refresh_snapshot_meta() is None
    finally:
        _cleanup(path)


def test_saving_new_snapshot_overwrites_previous() -> None:
    store, path = _store()
    try:
        store.save_refresh_snapshot({"created_at": "2026-01-01T00:00:00+00:00", "source": {"mode": "fallback"}})
        store.save_refresh_snapshot({"created_at": "2026-01-02T00:00:00+00:00", "source": {"mode": "sql"}})
        status = store.get_refresh_snapshot_status()
        assert status["snapshot"]["source"]["mode"] == "sql"
        assert status["snapshot"]["created_at"] == "2026-01-02T00:00:00+00:00"
    finally:
        _cleanup(path)
