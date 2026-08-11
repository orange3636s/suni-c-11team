"""Tests for RuntimeStore.save_alert_snapshot/list_alert_snapshots -- moved
out of the old tests/test_notify_dispatch.py (which tested the now-retired
src/notifications/dispatch.py alarm pipeline) since this immutable-snapshot
storage behavior is independent infrastructure, not part of that pipeline.
"""

from __future__ import annotations

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


def test_alert_snapshots_are_append_only_and_keep_model_provenance():
    store, path = _store()
    try:
        first_id = store.save_alert_snapshot(
            dataset_id="eval-a",
            model_id="model-a",
            model_version="pipeline-a",
            criteria_version="criteria-v1",
            payload={"items_top": [{"lot_wafer_id": "W1", "target_source": "predicted"}]},
            created_at="2026-08-10T00:00:00+00:00",
        )
        second_id = store.save_alert_snapshot(
            dataset_id="eval-a",
            model_id="model-b",
            model_version="pipeline-b",
            criteria_version="criteria-v2",
            payload={"items_top": []},
            created_at="2026-08-10T01:00:00+00:00",
        )
        history = store.list_alert_snapshots()
        assert [item["snapshot_id"] for item in history] == [second_id, first_id]
        assert history[1]["model_id"] == "model-a"
        assert history[1]["payload"]["items_top"][0]["target_source"] == "predicted"
    finally:
        _cleanup(path)
