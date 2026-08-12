"""Tests for RuntimeStore.list_alert_snapshots -- moved out of the old
tests/test_notify_dispatch.py (which tested the now-retired
src/notifications/dispatch.py alarm pipeline) since this immutable-snapshot
storage behavior is independent infrastructure, not part of that pipeline.

`save_alert_snapshot` (the writer this test used to call) had zero
production callers and was removed; `_seed_alert_snapshot` below inserts
directly into `alert_snapshots` to keep exercising the still-live reader.
"""

from __future__ import annotations

import json
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


def _seed_alert_snapshot(
    store: RuntimeStore,
    *,
    dataset_id: str,
    model_id: str | None,
    model_version: str | None,
    criteria_version: str,
    payload: dict,
    created_at: str,
) -> str:
    """Direct `alert_snapshots` insert standing in for the now-removed
    `RuntimeStore.save_alert_snapshot` -- that method had zero production
    callers (nothing writes alert snapshots anymore) but was the only
    seeding path for this test, which exercises the still-live
    `list_alert_snapshots` (used by api/routes/analysis.py)."""
    snapshot_id = f"alert_{uuid4().hex}"
    with store._connect() as connection:
        connection.execute(
            """INSERT INTO alert_snapshots
            (snapshot_id, created_at, dataset_id, model_id, model_version, criteria_version, payload_json)
            VALUES (?,?,?,?,?,?,?)""",
            (
                snapshot_id,
                created_at,
                dataset_id,
                model_id,
                model_version,
                criteria_version,
                json.dumps(payload),
            ),
        )
    return snapshot_id


def test_alert_snapshots_are_append_only_and_keep_model_provenance():
    store, path = _store()
    try:
        first_id = _seed_alert_snapshot(
            store,
            dataset_id="eval-a",
            model_id="model-a",
            model_version="pipeline-a",
            criteria_version="criteria-v1",
            payload={"items_top": [{"lot_wafer_id": "W1", "target_source": "predicted"}]},
            created_at="2026-08-10T00:00:00+00:00",
        )
        second_id = _seed_alert_snapshot(
            store,
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
