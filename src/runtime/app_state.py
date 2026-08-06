"""Server-persisted "latest result" for each of the three long-running
tabs (학습/원인 분석/사전 알람) -- spec: "학습·분석 결과 상태 유지". One row
per kind in RuntimeStore's `app_state` key-value table (never more than
one; a fresh result overwrites whatever was there). Scatter-plot point
coordinates are deliberately never part of any payload here (spec §3-1)
-- those stay cheap to refetch from /api/screening/scatter on demand and
would blow the size budget otherwise.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

# Bumped whenever a stored payload's shape changes incompatibly -- a
# stored record whose schema_version doesn't match is treated as absent
# (spec §3-5), not as a crash or a best-effort partial restore.
STATE_SCHEMA_VERSION = 1

StateKind = Literal["training", "analysis", "alarms"]
STATE_KEYS: dict[StateKind, str] = {
    "training": "latest_training",
    "analysis": "latest_analysis",
    "alarms": "latest_alarms",
}
# Field names a stored record may carry its dataset identity under --
# whichever ones are present are what `invalidate_state_for_dataset`
# matches against.
_DATASET_FIELDS = ("dataset", "train_dataset", "eval_dataset")


def save_state(store: RuntimeStore, kind: StateKind, *, dataset: dict[str, str], payload: dict[str, Any]) -> bool:
    """Best-effort -- a save failure must never surface as a training/
    analysis/alarm failure (spec §3-2), so every exception is swallowed
    here and only logged; callers get a bool back purely for their own
    "saved" response field, never something to raise on.
    """
    try:
        record = {
            "schema_version": STATE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **dataset,
            "payload": payload,
        }
        store.set_app_state(STATE_KEYS[kind], record)
        return True
    except Exception:
        logger.warning("최근 결과 저장 실패: kind=%s", kind, exc_info=True)
        return False


def _valid(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    if record.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    return record


def get_latest_state(store: RuntimeStore) -> dict[StateKind, dict[str, Any] | None]:
    raw = store.get_all_app_state(list(STATE_KEYS.values()))
    return {kind: _valid(raw.get(key)) for kind, key in STATE_KEYS.items()}


def _record_dataset_ids(record: dict[str, Any]) -> set[str]:
    return {record[field] for field in _DATASET_FIELDS if isinstance(record.get(field), str)}


def invalidate_state_for_dataset(store: RuntimeStore, dataset_id: str) -> list[StateKind]:
    """Deletes any of the 3 stored results that reference `dataset_id`
    (spec §3-5) -- called when that dataset is deleted. Never touches
    results for other datasets, and is never called on a fresh upload
    (spec: "데이터셋을 새로 업로드해도 기존 결과를 지우지 마라" -- uploading
    doesn't delete anything, so it never reaches this function at all).
    """
    deleted: list[StateKind] = []
    for kind, key in STATE_KEYS.items():
        record = store.get_app_state(key)
        if record and dataset_id in _record_dataset_ids(record):
            store.delete_app_state(key)
            deleted.append(kind)
    return deleted
