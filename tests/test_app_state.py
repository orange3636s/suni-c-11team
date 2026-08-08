"""Tests for the 학습·분석 결과 상태 유지 persistence layer:
RuntimeStore's app_state key-value table (src/runtime/store.py) and the
save/restore/invalidate helpers built on top of it (src/runtime/app_state.py).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.runtime.app_state import (
    STATE_SCHEMA_VERSION,
    get_latest_state,
    invalidate_state_for_dataset,
    save_state,
)
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


def test_app_state_roundtrip_via_store() -> None:
    store, path = _store()
    try:
        assert store.get_app_state("latest_training") is None
        store.set_app_state("latest_training", {"schema_version": 1, "dataset": "train", "payload": {"a": 1}})
        record = store.get_app_state("latest_training")
        assert record == {"schema_version": 1, "dataset": "train", "payload": {"a": 1}}
    finally:
        _cleanup(path)


def test_app_state_overwrites_not_accumulates() -> None:
    """spec §3-2: 각 종류당 1개만 유지 -- a second save replaces the first,
    never appends a second row."""
    store, path = _store()
    try:
        store.set_app_state("latest_analysis", {"dataset": "train"})
        store.set_app_state("latest_analysis", {"dataset": "test"})
        assert store.get_app_state("latest_analysis") == {"dataset": "test"}
    finally:
        _cleanup(path)


def test_get_all_app_state_returns_null_for_missing_keys() -> None:
    store, path = _store()
    try:
        store.set_app_state("latest_training", {"dataset": "train"})
        result = store.get_all_app_state(["latest_training", "latest_analysis", "latest_alarms"])
        assert result == {"latest_training": {"dataset": "train"}, "latest_analysis": None, "latest_alarms": None}
    finally:
        _cleanup(path)


def test_save_state_wraps_schema_version_and_created_at() -> None:
    store, path = _store()
    try:
        saved = save_state(store, "training", dataset={"dataset": "train"}, payload={"foo": "bar"})
        assert saved is True
        record = store.get_app_state("latest_training")
        assert record is not None
        assert record["schema_version"] == STATE_SCHEMA_VERSION
        assert record["dataset"] == "train"
        assert record["payload"] == {"foo": "bar"}
        assert "created_at" in record
    finally:
        _cleanup(path)


def test_save_state_never_raises_on_store_failure(monkeypatch) -> None:
    """spec §3-2: 저장 실패가 분석 실패로 이어지면 안 된다."""
    store, path = _store()
    try:
        monkeypatch.setattr(RuntimeStore, "set_app_state", lambda self, *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
        saved = save_state(store, "analysis", dataset={"dataset": "train"}, payload={})
        assert saved is False
    finally:
        _cleanup(path)


def test_get_latest_state_returns_null_for_all_three_when_empty() -> None:
    store, path = _store()
    try:
        result = get_latest_state(store)
        assert result == {"training": None, "analysis": None, "alarms": None}
    finally:
        _cleanup(path)


def test_get_latest_state_returns_all_three() -> None:
    store, path = _store()
    try:
        save_state(store, "training", dataset={"dataset": "train"}, payload={"p": "t"})
        save_state(store, "analysis", dataset={"dataset": "train"}, payload={"p": "a"})
        save_state(store, "alarms", dataset={"train_dataset": "train", "eval_dataset": "test"}, payload={"p": "al"})
        result = get_latest_state(store)
        assert result["training"]["payload"] == {"p": "t"}
        assert result["analysis"]["payload"] == {"p": "a"}
        assert result["alarms"]["payload"] == {"p": "al"}
        assert result["alarms"]["train_dataset"] == "train"
        assert result["alarms"]["eval_dataset"] == "test"
    finally:
        _cleanup(path)


def test_stale_schema_version_treated_as_absent() -> None:
    """spec §3-5: 저장된 결과의 스키마 버전이 현재와 다를 때 삭제(무효화)한다
    -- get_latest_state must never hand back a record whose shape the
    current code doesn't understand."""
    store, path = _store()
    try:
        store.set_app_state("latest_training", {"schema_version": 999, "dataset": "train", "payload": {}})
        result = get_latest_state(store)
        assert result["training"] is None
    finally:
        _cleanup(path)


def test_invalidate_state_for_dataset_deletes_only_matching_entries() -> None:
    store, path = _store()
    try:
        save_state(store, "training", dataset={"dataset": "train"}, payload={})
        save_state(store, "analysis", dataset={"dataset": "other_dataset"}, payload={})
        save_state(store, "alarms", dataset={"train_dataset": "train", "eval_dataset": "test"}, payload={})

        deleted = invalidate_state_for_dataset(store, "train")

        assert set(deleted) == {"training", "alarms"}
        assert store.get_app_state("latest_training") is None
        assert store.get_app_state("latest_alarms") is None
        # Untouched -- doesn't reference "train" on either side.
        assert store.get_app_state("latest_analysis") is not None
    finally:
        _cleanup(path)


def test_invalidate_state_for_dataset_no_op_when_nothing_matches() -> None:
    store, path = _store()
    try:
        save_state(store, "analysis", dataset={"dataset": "test"}, payload={})
        deleted = invalidate_state_for_dataset(store, "some-uploaded-uuid")
        assert deleted == []
        assert store.get_app_state("latest_analysis") is not None
    finally:
        _cleanup(path)


@pytest.mark.parametrize("kind", ["training", "analysis", "alarms"])
def test_save_state_kind_maps_to_expected_key(kind: str) -> None:
    store, path = _store()
    try:
        save_state(store, kind, dataset={"dataset": "train"}, payload={"k": kind})
        expected_key = {"training": "latest_training", "analysis": "latest_analysis", "alarms": "latest_alarms"}[kind]
        assert store.get_app_state(expected_key) is not None
    finally:
        _cleanup(path)
