"""Tests for RuntimeStore.promote_if_better -- the shared gate manual
training, the async training-job path, and the auto-ingest pipeline all
go through (자동 수집 파이프라인 §2-1).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.runtime.store import PROMOTION_TOLERANCE, RuntimeStore


def _store() -> tuple[RuntimeStore, Path]:
    root = Path(__file__).parent / ".tmp_runtime"
    root.mkdir(exist_ok=True)
    path = root / f"dashboard_{uuid4().hex}.db"
    return RuntimeStore(path), path


def _cleanup(path: Path) -> None:
    path.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def _metadata(r2: float, model_id: str) -> dict:
    return {"model_id": model_id, "metrics": {"test": {"r2": r2}}}


def test_first_model_promotes_without_gate():
    store, path = _store()
    try:
        result = store.promote_if_better(
            model_id="m1", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.1, "m1"),
        )
        assert result["active_model_id"] == "m1"
        events = store.list_promotion_events()
        assert len(events) == 1
        assert events[0]["promoted"] == 1
        assert "최초 모델" in events[0]["reason"]
    finally:
        _cleanup(path)


def test_better_challenger_is_promoted():
    store, path = _store()
    try:
        store.promote_if_better(model_id="m1", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.10, "m1"))
        result = store.promote_if_better(model_id="m2", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.20, "m2"))
        assert result["active_model_id"] == "m2"
    finally:
        _cleanup(path)


def test_challenger_within_tolerance_is_promoted():
    # 지시서 §2-1: 작은 노이즈(TOLERANCE 이내 하락)로는 승격이 막히지 않는다.
    store, path = _store()
    try:
        store.promote_if_better(model_id="m1", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.20, "m1"))
        slightly_worse = 0.20 - (PROMOTION_TOLERANCE / 2)
        result = store.promote_if_better(model_id="m2", pipeline_version="v1", dataset_version=0, metadata=_metadata(slightly_worse, "m2"))
        assert result["active_model_id"] == "m2"
    finally:
        _cleanup(path)


def test_challenger_beyond_tolerance_is_rejected():
    store, path = _store()
    try:
        store.promote_if_better(model_id="m1", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.20, "m1"))
        much_worse = 0.20 - (PROMOTION_TOLERANCE * 4)
        result = store.promote_if_better(model_id="m2", pipeline_version="v1", dataset_version=0, metadata=_metadata(much_worse, "m2"))
        # 챔피언 유지 -- 챌린저로 교체되지 않는다.
        assert result["active_model_id"] == "m1"
        active = store.active_model()
        assert active["active_model_id"] == "m1"

        events = store.list_promotion_events()
        assert events[0]["promoted"] == 0
        assert "저하" in events[0]["reason"]
    finally:
        _cleanup(path)


def test_incomparable_metric_promotes_by_default():
    store, path = _store()
    try:
        store.promote_if_better(model_id="m1", pipeline_version="v1", dataset_version=0, metadata={"model_id": "m1", "metrics": {}})
        result = store.promote_if_better(model_id="m2", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.5, "m2"))
        assert result["active_model_id"] == "m2"
    finally:
        _cleanup(path)


def test_rejected_challenger_is_not_discarded_from_history_but_not_active():
    store, path = _store()
    try:
        store.promote_if_better(model_id="m1", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.20, "m1"))
        store.promote_if_better(model_id="m2", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.01, "m2"))
        events = store.list_promotion_events()
        assert len(events) == 2
        assert events[0]["candidate_model_id"] == "m2"
        assert events[0]["promoted"] == 0
        assert store.active_model()["active_model_id"] == "m1"
    finally:
        _cleanup(path)
