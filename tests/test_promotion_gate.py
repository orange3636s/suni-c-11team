"""Tests for RuntimeStore.promote_if_better -- RB-4: the promotion gate
(R² threshold rejection) was removed. Every candidate is now promoted
unconditionally; this file verifies (a) that unconditional promotion
holds even for a much-worse challenger, and (b) that the performance
change is still recorded/reported, including a per-target regression
warning when any Y1~Y5 R² drops by >= REGRESSION_WARNING_RATIO
("교체는 하되 침묵하지 마라").
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


def _metadata(r2: float, model_id: str, target_metrics: dict | None = None) -> dict:
    metadata = {"model_id": model_id, "metrics": {"test": {"r2": r2}}}
    if target_metrics is not None:
        metadata["target_metrics"] = target_metrics
    return metadata


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


def test_much_worse_challenger_is_still_promoted_unconditionally():
    """RB-4: 게이트가 제거됐으므로 훨씬 나쁜 챌린저도 그대로 교체된다."""
    store, path = _store()
    try:
        store.promote_if_better(model_id="m1", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.20, "m1"))
        result = store.promote_if_better(model_id="m2", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.01, "m2"))
        assert result["active_model_id"] == "m2"
        active = store.active_model()
        assert active["active_model_id"] == "m2"

        events = store.list_promotion_events()
        assert events[0]["promoted"] == 1
        assert "0.2000" in events[0]["reason"] or "0.0100" in events[0]["reason"]
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


def test_no_challenger_is_ever_rejected_from_history():
    """RB-4: 이전에는 게이트 미달 챌린저가 promoted=0으로 이력에만 남고
    활성화되지 않았다 -- 이제는 모든 챌린저가 promoted=1로 활성화된다."""
    store, path = _store()
    try:
        store.promote_if_better(model_id="m1", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.20, "m1"))
        store.promote_if_better(model_id="m2", pipeline_version="v1", dataset_version=0, metadata=_metadata(0.01, "m2"))
        events = store.list_promotion_events()
        assert len(events) == 2
        assert events[0]["candidate_model_id"] == "m2"
        assert events[0]["promoted"] == 1
        assert store.active_model()["active_model_id"] == "m2"
    finally:
        _cleanup(path)


def test_per_target_regression_is_flagged_in_reason():
    """RB-4: 어느 모드든 R²가 50% 이상 떨어지면 교체는 하되 reason에
    "(저하)"로 남긴다 -- TrainingPanel이 이 문자열로 경고 스타일을
    고른다."""
    store, path = _store()
    try:
        store.promote_if_better(
            model_id="m1", pipeline_version="v1", dataset_version=0,
            metadata=_metadata(0.20, "m1", target_metrics={"Y2": {"r2": 0.35}, "Y1": {"r2": 0.24}}),
        )
        result = store.promote_if_better(
            model_id="m2", pipeline_version="v1", dataset_version=0,
            metadata=_metadata(0.20, "m2", target_metrics={"Y2": {"r2": 0.12}, "Y1": {"r2": 0.22}}),
        )
        assert result["active_model_id"] == "m2"  # 여전히 교체된다
        events = store.list_promotion_events()
        assert "Y2" in events[0]["reason"]
        assert "저하" in events[0]["reason"]
        assert "Y1" not in events[0]["reason"].split("Y2")[0]  # Y1은 50% 미만 하락이라 언급되지 않음
    finally:
        _cleanup(path)


def test_small_target_regression_is_not_flagged():
    store, path = _store()
    try:
        store.promote_if_better(
            model_id="m1", pipeline_version="v1", dataset_version=0,
            metadata=_metadata(0.20, "m1", target_metrics={"Y2": {"r2": 0.35}}),
        )
        store.promote_if_better(
            model_id="m2", pipeline_version="v1", dataset_version=0,
            metadata=_metadata(0.20, "m2", target_metrics={"Y2": {"r2": 0.30}}),
        )
        events = store.list_promotion_events()
        assert "저하" not in events[0]["reason"]
    finally:
        _cleanup(path)
