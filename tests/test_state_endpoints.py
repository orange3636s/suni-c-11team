"""Route-level tests for GET/POST /api/state/* (src/api/routes/state.py) --
calls the route functions directly (same pattern as test_model_deletion.py)
with `settings` monkeypatched to an isolated temp RuntimeStore, rather than
going through TestClient/HTTP.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.routes import state as state_routes
from api.schemas.state import AlarmsStateSaveRequest, AnalysisStateSaveRequest, TrainingStateSaveRequest


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime_db = tmp_path / "runtime" / f"dashboard_{uuid4().hex}.db"
    artifact_root = tmp_path / "runtime"
    test_settings = SimpleNamespace(runtime_db_path=runtime_db, runtime_artifact_dir=artifact_root)
    monkeypatch.setattr(state_routes, "settings", test_settings)
    return test_settings


def _fake_request() -> SimpleNamespace:
    # save_training_state reschedules the auto-ingest job via
    # request.app.state.scheduler -- no scheduler in these route-level
    # tests, so _apply_ingest_schedule's getattr(..., None) no-ops.
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


DEFAULT_NOTIFICATIONS = {
    "slack": {"connected": False, "target": None, "webhook_masked": None, "verified_at": None},
    "telegram": {"connected": False, "target": None, "chat_id_masked": None, "verified_at": None},
    "gmail": {"connected": False, "pending": False, "email": None, "verified_at": None},
    "conditions": {"grades": ["심각"], "timing": "on_analysis"},
}


def test_get_latest_empty(isolated_settings: SimpleNamespace) -> None:
    result = state_routes.get_latest()
    assert result == {
        "training": None,
        "analysis": None,
        "alarms": None,
        "notifications": DEFAULT_NOTIFICATIONS,
        "dataset_fallback_applied": False,
    }


def test_save_and_restore_training(isolated_settings: SimpleNamespace) -> None:
    response = state_routes.save_training_state(TrainingStateSaveRequest(dataset="train", payload={"performance": {"model_id": "m1"}}), _fake_request())
    assert response == {"saved": True}

    result = state_routes.get_latest()
    assert result["training"] is not None
    assert result["training"]["dataset"] == "train"
    assert result["training"]["payload"] == {"performance": {"model_id": "m1"}}
    assert result["analysis"] is None
    assert result["alarms"] is None


def test_save_and_restore_analysis(isolated_settings: SimpleNamespace) -> None:
    payload = {"activeTarget": "Y1", "paretoByTarget": {}, "scatterByKey": {}, "categoricalByKey": {}}
    state_routes.save_analysis_state(AnalysisStateSaveRequest(dataset="test", payload=payload))

    result = state_routes.get_latest()
    assert result["analysis"]["dataset"] == "test"
    assert result["analysis"]["payload"] == payload
    assert result["dataset_fallback_applied"] is False


def test_get_latest_drops_record_for_deleted_dataset(isolated_settings: SimpleNamespace) -> None:
    """지시서 CB: 저장된 결과가 더 이상 존재하지 않는 데이터셋(삭제된
    구버전 내장 데이터셋 등)을 가리키면 그 결과를 통째로 버리고 안내
    플래그를 세운다 -- dataset을 "train"으로 바꿔치기해 다른 스키마로
    계산된 옛 payload를 잘못된 라벨로 보여주지 않는다(부분 복원 금지)."""
    state_routes.save_analysis_state(
        AnalysisStateSaveRequest(
            dataset="a_since_deleted_dataset_id",
            payload={"activeTarget": "Y1", "paretoByTarget": {}, "scatterByKey": {}, "categoricalByKey": {}},
        )
    )

    result = state_routes.get_latest()

    assert result["analysis"] is None
    assert result["dataset_fallback_applied"] is True


def test_save_and_restore_alarms(isolated_settings: SimpleNamespace) -> None:
    payload = {"summary": {"a": 1}, "alarms": {"b": 2}, "recommendations": {"c": 3}}
    state_routes.save_alarms_state(AlarmsStateSaveRequest(train_dataset="train", eval_dataset="test", payload=payload))

    result = state_routes.get_latest()
    assert result["alarms"]["train_dataset"] == "train"
    assert result["alarms"]["eval_dataset"] == "test"
    assert result["alarms"]["payload"] == payload


def test_save_overwrites_previous_training_result(isolated_settings: SimpleNamespace) -> None:
    state_routes.save_training_state(TrainingStateSaveRequest(dataset="train", payload={"performance": {"model_id": "m1"}}), _fake_request())
    state_routes.save_training_state(TrainingStateSaveRequest(dataset="test", payload={"performance": {"model_id": "m2"}}), _fake_request())

    result = state_routes.get_latest()
    assert result["training"]["dataset"] == "test"
    assert result["training"]["payload"] == {"performance": {"model_id": "m2"}}


def test_get_latest_never_raises_on_store_error(isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """spec §3-3: 결과가 없으면 null, 실패해도 앱이 뜬다 (never a 500)."""
    monkeypatch.setattr(state_routes, "get_latest_state", lambda store: (_ for _ in ()).throw(OSError("db locked")))
    result = state_routes.get_latest()
    assert result == {
        "training": None,
        "analysis": None,
        "alarms": None,
        "notifications": DEFAULT_NOTIFICATIONS,
        "dataset_fallback_applied": False,
    }
