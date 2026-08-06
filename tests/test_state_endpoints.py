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


DEFAULT_NOTIFICATIONS = {
    "slack": {"connected": False, "target": None, "webhook_masked": None, "verified_at": None},
    "telegram": {"connected": False, "target": None, "chat_id_masked": None, "verified_at": None},
    "gmail": {"connected": False, "pending": False, "email": None, "verified_at": None},
    "conditions": {"grades": ["심각", "위험"], "timing": "on_analysis"},
}


def test_get_latest_empty(isolated_settings: SimpleNamespace) -> None:
    result = state_routes.get_latest()
    assert result == {"training": None, "analysis": None, "alarms": None, "notifications": DEFAULT_NOTIFICATIONS}


def test_save_and_restore_training(isolated_settings: SimpleNamespace) -> None:
    response = state_routes.save_training_state(TrainingStateSaveRequest(dataset="train", payload={"performance": {"model_id": "m1"}}))
    assert response == {"saved": True}

    result = state_routes.get_latest()
    assert result["training"] is not None
    assert result["training"]["dataset"] == "train"
    assert result["training"]["payload"] == {"performance": {"model_id": "m1"}}
    assert result["analysis"] is None
    assert result["alarms"] is None


def test_save_and_restore_analysis(isolated_settings: SimpleNamespace) -> None:
    payload = {"activeTarget": "Y1", "paretoByTarget": {}, "scatterByKey": {}, "categoricalByKey": {}}
    state_routes.save_analysis_state(AnalysisStateSaveRequest(dataset="mentorship_dataset_final", payload=payload))

    result = state_routes.get_latest()
    assert result["analysis"]["dataset"] == "mentorship_dataset_final"
    assert result["analysis"]["payload"] == payload


def test_save_and_restore_alarms(isolated_settings: SimpleNamespace) -> None:
    payload = {"summary": {"a": 1}, "alarms": {"b": 2}, "recommendations": {"c": 3}}
    state_routes.save_alarms_state(AlarmsStateSaveRequest(train_dataset="train", eval_dataset="test", payload=payload))

    result = state_routes.get_latest()
    assert result["alarms"]["train_dataset"] == "train"
    assert result["alarms"]["eval_dataset"] == "test"
    assert result["alarms"]["payload"] == payload


def test_save_overwrites_previous_training_result(isolated_settings: SimpleNamespace) -> None:
    state_routes.save_training_state(TrainingStateSaveRequest(dataset="train", payload={"performance": {"model_id": "m1"}}))
    state_routes.save_training_state(TrainingStateSaveRequest(dataset="test", payload={"performance": {"model_id": "m2"}}))

    result = state_routes.get_latest()
    assert result["training"]["dataset"] == "test"
    assert result["training"]["payload"] == {"performance": {"model_id": "m2"}}


def test_get_latest_never_raises_on_store_error(isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """spec §3-3: 결과가 없으면 null, 실패해도 앱이 뜬다 (never a 500)."""
    monkeypatch.setattr(state_routes, "get_latest_state", lambda store: (_ for _ in ()).throw(OSError("db locked")))
    result = state_routes.get_latest()
    assert result == {"training": None, "analysis": None, "alarms": None, "notifications": DEFAULT_NOTIFICATIONS}
