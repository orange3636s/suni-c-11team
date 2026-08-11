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
from api.schemas.state import AnalysisStateSaveRequest, TrainingStateSaveRequest


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
    "conditions": {"grades": ["심각"], "timing": ["on_analysis"]},
    "automation": {
        "enabled": False,
        "sql_host": "",
        "sql_port": "",
        "sql_db": "",
        "sql_user": "",
        "refresh_interval_minutes": 60,
        "last_run_at": None,
        "last_run_status": None,
        "last_run_sent_count": None,
    },
    "telegram_bot_username": None,
}


def test_get_latest_empty(isolated_settings: SimpleNamespace) -> None:
    result = state_routes.get_latest()
    assert result == {
        "training": None,
        "analysis": None,
        "alarms": None,
        "notifications": DEFAULT_NOTIFICATIONS,
        "dataset_fallback_applied": False,
        "degraded": False,
    }


def test_save_and_restore_training(isolated_settings: SimpleNamespace) -> None:
    # SD그룹: SQL 연결·refresh time·자동화 on/off는 더 이상 이 슬롯에
    # 저장하지 않는다("알림·자동화 설정" 팝업이 POST /api/notify/automation
    # 으로 저장한다) -- save_training_state는 더 이상 스케줄러를 건드리지
    # 않으므로 request 인자도 필요 없다.
    response = state_routes.save_training_state(TrainingStateSaveRequest(dataset="train", payload={"performance": {"model_id": "m1"}}))
    assert response == {"saved": True, "schedule_applied": True}

    result = state_routes.get_latest()
    assert result["training"] is not None
    assert result["training"]["dataset"] == "train"
    assert result["training"]["payload"] == {"performance": {"model_id": "m1"}}
    assert result["analysis"] is None
    assert result["alarms"] is None


def test_save_and_restore_analysis(isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    # 지시서 JA-1: save_analysis_state가 저장 시점에 FMEA를 채운다 --
    # 실제 _fmea_payload(원본 데이터셋 전수 스코어링 + SPC/ML 부트스트랩)
    # 는 이 라우팅 테스트에 비해 너무 느리므로 가벼운 스텁으로 대체한다.
    monkeypatch.setattr(state_routes, "_fmea_payload", lambda dataset, targets: {"dataset_id": dataset, "items": []})
    payload = {"activeTarget": "Y1", "paretoByTarget": {}, "scatterByKey": {}, "categoricalByKey": {}}
    state_routes.save_analysis_state(AnalysisStateSaveRequest(dataset="test", payload=payload))

    result = state_routes.get_latest()
    assert result["analysis"]["dataset"] == "test"
    assert result["analysis"]["payload"]["activeTarget"] == "Y1"
    assert result["analysis"]["payload"]["fmea"] == {"dataset_id": "test", "items": []}
    assert result["analysis"]["payload"]["fmeaError"] is None
    assert result["dataset_fallback_applied"] is False


def test_save_analysis_keeps_fmea_already_in_payload(
    isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JA-1: 이미 fmea가 실려 있으면(비어 있지 않으면) 다시 계산하지
    않는다 -- 계산이 불렸으면 실패하도록 만든 스텁으로 확인한다."""

    def _fail(dataset: str, targets):
        raise AssertionError("fmea가 이미 있으면 _fmea_payload를 다시 부르면 안 된다")

    monkeypatch.setattr(state_routes, "_fmea_payload", _fail)
    payload = {"activeTarget": "Y1", "paretoByTarget": {}, "fmea": {"dataset_id": "test", "items": [{"feature": "Step1_R1"}]}}
    state_routes.save_analysis_state(AnalysisStateSaveRequest(dataset="test", payload=payload))

    result = state_routes.get_latest()
    assert result["analysis"]["payload"]["fmea"] == {"dataset_id": "test", "items": [{"feature": "Step1_R1"}]}


def test_save_analysis_fmea_failure_does_not_block_save(
    isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JA-2: FMEA 계산이 실패해도 Pareto 등 나머지 payload는 그대로
    저장된다 -- 분석 저장 전체를 실패시키지 않는다."""

    def _boom(dataset: str, targets):
        raise RuntimeError("dataset unreadable")

    monkeypatch.setattr(state_routes, "_fmea_payload", _boom)
    payload = {"activeTarget": "Y1", "paretoByTarget": {"Y1": {"items": []}}, "measurementExpansion": {"total_wafers": 10}}
    response = state_routes.save_analysis_state(AnalysisStateSaveRequest(dataset="test", payload=payload))

    assert response == {"saved": True}
    result = state_routes.get_latest()
    assert result["analysis"]["payload"]["paretoByTarget"] == {"Y1": {"items": []}}
    assert result["analysis"]["payload"]["measurementExpansion"] == {"total_wafers": 10}
    assert result["analysis"]["payload"]["fmea"] is None
    assert result["analysis"]["payload"]["fmeaError"]


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


def test_save_overwrites_previous_training_result(isolated_settings: SimpleNamespace) -> None:
    state_routes.save_training_state(TrainingStateSaveRequest(dataset="train", payload={"performance": {"model_id": "m1"}}))
    state_routes.save_training_state(TrainingStateSaveRequest(dataset="test", payload={"performance": {"model_id": "m2"}}))

    result = state_routes.get_latest()
    assert result["training"]["dataset"] == "test"
    assert result["training"]["payload"] == {"performance": {"model_id": "m2"}}


def test_get_latest_never_raises_on_store_error(isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """spec §3-3: 결과가 없으면 null, 실패해도 앱이 뜬다 (never a 500).
    D-2: 이 실패는 "저장된 결과 없음"과 구분돼야 한다 -- degraded=True로
    표시해 사용자가 결과가 사라진 줄 알고 재분석을 다시 돌리지 않게 한다.
    """
    monkeypatch.setattr(state_routes, "get_latest_state", lambda store: (_ for _ in ()).throw(OSError("db locked")))
    result = state_routes.get_latest()
    assert result == {
        "training": None,
        "analysis": None,
        "alarms": None,
        "notifications": DEFAULT_NOTIFICATIONS,
        "dataset_fallback_applied": False,
        "degraded": True,
    }


def test_get_latest_reports_degraded_when_app_state_is_corrupted(isolated_settings: SimpleNamespace) -> None:
    """D-2: JSON이 깨진 저장 레코드는 get_app_state 레벨에서 조용히
    None이 되므로, degraded 플래그가 없으면 "저장된 적 없음"과 구분할
    방법이 없다."""
    store = state_routes._store()
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO app_state (state_key, value_json, updated_at) VALUES (?, ?, ?)",
            ("latest_training", "{not valid json", "2026-01-01T00:00:00+00:00"),
        )

    result = state_routes.get_latest()

    assert result["training"] is None
    assert result["degraded"] is True
