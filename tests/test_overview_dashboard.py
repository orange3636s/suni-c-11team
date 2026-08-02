from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

import pytest
from fastapi import HTTPException

from api.main import app
from api.routes import runtime as runtime_routes
from api.schemas.runtime import AnalysisOverviewResponse
from src.runtime.store import RuntimeStore


StoreFactory = Callable[[], RuntimeStore]


@pytest.fixture
def store_factory() -> Iterator[StoreFactory]:
    root = Path(__file__).parent / ".tmp_overview"
    root.mkdir(exist_ok=True)
    stores: list[RuntimeStore] = []

    def create() -> RuntimeStore:
        token = uuid4().hex
        store = RuntimeStore(root / f"{token}.db", root / f"{token}_artifacts")
        stores.append(store)
        return store

    yield create

    for store in stores:
        if store.artifact_root.exists():
            for path in sorted(store.artifact_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            store.artifact_root.rmdir()
        if store.path.exists():
            store.path.unlink()
    if root.exists() and not any(root.iterdir()):
        root.rmdir()


def _analysis_artifact(*, sparse: bool = False) -> dict[str, Any]:
    if sparse:
        return {"analysis_result": {"risk": {"critical_count": 0}}}
    analysis = {
        "created_at": "2026-08-02T01:00:00+00:00",
        "model": {
            "model_id": "model-a",
            "model_name": "Hybrid Multi-Y",
            "compatibility": "snapshot",
        },
        "dataset": {"filename": "train.csv", "row_count": 10},
        "metrics": {"test": {"r2": 0.91, "rmse": 2.5, "mae": 1.25}},
        "risk": {"normal_count": 7, "warning_count": 2, "critical_count": 1},
        "confidence": {"low_confidence_count": 0},
        "multi_y": {
            "average_direct_y": 92.0,
            "average_derived_y": 91.5,
            "average_ensemble_y": 91.8,
            "failure_rate_averages": {"Y1": 1.2},
            "fail_bit_count_averages": {"Y6": 3.0},
        },
        "feature_importance": {
            "global": [{"rank": 1, "feature": "Step1_R1", "mean_abs_shap": 0.7}],
            "steps": [{"rank": 1, "step": "1", "mean_abs_shap": 0.6}],
        },
        "risk_wafers": [{
            "identifier": "LOT01_W01",
            "predicted_value": 80.0,
            "risk_level": "danger",
            "top_harmful_features": ["Step1_R1"],
        }],
        "lot_summary": [{
            "lot_id": "LOT01",
            "wafer_count": 2,
            "average_predicted_yield": 84.0,
            "danger_count": 1,
            "warning_count": 1,
            "normal_count": 0,
            "danger_ratio": 0.5,
        }],
        "relationships": [{
            "rank": 1,
            "response": "Step1_R1",
            "defect": "Step1_D1",
            "chamber": "CH-1",
            "path_score": 0.4,
            "valid_count": 10,
        }],
        "statistics": {
            "numeric": [{
                "relation": "R vs Y",
                "feature": "Step1_R1",
                "target": "Y",
                "pearson": 0.2,
                "spearman": 0.18,
                "pearson_p_value": 0.03,
                "pearson_fdr_p_value": 0.04,
                "effect_size": 0.04,
                "valid_count": 10,
                "direction": "positive",
            }],
            "categorical": [],
        },
        "data_quality": {"r_measurement_coverage": 0.9},
        "warnings": ["검증 경고"],
    }
    return {
        "analysis_result": analysis,
        "response": {
            "analysis_result": analysis,
            "explanation": {
                "equipment_summary": [{
                    "rank": 1,
                    "equipment": "EQ-1",
                    "mean_abs_shap": 0.5,
                }],
            },
            "pareto": {
                "features": [{
                    "rank": 1,
                    "feature": "Step1_R1",
                    "impact": 0.7,
                    "cumulative_share": 1.0,
                }],
            },
        },
    }


def _complete_analysis(
    store: RuntimeStore,
    analysis_id: str,
    *,
    created_at: str,
    artifact: dict[str, Any] | None = None,
) -> None:
    store.start_analysis(
        analysis_id=analysis_id,
        source_filename=f"{analysis_id}.csv",
        model_id="model-a",
        created_at=created_at,
    )
    store.complete_analysis(
        analysis_id,
        metadata={
            "duration_ms": 5.0,
            "dataset_fingerprint": "fp",
            "model_name_snapshot": "Hybrid Multi-Y",
            "model_version_snapshot": "v1",
            "model_type_snapshot": "hybrid_multi_y",
            "schema_version": "v2",
            "row_count": 10,
            "lot_count": 1,
            "available_targets_json": '["Y", "Y1", "Y6"]',
            "default_target": "Y",
            "report_snapshot_available": 1,
        },
        summary={
            "average_predicted_yield": 91.8,
            "minimum_predicted_yield": 80.0,
            "critical_count": 1,
            "warning_count": 2,
            "normal_count": 7,
            "risk_lot_count": 1,
            "top_failure_target": "Y1",
        },
        methodology={},
        artifact=artifact if artifact is not None else _analysis_artifact(),
        warnings=["검증 경고"],
    )


def _set_status(store: RuntimeStore, analysis_id: str, status: str) -> None:
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "UPDATE analysis_runs SET status=? WHERE analysis_id=?",
            (status, analysis_id),
        )
        connection.commit()


def _overview(monkeypatch, store: RuntimeStore, analysis_id: str | None = None) -> dict[str, Any]:
    monkeypatch.setattr(runtime_routes, "get_runtime_store", lambda: store)
    response = runtime_routes.get_dashboard_overview(analysis_id)
    return AnalysisOverviewResponse.model_validate(response).model_dump(mode="json")


def test_runtime_overview_routes_are_registered() -> None:
    paths = app.openapi()["paths"]
    assert "/api/analyses/history" in paths
    assert "/api/analyses/history/{analysis_id}" in paths
    assert "/api/dashboard/overview" in paths
    assert "/api/api/dashboard/overview" not in paths
    parameters = paths["/api/dashboard/overview"]["get"]["parameters"]
    assert any(item["name"] == "analysis_id" and item["in"] == "query" for item in parameters)


def test_analysis_history_list_is_lightweight_searchable_and_paginated(store_factory: StoreFactory) -> None:
    store = store_factory()
    _complete_analysis(store, "analysis_first", created_at="2026-08-01T01:00:00+00:00")
    _complete_analysis(store, "analysis_second", created_at="2026-08-02T01:00:00+00:00")

    original = store._read_artifact_state
    store._read_artifact_state = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("artifact read"))
    try:
        listed = store.list_analyses({"limit": 1, "offset": 0, "sort": "newest"})
    finally:
        store._read_artifact_state = original
    assert listed["total"] == 2
    assert listed["items"][0]["analysis_id"] == "analysis_second"
    assert listed["items"][0]["artifact_available"] is True
    assert listed["items"][0]["model_name"] == "Hybrid Multi-Y"
    assert listed["items"][0]["average_predicted_yield"] == 91.8
    assert "artifact" not in listed["items"][0]
    assert store.list_analyses({"search": "analysis_first"})["items"][0]["analysis_id"] == "analysis_first"
    assert store.list_analyses({"status": "completed"})["total"] == 2
    assert store.list_analyses({"model_id": "model-a"})["total"] == 2
    assert store.list_analyses({"filename": "analysis_second.csv"})["total"] == 1
    assert store.list_analyses({"date_from": "2026-08-02T00:00:00+00:00"})["total"] == 1
    assert store.list_analyses({"date_to": "2026-08-01T23:59:59+00:00"})["total"] == 1
    assert store.list_analyses({"sort": "oldest"})["items"][0]["analysis_id"] == "analysis_first"


def test_overview_auto_selects_completed_before_newer_partial_and_failed(monkeypatch, store_factory: StoreFactory) -> None:
    store = store_factory()
    _complete_analysis(store, "analysis_completed", created_at="2026-08-01T01:00:00+00:00")
    _complete_analysis(store, "analysis_partial", created_at="2026-08-02T01:00:00+00:00")
    _set_status(store, "analysis_partial", "partial")
    store.start_analysis(
        analysis_id="analysis_failed",
        source_filename="failed.csv",
        model_id="model-a",
        created_at="2026-08-03T01:00:00+00:00",
    )
    store.fail_analysis("analysis_failed", "failed")

    body = _overview(monkeypatch, store)
    assert body["source"]["analysis_id"] == "analysis_completed"
    assert body["source"]["type"] == "analysis"
    assert body["model_metrics"] == {"r2": 0.91, "rmse": 2.5, "mae": 1.25}
    assert body["causes"]["top_equipment"][0]["equipment"] == "EQ-1"
    assert body["pareto"][0]["feature"] == "Step1_R1"
    assert body["relationships"][0]["pearson"] == 0.2


def test_overview_explicit_selection_and_unknown_id(monkeypatch, store_factory: StoreFactory) -> None:
    store = store_factory()
    _complete_analysis(store, "analysis_old", created_at="2026-08-01T01:00:00+00:00")
    _complete_analysis(store, "analysis_new", created_at="2026-08-02T01:00:00+00:00")
    selected = _overview(monkeypatch, store, "analysis_old")
    assert selected["source"]["analysis_id"] == "analysis_old"
    assert selected["source_label"] == "선택한 원인 분석"

    with pytest.raises(HTTPException) as raised:
        _overview(monkeypatch, store, "analysis_unknown")
    assert raised.value.status_code == 404
    assert raised.value.detail == "선택한 분석 이력을 찾을 수 없습니다."


def test_overview_uses_partial_only_when_completed_does_not_exist(monkeypatch, store_factory: StoreFactory) -> None:
    store = store_factory()
    _complete_analysis(store, "analysis_partial", created_at="2026-08-02T01:00:00+00:00")
    _set_status(store, "analysis_partial", "partial")
    response = _overview(monkeypatch, store)
    assert response["source"]["analysis_id"] == "analysis_partial"
    assert response["source"]["status"] == "partial"


def test_missing_and_corrupt_artifacts_keep_metadata_and_section_contract(monkeypatch, store_factory: StoreFactory) -> None:
    store = store_factory()
    _complete_analysis(store, "analysis_missing", created_at="2026-08-01T01:00:00+00:00")
    missing_path = store.artifact_root / "analyses" / "analysis_missing.json.gz"
    missing_path.unlink()
    missing_body = _overview(monkeypatch, store, "analysis_missing")
    assert missing_body["source"]["status"] == "artifact_missing"
    assert missing_body["source"]["artifact_available"] is False
    assert missing_body["summary"]["average_predicted_yield"] == 91.8
    assert missing_body["model_metrics"]["r2"] is None
    assert missing_body["pareto"] == []
    assert missing_body["availability"]["summary"] is True
    assert missing_body["availability"]["pareto"] is False

    _complete_analysis(store, "analysis_corrupt", created_at="2026-08-02T01:00:00+00:00")
    corrupt_path = store.artifact_root / "analyses" / "analysis_corrupt.json.gz"
    corrupt_path.write_bytes(b"not-a-gzip")
    corrupt = _overview(monkeypatch, store, "analysis_corrupt")
    assert corrupt["source"]["status"] == "artifact_corrupted"
    assert corrupt["source"]["artifact_status"] == "corrupted"


def test_legacy_sparse_snapshot_and_empty_analysis_contract(monkeypatch, store_factory: StoreFactory) -> None:
    store = store_factory()
    _complete_analysis(
        store,
        "analysis_legacy",
        created_at="2026-08-01T01:00:00+00:00",
        artifact=_analysis_artifact(sparse=True),
    )
    body = _overview(monkeypatch, store)
    assert body["summary"]["critical_count"] == 0
    assert body["model_metrics"] == {"r2": None, "rmse": None, "mae": None}
    assert body["risk_lots"] == []
    assert body["risk_wafers"] == []
    assert body["pareto"] == []
    assert body["relationships"] == []

    empty_store = store_factory()
    empty_store.start_prediction(
        prediction_id="prediction_only",
        source_filename="prediction.csv",
        model_id="model-a",
    )
    empty = _overview(monkeypatch, empty_store)
    assert empty["source"]["type"] == "empty"
    assert empty["source_type"] == "empty"
    assert empty["summary"]["average_predicted_yield"] is None


def test_invalid_legacy_summary_json_does_not_break_history_list(store_factory: StoreFactory) -> None:
    store = store_factory()
    _complete_analysis(store, "analysis_bad_json", created_at="2026-08-01T01:00:00+00:00")
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "UPDATE analysis_runs SET summary_json=? WHERE analysis_id=?",
            ("{broken", "analysis_bad_json"),
        )
        connection.commit()
    item = store.list_analyses({"limit": 10})["items"][0]
    assert item["summary"] is None
    assert item["metadata_decode_errors"] == ["summary"]
