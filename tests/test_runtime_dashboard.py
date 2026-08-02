from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.runtime.store import RuntimeStore, safe_runtime_call


def _store() -> tuple[RuntimeStore, Path]:
    root = Path(__file__).parent / ".tmp_runtime"
    root.mkdir(exist_ok=True)
    path = root / f"dashboard_{uuid4().hex}.db"
    return RuntimeStore(path), path


def _cleanup(path: Path) -> None:
    path.unlink()
    for name in ("predictions", "analyses"):
        directory = path.parent / name
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def test_alerts_are_real_risk_only_deduplicated_and_mutable() -> None:
    store, path = _store()
    try:
        predictions = [
            {"Lot_Wafer_ID": "LOT01_WF01", "predicted_Y": 82.0, "risk_level": "danger"},
            {"Lot_Wafer_ID": "LOT01_WF02", "predicted_Y": 88.0, "risk_level": "warning"},
            {"Lot_Wafer_ID": "LOT01_WF03", "predicted_Y": 96.0, "risk_level": "normal"},
        ]
        assert store.record_prediction_alerts(
            analysis_id="analysis-1", model_id="model-1", model_version=None,
            predictions=predictions, identifier_column="Lot_Wafer_ID",
        ) == 2
        assert store.record_prediction_alerts(
            analysis_id="analysis-1", model_id="model-1", model_version=None,
            predictions=predictions, identifier_column="Lot_Wafer_ID",
        ) == 0
        listed = store.list_alerts({"limit": 10})
        assert listed["total"] == 2
        assert store.alert_summary()["external_not_configured_count"] == 2
        updated = store.update_alert(listed["items"][0]["alert_id"], "Acknowledged")
        assert updated and updated["acknowledged_at"]
        resolved = store.update_alert(listed["items"][0]["alert_id"], "Resolved")
        assert resolved and resolved["resolved_at"]
    finally:
        _cleanup(path)


def test_run_history_records_latency_and_error_count_inputs() -> None:
    store, path = _store()
    try:
        store.record_run(event_type="predict", duration_ms=12.5, status="success", row_count=20)
        store.record_run(event_type="report", duration_ms=7.5, status="failed", error_type="RuntimeError")
        runs = store.list_runs()
        assert len(runs) == 2
        assert {run["status"] for run in runs} == {"success", "failed"}
        assert sum(run["duration_ms"] for run in runs) == 20.0
    finally:
        _cleanup(path)


def test_runtime_store_failure_isolated_from_main_request(monkeypatch) -> None:
    monkeypatch.setattr(RuntimeStore, "record_run", lambda self, **values: (_ for _ in ()).throw(OSError("db unavailable")))
    assert safe_runtime_call("record_run", event_type="predict") is None


def test_prediction_history_metadata_artifact_list_detail_and_delete() -> None:
    store, path = _store()
    prediction_id = "prediction_test_history"
    try:
        assert store.start_prediction(
            prediction_id=prediction_id, source_filename="sample.csv",
            model_id="model-1", warning_threshold=90, critical_threshold=85,
        ) == prediction_id
        assert store.complete_prediction(
            prediction_id,
            metadata={
                "duration_ms": 12.0, "dataset_fingerprint": "fingerprint",
                "model_name_snapshot": "Model", "model_version_snapshot": "v1",
                "model_type_snapshot": "single", "schema_version": "v2",
                "row_count": 2, "lot_count": 1, "final_strategy": "direct",
            },
            summary={"average_predicted_yield": 91.5, "critical_count": 0},
            preprocessing={"missing_indicator": True},
            artifact={"rows": [{"Lot_Wafer_ID": "LOT01_W01"}]},
            warnings=[],
        ) is True
        listed = store.list_predictions({"limit": 10})
        assert listed["total"] == 1
        assert "rows" not in listed["items"][0]
        detail = store.get_prediction(prediction_id)
        assert detail and detail["artifact"]["rows"][0]["Lot_Wafer_ID"] == "LOT01_W01"
        assert detail["metadata"]["summary"]["average_predicted_yield"] == 91.5
        assert store.delete_prediction(prediction_id) is True
        assert store.get_prediction(prediction_id) is None
    finally:
        _cleanup(path)


def test_analysis_history_link_survives_prediction_deletion_and_is_latest() -> None:
    store, path = _store()
    prediction_id = "prediction_linked"
    analysis_id = "analysis_linked"
    try:
        store.start_prediction(prediction_id=prediction_id, source_filename="sample.csv", model_id="model-1")
        store.complete_prediction(
            prediction_id,
            metadata={
                "duration_ms": 1.0, "dataset_fingerprint": "fp",
                "model_name_snapshot": "Model", "model_version_snapshot": None,
                "model_type_snapshot": "single", "schema_version": None,
                "row_count": 1, "lot_count": 1, "final_strategy": None,
            }, summary={}, preprocessing={}, artifact={"rows": []}, warnings=[],
        )
        store.start_analysis(
            analysis_id=analysis_id, prediction_id=prediction_id,
            source_filename="sample.csv", model_id="model-1",
        )
        store.complete_analysis(
            analysis_id,
            metadata={
                "duration_ms": 2.0, "dataset_fingerprint": "fp",
                "model_name_snapshot": "Model", "model_version_snapshot": None,
                "model_type_snapshot": "single", "schema_version": None,
                "row_count": 1, "lot_count": 1,
                "available_targets_json": '["Y"]', "default_target": "Y",
                "report_snapshot_available": 1,
            }, summary={"average_predicted_yield": 90.0}, methodology={},
            artifact={"analysis_result": {"risk": {}}}, warnings=[],
        )
        assert store.latest_completed("analysis")["metadata"]["analysis_id"] == analysis_id
        assert store.delete_prediction(prediction_id) is True
        detail = store.get_analysis(analysis_id)
        assert detail and detail["source_prediction_deleted"] is True
        assert detail["artifact"] is not None
        assert store.delete_analysis(analysis_id) is True
    finally:
        _cleanup(path)
