from __future__ import annotations

import asyncio
import json
import shutil
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pandas as pd
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from api.main import app, health_check
from api.routes import data as data_routes
from src.runtime.operation_coordinator import (
    HEAVY_JOB_MESSAGE,
    ActiveOperationError,
    OperationCoordinator,
    operation_coordinator,
)
from src.runtime.store import RuntimeStore
from src.runtime.training_jobs import TrainingJobManager, new_training_job_id


@pytest.fixture
def job_root() -> Any:
    parent = (Path(__file__).parent / ".training_job_cases").resolve()
    parent.mkdir(exist_ok=True)
    root = (parent / uuid4().hex).resolve()
    root.mkdir()
    try:
        yield root
    finally:
        if root.exists() and root.parent == parent:
            shutil.rmtree(root)
        try:
            parent.rmdir()
        except OSError:
            pass


def _manager(job_root: Path) -> tuple[TrainingJobManager, RuntimeStore, OperationCoordinator]:
    runtime = job_root / "runtime"
    runtime.mkdir()
    store = RuntimeStore(runtime / "dashboard.db", runtime)
    coordinator = OperationCoordinator()
    manager = TrainingJobManager(
        store=store,
        input_root=runtime / "training_jobs",
        coordinator=coordinator,
    )
    return manager, store, coordinator


def _wait_for_terminal(
    manager: TrainingJobManager,
    job_id: str,
    *,
    timeout: float = 5,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = manager.get(job_id)
        if row is not None and row["status"] in {
            "completed",
            "failed",
            "interrupted",
        }:
            return row
        time.sleep(0.01)
    raise AssertionError("학습 Job이 제한 시간 안에 종료되지 않았습니다.")


def _wait_for_gate_release(coordinator: OperationCoordinator) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if coordinator.snapshot()["training"] == 0:
            return
        time.sleep(0.01)
    raise AssertionError("학습 Job 종료 후 공통 Gate가 해제되지 않았습니다.")


def _compact_result() -> dict[str, Any]:
    return {
        "model_id": "Y_HGBR_20260802_120000",
        "target": "Y",
        "best_model": "HistGradientBoostingRegressor",
        "test_metrics": {"r2": 0.8, "rmse": 1.2, "mae": 0.9, "mse": 1.44},
        "feature_count": 12,
        "warning_count": 0,
    }


def _asgi_request(method: str, path: str) -> tuple[int, dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    start = next(item for item in messages if item["type"] == "http.response.start")
    body = b"".join(
        item.get("body", b"")
        for item in messages
        if item["type"] == "http.response.body"
    )
    return int(start["status"]), json.loads(body.decode("utf-8"))


def test_operation_gate_allows_only_one_heavy_job_and_releases_on_error() -> None:
    coordinator = OperationCoordinator()

    with pytest.raises(RuntimeError):
        with coordinator.job("training"):
            with pytest.raises(ActiveOperationError) as conflict:
                with coordinator.job("analysis"):
                    pass
            assert str(conflict.value) == HEAVY_JOB_MESSAGE
            raise RuntimeError("training failed")

    with coordinator.job("prediction"):
        assert coordinator.snapshot()["prediction"] == 1
    assert coordinator.snapshot()["prediction"] == 0


def test_training_job_persists_compact_status_and_releases_gate(job_root: Path) -> None:
    manager, _, coordinator = _manager(job_root)
    job_id = new_training_job_id()
    input_path = manager.allocate_input_path(job_id)
    input_path.write_bytes(b"Y,Step1_R1\n90,1\n")

    def runner(progress):
        progress("Y 모델 학습", 55)
        return _compact_result()

    try:
        manager.submit(
            job_id=job_id,
            source_filename="training.csv",
            input_path=input_path,
            runner=runner,
        )
        row = _wait_for_terminal(manager, job_id)

        assert row["status"] == "completed"
        assert row["stage"] == "학습 완료"
        assert row["progress"] == 100
        assert row["result"] == _compact_result()
        assert row["error_message"] is None
        assert row["elapsed_seconds"] >= 0
        _wait_for_gate_release(coordinator)
        assert coordinator.snapshot()["training"] == 0
        assert not input_path.parent.exists()
    finally:
        manager.shutdown()


def test_failed_training_job_releases_gate_and_cleans_input(job_root: Path) -> None:
    manager, _, coordinator = _manager(job_root)
    job_id = new_training_job_id()
    input_path = manager.allocate_input_path(job_id)
    input_path.write_bytes(b"Y\n90\n")

    def fail(_progress):
        raise RuntimeError("internal secret detail")

    try:
        manager.submit(
            job_id=job_id,
            source_filename="training.csv",
            input_path=input_path,
            runner=fail,
        )
        row = _wait_for_terminal(manager, job_id)

        assert row["status"] == "failed"
        assert row["error_message"] == "모델 학습 중 서버 오류가 발생했습니다."
        assert "internal secret detail" not in row["error_message"]
        _wait_for_gate_release(coordinator)
        assert coordinator.snapshot()["training"] == 0
        assert not input_path.parent.exists()
    finally:
        manager.shutdown()


def test_startup_recovery_marks_running_job_interrupted(job_root: Path) -> None:
    manager, store, _ = _manager(job_root)
    job_id = new_training_job_id()
    input_path = manager.allocate_input_path(job_id)
    input_path.write_bytes(b"Y\n90\n")
    store.create_training_job(job_id, source_filename="training.csv")
    store.start_training_job(job_id)

    try:
        assert manager.recover_interrupted() == 1
        row = manager.get(job_id)
        assert row is not None
        assert row["status"] == "interrupted"
        assert row["progress"] == 5
        assert row["error_message"] == "서버가 재시작되어 학습이 중단되었습니다."
        assert not input_path.parent.exists()
        assert manager.recover_interrupted() == 0
    finally:
        manager.shutdown()


def test_startup_recovery_removes_only_allowlisted_orphan_inputs(job_root: Path) -> None:
    manager, _, _ = _manager(job_root)
    orphan_id = new_training_job_id()
    orphan_input = manager.allocate_input_path(orphan_id)
    orphan_input.write_bytes(b"Y\n90\n")
    unknown = manager.input_root / "keep-this-directory"
    unknown.mkdir()
    (unknown / "source.csv").write_bytes(b"keep")

    try:
        assert manager.recover_interrupted() == 0
        assert not orphan_input.parent.exists()
        assert (unknown / "source.csv").read_bytes() == b"keep"
    finally:
        manager.shutdown()


def test_train_job_api_returns_accepted_then_summary(
    job_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _ = _manager(job_root)
    monkeypatch.setattr(data_routes, "get_training_job_manager", lambda: manager)

    def fake_run(_path, _filename, _options, progress):
        progress("Y 모델 학습", 75)
        return _compact_result()

    monkeypatch.setattr(data_routes, "_run_persisted_training_job", fake_run)
    upload = UploadFile(
        file=BytesIO(b"Y,Step1_R1\n90,1\n"),
        filename="training.csv",
    )
    try:
        accepted = asyncio.run(data_routes.create_training_job(upload))
        assert accepted.status == "queued"
        route = next(
            route
            for route in data_routes.router.routes
            if getattr(route, "path", None) == "/api/train/jobs"
        )
        assert route.status_code == 202

        row = _wait_for_terminal(manager, accepted.job_id)
        response = data_routes.get_training_job(accepted.job_id)
        assert row["status"] == "completed"
        assert response.result is not None
        assert response.result.model_id == _compact_result()["model_id"]
        assert response.result.test_metrics is not None
        assert response.result.test_metrics.r2 == 0.8
    finally:
        manager.shutdown()


def test_train_job_api_runs_real_training_and_saves_model(
    job_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, coordinator = _manager(job_root)
    model_dir = job_root / "models"
    model_dir.mkdir()
    monkeypatch.setattr(data_routes, "get_training_job_manager", lambda: manager)
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_dir)
    fixture = Path(__file__).parent / "fixtures" / "training_sample.csv"
    source = pd.read_csv(fixture)
    frame = pd.concat([source] * 3, ignore_index=True)
    frame["Lot_Wafer_ID"] = [
        f"LOT{index // 5:02d}_WF{index % 5 + 1:02d}" for index in range(len(frame))
    ]
    frame["Lot_ID"] = frame["Lot_Wafer_ID"].str.extract(r"^(LOT\d+)", expand=False)
    total_failure = np.clip(100.0 - frame["Y"], 0.0, None)
    for index, weight in enumerate((0.10, 0.15, 0.20, 0.25, 0.30), 1):
        frame[f"Y{index}"] = total_failure * weight
    frame["Step1_Config"] = frame.pop("Step1_EQ")
    upload = UploadFile(
        file=BytesIO(frame.to_csv(index=False).encode("utf-8")),
        filename=fixture.name,
    )
    try:
        accepted = asyncio.run(data_routes.create_training_job(upload))
        row = _wait_for_terminal(manager, accepted.job_id, timeout=30)
        _wait_for_gate_release(coordinator)

        assert row["status"] == "completed", row.get("error_message")
        result = row["result"]
        assert result is not None
        assert result["model_id"]
        assert (model_dir / f'{result["model_id"]}.joblib').is_file()
        assert (model_dir / f'{result["model_id"]}.json').is_file()
        assert result["test_metrics"]["rmse"] is not None
    finally:
        manager.shutdown()


def test_train_job_api_returns_409_while_another_heavy_job_runs(
    job_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, coordinator = _manager(job_root)
    monkeypatch.setattr(data_routes, "get_training_job_manager", lambda: manager)
    upload = UploadFile(
        file=BytesIO(b"Y,Step1_R1\n90,1\n"),
        filename="training.csv",
    )
    try:
        with coordinator.job("analysis"):
            with pytest.raises(HTTPException) as conflict:
                asyncio.run(data_routes.create_training_job(upload))
        assert conflict.value.status_code == 409
        assert conflict.value.detail == HEAVY_JOB_MESSAGE
        input_root = job_root / "runtime" / "training_jobs"
        assert not list(input_root.iterdir())
    finally:
        manager.shutdown()


def test_health_is_independent_from_heavy_operation_gate() -> None:
    coordinator = OperationCoordinator()
    with coordinator.job("training"):
        assert asyncio.run(health_check()) == {"status": "ok"}


def test_http_health_stays_200_and_second_heavy_request_gets_exact_409() -> None:
    with operation_coordinator.job("training"):
        health_status, health_body = _asgi_request("GET", "/health")
        conflict_status, conflict_body = _asgi_request("POST", "/api/predict")

    assert health_status == 200
    assert health_body == {"status": "ok"}
    assert conflict_status == 409
    assert conflict_body == {"detail": HEAVY_JOB_MESSAGE}


def test_model_listing_stays_available_during_training_gate(
    job_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = job_root / "models"
    model_dir.mkdir()
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_dir)

    with operation_coordinator.job("training"):
        models_status, models_body = _asgi_request("GET", "/api/models")

    assert models_status == 200
    assert models_body == {
        "success": True,
        "models": [],
        "warnings": [],
        "total": 0,
    }


def test_missing_training_job_returns_404(
    job_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _ = _manager(job_root)
    monkeypatch.setattr(data_routes, "get_training_job_manager", lambda: manager)
    try:
        with pytest.raises(HTTPException) as missing:
            data_routes.get_training_job(new_training_job_id())
        assert missing.value.status_code == 404
    finally:
        manager.shutdown()


def test_predict_response_is_preview_but_history_artifact_keeps_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "Lot_Wafer_ID": f"LOT_{index}",
            "Lot_ID": "LOT",
            "predicted_Y": float(90 - index / 10),
            "risk_level": "normal",
            "confidence": "high",
        }
        for index in range(25)
    ]
    result = SimpleNamespace(
        model_id="model_1",
        target="Y",
        model_name="HGBR",
        total_rows=25,
        average_prediction=88.8,
        normal_count=25,
        warning_count=0,
        danger_count=0,
        evaluation=None,
        identifier_column="Lot_Wafer_ID",
        predictions=rows,
        warnings=[],
        truncated=False,
        preprocessing_summary={},
    )
    loaded = SimpleNamespace(metadata={"model_version": "1", "model_type": "HGBR"})
    captured: dict[str, Any] = {}

    async def fake_run(*_args, **_kwargs):
        return "prediction.csv", pd.DataFrame({"Y": range(25)}), loaded, result

    def fake_runtime(method: str, **values: Any):
        if method == "start_prediction":
            return values["prediction_id"]
        if method == "complete_prediction":
            captured.update(values)
            return True
        return True

    monkeypatch.setattr(data_routes, "_run_prediction", fake_run)
    monkeypatch.setattr(
        data_routes,
        "_latest_model",
        lambda: SimpleNamespace(model_id="model_1"),
    )
    monkeypatch.setattr(data_routes, "safe_runtime_call", fake_runtime)
    upload = UploadFile(file=BytesIO(b"Y\n90\n"), filename="prediction.csv")

    response = asyncio.run(data_routes.predict_csv(upload))

    assert len(response.predictions) == 10
    assert response.preview_row_count == 10
    assert response.truncated is True
    assert response.artifact_available is False
    assert captured == {}


def test_relationship_browser_snapshot_is_compact_without_mutating_artifact() -> None:
    wafer_rows = [
        {
            "identifier": f"W{index}",
            "direct_y": float(index),
            "derived_y": float(index),
            "ensemble_y": float(index),
            "direct_derived_gap": 0.0,
            "failure_rates": {},
            "fail_bit_counts": {},
        }
        for index in range(600)
    ]
    lots = [
        {
            "lot_id": f"L{lot_index}",
            "wafer_list": [
                {"identifier": f"L{lot_index}_W{wafer_index}"}
                for wafer_index in range(60)
            ],
        }
        for lot_index in range(7)
    ]
    full_lot = {"total_lot_count": 7, "lots": lots}
    full = {
        "analysis_id": "analysis_1",
        "multi_y": {
            "direct_y": list(range(600)),
            "derived_y": list(range(600)),
            "ensemble_y": list(range(600)),
            "failure_rates": {"Y1": list(range(600))},
            "fail_bit_counts": {"Y6": list(range(600))},
            "average_direct_y": 90.0,
            "average_derived_y": 89.0,
            "average_ensemble_y": 89.5,
            "ensemble_weight": 0.5,
            "failure_rate_averages": {"Y1": 1.0},
            "fail_bit_count_averages": {"Y6": 2.0},
            "wafer_results": wafer_rows,
        },
        "lot_analysis": full_lot,
        "lot_summary": [{"lot_id": f"L{index}"} for index in range(7)],
    }
    explanation = SimpleNamespace(
        wafer_explanations=[
            SimpleNamespace(identifier=f"W{index}")
            for index in range(0, 600, 2)
        ]
    )

    compact_lot = data_routes._compact_lot_analysis(full_lot)
    compact = data_routes._compact_analysis_result(
        full,
        explanation,
        compact_lot,
    )

    assert "direct_y" not in compact["multi_y"]
    assert len(compact["multi_y"]["wafer_results"]) == 300
    assert compact["multi_y"]["wafer_results_truncated"] is True
    assert len(compact["lot_analysis"]["lots"]) == 7
    assert all(
        len(lot["wafer_list"]) == 60
        for lot in compact["lot_analysis"]["lots"]
    )
    assert compact["lot_analysis"]["lot_list_truncated"] is False
    assert len(compact["lot_summary"]) == 5
    assert len(full["multi_y"]["direct_y"]) == 600
    assert len(full["multi_y"]["wafer_results"]) == 600
    assert len(full["lot_analysis"]["lots"]) == 7
    assert len(full["lot_analysis"]["lots"][0]["wafer_list"]) == 60
    assert len(full["lot_summary"]) == 7
