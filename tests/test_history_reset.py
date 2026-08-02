from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sklearn.linear_model import Ridge

from api.main import app
from api.routes import admin as admin_routes
from api.routes import data as data_routes
from api.routes import runtime as runtime_routes
from src.runtime.history_reset import (
    HistoryResetService,
    UnsafeResetPathError,
)
from src.runtime.operation_coordinator import OperationCoordinator
from src.runtime.store import RuntimeStore
from src.ml.model_io import save_model_bundle


RESET_BODY = {"confirmation": "RESET_ALL_HISTORY"}
ACTIVE_JOB_MESSAGE = (
    "현재 실행 중인 작업이 있습니다. 작업 완료 후 다시 시도해 주세요."
)


@dataclass
class ResetEnvironment:
    root: Path
    model_dir: Path
    runtime_dir: Path
    store: RuntimeStore
    coordinator: OperationCoordinator
    service: HistoryResetService


@pytest.fixture
def tmp_path() -> Any:
    """Workspace-local equivalent for Windows environments with locked TEMP ACLs."""
    parent = (Path(__file__).parent / ".history_reset_cases").resolve()
    parent.mkdir(exist_ok=True)
    root = (parent / uuid4().hex).resolve()
    if root.parent != parent:
        raise RuntimeError("테스트 임시 경로가 허용된 부모를 벗어났습니다.")
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


@pytest.fixture
def reset_environment(tmp_path: Path) -> ResetEnvironment:
    model_dir = tmp_path / "models"
    runtime_dir = tmp_path / "runtime"
    model_dir.mkdir()
    runtime_dir.mkdir()
    store = RuntimeStore(runtime_dir / "dashboard.db", runtime_dir)
    coordinator = OperationCoordinator()
    return ResetEnvironment(
        root=tmp_path,
        model_dir=model_dir,
        runtime_dir=runtime_dir,
        store=store,
        coordinator=coordinator,
        service=HistoryResetService(
            model_dir=model_dir,
            store=store,
            coordinator=coordinator,
        ),
    )


def _configure_api(
    monkeypatch: pytest.MonkeyPatch,
    environment: ResetEnvironment,
) -> None:
    monkeypatch.setattr(admin_routes, "_RESET_LIMITER", admin_routes.ResetRateLimiter())
    monkeypatch.setattr(
        admin_routes,
        "get_history_reset_service",
        lambda: environment.service,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", environment.model_dir)
    monkeypatch.setattr(
        runtime_routes,
        "get_runtime_store",
        lambda: environment.store,
    )


def _asgi_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    secret: str | None = None,
) -> tuple[int, Any]:
    body = (
        json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        if json_body is not None
        else b""
    )
    headers: list[tuple[bytes, bytes]] = [(b"host", b"testserver")]
    if json_body is not None:
        headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ]
        )
    if secret is not None:
        headers.append((b"x-admin-reset-secret", secret.encode("utf-8")))

    messages: list[dict[str, Any]] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
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
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    try:
        asyncio.run(app(scope, receive, send))
    except Exception:
        # Starlette sends the production 500 response and then re-raises so
        # test servers can optionally expose the original application error.
        if not any(message["type"] == "http.response.start" for message in messages):
            raise

    status_code = next(
        int(message["status"])
        for message in messages
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        bytes(message.get("body", b""))
        for message in messages
        if message["type"] == "http.response.body"
    )
    if not response_body:
        return status_code, None
    try:
        return status_code, json.loads(response_body)
    except json.JSONDecodeError:
        return status_code, response_body.decode("utf-8", errors="replace")


def _write_flat_model(
    environment: ResetEnvironment,
    model_id: str = "flat_model",
    *,
    metadata: str = '{"model_id":"flat_model"}',
) -> tuple[Path, Path]:
    bundle = environment.model_dir / f"{model_id}.joblib"
    metadata_path = environment.model_dir / f"{model_id}.json"
    bundle.write_bytes(b"test model bundle")
    metadata_path.write_text(metadata, encoding="utf-8")
    return bundle, metadata_path


def _write_hybrid_model(
    environment: ResetEnvironment,
    model_id: str = "hybrid_model",
) -> Path:
    directory = environment.model_dir / model_id
    directory.mkdir()
    (directory / "bundle.joblib").write_bytes(b"hybrid bundle")
    (directory / "metadata.json").write_text(
        '{"model_id":"hybrid_model"}', encoding="utf-8"
    )
    (directory / "oof_predictions.json.gz").write_bytes(b"oof")
    (directory / "fold_assignments.json.gz").write_bytes(b"folds")
    return directory


def _add_prediction(
    environment: ResetEnvironment,
    prediction_id: str = "prediction_reset_test",
    *,
    complete: bool = True,
) -> Path:
    environment.store.start_prediction(
        prediction_id=prediction_id,
        source_filename="source.csv",
        model_id="flat_model",
        warning_threshold=90.0,
        critical_threshold=85.0,
    )
    artifact = (
        environment.runtime_dir / "predictions" / f"{prediction_id}.json.gz"
    )
    if complete:
        environment.store.complete_prediction(
            prediction_id,
            metadata={
                "duration_ms": 10.0,
                "dataset_fingerprint": "prediction-fingerprint",
                "model_name_snapshot": "Flat Model",
                "model_version_snapshot": "v1",
                "model_type_snapshot": "single",
                "schema_version": "v2",
                "row_count": 1,
                "lot_count": 1,
                "final_strategy": "direct",
            },
            summary={"average_predicted_yield": 91.0},
            preprocessing={},
            artifact={"rows": [{"Lot_Wafer_ID": "LOT01_W01"}]},
            warnings=[],
        )
    return artifact


def _add_analysis(
    environment: ResetEnvironment,
    analysis_id: str = "analysis_reset_test",
    *,
    complete: bool = True,
) -> Path:
    environment.store.start_analysis(
        analysis_id=analysis_id,
        prediction_id=None,
        source_filename="source.csv",
        model_id="flat_model",
    )
    artifact = environment.runtime_dir / "analyses" / f"{analysis_id}.json.gz"
    if complete:
        environment.store.complete_analysis(
            analysis_id,
            metadata={
                "duration_ms": 20.0,
                "dataset_fingerprint": "analysis-fingerprint",
                "model_name_snapshot": "Flat Model",
                "model_version_snapshot": "v1",
                "model_type_snapshot": "single",
                "schema_version": "v2",
                "row_count": 1,
                "lot_count": 1,
                "available_targets_json": '["Y1", "Y2", "Y3", "Y4", "Y5"]',
                "default_target": "Y1",
            },
            summary={"critical_count": 1},
            methodology={"analysis_unit": "wafer"},
            artifact={"analysis_result": {"risk": {"critical_count": 1}}},
            warnings=[],
        )
    return artifact


def _delete_history() -> tuple[int, Any]:
    return _asgi_request("POST", "/api/admin/history/reset", json_body=RESET_BODY)


def test_full_reset_deletes_all_scoped_data_and_empty_apis(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = reset_environment
    _configure_api(monkeypatch, environment)
    _write_flat_model(environment)
    _write_hybrid_model(environment)
    prediction_artifact = _add_prediction(environment)
    analysis_artifact = _add_analysis(environment)
    source_csv = environment.root / "source.csv"
    source_csv.write_text("Y\n95\n", encoding="utf-8")

    summary_status, summary = _asgi_request(
        "GET", "/api/admin/history/reset/summary"
    )
    assert summary_status == 200
    assert summary == {
        "model_count": 2,
        "prediction_history_count": 1,
        "analysis_history_count": 1,
        "model_artifact_count": 6,
        "prediction_artifact_count": 1,
        "analysis_artifact_count": 1,
    }

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload == {
        "success": True,
        "deleted": {
            "models": 2,
            "model_files": 6,
            "prediction_histories": 1,
            "prediction_artifacts": 1,
            "analysis_histories": 1,
            "analysis_artifacts": 1,
        },
        "preserved": {
            "alert_logs": True,
            "automation_runs": True,
            "source_csv": True,
        },
    }
    assert source_csv.is_file()
    assert not prediction_artifact.exists()
    assert not analysis_artifact.exists()

    models_status, models = _asgi_request("GET", "/api/models", secret=None)
    predictions_status, predictions = _asgi_request(
        "GET", "/api/predictions/history", secret=None
    )
    analyses_status, analyses = _asgi_request(
        "GET", "/api/analyses/history", secret=None
    )
    overview_status, overview = _asgi_request(
        "GET", "/api/dashboard/overview", secret=None
    )
    assert models_status == predictions_status == analyses_status == overview_status == 200
    assert models["models"] == []
    assert models["total"] == 0
    assert predictions["items"] == [] and predictions["total"] == 0
    assert analyses["items"] == [] and analyses["total"] == 0
    assert overview["source"]["status"] == "empty"
    assert overview["source"]["analysis_id"] is None


def test_real_model_and_histories_are_visible_then_empty_after_reset(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = reset_environment
    _configure_api(monkeypatch, environment)
    fitted_model = Ridge(alpha=1.0).fit(
        [[1.0], [2.0], [3.0]],
        [91.0, 92.0, 93.0],
    )
    _, _, metadata = save_model_bundle(
        fitted_model,
        target="Y",
        model_name="Isolated Reset Verification",
        feature_columns=["Step1_R1"],
        metrics={
            "test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0},
        },
        random_state=42,
        split_method="group",
        model_dir=environment.model_dir,
    )
    prediction_artifact = _add_prediction(environment)
    analysis_artifact = _add_analysis(environment)

    before_models_status, before_models = _asgi_request(
        "GET", "/api/models", secret=None
    )
    before_predictions_status, before_predictions = _asgi_request(
        "GET", "/api/predictions/history", secret=None
    )
    before_analyses_status, before_analyses = _asgi_request(
        "GET", "/api/analyses/history", secret=None
    )
    assert (
        before_models_status
        == before_predictions_status
        == before_analyses_status
        == 200
    )
    assert before_models["total"] == 1
    assert before_models["models"][0]["model_id"] == metadata["model_id"]
    assert before_predictions["total"] == 1
    assert before_analyses["total"] == 1
    assert prediction_artifact.is_file()
    assert analysis_artifact.is_file()

    reset_status, reset_payload = _delete_history()

    assert reset_status == 200
    assert reset_payload["deleted"] == {
        "models": 1,
        "model_files": 2,
        "prediction_histories": 1,
        "prediction_artifacts": 1,
        "analysis_histories": 1,
        "analysis_artifacts": 1,
    }

    after_models_status, after_models = _asgi_request(
        "GET", "/api/models", secret=None
    )
    after_predictions_status, after_predictions = _asgi_request(
        "GET", "/api/predictions/history", secret=None
    )
    after_analyses_status, after_analyses = _asgi_request(
        "GET", "/api/analyses/history", secret=None
    )
    overview_status, overview = _asgi_request(
        "GET", "/api/dashboard/overview", secret=None
    )
    assert (
        after_models_status
        == after_predictions_status
        == after_analyses_status
        == overview_status
        == 200
    )
    assert after_models["models"] == [] and after_models["total"] == 0
    assert (
        after_predictions["items"] == []
        and after_predictions["total"] == 0
    )
    assert after_analyses["items"] == [] and after_analyses["total"] == 0
    assert overview["source"]["status"] == "empty"


def test_missing_confirmation_returns_400_without_mutation(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, metadata = _write_flat_model(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _asgi_request("POST", "/api/admin/history/reset")

    assert status_code == 400
    assert payload["detail"] == "초기화 확인값이 올바르지 않습니다."
    assert bundle.is_file() and metadata.is_file()


def test_wrong_confirmation_returns_400_without_mutation(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, metadata = _write_flat_model(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _asgi_request(
        "POST",
        "/api/admin/history/reset",
        json_body={"confirmation": "WRONG"},
    )

    assert status_code == 400
    assert payload["detail"] == "초기화 확인값이 올바르지 않습니다."
    assert bundle.is_file() and metadata.is_file()


def test_reset_requires_no_secret_header(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _write_flat_model(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _asgi_request(
        "POST", "/api/admin/history/reset", json_body=RESET_BODY, secret=None
    )

    assert status_code == 200
    assert payload["success"] is True
    assert not bundle.exists()


def test_same_ip_rate_limit_returns_429(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, reset_environment)
    for _ in range(3):
        status_code, _ = _asgi_request(
            "POST",
            "/api/admin/history/reset",
            json_body={"confirmation": "WRONG"},
        )
        assert status_code == 400

    status_code, payload = _asgi_request(
        "POST",
        "/api/admin/history/reset",
        json_body=RESET_BODY,
    )
    assert status_code == 429
    assert "너무 많" in payload["detail"]


def test_active_training_returns_409_without_mutation(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _write_flat_model(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    with reset_environment.coordinator.job("training"):
        status_code, payload = _delete_history()

    assert status_code == 409
    assert payload["detail"] == ACTIVE_JOB_MESSAGE
    assert bundle.is_file()


def test_active_prediction_returns_409_without_mutation(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _write_flat_model(reset_environment)
    _add_prediction(reset_environment, complete=False)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 409
    assert payload["detail"] == ACTIVE_JOB_MESSAGE
    assert bundle.is_file()
    assert reset_environment.store.history_reset_counts()[
        "prediction_history_count"
    ] == 1


def test_active_analysis_returns_409_without_mutation(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _write_flat_model(reset_environment)
    _add_analysis(reset_environment, complete=False)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 409
    assert payload["detail"] == ACTIVE_JOB_MESSAGE
    assert bundle.is_file()
    assert reset_environment.store.history_reset_counts()[
        "analysis_history_count"
    ] == 1


def test_model_only_reset(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, metadata = _write_flat_model(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload["deleted"]["models"] == 1
    assert payload["deleted"]["model_files"] == 2
    assert payload["deleted"]["prediction_histories"] == 0
    assert payload["deleted"]["analysis_histories"] == 0
    assert not bundle.exists() and not metadata.exists()


def test_prediction_only_reset(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _add_prediction(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload["deleted"]["models"] == 0
    assert payload["deleted"]["prediction_histories"] == 1
    assert payload["deleted"]["prediction_artifacts"] == 1
    assert reset_environment.store.list_predictions({})["total"] == 0
    assert not artifact.exists()


def test_analysis_only_reset(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _add_analysis(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload["deleted"]["models"] == 0
    assert payload["deleted"]["analysis_histories"] == 1
    assert payload["deleted"]["analysis_artifacts"] == 1
    assert reset_environment.store.list_analyses({})["total"] == 0
    assert not artifact.exists()


def test_empty_reset_is_successful_and_idempotent(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload["success"] is True
    assert all(value == 0 for value in payload["deleted"].values())


def test_missing_artifact_does_not_block_history_reset(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _add_prediction(reset_environment)
    artifact.unlink()
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload["deleted"]["prediction_histories"] == 1
    assert payload["deleted"]["prediction_artifacts"] == 0
    assert reset_environment.store.list_predictions({})["total"] == 0


def test_corrupt_model_metadata_is_still_removed(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, metadata = _write_flat_model(
        reset_environment,
        model_id="corrupt_model",
        metadata="{not valid json",
    )
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload["deleted"]["models"] == 1
    assert payload["deleted"]["model_files"] == 2
    assert not bundle.exists() and not metadata.exists()


def test_unrelated_files_and_source_csv_are_preserved(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_flat_model(reset_environment)
    model_readme = reset_environment.model_dir / "README.txt"
    runtime_note = reset_environment.runtime_dir / "predictions" / "keep.csv"
    source_csv = reset_environment.root / "original-source.csv"
    model_readme.write_text("keep model note", encoding="utf-8")
    runtime_note.parent.mkdir(exist_ok=True)
    runtime_note.write_text("keep runtime note", encoding="utf-8")
    source_csv.write_text("Y\n99\n", encoding="utf-8")
    _configure_api(monkeypatch, reset_environment)

    status_code, _ = _delete_history()

    assert status_code == 200
    assert model_readme.read_text(encoding="utf-8") == "keep model note"
    assert runtime_note.read_text(encoding="utf-8") == "keep runtime note"
    assert source_csv.read_text(encoding="utf-8") == "Y\n99\n"


def test_alert_logs_are_preserved(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_environment.store.record_prediction_alerts(
        analysis_id="analysis_preserved_alert",
        model_id="flat_model",
        model_version="v1",
        predictions=[
            {
                "Lot_Wafer_ID": "LOT01_W01",
                "predicted_Y": 80.0,
                "risk_level": "danger",
            }
        ],
        identifier_column="Lot_Wafer_ID",
    )
    _add_analysis(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, _ = _delete_history()

    assert status_code == 200
    assert reset_environment.store.list_alerts({"limit": 10})["total"] == 1


def test_automation_runs_are_preserved(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_environment.store.record_run(
        event_type="analyze",
        model_id="flat_model",
        status="success",
    )
    _add_prediction(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    status_code, _ = _delete_history()

    assert status_code == 200
    runs = reset_environment.store.list_runs()
    assert len(runs) == 1
    assert runs[0]["event_type"] == "analyze"


def test_two_consecutive_resets_are_both_successful(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_flat_model(reset_environment)
    _add_prediction(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    first_status, first = _delete_history()
    second_status, second = _delete_history()

    assert first_status == second_status == 200
    assert first["deleted"]["models"] == 1
    assert first["deleted"]["prediction_histories"] == 1
    assert all(value == 0 for value in second["deleted"].values())


def test_symlink_model_is_rejected_without_touching_target(
    reset_environment: ResetEnvironment,
) -> None:
    outside = reset_environment.root / "outside-model.joblib"
    outside.write_bytes(b"must remain")
    linked = reset_environment.model_dir / "linked_model.joblib"
    try:
        linked.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"현재 환경에서 심볼릭 링크를 만들 수 없습니다: {exc}")

    with pytest.raises(UnsafeResetPathError):
        reset_environment.service.reset()

    assert linked.is_symlink()
    assert outside.read_bytes() == b"must remain"


def test_traversal_like_paths_are_never_deletion_targets(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_model_name = reset_environment.model_dir / "bad..model.joblib"
    unsafe_model_name.write_bytes(b"must remain")
    outside_artifact = reset_environment.root / "outside-prediction.json.gz"
    outside_artifact.write_bytes(b"must remain")
    prediction_id = "prediction_outside_path"
    reset_environment.store.start_prediction(
        prediction_id=prediction_id,
        source_filename="source.csv",
        model_id="flat_model",
    )
    reset_environment.store.fail_prediction(prediction_id, "fixture")
    with reset_environment.store._connect() as connection:
        connection.execute(
            "UPDATE prediction_runs SET artifact_path=? WHERE prediction_id=?",
            (str(outside_artifact), prediction_id),
        )
    _configure_api(monkeypatch, reset_environment)

    status_code, payload = _delete_history()

    assert status_code == 200
    assert payload["deleted"]["models"] == 0
    assert payload["deleted"]["prediction_histories"] == 1
    assert unsafe_model_name.read_bytes() == b"must remain"
    assert outside_artifact.read_bytes() == b"must remain"


def test_database_failure_returns_500_and_restores_files_and_rows(
    reset_environment: ResetEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, metadata = _write_flat_model(reset_environment)
    prediction_artifact = _add_prediction(reset_environment)
    analysis_artifact = _add_analysis(reset_environment)
    _configure_api(monkeypatch, reset_environment)

    def fail_after_deletes(connection: sqlite3.Connection) -> dict[str, int]:
        connection.execute("DELETE FROM analysis_runs")
        connection.execute("DELETE FROM prediction_runs")
        raise sqlite3.OperationalError("forced transaction failure")

    monkeypatch.setattr(
        reset_environment.store,
        "delete_reset_history_rows",
        fail_after_deletes,
    )

    status_code, payload = _delete_history()

    assert status_code == 500
    assert payload == {"detail": "이력 초기화 중 서버 오류가 발생했습니다."}
    assert bundle.is_file() and metadata.is_file()
    assert prediction_artifact.is_file() and analysis_artifact.is_file()
    assert reset_environment.store.history_reset_counts() == {
        "prediction_history_count": 1,
        "analysis_history_count": 1,
    }
    assert not (reset_environment.model_dir / ".history-reset").exists()
    assert not (reset_environment.runtime_dir / ".history-reset").exists()
