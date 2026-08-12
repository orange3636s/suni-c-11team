from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import gzip
import json
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.linear_model import Ridge

from api.main import app
from api.routes import data as data_routes
from src.ml import inference as inference_module
from src.ml.inference import (
    InvalidModelIdError,
    ModelDeletionError,
    ModelNotFoundError,
    delete_model_bundle,
    list_prediction_models,
)
from src.ml.model_io import save_model_bundle
from src.runtime.store import RuntimeStore


def _asgi_request(method: str, path: str) -> tuple[int, bytes]:
    messages: list[dict[str, object]] = []
    request_sent = False

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
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
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    status_code = next(
        int(message["status"])
        for message in messages
        if message["type"] == "http.response.start"
    )
    body = b"".join(
        bytes(message.get("body", b""))
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status_code, body


def _seed_run(store: RuntimeStore, *, event_type: str, model_id: str, status: str = "success") -> None:
    """Direct `runs` table insert standing in for the now-removed
    `RuntimeStore.record_run` -- that method had zero production callers
    (nothing writes prediction/analysis history anymore) but was the only
    seeding path for these deletion tests, which exercise the still-live
    `model_reference_counts`/deletion-history-preservation behavior."""
    now = datetime.now(timezone.utc).isoformat()
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO runs VALUES (:run_id,:event_type,:model_id,:started_at,:completed_at,:duration_ms,:row_count,:status,:error_type,:critical_count,:warning_count,:schema_version,:filename)",
            {
                "run_id": f"run_{uuid4().hex}",
                "event_type": event_type,
                "model_id": model_id,
                "started_at": now,
                "completed_at": now,
                "duration_ms": 0.0,
                "row_count": None,
                "status": status,
                "error_type": None,
                "critical_count": None,
                "warning_count": None,
                "schema_version": None,
                "filename": None,
            },
        )


@pytest.fixture(scope="module")
def real_ridge_model() -> Ridge:
    dataframe = pd.read_csv(Path(__file__).parent / "fixtures" / "training_sample.csv")
    return Ridge(alpha=1.0).fit(dataframe[["Step1_R1"]], dataframe["Y"])


def test_model_delete_openapi_contract_is_delete_only() -> None:
    operations = app.openapi()["paths"]["/api/models/{model_id}"]

    assert "delete" in operations
    assert "post" not in operations
    assert "put" not in operations
    assert set(operations["delete"]["responses"]) >= {"200", "400", "404", "500"}


def test_model_deletion_preserves_runtime_history(real_ridge_model: Ridge) -> None:
    root = Path(__file__).parent / ".tmp_model_deletion" / uuid4().hex
    root.mkdir(parents=True)
    model_path, metadata_path, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="Deletion Test",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=root,
    )
    store = RuntimeStore(root / "runtime.db")
    _seed_run(store, event_type="predict", model_id=metadata["model_id"])
    assert store.model_reference_counts(metadata["model_id"])["prediction_history_count"] == 1

    removed = delete_model_bundle(metadata["model_id"], root).deleted_files

    assert model_path.name in removed
    assert metadata_path.name in removed
    assert not model_path.exists()
    assert store.model_reference_counts(metadata["model_id"])["prediction_history_count"] == 1
    models, _ = list_prediction_models(root)
    assert models == []

    (root / "runtime.db").unlink()
    root.rmdir()
    root.parent.rmdir()


def _record_history(store: RuntimeStore, model_id: str) -> None:
    _seed_run(store, event_type="predict", model_id=model_id)
    _seed_run(store, event_type="analyze", model_id=model_id)


@pytest.fixture
def deletion_root() -> Path:
    parent = Path(__file__).parent / ".tmp_model_delete_api"
    root = parent / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


@pytest.mark.parametrize(
    ("model_kind", "bundle_layout"),
    [
        ("legacy", "legacy"),
        ("single", "legacy"),
        ("ensemble", "legacy"),
        ("hybrid_multi_y", "hybrid"),
    ],
)
def test_delete_api_uses_model_id_and_preserves_history(
    deletion_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_ridge_model: Ridge,
    model_kind: str,
    bundle_layout: str,
) -> None:
    model_root = deletion_root / "models"
    model_path, metadata_path, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="Deletion Test",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    model_id = metadata["model_id"]
    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
    stored["model_type"] = model_kind
    metadata_path.write_text(json.dumps(stored), encoding="utf-8")
    if bundle_layout == "hybrid":
        bundle_dir = model_root / model_id
        bundle_dir.mkdir()
        shutil.move(model_path, bundle_dir / "bundle.joblib")
        shutil.move(metadata_path, bundle_dir / "metadata.json")
        with gzip.open(bundle_dir / "oof_predictions.json.gz", "wt", encoding="utf-8") as handle:
            json.dump({"Y": [90.0, 91.0]}, handle)
        with gzip.open(bundle_dir / "fold_assignments.json.gz", "wt", encoding="utf-8") as handle:
            json.dump([0, 1], handle)
        metadata_path = bundle_dir / "metadata.json"
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored["model_type"] = "hybrid_multi_y"
        stored["bundle_type"] = "hybrid_multi_y"
        metadata_path.write_text(json.dumps(stored), encoding="utf-8")

    runtime_db = deletion_root / "runtime" / "dashboard.db"
    artifact_root = deletion_root / "runtime"
    store = RuntimeStore(runtime_db, artifact_root)
    _record_history(store, model_id)
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_db,
        runtime_artifact_dir=artifact_root,
        model_dir=model_root,
        max_prediction_history=100,
        max_analysis_history=50,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_root)
    monkeypatch.setattr(data_routes, "settings", test_settings)

    listed_before = data_routes.get_models()
    assert model_id in {item.model_id for item in listed_before.models}

    deleted = data_routes.delete_model(model_id)
    expected_deleted = (
        {
            f"{model_id}/bundle.joblib",
            f"{model_id}/metadata.json",
            f"{model_id}/oof_predictions.json.gz",
            f"{model_id}/fold_assignments.json.gz",
        }
        if bundle_layout == "hybrid"
        else {model_path.name, metadata_path.name}
    )
    assert deleted.deleted is True
    assert deleted.model_id == model_id
    assert set(deleted.deleted_files) == expected_deleted
    assert deleted.missing_files == []
    assert deleted.metadata_deleted is True
    assert deleted.bundle_deleted is True
    assert deleted.removed_files == deleted.deleted_files
    assert deleted.registry_removed is True
    assert deleted.prediction_history_kept is True
    assert deleted.analysis_history_kept is True
    assert deleted.prediction_history_count == 1
    assert deleted.analysis_history_count == 1
    assert not (model_root / ".deleting").exists()

    listed_after = data_routes.get_models()
    assert model_id not in {item.model_id for item in listed_after.models}
    with pytest.raises(HTTPException) as missing_detail:
        data_routes.get_model_detail(model_id)
    assert missing_detail.value.status_code == 404
    with pytest.raises(HTTPException) as already_deleted:
        data_routes.delete_model(model_id)
    assert already_deleted.value.status_code == 404


@pytest.mark.parametrize("remaining_kind", ["metadata", "bundle"])
def test_delete_api_cleans_incomplete_flat_model(
    deletion_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_ridge_model: Ridge,
    remaining_kind: str,
) -> None:
    model_root = deletion_root / "models"
    model_path, metadata_path, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name=f"Incomplete {remaining_kind}",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    if remaining_kind == "metadata":
        model_path.unlink()
        expected_file = metadata_path.name
    else:
        metadata_path.unlink()
        expected_file = model_path.name
    runtime_root = deletion_root / "runtime"
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_root / "dashboard.db",
        runtime_artifact_dir=runtime_root,
        model_dir=model_root,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_root)
    monkeypatch.setattr(data_routes, "settings", test_settings)

    response = data_routes.delete_model(metadata["model_id"])

    assert response.deleted_files == [expected_file]
    expected_missing = (
        f"{metadata['model_id']}.joblib"
        if remaining_kind == "metadata"
        else f"{metadata['model_id']}.json"
    )
    assert response.missing_files == [expected_missing]
    assert response.metadata_deleted is (remaining_kind == "metadata")
    assert response.bundle_deleted is (remaining_kind == "bundle")
    assert not model_path.exists()
    assert not metadata_path.exists()


@pytest.mark.parametrize("remaining_kind", ["metadata", "bundle"])
def test_delete_api_cleans_incomplete_hybrid_bundle(
    deletion_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_ridge_model: Ridge,
    remaining_kind: str,
) -> None:
    model_root = deletion_root / "models"
    model_path, metadata_path, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name=f"Incomplete Hybrid {remaining_kind}",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    bundle_dir = model_root / metadata["model_id"]
    bundle_dir.mkdir()
    if remaining_kind == "metadata":
        model_path.unlink()
        shutil.move(metadata_path, bundle_dir / "metadata.json")
        expected_file = f"{metadata['model_id']}/metadata.json"
    else:
        metadata_path.unlink()
        shutil.move(model_path, bundle_dir / "bundle.joblib")
        expected_file = f"{metadata['model_id']}/bundle.joblib"
    runtime_root = deletion_root / "runtime"
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_root / "dashboard.db",
        runtime_artifact_dir=runtime_root,
        model_dir=model_root,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_root)
    monkeypatch.setattr(data_routes, "settings", test_settings)

    response = data_routes.delete_model(metadata["model_id"])

    assert response.deleted_files == [expected_file]
    expected_missing = {
        f"{metadata['model_id']}/{name}"
        for name in (
            "bundle.joblib",
            "metadata.json",
            "oof_predictions.json.gz",
            "fold_assignments.json.gz",
        )
    } - {expected_file}
    assert set(response.missing_files) == expected_missing
    assert response.metadata_deleted is (remaining_kind == "metadata")
    assert response.bundle_deleted is (remaining_kind == "bundle")
    assert not bundle_dir.exists()


def test_delete_http_endpoint_returns_200_and_detail_becomes_404(
    deletion_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_ridge_model: Ridge,
) -> None:
    model_root = deletion_root / "models"
    model_path, metadata_path, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="HTTP Route Deletion",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    runtime_root = deletion_root / "runtime"
    runtime_db = runtime_root / "dashboard.db"
    store = RuntimeStore(runtime_db, runtime_root)
    _record_history(store, metadata["model_id"])
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_db,
        runtime_artifact_dir=runtime_root,
        model_dir=model_root,
        max_prediction_history=100,
        max_analysis_history=50,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_root)
    monkeypatch.setattr(data_routes, "settings", test_settings)

    path = f"/api/models/{metadata['model_id']}"
    list_before_status, list_before_body = _asgi_request("GET", "/api/models")
    response_status, response_body = _asgi_request("DELETE", path)
    list_after_status, list_after_body = _asgi_request("GET", "/api/models")
    detail_status, _ = _asgi_request("GET", path)
    duplicate_status, _ = _asgi_request("DELETE", path)

    assert list_before_status == 200
    assert metadata["model_id"] in {
        item["model_id"] for item in json.loads(list_before_body)["models"]
    }
    assert response_status == 200
    payload = json.loads(response_body)
    assert payload["deleted"] is True
    assert payload["model_id"] == metadata["model_id"]
    assert set(payload["deleted_files"]) == {model_path.name, metadata_path.name}
    assert payload["missing_files"] == []
    assert payload["metadata_deleted"] is True
    assert payload["bundle_deleted"] is True
    assert payload["prediction_history_kept"] is True
    assert payload["analysis_history_kept"] is True
    assert list_after_status == 200
    assert metadata["model_id"] not in {
        item["model_id"] for item in json.loads(list_after_body)["models"]
    }
    assert detail_status == 404
    assert duplicate_status == 404


def test_delete_rejects_flat_symlink_without_touching_target(
    deletion_root: Path,
) -> None:
    model_root = deletion_root / "models"
    model_root.mkdir()
    outside_target = deletion_root / "outside-model.joblib"
    outside_target.write_bytes(b"must remain")
    model_id = "linked_model"
    linked_model = model_root / f"{model_id}.joblib"
    try:
        linked_model.symlink_to(outside_target)
    except OSError as exc:
        pytest.skip(f"현재 환경에서 심볼릭 링크를 만들 수 없습니다: {exc}")

    with pytest.raises(ModelDeletionError):
        delete_model_bundle(model_id, model_root)

    assert linked_model.is_symlink()
    assert outside_target.read_bytes() == b"must remain"


def test_delete_maps_resolved_path_escape_to_deletion_error(
    deletion_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H-1: 심볼릭 링크 생성 권한이 없는 환경(예: 관리자 권한 없는
    Windows)에서도, `_model_paths`가 (평소 심볼릭 링크가 root 밖으로
    resolve될 때 던지는) `InvalidModelIdError`를 삭제 경로가
    `ModelDeletionError`로 다시 던지는지 심볼릭 링크 없이 확인한다 --
    `test_delete_rejects_flat_symlink_without_touching_target`은 이
    환경에서 skip되므로 별도로 검증이 필요하다."""
    model_root = deletion_root / "models"
    model_root.mkdir()

    def _raise_invalid(model_id: str, model_dir: object) -> tuple[Path, Path]:
        raise InvalidModelIdError("유효하지 않은 모델 ID입니다.")

    monkeypatch.setattr(inference_module, "_model_paths", _raise_invalid)

    with pytest.raises(ModelDeletionError):
        delete_model_bundle("some_model", model_root)


def test_delete_rejects_bundle_junction_without_touching_target(
    deletion_root: Path,
) -> None:
    winapi = pytest.importorskip("_winapi")
    model_root = deletion_root / "models"
    model_root.mkdir()
    outside_bundle = deletion_root / "outside-bundle"
    outside_bundle.mkdir()
    outside_metadata = outside_bundle / "metadata.json"
    outside_metadata.write_text('{"must": "remain"}', encoding="utf-8")
    model_id = "junction_model"
    linked_bundle = model_root / model_id
    winapi.CreateJunction(str(outside_bundle), str(linked_bundle))
    try:
        assert linked_bundle.is_junction()
        with pytest.raises(ModelDeletionError):
            delete_model_bundle(model_id, model_root)
        assert outside_metadata.read_text(encoding="utf-8") == '{"must": "remain"}'
    finally:
        linked_bundle.rmdir()


def test_delete_api_preserves_other_model_and_rejects_path_traversal(
    deletion_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_ridge_model: Ridge,
) -> None:
    model_root = deletion_root / "models"
    target_model, target_metadata, target = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="Delete Target",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    sentinel_model, sentinel_metadata, _ = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="Keep Sentinel",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    runtime_root = deletion_root / "runtime"
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_root / "dashboard.db",
        runtime_artifact_dir=runtime_root,
        model_dir=model_root,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_root)
    monkeypatch.setattr(data_routes, "settings", test_settings)

    data_routes.delete_model(target["model_id"])

    assert not target_model.exists()
    assert not target_metadata.exists()
    assert sentinel_model.is_file()
    assert sentinel_metadata.is_file()
    with pytest.raises(HTTPException) as invalid:
        data_routes.delete_model("..\\outside")
    assert invalid.value.status_code == 400
    with pytest.raises(HTTPException) as missing:
        data_routes.delete_model("valid_but_missing")
    assert missing.value.status_code == 404


def test_delete_route_reports_partial_failure_and_recovers_on_retry(
    deletion_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_ridge_model: Ridge,
) -> None:
    model_root = deletion_root / "models"
    model_path, metadata_path, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="HTTP Delete",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    runtime_root = deletion_root / "runtime"
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_root / "dashboard.db",
        runtime_artifact_dir=runtime_root,
        model_dir=model_root,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_root)
    monkeypatch.setattr(data_routes, "settings", test_settings)

    original_unlink = Path.unlink

    def fail_metadata_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == metadata_path.name and ".deleting" in path.parts:
            raise PermissionError("test permission error")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_metadata_unlink)
    with pytest.raises(HTTPException) as failed:
        data_routes.delete_model(metadata["model_id"])
    assert failed.value.status_code == 500
    assert metadata_path.name in " ".join(failed.value.detail["errors"])
    assert not model_path.exists()
    assert not metadata_path.exists()
    assert (model_root / ".deleting" / metadata["model_id"] / metadata_path.name).is_file()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = data_routes.delete_model(metadata["model_id"])
    assert recovered.deleted is True
    with pytest.raises(HTTPException) as detail_after:
        data_routes.get_model_detail(metadata["model_id"])
    assert detail_after.value.status_code == 404
    assert not (model_root / ".deleting").exists()


def test_empty_staging_directory_is_cleaned_and_returns_not_found(
    deletion_root: Path,
) -> None:
    model_root = deletion_root / "models"
    model_id = "empty_staged_model"
    (model_root / ".deleting" / model_id).mkdir(parents=True)

    with pytest.raises(ModelNotFoundError):
        delete_model_bundle(model_id, model_root)

    assert not (model_root / ".deleting").exists()


def test_split_flat_staging_is_recovered_before_delete(
    deletion_root: Path,
    real_ridge_model: Ridge,
) -> None:
    model_root = deletion_root / "models"
    model_path, metadata_path, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="Split Staging Recovery",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )
    staged_dir = model_root / ".deleting" / metadata["model_id"]
    staged_dir.mkdir(parents=True)
    shutil.move(metadata_path, staged_dir / metadata_path.name)

    deleted = delete_model_bundle(metadata["model_id"], model_root).deleted_files

    assert set(deleted) == {model_path.name, metadata_path.name}
    assert not model_path.exists()
    assert not metadata_path.exists()
    assert not (model_root / ".deleting").exists()


def test_unknown_hybrid_artifact_blocks_delete_without_mutation(
    deletion_root: Path,
) -> None:
    model_root = deletion_root / "models"
    model_id = "hybrid_with_unknown_artifact"
    bundle_dir = model_root / model_id
    bundle_dir.mkdir(parents=True)
    bundle_path = bundle_dir / "bundle.joblib"
    metadata_path = bundle_dir / "metadata.json"
    unknown_path = bundle_dir / "unexpected.bin"
    bundle_path.write_bytes(b"bundle")
    metadata_path.write_text("{}", encoding="utf-8")
    unknown_path.write_bytes(b"keep")

    with pytest.raises(ModelDeletionError):
        delete_model_bundle(model_id, model_root)

    assert bundle_path.is_file()
    assert metadata_path.is_file()
    assert unknown_path.is_file()
    assert not (model_root / ".deleting").exists()


def test_concurrent_delete_same_model_returns_success_and_not_found(
    deletion_root: Path,
    real_ridge_model: Ridge,
) -> None:
    model_root = deletion_root / "models"
    _, _, metadata = save_model_bundle(
        real_ridge_model,
        target="Y",
        model_name="Concurrent Delete",
        feature_columns=["Step1_R1"],
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_root,
    )

    def attempt_delete() -> str:
        try:
            delete_model_bundle(metadata["model_id"], model_root)
            return "deleted"
        except ModelNotFoundError:
            return "not_found"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: attempt_delete(), range(2)))

    assert sorted(outcomes) == ["deleted", "not_found"]
    assert not (model_root / ".deleting").exists()
