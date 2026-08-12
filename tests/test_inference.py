import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.dummy import DummyRegressor

import api.routes.data as data_routes
import src.ml.inference as inference_module
from src.ml.dataset import prepare_dataset
from src.ml.inference import (
    get_prediction_model_detail,
    InferenceInputError,
    list_prediction_models,
    load_prediction_model,
    prepare_inference_features,
)
from src.ml.model_io import save_model_bundle
from src.preprocessing import preprocess_dataframe


@pytest.fixture(scope="module")
def inference_environment():
    temporary_root = Path(__file__).parent / ".tmp_inference_models"
    temporary_root.mkdir(exist_ok=True)
    model_dir = temporary_root / f"run_{uuid4().hex}"
    model_dir.mkdir()
    fixture_path = Path(__file__).parent / "fixtures" / "training_sample.csv"
    dataframe = pd.read_csv(fixture_path)
    processed, _ = preprocess_dataframe(dataframe)
    dataset = prepare_dataset(processed)
    estimator = DummyRegressor(strategy="mean").fit(
        dataset.features[dataset.numeric_columns], dataset.target
    )
    model_path, _, _ = save_model_bundle(
        estimator,
        target="Y",
        model_name="DummyRegressor",
        feature_columns=dataset.feature_columns,
        metrics={"test": {"r2": 0.0, "rmse": 1.0, "mae": 1.0}},
        random_state=42,
        split_method="group",
        model_dir=model_dir,
    )
    yield {
        "model_dir": model_dir,
        "model_id": model_path.stem,
        "dataframe": dataframe,
        "feature_columns": dataset.feature_columns,
    }
    for generated_file in model_dir.iterdir():
        if generated_file.is_dir():
            for nested in generated_file.iterdir():
                nested.unlink()
            generated_file.rmdir()
        else:
            generated_file.unlink()
    model_dir.rmdir()
    if not any(temporary_root.iterdir()):
        temporary_root.rmdir()


def test_valid_model_is_listed(inference_environment) -> None:
    models, warnings = list_prediction_models(
        inference_environment["model_dir"]
    )

    assert warnings == []
    assert len(models) == 1
    assert models[0]["model_id"] == inference_environment["model_id"]
    assert models[0]["target"] == "Y"
    assert models[0]["feature_count"] == len(
        inference_environment["feature_columns"]
    )
    assert models[0]["available_targets"] == ["Y"]
    assert models[0]["available"] is True
    assert models[0]["loadable"] is True
    assert models[0]["compatibility_status"] == "legacy"
    assert models[0]["incompatibility_reason"] is None


def test_models_api_returns_valid_models(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        inference_environment["model_dir"],
    )

    response = data_routes.get_models()

    assert response.success is True
    assert response.models[0].model_id == inference_environment["model_id"]
    assert response.models[0].available_targets == ["Y"]
    assert response.models[0].available is True
    assert response.models[0].loadable is True
    assert response.models[0].compatibility_status == "legacy"
    assert response.models[0].incompatibility_reason is None


def test_model_list_is_sorted_by_created_at_descending(
    inference_environment,
) -> None:
    model_dir = inference_environment["model_dir"]
    source_id = inference_environment["model_id"]
    source_metadata = json.loads(
        (model_dir / f"{source_id}.json").read_text(encoding="utf-8")
    )
    older_id = "Y_older_model_20200101_000000"
    older_metadata = {**source_metadata, "created_at": "2020-01-01T00:00:00+00:00"}
    older_json = model_dir / f"{older_id}.json"
    older_model = model_dir / f"{older_id}.joblib"
    older_json.write_text(
        json.dumps(older_metadata),
        encoding="utf-8",
    )
    older_model.write_bytes(
        (model_dir / f"{source_id}.joblib").read_bytes()
    )
    try:
        models, _ = list_prediction_models(model_dir)
    finally:
        older_json.unlink()
        older_model.unlink()

    assert models[0]["model_id"] == source_id
    assert models[-1]["model_id"] == older_id


def test_model_detail_handles_legacy_missing_optional_metadata(
    inference_environment,
) -> None:
    detail = get_prediction_model_detail(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    assert detail["model_id"] == inference_environment["model_id"]
    assert detail["dataset_split"] == {}
    assert detail["dataset_rows"] == {}
    assert detail["training_time_seconds"] is None
    assert detail["metrics"]["test"]["mse"] is None
    assert detail["storage_status"] == "available"
    assert detail["available"] is True
    assert detail["loadable"] is True
    assert detail["compatibility_status"] == "legacy"
    assert detail["incompatibility_reason"] is None


def test_model_detail_api_returns_optional_fields(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        inference_environment["model_dir"],
    )

    response = data_routes.get_model_detail(
        inference_environment["model_id"]
    )

    assert response.success is True
    assert response.model_id == inference_environment["model_id"]
    assert response.metrics["test"].mse is None
    assert response.available is True
    assert response.loadable is True
    assert response.compatibility_status == "legacy"
    assert response.incompatibility_reason is None


def test_model_detail_api_rejects_missing_model(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        inference_environment["model_dir"],
    )

    with pytest.raises(HTTPException) as error:
        data_routes.get_model_detail("missing-model")

    assert error.value.status_code == 404


def test_model_detail_normalizes_nullable_fields_and_legacy_aliases(
    inference_environment,
) -> None:
    model_dir = inference_environment["model_dir"]
    model_id = inference_environment["model_id"]
    metadata_path = model_dir / f"{model_id}.json"
    original = metadata_path.read_text(encoding="utf-8")
    metadata = json.loads(original)
    metadata.update({
        "metrics": None,
        "validation_metrics": {"r2": 0.7, "rmse": 1.2, "mae": 0.8},
        "dataset_rows": None,
        "dataset_split": None,
        "train_size": 80,
        "validation_size": 10,
        "test_size": 10,
        "target_metrics": None,
        "outer_fold_metrics": None,
        "available_targets": None,
        "risk_metrics": [],
    })
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    try:
        detail = get_prediction_model_detail(model_id, model_dir)
    finally:
        metadata_path.write_text(original, encoding="utf-8")

    assert detail["metrics"]["validation"]["r2"] == 0.7
    assert detail["dataset_rows"] == {"train": 80, "validation": 10, "test": 10}
    assert detail["dataset_split"] == {}
    assert detail["target_metrics"] == {}
    assert detail["outer_fold_metrics"] == []
    assert detail["available_targets"] == ["Y"]
    assert detail["risk_metrics"] == {}


def test_model_detail_rejects_corrupt_metadata_with_clear_status(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = inference_environment["model_dir"]
    model_id = inference_environment["model_id"]
    metadata_path = model_dir / f"{model_id}.json"
    original = metadata_path.read_text(encoding="utf-8")
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_dir)
    metadata_path.write_text("{not-json", encoding="utf-8")
    try:
        with pytest.raises(HTTPException) as error:
            data_routes.get_model_detail(model_id)
    finally:
        metadata_path.write_text(original, encoding="utf-8")

    assert error.value.status_code == 422
    assert "유효하지 않은 모델 메타데이터" in str(error.value.detail)


def test_empty_model_directory_returns_empty_list() -> None:
    empty_dir = (
        Path(__file__).parent
        / ".tmp_inference_models"
        / f"empty_{uuid4().hex}"
    )
    empty_dir.mkdir(parents=True)
    try:
        models, warnings = list_prediction_models(empty_dir)
    finally:
        empty_dir.rmdir()

    assert models == []
    assert warnings == []


def test_broken_model_metadata_is_skipped(inference_environment) -> None:
    model_dir = inference_environment["model_dir"]
    broken_json = model_dir / "broken.json"
    broken_model = model_dir / "broken.joblib"
    broken_json.write_text("{not-json", encoding="utf-8")
    broken_model.write_bytes(b"broken")
    try:
        models, warnings = list_prediction_models(model_dir)
    finally:
        broken_json.unlink()
        broken_model.unlink()

    assert len(models) == 1
    assert any("broken" in warning for warning in warnings)


def test_broken_joblib_is_listed_as_unavailable(inference_environment) -> None:
    model_dir = inference_environment["model_dir"]
    source_metadata = model_dir / (
        f"{inference_environment['model_id']}.json"
    )
    broken_json = model_dir / "broken_model.json"
    broken_model = model_dir / "broken_model.joblib"
    broken_json.write_text(
        source_metadata.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    broken_model.write_bytes(b"not-a-joblib-model")
    try:
        models, warnings = list_prediction_models(model_dir)
    finally:
        broken_json.unlink()
        broken_model.unlink()

    assert len(models) == 2
    broken = next(model for model in models if model["model_id"] == "broken_model")
    assert broken["available"] is False
    assert broken["loadable"] is False
    assert broken["compatibility_status"] == "load_error"
    assert broken["incompatibility_reason"] == "모델 파일을 불러올 수 없습니다."
    assert any("broken_model" in warning for warning in warnings)


def test_missing_xgboost_dependency_is_visible_in_list_and_detail(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = inference_environment["model_dir"]
    source_id = inference_environment["model_id"]
    dependency_id = "Y_xgboost_dependency_missing"
    dependency_json = model_dir / f"{dependency_id}.json"
    dependency_model = model_dir / f"{dependency_id}.joblib"
    dependency_metadata = json.loads(
        (model_dir / f"{source_id}.json").read_text(encoding="utf-8")
    )
    dependency_metadata["model_name"] = "XGBoostRegressor"
    dependency_metadata["model_type"] = "XGBoostRegressor"
    dependency_json.write_text(
        json.dumps(dependency_metadata),
        encoding="utf-8",
    )
    dependency_model.write_bytes(
        (model_dir / f"{source_id}.joblib").read_bytes()
    )
    original_find_spec = inference_module.importlib.util.find_spec
    monkeypatch.setattr(
        inference_module.importlib.util,
        "find_spec",
        lambda name: None if name == "xgboost" else original_find_spec(name),
    )
    monkeypatch.setattr(
        inference_module,
        "load_model",
        lambda path: (_ for _ in ()).throw(
            AssertionError(f"목록/상세 조회에서 모델을 load했습니다: {path}")
        ),
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_dir)
    try:
        models, warnings = list_prediction_models(model_dir)
        detail = data_routes.get_model_detail(dependency_id)
    finally:
        dependency_json.unlink()
        dependency_model.unlink()

    unavailable = next(
        model for model in models if model["model_id"] == dependency_id
    )
    expected_reason = "xgboost가 설치되어 있지 않습니다."
    assert unavailable["available"] is False
    assert unavailable["loadable"] is False
    assert unavailable["compatibility_status"] == "dependency_missing"
    assert unavailable["incompatibility_reason"] == expected_reason
    assert any(
        dependency_id in warning and expected_reason in warning
        for warning in warnings
    )
    assert detail.model_id == dependency_id
    assert detail.available is False
    assert detail.loadable is False
    assert detail.compatibility_status == "dependency_missing"
    assert detail.incompatibility_reason == expected_reason


def test_unknown_model_id_is_rejected(inference_environment) -> None:
    with pytest.raises(InferenceInputError, match="존재하지"):
        load_prediction_model(
            "not_existing",
            inference_environment["model_dir"],
        )


@pytest.mark.parametrize(
    "model_id",
    ["../secret", "..\\secret", "folder/model", "C:\\secret"],
)
def test_model_id_path_traversal_is_rejected(
    inference_environment,
    model_id: str,
) -> None:
    with pytest.raises(InferenceInputError, match="유효하지 않은 모델 ID"):
        load_prediction_model(
            model_id,
            inference_environment["model_dir"],
        )


def test_inference_features_restore_training_order(
    inference_environment,
) -> None:
    dataframe = inference_environment["dataframe"]
    processed, _ = preprocess_dataframe(dataframe)
    reversed_dataframe = processed.loc[:, list(reversed(processed.columns))]

    features, _ = prepare_inference_features(
        reversed_dataframe,
        inference_environment["feature_columns"],
    )

    assert list(features.columns) == inference_environment["feature_columns"]
