import asyncio
import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile

import api.routes.data as data_routes
import src.ml.inference as inference_module
from src.ml.dataset import prepare_dataset, split_dataset
from src.ml.inference import (
    get_prediction_model_detail,
    InferenceInputError,
    list_prediction_models,
    load_prediction_model,
    predict_dataframe,
    prepare_inference_features,
)
from src.ml.model_io import save_model_bundle
from src.ml.training import train_regression_models
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
    split = split_dataset(dataset)
    training = train_regression_models(dataset, split)
    model_path, _, _ = save_model_bundle(
        training.best_model,
        target="Y",
        model_name=training.best_model_name,
        feature_columns=dataset.feature_columns,
        metrics={
            name: values.as_dict()
            for name, values in training.metrics.items()
        },
        random_state=42,
        split_method=split.split_method,
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


def _upload(dataframe: pd.DataFrame) -> UploadFile:
    return UploadFile(
        file=BytesIO(dataframe.to_csv(index=False).encode("utf-8")),
        filename="prediction.csv",
    )


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
        "target_ensemble_configs": None,
        "ensemble_config": {
            "Y": {"selected_type": "weighted", "weights": {"ridge": 1.0}}
        },
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
    assert detail["target_ensemble_configs"]["Y"]["selected_type"] == "weighted"
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


def test_missing_required_feature_is_rejected(
    inference_environment,
) -> None:
    dataframe = inference_environment["dataframe"].drop(
        columns=["Step1_R1"]
    )
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    with pytest.raises(InferenceInputError, match="feature가 누락.*Step1_R1"):
        predict_dataframe(dataframe, loaded)


def test_extra_feature_is_ignored_with_warning(
    inference_environment,
) -> None:
    dataframe = inference_environment["dataframe"].copy()
    dataframe["Step9_R1"] = range(len(dataframe))
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    result = predict_dataframe(dataframe, loaded)

    assert any("Step9_R1" in warning for warning in result.warnings)
    assert len(result.predictions) == len(dataframe)


def test_prediction_without_target_succeeds(
    inference_environment,
) -> None:
    dataframe = inference_environment["dataframe"].drop(columns=["Y"])
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    result = predict_dataframe(dataframe, loaded)

    assert result.evaluation is None
    assert all("actual_Y" not in row for row in result.predictions)


def test_prediction_with_target_has_evaluation(
    inference_environment,
) -> None:
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    result = predict_dataframe(
        inference_environment["dataframe"],
        loaded,
    )

    assert result.evaluation is not None
    assert result.evaluation.rmse is not None
    assert all("actual_Y" in row for row in result.predictions)
    assert all("absolute_error" in row for row in result.predictions)


@pytest.mark.parametrize(
    ("combined", "expected_lot", "expected_wafer", "expected_slot"),
    [
        ("LOT001_W01", "LOT001", "W01", 1),
        ("LOT001W02", "LOT001", "W02", 2),
        ("LOT001-W03", "LOT001", "W03", 3),
        ("LOT001_WAFER04", "LOT001", "W04", 4),
        ("LOT001_WF05", "LOT001", "W05", 5),
    ],
)
def test_prediction_rows_include_canonical_identifiers(
    inference_environment,
    combined: str,
    expected_lot: str,
    expected_wafer: str,
    expected_slot: int,
) -> None:
    dataframe = inference_environment["dataframe"].copy()
    dataframe.loc[dataframe.index[0], "Lot_Wafer_ID"] = combined
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    row = predict_dataframe(dataframe, loaded).predictions[0]

    assert row["Lot_Wafer_ID"] == combined
    assert row["Lot_ID"] == expected_lot
    assert row["Wafer_ID"] == expected_wafer
    assert row["Wafer_Slot"] == expected_slot


def test_prediction_identifier_source_fields_take_priority(
    inference_environment,
) -> None:
    dataframe = inference_environment["dataframe"].copy()
    dataframe["Lot_ID"] = "SOURCE_LOT"
    dataframe["Wafer_ID"] = "RAW_WAFER"
    dataframe["Wafer_Slot"] = 7
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    row = predict_dataframe(dataframe, loaded).predictions[0]

    assert row["Lot_ID"] == "SOURCE_LOT"
    assert row["Wafer_ID"] == "RAW_WAFER"
    assert row["Wafer_Slot"] == 7


def test_prediction_identifier_legacy_lowercase_fields_are_restored(
    inference_environment,
) -> None:
    dataframe = inference_environment["dataframe"].copy().drop(
        columns=["Lot_Wafer_ID"]
    )
    dataframe["lot_id"] = "LEGACY_LOT"
    dataframe["wafer_id"] = "W09"
    dataframe["wafer_slot"] = 9
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    result = predict_dataframe(dataframe, loaded)
    row = result.predictions[0]

    assert result.identifier_column == "row_id"
    assert row["Lot_Wafer_ID"] == "LEGACY_LOT_W09"
    assert row["Lot_ID"] == "LEGACY_LOT"
    assert row["Wafer_ID"] == "W09"
    assert row["Wafer_Slot"] == 9


def test_prediction_identifier_parse_failure_is_safe(
    inference_environment,
) -> None:
    dataframe = inference_environment["dataframe"].copy()
    dataframe.loc[dataframe.index[0], "Lot_Wafer_ID"] = "UNPARSEABLE"
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    row = predict_dataframe(dataframe, loaded).predictions[0]

    assert row["Lot_Wafer_ID"] == "UNPARSEABLE"
    assert row["Lot_ID"] is None
    assert row["Wafer_ID"] is None
    assert row["Wafer_Slot"] is None


def test_y_risk_counts_match_predictions(inference_environment) -> None:
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    result = predict_dataframe(
        inference_environment["dataframe"],
        loaded,
        warning_threshold=96,
        danger_threshold=93,
    )
    risk_levels = [row["risk_level"] for row in result.predictions]

    assert result.normal_count == risk_levels.count("normal")
    assert result.warning_count == risk_levels.count("warning")
    assert result.danger_count == risk_levels.count("danger")
    assert sum(
        [
            result.normal_count,
            result.warning_count,
            result.danger_count,
        ]
    ) == len(result.predictions)


def test_invalid_thresholds_are_rejected(inference_environment) -> None:
    loaded = load_prediction_model(
        inference_environment["model_id"],
        inference_environment["model_dir"],
    )

    with pytest.raises(InferenceInputError, match="주의 기준값"):
        predict_dataframe(
            inference_environment["dataframe"],
            loaded,
            warning_threshold=90,
            danger_threshold=95,
        )


def test_predict_api_response_is_json_serializable(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        inference_environment["model_dir"],
    )
    data_routes._runtime_store().promote_model(
        model_id=inference_environment["model_id"],
        pipeline_version="direct_y_v1",
        dataset_version=0,
        metadata={"target": "Y"},
    )
    dataframe = inference_environment["dataframe"].drop(columns=["Y"])

    response = asyncio.run(
        data_routes.predict_csv(
            _upload(dataframe),
            warning_threshold=95,
            danger_threshold=90,
        )
    )
    serialized = json.loads(response.model_dump_json())

    assert response.success is True
    assert serialized["summary"]["total_rows"] == len(dataframe)
    assert serialized["predictions"]


def test_predict_download_returns_csv(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        inference_environment["model_dir"],
    )
    data_routes._runtime_store().promote_model(
        model_id=inference_environment["model_id"],
        pipeline_version="direct_y_v1",
        dataset_version=0,
        metadata={"target": "Y"},
    )

    response = asyncio.run(
        data_routes.download_predictions(
            _upload(inference_environment["dataframe"]),
            warning_threshold=95,
            danger_threshold=90,
        )
    )
    decoded = response.body.decode("utf-8-sig")

    assert response.status_code == 200
    assert "predicted_Y" in decoded.splitlines()[0]
    assert "risk_level" in decoded.splitlines()[0]


def test_predict_api_unknown_model_returns_400(
    inference_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        inference_environment["model_dir"],
    )
    data_routes._runtime_store().promote_model(
        model_id="missing",
        pipeline_version="direct_y_v1",
        dataset_version=0,
        metadata={"target": "Y"},
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            data_routes.predict_csv(
                _upload(inference_environment["dataframe"]),
                warning_threshold=95,
                danger_threshold=90,
            )
        )

    assert error.value.status_code == 409
