import asyncio
import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile

import api.routes.data as data_routes
from src.ml.dataset import prepare_dataset, split_dataset
from src.ml.inference import (
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


def test_broken_joblib_is_skipped(inference_environment) -> None:
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

    assert len(models) == 1
    assert any("broken_model" in warning for warning in warnings)


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
    dataframe = inference_environment["dataframe"].drop(columns=["Y"])

    response = asyncio.run(
        data_routes.predict_csv(
            _upload(dataframe),
            model_id=inference_environment["model_id"],
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

    response = asyncio.run(
        data_routes.download_predictions(
            _upload(inference_environment["dataframe"]),
            model_id=inference_environment["model_id"],
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

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            data_routes.predict_csv(
                _upload(inference_environment["dataframe"]),
                model_id="missing",
                warning_threshold=95,
                danger_threshold=90,
            )
        )

    assert error.value.status_code == 400
