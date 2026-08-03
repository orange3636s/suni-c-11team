import asyncio
import json
import shutil
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor

import api.routes.data as data_routes
import src.ml.dataset as dataset_module
import src.ml.training as training_module
from src.ml.dataset import prepare_dataset, split_dataset
from src.ml.evaluation import evaluate_regression
from src.ml.model_io import load_model, save_model_bundle
from src.ml.training import (
    OutlierPolicyTransformer,
    _build_preprocessor,
    missing_strategy_for_model,
    outlier_strategy_for_model,
    train_regression_models,
)
from src.preprocessing import preprocess_dataframe


class FailingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, features, target):
        del features, target
        raise RuntimeError("intentional failure")


@pytest.fixture
def training_dataframe() -> pd.DataFrame:
    random = np.random.default_rng(42)
    row_count = 100
    response = random.normal(size=row_count)
    return pd.DataFrame(
        {
            "Lot_Wafer_ID": [
                f"LOT{index // 4:02d}_WF{index % 4 + 1:02d}"
                for index in range(row_count)
            ],
            "Y": 90 + 2.5 * response + random.normal(0, 0.2, row_count),
            "Step1_R1": response,
            "Step1_D1": random.normal(size=row_count),
            "Step1_EQ": [
                "EQ_A" if index % 2 else "EQ_B"
                for index in range(row_count)
            ],
        }
    )


@pytest.fixture
def prepared_training_data(training_dataframe: pd.DataFrame):
    processed, _ = preprocess_dataframe(training_dataframe)
    dataset = prepare_dataset(processed, target="Y")
    split = split_dataset(dataset)
    return dataset, split


@pytest.fixture
def model_output_dir():
    temporary_root = Path(__file__).parent / ".tmp_models"
    temporary_root.mkdir(exist_ok=True)
    output_dir = temporary_root / f"training_{uuid4().hex}"
    output_dir.mkdir()
    yield output_dir
    for generated_file in output_dir.iterdir():
        if generated_file.is_dir():
            shutil.rmtree(generated_file)
        else:
            generated_file.unlink()
    output_dir.rmdir()
    if not any(temporary_root.iterdir()):
        temporary_root.rmdir()


def _as_upload(dataframe: pd.DataFrame, filename: str = "training.csv") -> UploadFile:
    content = dataframe.to_csv(index=False).encode("utf-8")
    return UploadFile(file=BytesIO(content), filename=filename)


def _automatic_targets(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    frame["Lot_ID"] = frame["Lot_Wafer_ID"].str.extract(r"^(LOT\d+)", expand=False)
    total_failure = np.clip(100.0 - pd.to_numeric(frame["Y"], errors="coerce"), 0.0, None)
    weights = np.asarray([0.10, 0.15, 0.20, 0.25, 0.30])
    for index, weight in enumerate(weights, 1):
        frame[f"Y{index}"] = total_failure * weight
    frame["Step1_Config"] = frame.pop("Step1_EQ")
    return frame


def test_y_target_training_succeeds(prepared_training_data) -> None:
    dataset, split = prepared_training_data

    result = train_regression_models(dataset, split)

    assert result.best_model_name
    assert len(result.model_comparison) == 1
    assert sum(item.selected for item in result.model_comparison) == 1
    assert result.outlier_strategy == outlier_strategy_for_model(result.best_model_name)
    assert result.outlier_indicator is (result.outlier_strategy == "flag_only")
    assert set(result.model_strategies.values()) <= {"native", "median"}
    assert result.model_outlier_strategies["HistGradientBoostingRegressor"] == "flag_only"
    assert result.fallback_used is False


def test_model_specific_missing_and_outlier_policies() -> None:
    assert missing_strategy_for_model("HistGradientBoostingRegressor") == "native"
    assert outlier_strategy_for_model("HistGradientBoostingRegressor") == "flag_only"
    assert missing_strategy_for_model("XGBoostRegressor") == "native"
    assert outlier_strategy_for_model("XGBoostRegressor") == "flag_only"
    assert missing_strategy_for_model("CatBoostRegressor") == "native"
    assert outlier_strategy_for_model("CatBoostRegressor") == "flag_only"
    assert missing_strategy_for_model("RandomForestRegressor") == "median"
    assert outlier_strategy_for_model("RandomForestRegressor") == "flag_only"
    assert missing_strategy_for_model("Ridge") == "median"
    assert outlier_strategy_for_model("Ridge") == "iqr"


def test_outlier_flag_only_preserves_raw_values_and_adds_indicators() -> None:
    values = np.asarray([[1.0], [2.0], [3.0], [4.0], [100.0], [np.nan]])
    transformer = OutlierPolicyTransformer(strategy="flag_only").fit(values[:5])

    transformed = transformer.transform(values)

    assert transformed.shape == (6, 2)
    assert transformed[4, 0] == 100.0
    assert transformed[4, 1] == 1.0
    assert np.isnan(transformed[5, 0])


def test_linear_iqr_policy_clips_without_adding_indicator() -> None:
    values = np.asarray([[1.0], [2.0], [3.0], [4.0], [100.0]])
    transformer = OutlierPolicyTransformer(strategy="iqr").fit(values[:4])

    transformed = transformer.transform(values)

    assert transformed.shape == (5, 1)
    assert transformed[4, 0] < 100.0
    assert transformer.get_feature_names_out(["Step1_R1"]).tolist() == ["Step1_R1"]


def test_model_preprocessor_uses_native_only_when_requested(training_dataframe: pd.DataFrame) -> None:
    training_dataframe.loc[0, "Step1_R1"] = np.nan
    dataset = prepare_dataset(training_dataframe)
    native = _build_preprocessor(dataset, missing_strategy="native", outlier_strategy="flag_only")
    median = _build_preprocessor(dataset, missing_strategy="median", outlier_strategy="flag_only")

    native.fit(dataset.features, dataset.target)
    median.fit(dataset.features, dataset.target)

    assert "imputer" not in native.named_transformers_["numeric"].named_steps
    assert "imputer" in median.named_transformers_["numeric"].named_steps
    assert np.isnan(native.transform(dataset.features)).any()
    assert not np.isnan(median.transform(dataset.features)).any()


def test_unknown_target_is_rejected(training_dataframe: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="지원하지 않는 목표 변수"):
        prepare_dataset(training_dataframe, target="UNKNOWN")


def test_string_target_is_converted_to_numeric(
    training_dataframe: pd.DataFrame,
) -> None:
    training_dataframe["Y"] = training_dataframe["Y"].map(
        lambda value: f"{value:.6f}"
    )

    dataset = prepare_dataset(training_dataframe)

    assert pd.api.types.is_float_dtype(dataset.target)
    assert len(dataset.target) == len(training_dataframe)


def test_numeric_feature_strings_are_converted(
    training_dataframe: pd.DataFrame,
) -> None:
    training_dataframe["Step1_R1"] = training_dataframe["Step1_R1"].map(
        lambda value: f"{value:.6f}"
    )

    dataset = prepare_dataset(training_dataframe)

    assert pd.api.types.is_numeric_dtype(dataset.features["Step1_R1"])
    assert isinstance(dataset.features.loc[:, "Step1_R1"], pd.Series)


def test_duplicate_dataframe_feature_column_is_rejected(
    training_dataframe: pd.DataFrame,
) -> None:
    duplicate_dataframe = training_dataframe.copy()
    duplicate_dataframe.columns = [
        "Lot_Wafer_ID",
        "Y",
        "Step1_R1",
        "Step1_R1",
        "Step1_EQ",
    ]

    with pytest.raises(ValueError, match="중복된 컬럼명.*Step1_R1"):
        prepare_dataset(duplicate_dataframe)


def test_duplicate_target_column_is_rejected(
    training_dataframe: pd.DataFrame,
) -> None:
    duplicate_target = pd.concat(
        [
            training_dataframe,
            training_dataframe[["Y"]],
        ],
        axis=1,
    )

    with pytest.raises(ValueError, match="목표 변수 'Y' 컬럼이 중복"):
        prepare_dataset(duplicate_target)


def test_detected_feature_duplicates_are_deduplicated_in_order(
    training_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_module,
        "detect_feature_columns",
        lambda columns, schema: {
            "r_columns": ["Step1_R1", "Step1_R1"],
            "d_columns": ["Step1_D1", "Step1_R1"],
            "eq_columns": ["Step1_EQ", "Step1_EQ"],
            "target_columns": ["Y"],
        },
    )

    dataset = prepare_dataset(training_dataframe)

    assert dataset.feature_columns == [
        "Step1_R1",
        "Step1_D1",
        "Step1_EQ",
    ]
    assert dataset.features.columns.is_unique


def test_missing_target_rows_are_aligned_and_removed(
    training_dataframe: pd.DataFrame,
) -> None:
    training_dataframe.loc[[1, 7, 11], "Y"] = np.nan

    dataset = prepare_dataset(training_dataframe)

    assert len(dataset.features) == 97
    assert len(dataset.target) == 97
    assert dataset.features.index.equals(dataset.target.index)


def test_dataset_without_features_is_rejected(
    training_dataframe: pd.DataFrame,
) -> None:
    no_features = training_dataframe[["Lot_Wafer_ID", "Y"]]

    with pytest.raises(ValueError, match="feature"):
        prepare_dataset(no_features)


def test_all_missing_target_is_rejected(
    training_dataframe: pd.DataFrame,
) -> None:
    training_dataframe["Y"] = np.nan

    with pytest.raises(ValueError, match="모두 결측"):
        prepare_dataset(training_dataframe)


def test_small_dataset_is_rejected_with_clear_error(
    training_dataframe: pd.DataFrame,
) -> None:
    small_dataframe = training_dataframe.head(9)

    with pytest.raises(ValueError, match="최소 10개"):
        prepare_dataset(small_dataframe)


def test_train_validation_test_split_ratio(
    prepared_training_data,
) -> None:
    _, split = prepared_training_data

    assert split.row_counts == {
        "train_rows": 68,
        "validation_rows": 16,
        "test_rows": 16,
    }


def test_custom_train_validation_test_split_ratio(
    training_dataframe: pd.DataFrame,
) -> None:
    dataset = prepare_dataset(training_dataframe)

    split = split_dataset(
        dataset,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    expected = {
        "train_rows": 70,
        "validation_rows": 15,
        "test_rows": 15,
    }
    assert sum(split.row_counts.values()) == 100
    assert all(
        abs(split.row_counts[key] - value) <= 4
        for key, value in expected.items()
    )


def test_same_lot_is_not_shared_between_splits(
    prepared_training_data,
) -> None:
    _, split = prepared_training_data
    train_groups = set(split.train_groups)
    validation_groups = set(split.validation_groups)
    test_groups = set(split.test_groups)

    assert split.group_split_used is True
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_insufficient_groups_fall_back_to_random_split(
    training_dataframe: pd.DataFrame,
) -> None:
    training_dataframe["Lot_Wafer_ID"] = [
        f"UNPARSEABLE-{index}" for index in range(len(training_dataframe))
    ]
    dataset = prepare_dataset(training_dataframe)

    split = split_dataset(dataset)

    assert split.group_split_used is False
    assert split.row_counts == {
        "train_rows": 69,
        "validation_rows": 16,
        "test_rows": 15,
    }
    assert any("random split" in warning for warning in split.warnings)


def test_selected_model_has_all_metrics(prepared_training_data) -> None:
    dataset, split = prepared_training_data
    result = train_regression_models(dataset, split)

    assert set(result.metrics) == {"train", "validation", "test"}
    for metrics in result.metrics.values():
        assert metrics.rmse is not None
        assert metrics.mae is not None


def test_rmse_and_single_row_r2_are_json_safe() -> None:
    metrics = evaluate_regression(
        np.array([1.0]),
        np.array([2.0]),
    )

    assert metrics.r2 is None
    assert metrics.rmse == pytest.approx(1.0)
    assert metrics.mae == pytest.approx(1.0)


def test_one_model_failure_does_not_stop_other_models(
    prepared_training_data,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, split = prepared_training_data
    monkeypatch.setattr(
        training_module,
        "_candidate_estimators",
        lambda random_state: {
            "FailingRegressor": FailingRegressor(),
            "DummyRegressor": DummyRegressor(),
        },
    )

    result = train_regression_models(dataset, split)
    failed = next(
        item
        for item in result.model_comparison
        if item.model_name == "FailingRegressor"
    )

    assert result.best_model_name == "DummyRegressor"
    assert failed.status == "failed"
    assert failed.validation is None
    assert failed.error_message


def test_model_and_metadata_are_saved(
    prepared_training_data,
    model_output_dir: Path,
) -> None:
    dataset, split = prepared_training_data
    result = train_regression_models(dataset, split)
    metrics = {
        name: values.as_dict()
        for name, values in result.metrics.items()
    }

    model_path, metadata_path, metadata = save_model_bundle(
        result.best_model,
        target="Y",
        model_name=result.best_model_name,
        feature_columns=dataset.feature_columns,
        metrics=metrics,
        random_state=42,
        split_method=split.split_method,
        model_dir=model_output_dir,
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert load_model(model_path) is not None
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


def test_model_filename_is_windows_safe(
    prepared_training_data,
    model_output_dir: Path,
) -> None:
    dataset, split = prepared_training_data
    result = train_regression_models(dataset, split)

    model_path, metadata_path, _ = save_model_bundle(
        result.best_model,
        target="Y",
        model_name="Unsafe:Model/Name",
        feature_columns=dataset.feature_columns,
        metrics={
            name: values.as_dict()
            for name, values in result.metrics.items()
        },
        random_state=42,
        split_method=split.split_method,
        model_dir=model_output_dir,
    )

    assert ":" not in model_path.name
    assert "/" not in model_path.name
    assert ":" not in metadata_path.name


def test_train_api_response_is_json_serializable(
    training_dataframe: pd.DataFrame,
    model_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_output_dir)

    response = asyncio.run(
        data_routes.train_model(_as_upload(_automatic_targets(training_dataframe)))
    )
    serialized = response.model_dump_json()

    assert response.success is True
    assert response.target == "Y"
    assert response.split.group_split_used is True
    assert json.loads(serialized)["metrics"]["test"]["rmse"] is not None
    assert (model_output_dir / response.artifacts.model_file).exists()
    metadata_path = model_output_dir / response.artifacts.metadata_file
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_rows"] == {
        "train": response.split.train_rows,
        "test": response.split.test_rows,
    }
    assert metadata["source_filename"] == "training.csv"
    assert set(metadata["available_targets"]) == {"Y1", "Y2", "Y3", "Y4", "Y5"}


def test_train_api_succeeds_with_fixture_csv(
    model_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "training_sample.csv"
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_output_dir)
    fixture_frame = _automatic_targets(pd.read_csv(fixture_path))
    upload = _as_upload(fixture_frame, fixture_path.name)

    response = asyncio.run(data_routes.train_model(upload))

    assert response.success is True
    assert response.target == "Y"
    assert response.split.train_rows > 0
    assert response.split.validation_rows == 0
    assert response.split.test_rows > 0
    assert json.loads(response.model_dump_json())["best_model"]


def test_train_api_rejects_non_csv() -> None:
    upload = UploadFile(file=BytesIO(b"not csv"), filename="training.txt")

    with pytest.raises(HTTPException) as error:
        asyncio.run(data_routes.train_model(upload))

    assert error.value.status_code == 400


def test_unexpected_training_error_returns_json_detail(
    training_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        data_routes,
        "train_and_evaluate",
        raise_unexpected_error,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            data_routes.train_model(_as_upload(_automatic_targets(training_dataframe)))
        )

    assert error.value.status_code == 500
    assert "모델 학습" in error.value.detail
