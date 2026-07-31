from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.column_detection import detect_feature_columns
from src.data_validation import load_data_schema, validate_dataframe
from src.ml.evaluation import RegressionMetrics, evaluate_regression
from src.ml.model_io import (
    DEFAULT_MODEL_DIR,
    load_metadata,
    load_model,
)
from src.preprocessing import preprocess_dataframe


logger = logging.getLogger(__name__)

DEFAULT_WARNING_THRESHOLD = 95.0
DEFAULT_DANGER_THRESHOLD = 90.0
MAX_PREDICTION_ROWS = 5000
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REQUIRED_METADATA_FIELDS = {
    "target",
    "model_name",
    "feature_columns",
    "created_at",
    "metrics",
    "split_method",
}


class InferenceInputError(ValueError):
    """사용자가 입력 데이터나 모델 선택을 수정할 수 있는 오류."""


class ModelLoadError(RuntimeError):
    """모델 파일 접근 또는 역직렬화 오류."""


@dataclass
class LoadedPredictionModel:
    model_id: str
    model: Any
    metadata: dict[str, Any]


@dataclass
class PredictionResult:
    model_id: str
    target: str
    model_name: str
    identifier_column: str
    predictions: list[dict[str, Any]]
    total_rows: int
    average_prediction: float
    normal_count: int
    warning_count: int
    danger_count: int
    evaluation: RegressionMetrics | None = None
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False


def _validate_model_id(model_id: str) -> None:
    if (
        not model_id
        or not MODEL_ID_PATTERN.fullmatch(model_id)
        or ".." in model_id
        or "/" in model_id
        or "\\" in model_id
    ):
        raise InferenceInputError("유효하지 않은 모델 ID입니다.")


def _model_paths(
    model_id: str,
    model_dir: str | Path,
) -> tuple[Path, Path]:
    _validate_model_id(model_id)
    root = Path(model_dir).resolve()
    model_path = (root / f"{model_id}.joblib").resolve()
    metadata_path = (root / f"{model_id}.json").resolve()
    if model_path.parent != root or metadata_path.parent != root:
        raise InferenceInputError("유효하지 않은 모델 ID입니다.")
    return model_path, metadata_path


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing_fields = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    version = metadata.get("sklearn_version") or metadata.get(
        "scikit_learn_version"
    )
    if missing_fields or not version:
        missing = [*missing_fields]
        if not version:
            missing.append("sklearn_version")
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다. 누락 항목: "
            + ", ".join(missing)
        )
    if not isinstance(metadata["feature_columns"], list) or not all(
        isinstance(column, str)
        for column in metadata["feature_columns"]
    ):
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다: feature_columns"
        )
    metrics = metadata.get("metrics")
    test_metrics = metrics.get("test") if isinstance(metrics, dict) else None
    if not isinstance(test_metrics, dict) or not {
        "r2",
        "rmse",
        "mae",
    }.issubset(test_metrics):
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다: metrics.test"
        )


def load_prediction_model(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> LoadedPredictionModel:
    model_path, metadata_path = _model_paths(model_id, model_dir)
    if not metadata_path.is_file() or not model_path.is_file():
        raise InferenceInputError(
            "존재하지 않거나 사용할 수 없는 모델 ID입니다."
        )
    try:
        metadata = load_metadata(metadata_path)
    except (OSError, ValueError) as exc:
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다."
        ) from exc
    _validate_metadata(metadata)
    try:
        model = load_model(model_path)
    except Exception as exc:
        raise ModelLoadError("모델 파일을 불러오지 못했습니다.") from exc
    if not callable(getattr(model, "predict", None)):
        raise ModelLoadError("예측 가능한 모델 파일이 아닙니다.")
    return LoadedPredictionModel(
        model_id=model_id,
        model=model,
        metadata=metadata,
    )


def list_prediction_models(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(model_dir)
    if not root.exists():
        return [], []
    models: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        metadata_paths = sorted(root.glob("*.json"), reverse=True)
    except OSError as exc:
        raise ModelLoadError("모델 폴더를 읽지 못했습니다.") from exc

    for metadata_path in metadata_paths:
        model_id = metadata_path.stem
        model_path = root / f"{model_id}.joblib"
        if not model_path.is_file():
            warnings.append(
                f"{model_id}: 대응하는 joblib 모델 파일이 없습니다."
            )
            continue
        try:
            metadata = load_metadata(metadata_path)
            _validate_metadata(metadata)
            model = load_model(model_path)
            if not callable(getattr(model, "predict", None)):
                raise ValueError("예측 가능한 모델이 아닙니다.")
            test_metrics = metadata.get("metrics", {}).get("test", {})
            models.append(
                {
                    "model_id": model_id,
                    "target": str(metadata["target"]),
                    "model_name": str(metadata["model_name"]),
                    "created_at": str(metadata["created_at"]),
                    "test_metrics": {
                        "r2": test_metrics.get("r2"),
                        "rmse": test_metrics.get("rmse"),
                        "mae": test_metrics.get("mae"),
                    },
                    "feature_count": len(metadata["feature_columns"]),
                }
            )
        except Exception as exc:
            logger.warning(
                "모델 메타데이터 제외: %s",
                model_id,
                exc_info=exc,
            )
            warnings.append(
                f"{model_id}: 유효하지 않은 모델 메타데이터입니다."
            )
    models.sort(key=lambda item: str(item["created_at"]), reverse=True)
    return models, warnings


def get_prediction_model_detail(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    model_path, metadata_path = _model_paths(model_id, model_dir)
    if not metadata_path.is_file():
        raise InferenceInputError("존재하지 않는 모델 ID입니다.")
    try:
        metadata = load_metadata(metadata_path)
    except (OSError, ValueError) as exc:
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다."
        ) from exc

    feature_names = metadata.get("feature_columns")
    if not isinstance(feature_names, list):
        feature_names = []
    metrics_source = metadata.get("metrics")
    metrics: dict[str, dict[str, Any]] = {}
    if isinstance(metrics_source, dict):
        for split_name in ("train", "validation", "test"):
            split_metrics = metrics_source.get(split_name)
            if isinstance(split_metrics, dict):
                metrics[split_name] = {
                    key: split_metrics.get(key)
                    for key in ("r2", "rmse", "mse", "mae")
                }

    dataset_split = metadata.get("dataset_split")
    dataset_rows = metadata.get("dataset_rows")
    preprocessing_config = metadata.get("preprocessing_config")
    return {
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "model_version": metadata.get("model_version"),
        "created_at": metadata.get("created_at"),
        "target": metadata.get("target"),
        "feature_count": metadata.get("feature_count", len(feature_names)),
        "feature_names": feature_names,
        "dataset_split": (
            dataset_split if isinstance(dataset_split, dict) else None
        ),
        "dataset_rows": (
            dataset_rows if isinstance(dataset_rows, dict) else None
        ),
        "metrics": metrics,
        "random_seed": metadata.get("random_state"),
        "split_method": metadata.get("split_method"),
        "preprocessing_version": metadata.get("preprocessing_version"),
        "preprocessing_config": (
            preprocessing_config
            if isinstance(preprocessing_config, dict)
            else None
        ),
        "training_time_seconds": metadata.get("training_time_seconds"),
        "source_filename": metadata.get("source_filename"),
        "model_file": metadata.get("model_file", model_path.name),
        "metadata_file": metadata.get(
            "metadata_file",
            metadata_path.name,
        ),
        "storage_status": (
            "available" if model_path.is_file() else "model_file_missing"
        ),
        "champion": metadata.get("champion"),
        "sklearn_version": (
            metadata.get("sklearn_version")
            or metadata.get("scikit_learn_version")
        ),
    }


def prepare_inference_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()
    if duplicate_columns:
        raise InferenceInputError(
            "예측 데이터에 중복된 컬럼명이 있습니다: "
            + ", ".join(dict.fromkeys(duplicate_columns))
        )
    ordered_features = list(dict.fromkeys(feature_columns))
    missing_features = [
        column for column in ordered_features
        if column not in dataframe.columns
    ]
    if missing_features:
        raise InferenceInputError(
            "예측에 필요한 feature가 누락되었습니다: "
            + ", ".join(missing_features)
        )

    schema = load_data_schema()
    detected = detect_feature_columns(list(dataframe.columns), schema)
    detected_features = set(
        [
            *detected["r_columns"],
            *detected["d_columns"],
            *detected["eq_columns"],
        ]
    )
    extra_features = [
        column
        for column in dataframe.columns
        if column in detected_features and column not in ordered_features
    ]
    warnings: list[str] = []
    if extra_features:
        warnings.append(
            "학습에 사용하지 않은 feature를 제외했습니다: "
            + ", ".join(extra_features)
        )
    features = dataframe.loc[:, ordered_features].copy()
    if features.empty:
        raise InferenceInputError("유효한 예측 행이 없습니다.")
    return features, warnings


def _identifier_values(
    dataframe: pd.DataFrame,
) -> tuple[str, pd.Series]:
    schema = load_data_schema()
    id_column = schema["id_column"]
    if id_column in dataframe.columns:
        return id_column, dataframe[id_column].astype("string")
    row_ids = pd.Series(
        [
            f"ROW_{index:06d}"
            for index in range(1, len(dataframe) + 1)
        ],
        index=dataframe.index,
        dtype="string",
    )
    return "row_id", row_ids


def _finite_float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _risk_level(
    prediction: float,
    warning_threshold: float,
    danger_threshold: float,
) -> str:
    if prediction >= warning_threshold:
        return "normal"
    if prediction >= danger_threshold:
        return "warning"
    return "danger"


def predict_dataframe(
    dataframe: pd.DataFrame,
    loaded: LoadedPredictionModel,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    danger_threshold: float = DEFAULT_DANGER_THRESHOLD,
    max_rows: int | None = MAX_PREDICTION_ROWS,
) -> PredictionResult:
    if (
        not math.isfinite(warning_threshold)
        or not math.isfinite(danger_threshold)
        or warning_threshold <= danger_threshold
    ):
        raise InferenceInputError(
            "주의 기준값은 위험 기준값보다 커야 합니다."
        )
    validation = validate_dataframe(
        dataframe,
        require_id=False,
        require_yield=False,
    )
    if not validation["is_valid"]:
        raise InferenceInputError(
            "예측 데이터 검증에 실패했습니다: "
            + " ".join(validation["errors"])
        )
    processed, preprocessing_report = preprocess_dataframe(dataframe)
    feature_columns = list(loaded.metadata["feature_columns"])
    features, feature_warnings = prepare_inference_features(
        processed,
        feature_columns,
    )
    if len(features) == 0:
        raise InferenceInputError("유효한 예측 행이 없습니다.")

    try:
        raw_predictions = np.asarray(
            loaded.model.predict(features),
            dtype=float,
        )
    except Exception as exc:
        raise ModelLoadError("모델 예측 실행에 실패했습니다.") from exc
    if raw_predictions.ndim != 1 or len(raw_predictions) != len(features):
        raise ModelLoadError("모델 예측 결과의 행 수가 올바르지 않습니다.")
    if not np.isfinite(raw_predictions).all():
        raise ModelLoadError("모델 예측 결과에 유효하지 않은 값이 있습니다.")

    target = str(loaded.metadata["target"])
    display_predictions = raw_predictions.copy()
    warnings = list(
        dict.fromkeys(
            [
                *preprocessing_report["warnings"],
                *feature_warnings,
            ]
        )
    )
    if target == "Y":
        clipped = np.clip(display_predictions, 0.0, 100.0)
        clipped_count = int(np.count_nonzero(clipped != display_predictions))
        if clipped_count:
            warnings.append(
                f"표시 범위를 위해 {clipped_count}개 Y 예측값을 "
                "0~100으로 조정했습니다."
            )
        display_predictions = clipped
    else:
        warnings.append(
            f"{target} 목표 변수의 위험 등급 정책이 정의되지 않아 "
            "위험 등급을 계산하지 않았습니다."
        )

    identifier_column, identifiers = _identifier_values(dataframe)
    actual_values: pd.Series | None = None
    evaluation: RegressionMetrics | None = None
    if target in dataframe.columns:
        actual_values = pd.to_numeric(
            dataframe[target],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        evaluation_mask = actual_values.notna()
        if evaluation_mask.any():
            evaluation = evaluate_regression(
                actual_values.loc[evaluation_mask],
                raw_predictions[evaluation_mask.to_numpy()],
            )

    prediction_column = f"predicted_{target}"
    prediction_rows: list[dict[str, Any]] = []
    normal_count = 0
    warning_count = 0
    danger_count = 0
    for position, (index, identifier) in enumerate(identifiers.items()):
        prediction = float(display_predictions[position])
        row: dict[str, Any] = {
            identifier_column: (
                None if pd.isna(identifier) else str(identifier)
            ),
            prediction_column: prediction,
        }
        if target == "Y":
            risk = _risk_level(
                prediction,
                warning_threshold,
                danger_threshold,
            )
            row["risk_level"] = risk
            if risk == "normal":
                normal_count += 1
            elif risk == "warning":
                warning_count += 1
            else:
                danger_count += 1
        else:
            row["risk_level"] = None

        if actual_values is not None:
            actual = _finite_float(actual_values.loc[index])
            row[f"actual_{target}"] = actual
            if actual is not None:
                residual = actual - float(raw_predictions[position])
                row["residual"] = residual
                row["absolute_error"] = abs(residual)
            else:
                row["residual"] = None
                row["absolute_error"] = None
        prediction_rows.append(row)

    truncated = (
        max_rows is not None and len(prediction_rows) > max_rows
    )
    if truncated:
        warnings.append(
            f"화면 응답은 최대 {max_rows}행까지만 표시합니다."
        )
        prediction_rows = prediction_rows[:max_rows]

    return PredictionResult(
        model_id=loaded.model_id,
        target=target,
        model_name=str(loaded.metadata["model_name"]),
        identifier_column=identifier_column,
        predictions=prediction_rows,
        total_rows=len(dataframe),
        average_prediction=float(np.mean(display_predictions)),
        normal_count=normal_count,
        warning_count=warning_count,
        danger_count=danger_count,
        evaluation=evaluation,
        warnings=warnings,
        truncated=truncated,
    )
