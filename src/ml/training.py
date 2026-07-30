from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from src.ml.dataset import (
    RANDOM_STATE,
    DatasetSplit,
    PreparedDataset,
)
from src.ml.evaluation import RegressionMetrics, evaluate_regression


logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    model_name: str
    status: str
    validation: RegressionMetrics | None = None
    selected: bool = False
    error_message: str | None = None


@dataclass
class TrainingResult:
    best_model_name: str
    best_model: Any
    metrics: dict[str, RegressionMetrics]
    model_comparison: list[ModelComparison]
    warnings: list[str] = field(default_factory=list)


def _build_preprocessor(dataset: PreparedDataset) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if dataset.numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                dataset.numeric_columns,
            )
        )
    if dataset.categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                dataset.categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _candidate_estimators(random_state: int) -> dict[str, Any]:
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            random_state=random_state,
        ),
    }


def train_regression_models(
    dataset: PreparedDataset,
    split: DatasetSplit,
    random_state: int = RANDOM_STATE,
) -> TrainingResult:
    fitted_models: dict[str, Pipeline] = {}
    comparisons: list[ModelComparison] = []
    warnings: list[str] = []

    for model_name, estimator in _candidate_estimators(random_state).items():
        logger.info("모델 학습 시작: %s", model_name)
        pipeline = Pipeline(
            steps=[
                ("features", _build_preprocessor(dataset)),
                ("model", estimator),
            ]
        )
        try:
            with threadpool_limits(limits=1):
                pipeline.fit(split.x_train, split.y_train)
                validation_prediction = pipeline.predict(
                    split.x_validation
                )
            validation_metrics = evaluate_regression(
                split.y_validation,
                validation_prediction,
            )
        except Exception:
            logger.exception("모델 학습 실패: %s", model_name)
            error_message = (
                "모델 입력 데이터 또는 실행 환경을 처리하지 못했습니다."
            )
            warnings.append(f"{model_name} 학습 실패: {error_message}")
            comparisons.append(
                ModelComparison(
                    model_name=model_name,
                    status="failed",
                    error_message=error_message,
                )
            )
            continue

        fitted_models[model_name] = pipeline
        comparisons.append(
            ModelComparison(
                model_name=model_name,
                status="success",
                validation=validation_metrics,
            )
        )
        logger.info("모델 학습 완료: %s", model_name)

    successful_comparisons = [
        item for item in comparisons if item.status == "success"
    ]
    if not successful_comparisons:
        raise ValueError("학습에 성공한 모델이 없습니다.")

    def ranking_key(item: ModelComparison) -> tuple[float, float]:
        if item.validation is None:
            return float("inf"), float("inf")
        rmse = (
            item.validation.rmse
            if item.validation.rmse is not None
            else float("inf")
        )
        r2 = item.validation.r2 if item.validation.r2 is not None else float("-inf")
        return rmse, -r2

    best_comparison = min(successful_comparisons, key=ranking_key)
    best_comparison.selected = True
    best_model = fitted_models[best_comparison.model_name]
    if best_comparison.validation is None:
        raise ValueError("선택된 모델의 Validation 지표가 없습니다.")
    with threadpool_limits(limits=1):
        metrics = {
            "train": evaluate_regression(
                split.y_train,
                best_model.predict(split.x_train),
            ),
            "validation": best_comparison.validation,
            "test": evaluate_regression(
                split.y_test,
                best_model.predict(split.x_test),
            ),
        }
    return TrainingResult(
        best_model_name=best_comparison.model_name,
        best_model=best_model,
        metrics=metrics,
        model_comparison=comparisons,
        warnings=warnings,
    )
