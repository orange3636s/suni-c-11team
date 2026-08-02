from __future__ import annotations

import gc
import os
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
from src.ml.memory_usage import log_memory_stage


logger = logging.getLogger(__name__)
NATIVE_MISSING_MODELS = {
    "HistGradientBoostingRegressor",
    "HistGradientBoostingPoisson",
    "XGBoostRegressor",
    "CatBoostRegressor",
}
FLAG_ONLY_OUTLIER_MODELS = {
    "HistGradientBoostingRegressor",
    "HistGradientBoostingPoisson",
    "XGBoostRegressor",
    "CatBoostRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
}


def missing_strategy_for_model(model_name: str) -> str:
    return "native" if model_name in NATIVE_MISSING_MODELS else "median"


def outlier_strategy_for_model(model_name: str) -> str:
    return "flag_only" if model_name in FLAG_ONLY_OUTLIER_MODELS else "iqr"


class OutlierPolicyTransformer(TransformerMixin, BaseEstimator):
    """Learn train-only IQR boundaries and preserve raw values for flag_only."""

    def __init__(
        self,
        strategy: str = "flag_only",
        lower_multiplier: float = 1.5,
        upper_multiplier: float = 1.5,
    ) -> None:
        self.strategy = strategy
        self.lower_multiplier = lower_multiplier
        self.upper_multiplier = upper_multiplier

    def fit(self, x: Any, y: Any = None) -> "OutlierPolicyTransformer":
        if self.strategy not in {"flag_only", "iqr", "none"}:
            raise ValueError(f"지원하지 않는 이상치 처리 전략입니다: {self.strategy}")
        values = np.asarray(x, dtype=np.float32)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(
            list(getattr(x, "columns", [f"x{index}" for index in range(values.shape[1])])),
            dtype=object,
        )
        lower: list[float] = []
        upper: list[float] = []
        for index in range(values.shape[1]):
            observed = values[:, index][np.isfinite(values[:, index])]
            if len(observed) < 4:
                lower.append(float("nan"))
                upper.append(float("nan"))
                continue
            q1, q3 = np.quantile(observed, [0.25, 0.75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                lower.append(float("nan"))
                upper.append(float("nan"))
                continue
            lower.append(float(q1 - self.lower_multiplier * iqr))
            upper.append(float(q3 + self.upper_multiplier * iqr))
        self.lower_bounds_ = np.asarray(lower, dtype=np.float32)
        self.upper_bounds_ = np.asarray(upper, dtype=np.float32)
        return self

    def transform(self, x: Any) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        valid_bounds = np.isfinite(self.lower_bounds_) & np.isfinite(self.upper_bounds_)
        flags = np.zeros_like(values, dtype=np.float32)
        if valid_bounds.any():
            flags[:, valid_bounds] = (
                (values[:, valid_bounds] < self.lower_bounds_[valid_bounds])
                | (values[:, valid_bounds] > self.upper_bounds_[valid_bounds])
            ).astype(np.float32)
        if self.strategy == "flag_only":
            return np.hstack([values, flags]).astype(np.float32, copy=False)
        if self.strategy == "iqr":
            clipped = values.copy()
            clipped[:, valid_bounds] = np.clip(
                clipped[:, valid_bounds],
                self.lower_bounds_[valid_bounds],
                self.upper_bounds_[valid_bounds],
            )
            return clipped.astype(np.float32, copy=False)
        return values.astype(np.float32, copy=False)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = np.asarray(
            input_features if input_features is not None else self.feature_names_in_,
            dtype=object,
        )
        if self.strategy != "flag_only":
            return names
        return np.concatenate([names, np.asarray([f"{name}_outlier" for name in names], dtype=object)])


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
    missing_strategy: str = "median"
    outlier_strategy: str = "flag_only"
    missing_indicator: bool = True
    outlier_indicator: bool = True
    model_strategies: dict[str, str] = field(default_factory=dict)
    model_outlier_strategies: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False


def _build_preprocessor(
    dataset: PreparedDataset,
    *,
    missing_strategy: str = "median",
    outlier_strategy: str = "flag_only",
) -> ColumnTransformer:
    if missing_strategy not in {"native", "median"}:
        raise ValueError(f"지원하지 않는 모델 결측치 전략입니다: {missing_strategy}")
    transformers: list[tuple[str, Any, list[str]]] = []
    numeric_columns = [column for column in dataset.numeric_columns if not column.endswith("_missing")]
    indicator_columns = [column for column in dataset.numeric_columns if column.endswith("_missing")]
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [
            ("outliers", OutlierPolicyTransformer(strategy=outlier_strategy)),
        ]
        if missing_strategy == "median":
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric_columns,
            )
        )
    if indicator_columns:
        transformers.append(("missing_indicators", "passthrough", indicator_columns))
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
                                encoded_missing_value=-1,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                dataset.categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def _candidate_estimators(random_state: int) -> dict[str, Any]:
    return {
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=random_state,
        ),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        ),
        "DummyRegressor": DummyRegressor(strategy="mean"),
    }


def train_regression_models(
    dataset: PreparedDataset,
    split: DatasetSplit,
    random_state: int = RANDOM_STATE,
) -> TrainingResult:
    comparisons: list[ModelComparison] = []
    warnings: list[str] = []
    best_model: Pipeline | None = None
    best_model_name: str | None = None
    best_rank = (float("inf"), float("inf"))

    log_memory_stage(
        logger,
        "legacy_training_start",
        rows=len(dataset.features),
        features=len(dataset.feature_columns),
    )

    for model_name, estimator in _candidate_estimators(random_state).items():
        logger.info("모델 학습 시작: %s", model_name)
        pipeline = Pipeline(
            steps=[
                (
                    "features",
                    _build_preprocessor(
                        dataset,
                        missing_strategy=missing_strategy_for_model(model_name),
                        outlier_strategy=outlier_strategy_for_model(model_name),
                    ),
                ),
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
            del pipeline
            gc.collect()
            log_memory_stage(
                logger,
                "legacy_candidate_failed",
                model=model_name,
            )
            continue

        rmse = (
            validation_metrics.rmse
            if validation_metrics.rmse is not None
            else float("inf")
        )
        r2 = (
            validation_metrics.r2
            if validation_metrics.r2 is not None
            else float("-inf")
        )
        candidate_rank = (rmse, -r2)
        if candidate_rank < best_rank:
            previous_model = best_model
            best_model = pipeline
            best_model_name = model_name
            best_rank = candidate_rank
            if previous_model is not None:
                del previous_model
        else:
            del pipeline
        comparisons.append(
            ModelComparison(
                model_name=model_name,
                status="success",
                validation=validation_metrics,
            )
        )
        logger.info("모델 학습 완료: %s", model_name)
        gc.collect()
        log_memory_stage(
            logger,
            "legacy_candidate_complete",
            model=model_name,
        )

    successful_comparisons = [
        item for item in comparisons if item.status == "success"
    ]
    if not successful_comparisons:
        raise ValueError("학습에 성공한 모델이 없습니다.")

    if best_model is None or best_model_name is None:
        raise ValueError("학습에 성공한 모델이 없습니다.")
    best_comparison = next(
        item
        for item in successful_comparisons
        if item.model_name == best_model_name
    )
    best_comparison.selected = True
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
    result = TrainingResult(
        best_model_name=best_comparison.model_name,
        best_model=best_model,
        metrics=metrics,
        model_comparison=comparisons,
        warnings=warnings,
        missing_strategy=missing_strategy_for_model(best_comparison.model_name),
        outlier_strategy=outlier_strategy_for_model(best_comparison.model_name),
        missing_indicator=any(column.endswith("_missing") for column in dataset.numeric_columns),
        outlier_indicator=outlier_strategy_for_model(best_comparison.model_name) == "flag_only",
        model_strategies={
            item.model_name: missing_strategy_for_model(item.model_name)
            for item in successful_comparisons
        },
        model_outlier_strategies={
            item.model_name: outlier_strategy_for_model(item.model_name)
            for item in successful_comparisons
        },
        fallback_used=False,
    )
    log_memory_stage(
        logger,
        "legacy_training_complete",
        selected=result.best_model_name,
    )
    return result
