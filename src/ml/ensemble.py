from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


logger = logging.getLogger(__name__)

EnsembleSize = Literal["auto", "2", "3"]
EnsembleMethod = Literal[
    "auto", "simple_average", "weighted_average", "stacking"
]


@dataclass(frozen=True)
class EnsembleOptions:
    enabled: bool = True
    size: EnsembleSize = "auto"
    method: EnsembleMethod = "auto"
    min_improvement: float = 0.01
    diversity_check: bool = True
    max_base_models: int = 3

    def validate(self) -> None:
        if self.size not in {"auto", "2", "3"}:
            raise ValueError("앙상블 크기는 auto, 2, 3 중 하나여야 합니다.")
        if self.method not in {
            "auto", "simple_average", "weighted_average", "stacking"
        }:
            raise ValueError("지원하지 않는 앙상블 결합 방식입니다.")
        if not 0.0 <= self.min_improvement <= 1.0:
            raise ValueError("최소 개선 기준은 0~1 사이여야 합니다.")
        if self.max_base_models not in {2, 3}:
            raise ValueError("최대 Base Model 수는 2 또는 3이어야 합니다.")


@dataclass
class EnsembleRegressor:
    """A persistable prediction-only wrapper for selected fitted base models."""

    models: dict[str, Any]
    weights: dict[str, float]
    method: str = "weighted_average"
    meta_model: Any | None = None

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        predictions = [
            np.asarray(model.predict(features), dtype=float)
            for model in self.models.values()
        ]
        return np.column_stack(predictions)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        matrix = self._matrix(features)
        if self.method == "stacking" and self.meta_model is not None:
            return np.asarray(self.meta_model.predict(matrix), dtype=float)
        weights = np.asarray(
            [self.weights[name] for name in self.models], dtype=float
        )
        return matrix @ weights

    def prediction_spread(self, features: pd.DataFrame) -> np.ndarray:
        matrix = self._matrix(features)
        return np.std(matrix, axis=1) if matrix.shape[1] > 1 else np.zeros(len(matrix))


@dataclass
class TargetSelection:
    model: Any
    selected_type: str
    method: str
    base_models: list[str]
    weights: dict[str, float]
    oof_prediction: np.ndarray
    metrics: dict[str, float | None]
    best_single_name: str
    best_single_metrics: dict[str, float | None]
    improvement_over_single: dict[str, float | None]
    prediction_correlations: dict[str, dict[str, float | None]]
    residual_correlations: dict[str, dict[str, float | None]]
    fold_rmse: list[float]
    agreement: dict[str, float | None]
    warnings: list[str] = field(default_factory=list)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
    }


def _value(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def optimize_non_negative_weights(
    predictions: np.ndarray, actual: np.ndarray
) -> np.ndarray:
    """Least-squares weights projected onto the probability simplex."""
    count = predictions.shape[1]
    try:
        weights, *_ = np.linalg.lstsq(predictions, actual, rcond=None)
        weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    except np.linalg.LinAlgError:
        weights = np.ones(count, dtype=float)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.full(count, 1.0 / count)
    return weights / total


def _correlation(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 1:
        return np.ones((1, 1), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.corrcoef(values, rowvar=False)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _named_matrix(matrix: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    return {
        left: {right: float(matrix[i, j]) for j, right in enumerate(names)}
        for i, left in enumerate(names)
    }


def _fold_rmse(
    actual: np.ndarray, predicted: np.ndarray, fold_ids: np.ndarray
) -> list[float]:
    return [
        float(np.sqrt(mean_squared_error(actual[fold_ids == fold], predicted[fold_ids == fold])))
        for fold in sorted(set(fold_ids.tolist()))
    ]


def _allowed_sizes(options: EnsembleOptions, available: int) -> list[int]:
    maximum = min(options.max_base_models, available)
    if options.size == "auto":
        return [size for size in (2, 3) if size <= maximum]
    requested = int(options.size)
    return [requested] if requested <= maximum else []


def select_target_ensemble(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    candidate_factory: Callable[[], dict[str, Any]],
    *,
    options: EnsembleOptions | None = None,
    folds: int = 3,
    prediction_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> TargetSelection:
    """Select only from group-OOF predictions, then refit on all development rows."""
    options = options or EnsembleOptions()
    options.validate()
    y = np.asarray(target, dtype=float)
    group_values = np.asarray(groups)
    n_splits = min(folds, len(np.unique(group_values)))
    if n_splits < 2:
        raise ValueError("Inner Group CV를 위한 Lot 그룹이 2개 이상 필요합니다.")
    transform = prediction_transform or (lambda values: values)
    candidates = candidate_factory()
    predictions: dict[str, np.ndarray] = {}
    fold_ids = np.full(len(features), -1, dtype=int)
    warnings: list[str] = []
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(features, y, group_values))
    for fold, (_, holdout) in enumerate(splits):
        fold_ids[holdout] = fold
    for name, estimator in candidates.items():
        oof = np.full(len(features), np.nan, dtype=float)
        try:
            for train_index, holdout_index in splits:
                fitted = clone(estimator)
                fitted.fit(features.iloc[train_index], y[train_index])
                oof[holdout_index] = transform(
                    np.asarray(fitted.predict(features.iloc[holdout_index]), dtype=float)
                )
            if not np.isfinite(oof).all():
                raise ValueError("OOF prediction이 완성되지 않았습니다.")
            predictions[name] = oof
        except Exception as exc:
            logger.warning("앙상블 Base Model 제외: %s", name, exc_info=True)
            warnings.append(f"{name} 후보 제외: {type(exc).__name__}")
    if not predictions:
        raise ValueError("OOF 학습에 성공한 Base Model이 없습니다.")

    single_metrics = {
        name: regression_metrics(y, prediction)
        for name, prediction in predictions.items()
    }
    ranked = sorted(
        predictions,
        key=lambda name: (
            _value(single_metrics[name]["rmse"], float("inf")),
            -_value(single_metrics[name]["r2"], float("-inf")),
        ),
    )
    best_single = ranked[0]
    best_prediction = predictions[best_single]
    best_metrics = single_metrics[best_single]
    best_spec: tuple[str, list[str], np.ndarray, Any | None] = (
        "single", [best_single], np.ones(1), None
    )

    methods = (
        [options.method]
        if options.method != "auto"
        else ["simple_average", "weighted_average", "stacking"]
    )
    top_names = ranked[: min(max(options.max_base_models, 3), len(ranked))]
    candidate_scores: list[tuple[float, float, float, tuple[str, list[str], np.ndarray, Any | None], np.ndarray]] = []
    if options.enabled:
        all_matrix = np.column_stack([predictions[name] for name in top_names])
        all_correlation = _correlation(all_matrix)
        for size in _allowed_sizes(options, len(top_names)):
            for names_tuple in itertools.combinations(top_names, size):
                names = list(names_tuple)
                indices = [top_names.index(name) for name in names]
                if options.diversity_check and size > 1:
                    pair_values = [
                        abs(float(all_correlation[left, right]))
                        for left in indices
                        for right in indices
                        if left < right
                    ]
                    # Perfectly duplicated OOF predictions add no useful diversity.
                    if pair_values and min(pair_values) >= 0.9999:
                        continue
                matrix = np.column_stack([predictions[name] for name in names])
                for method in methods:
                    meta_model = None
                    if method == "simple_average":
                        weights = np.full(size, 1.0 / size)
                        combined = matrix @ weights
                    elif method == "weighted_average":
                        weights = optimize_non_negative_weights(matrix, y)
                        combined = matrix @ weights
                    elif method == "stacking":
                        if len(y) < max(30, size * 10):
                            continue
                        # Cross-fit the simple regularized stacker using the same
                        # group boundaries; no in-sample base predictions are used.
                        combined = np.zeros(len(y), dtype=float)
                        for train_index, holdout_index in splits:
                            stacker = Ridge(alpha=1.0)
                            stacker.fit(matrix[train_index], y[train_index])
                            combined[holdout_index] = stacker.predict(matrix[holdout_index])
                        meta_model = Ridge(alpha=1.0).fit(matrix, y)
                        weights = np.full(size, 1.0 / size)
                    else:
                        continue
                    metric = regression_metrics(y, combined)
                    fold_values = _fold_rmse(y, combined, fold_ids)
                    candidate_scores.append((
                        _value(metric["rmse"], float("inf")),
                        float(np.std(fold_values)),
                        -_value(metric["r2"], float("-inf")),
                        (method, names, weights, meta_model),
                        combined,
                    ))

    if candidate_scores:
        _, _, _, spec, combined = min(candidate_scores, key=lambda item: item[:3])
        ensemble_metrics = regression_metrics(y, combined)
        single_rmse = _value(best_metrics["rmse"], float("inf"))
        ensemble_rmse = _value(ensemble_metrics["rmse"], float("inf"))
        relative_improvement = (
            (single_rmse - ensemble_rmse) / single_rmse if single_rmse > 0 else 0.0
        )
        r2_gain = _value(ensemble_metrics["r2"], 0.0) - _value(best_metrics["r2"], 0.0)
        ensemble_folds = _fold_rmse(y, combined, fold_ids)
        single_folds = _fold_rmse(y, best_prediction, fold_ids)
        stability_gain = float(np.std(single_folds) - np.std(ensemble_folds))
        if relative_improvement >= options.min_improvement or (
            options.min_improvement == 0 and r2_gain > 0
        ) or (relative_improvement >= 0 and stability_gain > 0.01 * max(np.mean(single_folds), 1e-9)):
            best_spec = spec
            best_prediction = combined

    method, selected_names, selected_weights, meta_model = best_spec
    fitted_models: dict[str, Any] = {}
    for name in selected_names:
        fitted = clone(candidates[name])
        fitted.fit(features, y)
        fitted_models[name] = fitted
    if method == "single":
        selected_model = fitted_models[selected_names[0]]
        weights = {selected_names[0]: 1.0}
    else:
        weights = {
            name: float(selected_weights[index])
            for index, name in enumerate(selected_names)
        }
        selected_model = EnsembleRegressor(
            models=fitted_models,
            weights=weights,
            method=method,
            meta_model=meta_model,
        )

    selected_matrix = np.column_stack([predictions[name] for name in selected_names])
    selected_metrics = regression_metrics(y, best_prediction)
    selected_folds = _fold_rmse(y, best_prediction, fold_ids)
    prediction_corr = _correlation(selected_matrix)
    residual_corr = _correlation(y[:, None] - selected_matrix)
    spread = np.std(selected_matrix, axis=1) if len(selected_names) > 1 else np.zeros(len(y))
    single_rmse = _value(best_metrics["rmse"], 0.0)
    selected_rmse = _value(selected_metrics["rmse"], 0.0)
    return TargetSelection(
        model=selected_model,
        selected_type="single" if method == "single" else f"{len(selected_names)}-model-ensemble",
        method=method,
        base_models=selected_names,
        weights=weights,
        oof_prediction=best_prediction,
        metrics=selected_metrics,
        best_single_name=best_single,
        best_single_metrics=best_metrics,
        improvement_over_single={
            "rmse_relative": ((single_rmse - selected_rmse) / single_rmse) if single_rmse else 0.0,
            "r2_absolute": _value(selected_metrics["r2"], 0.0) - _value(best_metrics["r2"], 0.0),
        },
        prediction_correlations=_named_matrix(prediction_corr, selected_names),
        residual_correlations=_named_matrix(residual_corr, selected_names),
        fold_rmse=selected_folds,
        agreement={
            "mean_prediction_spread": float(np.mean(spread)),
            "max_prediction_spread": float(np.max(spread)),
            "mean_pairwise_correlation": (
                float(np.mean(prediction_corr[np.triu_indices(len(selected_names), 1)]))
                if len(selected_names) > 1 else 1.0
            ),
        },
        warnings=warnings,
    )


def selection_metadata(selection: TargetSelection) -> dict[str, Any]:
    return {
        "selected_type": selection.selected_type,
        "method": selection.method,
        "base_models": selection.base_models,
        "weights": selection.weights,
        "best_single_model": selection.best_single_name,
        "best_single_metrics": selection.best_single_metrics,
        "ensemble_metrics": selection.metrics,
        "improvement_over_single": selection.improvement_over_single,
        "prediction_correlations": selection.prediction_correlations,
        "residual_correlations": selection.residual_correlations,
        "fold_rmse": selection.fold_rmse,
        "fold_rmse_std": float(np.std(selection.fold_rmse)),
        "worst_fold_rmse": max(selection.fold_rmse),
        "agreement": selection.agreement,
    }
