"""Per-target GBDT pipeline built on the Pareto-selected screening factors.

Each of Y1..Y5 gets its own HistGradientBoostingRegressor trained only on
the factor(s) that survived screening (src.analysis.screening) for that
target. Feature engineering follows the screening result directly:

  cols[f]            = raw value, NaN preserved (HGBR handles NaN natively)
  cols[f + "_miss"]  = missingness flag (measurement selection is itself signal)
  cols[f + "_dev"]   = |value - optimal_center| when the factor is u_shape

`optimal_center` is estimated by the screening selector, which is always
run on the training data passed to `fit_target_pipeline` -- callers must
never pass test/held-out rows into that call, or the center estimate leaks.

Final yield is derived as clip(100 - sum(clip(pred_Y1..Y5, 0, None)), 0, 100).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import (
    DEFAULT_CUTOFF,
    DEFAULT_FDR_ALPHA,
    ParetoFactor,
    select_pareto_factors,
)

FAIL_RATE_TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"]
FINAL_YIELD_COLUMN = "Y"

HGBR_PARAMS = dict(
    max_iter=300,
    learning_rate=0.06,
    max_depth=6,
    random_state=42,
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_features(df: pd.DataFrame, factors: list[ParetoFactor]) -> pd.DataFrame:
    """Build the engineered feature frame for a fixed set of screening factors.

    `factors` must come from a selector run on training data only; this
    function only reads `df` for the raw factor values, never re-derives
    `optimal_center`.
    """
    columns: dict[str, pd.Series] = {}
    for factor in factors:
        if factor.kind == "Config":
            raw = df[factor.feature].astype("category")
            columns[factor.feature] = raw
            columns[f"{factor.feature}_miss"] = raw.isna().astype("int8")
            continue
        raw = _numeric(df[factor.feature])
        columns[factor.feature] = raw.astype("float32")
        columns[f"{factor.feature}_miss"] = raw.isna().astype("int8")
        if factor.relation_shape == "u_shape" and factor.optimal_center is not None:
            columns[f"{factor.feature}_dev"] = (raw - factor.optimal_center).abs().astype("float32")
    return pd.DataFrame(columns, index=df.index)


@dataclass
class TargetPipelineResult:
    target: str
    factors: list[ParetoFactor]
    feature_columns: list[str]
    model: HistGradientBoostingRegressor
    no_significant_factor: bool = False
    baseline_value: float = 0.0


def fit_target_pipeline(
    train_df: pd.DataFrame,
    schema: Schema,
    target: str,
    *,
    cutoff: float = DEFAULT_CUTOFF,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> TargetPipelineResult:
    selection = select_pareto_factors(train_df, schema, target, cutoff=cutoff, fdr_alpha=fdr_alpha)
    baseline_value = float(_numeric(train_df[target]).mean())

    if selection.no_significant_factor or not selection.factors:
        return TargetPipelineResult(
            target=target,
            factors=[],
            feature_columns=[],
            model=None,  # type: ignore[arg-type]
            no_significant_factor=True,
            baseline_value=baseline_value,
        )

    features = build_features(train_df, selection.factors)
    model = HistGradientBoostingRegressor(**HGBR_PARAMS)
    model.fit(features, _numeric(train_df[target]))
    return TargetPipelineResult(
        target=target,
        factors=selection.factors,
        feature_columns=list(features.columns),
        model=model,
        baseline_value=baseline_value,
    )


def predict_target(result: TargetPipelineResult, df: pd.DataFrame) -> np.ndarray:
    if result.no_significant_factor:
        return np.full(len(df), result.baseline_value, dtype=np.float32)
    features = build_features(df, result.factors).reindex(columns=result.feature_columns)
    return np.asarray(result.model.predict(features), dtype=np.float32)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    return {
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "mae": float(mean_absolute_error(actual, predicted)),
        "n": int(len(actual)),
    }


@dataclass
class PipelineEvaluation:
    target_results: dict[str, TargetPipelineResult]
    test_predictions: dict[str, np.ndarray]
    final_yield_prediction: np.ndarray
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Schema,
    *,
    cutoff: float = DEFAULT_CUTOFF,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> PipelineEvaluation:
    target_results: dict[str, TargetPipelineResult] = {}
    test_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    for target in FAIL_RATE_TARGETS:
        result = fit_target_pipeline(train_df, schema, target, cutoff=cutoff, fdr_alpha=fdr_alpha)
        target_results[target] = result
        prediction = predict_target(result, test_df)
        test_predictions[target] = prediction
        metrics[target] = _metrics(_numeric(test_df[target]).to_numpy(), prediction)

    clipped_sum = np.zeros(len(test_df), dtype=np.float32)
    for target in FAIL_RATE_TARGETS:
        clipped_sum += np.maximum(test_predictions[target], 0.0)
    final_yield_prediction = np.clip(100.0 - clipped_sum, 0.0, 100.0)
    metrics[FINAL_YIELD_COLUMN] = _metrics(
        _numeric(test_df[FINAL_YIELD_COLUMN]).to_numpy(), final_yield_prediction
    )

    return PipelineEvaluation(
        target_results=target_results,
        test_predictions=test_predictions,
        final_yield_prediction=final_yield_prediction,
        metrics=metrics,
    )
