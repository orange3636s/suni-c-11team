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
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import (
    DEFAULT_FDR_ALPHA,
    ParetoFactor,
    effective_confidence_tier,
    select_primary_factor,
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
    no_factor_available: bool = False
    baseline_value: float = 0.0


def fit_target_pipeline(
    train_df: pd.DataFrame,
    schema: Schema,
    target: str,
    *,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> TargetPipelineResult:
    """The model always uses the single strongest (by eps2) factor for this
    target, regardless of its p-value -- confidence is communicated via a
    tier badge on the summary card, not by falling back to a baseline
    model. The baseline-constant fallback only fires when NO factor in the
    whole pool clears its own minimum sample size (`select_primary_factor`
    returns None) -- see that function's docstring for what "분석 불가"
    means precisely.
    """
    factor = select_primary_factor(train_df, schema, target, fdr_alpha=fdr_alpha)
    baseline_value = float(_numeric(train_df[target]).mean())

    if factor is None:
        return TargetPipelineResult(
            target=target,
            factors=[],
            feature_columns=[],
            model=None,  # type: ignore[arg-type]
            no_factor_available=True,
            baseline_value=baseline_value,
        )

    features = build_features(train_df, [factor])
    model = HistGradientBoostingRegressor(**HGBR_PARAMS)
    model.fit(features, _numeric(train_df[target]))
    return TargetPipelineResult(
        target=target,
        factors=[factor],
        feature_columns=list(features.columns),
        model=model,
        baseline_value=baseline_value,
    )


def predict_target(result: TargetPipelineResult, df: pd.DataFrame) -> np.ndarray:
    if result.no_factor_available:
        return np.full(len(df), result.baseline_value, dtype=np.float32)
    features = build_features(df, result.factors).reindex(columns=result.feature_columns)
    return np.asarray(result.model.predict(features), dtype=np.float32)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    mse = float(mean_squared_error(actual, predicted))
    return {
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "rmse": mse**0.5,
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
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
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> PipelineEvaluation:
    target_results: dict[str, TargetPipelineResult] = {}
    test_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    for target in FAIL_RATE_TARGETS:
        result = fit_target_pipeline(train_df, schema, target, fdr_alpha=fdr_alpha)
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


def target_metrics_summary(evaluation: PipelineEvaluation) -> dict[str, dict[str, object]]:
    """Per-target detail for the training tab's 5 performance cards. The
    primary factor is always surfaced when one exists -- see
    `select_primary_factor`'s docstring: `no_factor_available` only fires
    when every candidate fails its own minimum-sample-size gate, never
    because of a low p-value.
    """
    summary: dict[str, dict[str, object]] = {}
    for target in FAIL_RATE_TARGETS:
        result = evaluation.target_results[target]
        metrics = evaluation.metrics[target]
        if result.no_factor_available:
            summary[target] = {
                "no_factor_available": True,
                "feature": None,
                "kind": None,
                "eps2": None,
                "contribution_pct": None,
                "relation_shape": None,
                "optimal_center": None,
                "cumulative_pct": None,
                "p_value": None,
                "q_value": None,
                "confidence_tier": None,
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "n": metrics["n"],
            }
            continue
        factor = result.factors[0]
        summary[target] = {
            "no_factor_available": False,
            "feature": factor.feature,
            "kind": factor.kind,
            "eps2": factor.eps2,
            "contribution_pct": factor.contribution_pct,
            "relation_shape": factor.relation_shape,
            "optimal_center": factor.optimal_center,
            "cumulative_pct": factor.cumulative_pct,
            "p_value": factor.p_value,
            "q_value": factor.q_value,
            "confidence_tier": effective_confidence_tier(factor.eps2, factor.p_value, under_sampled=factor.under_sampled),
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "n": metrics["n"],
        }
    return summary


def build_hybrid_training_result(
    evaluation: PipelineEvaluation,
    *,
    source_filename: str,
    dataset_rows: dict[str, int],
    train_lots: list[str],
    test_lots: list[str],
    split_method: str,
):
    """Package a PipelineEvaluation into the existing hybrid bundle format so
    it can be persisted/listed/deleted through the already-kept
    save_hybrid_bundle / list_prediction_models / delete_model_bundle
    machinery, without that machinery needing to know pipeline.py exists.

    The bundle's `.predict()` path is intentionally left unusable here (each
    target model uses a different, small feature set -- there is no more
    live prediction-serving endpoint to call it from anyway).
    """
    import sklearn

    from src.ml.hybrid import (
        HybridMultiYBundle,
        HybridTrainingResult,
        PIPELINE_VERSION,
    )

    target_models = {target: evaluation.target_results[target].model for target in FAIL_RATE_TARGETS}
    all_feature_columns = sorted(
        {
            column
            for target in FAIL_RATE_TARGETS
            for column in evaluation.target_results[target].feature_columns
        }
    )
    bundle = HybridMultiYBundle(feature_columns=all_feature_columns, target_models=target_models)

    target_metrics = target_metrics_summary(evaluation)
    metadata = {
        "schema_version": "semicon_yield_v2",
        "pipeline_version": "screening_pareto_pipeline_v1",
        "model_version": "screening_pareto_pipeline_v1",
        "model_type": "HistGradientBoostingRegressor",
        "bundle_type": "screening_pareto_pipeline",
        "model_name": "스크리닝 기반 Y1~Y5 GBDT",
        "target": FINAL_YIELD_COLUMN,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_filename": source_filename,
        "feature_columns": all_feature_columns,
        "feature_count": len(all_feature_columns),
        "available_targets": FAIL_RATE_TARGETS,
        "analysis_only_targets": [FINAL_YIELD_COLUMN, *[f"Y{i}" for i in range(6, 11)]],
        "split_method": split_method,
        "dataset_rows": dataset_rows,
        "target_metrics": target_metrics,
        "final_y_formula": "clip(100 - sum(max(predicted_Y1..predicted_Y5, 0)), 0, 100)",
        "metrics": {"test": {k: v for k, v in evaluation.metrics[FINAL_YIELD_COLUMN].items() if k != "n"}},
        "final_y_metrics": {"test": evaluation.metrics[FINAL_YIELD_COLUMN]},
        "target_model_artifacts": {target: f"target_{target}.joblib" for target in FAIL_RATE_TARGETS},
        "split_metadata": {"lot_assignments": {"train": sorted(train_lots), "test": sorted(test_lots)}},
        "random_state": HGBR_PARAMS["random_state"],
        "hgbr_params": HGBR_PARAMS,
        "sklearn_version": sklearn.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    oof_predictions = {
        **{f"test_{target}": evaluation.test_predictions[target] for target in FAIL_RATE_TARGETS},
        "test_predicted_Y": evaluation.final_yield_prediction,
    }

    return HybridTrainingResult(
        bundle=bundle,
        metadata=metadata,
        warnings=[],
        oof_predictions=oof_predictions,
    )
