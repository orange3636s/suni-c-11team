"""Per-target GBDT pipeline built on the Pareto-selected screening factors.

Each of Y1..Y5 gets its own LGBMRegressor trained only on the factor(s)
that survived screening (src.analysis.screening) for that target. Feature
engineering follows the screening result directly:

  cols[f]            = raw value, NaN preserved (LightGBM handles NaN natively)
  cols[f + "_miss"]  = missingness flag (measurement selection is itself signal)
  cols[f + "_dev"]   = |value - optimal_center| when the factor is u_shape

`optimal_center` is estimated by the screening selector, which is always
run on the training data passed to `fit_target_pipeline` -- callers must
never pass test/held-out rows into that call, or the center estimate leaks.

Final yield is derived as clip(100 - sum(clip(pred_Y1..Y5, 0, None)), 0, 100).

ND: LightGBM replaced HistGradientBoostingRegressor here (previously the
only production model, single-factor per target) -- it was fastest in a
4-way library comparison (LightGBM/sklearn HistGBDT/CatBoost/XGBoost,
32% faster than HistGBDT), fully deterministic across repeated runs at a
fixed seed (CatBoost had R² up +0.029 but a seed-to-seed SD of 0.0041,
which would make the 83-case golden-test suite flake), and handles the
85%/95%-missing R/D columns natively with no imputation step. No
hyperparameter tuning: the factor-measurement rate is the actual bottleneck
here (R² triples once the core factors are measured), so tuning this model
has nothing to gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import (
    DEFAULT_FDR_ALPHA,
    ParetoFactor,
    effective_confidence_tier,
    select_primary_factor,
)
from src.ml.feature_builder import FactorFeatureSpec, build_feature_frame, numeric_series, trained_categories_from_model

FAIL_RATE_TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"]
FINAL_YIELD_COLUMN = "Y"

# 전체 Y의 "저수율 웨이퍼 식별력"(AUC)을 재는 기준 분위수 -- 실제 Y가
# 하위 10%에 들어가는지를 양성 라벨로 삼는다. 학습 팝업의 "하위 10%
# 식별력" 문구는 이 상수를 그대로 가리키므로, 이 값을 바꾸면 프런트
# 문구도 함께 바꿔야 한다.
AUC_BOTTOM_DECILE = 0.10

# ND: fixed, untuned config -- see module docstring. Never add hyperparameter
# search here; the bottleneck is measurement coverage, not model capacity.
LGBM_PARAMS = dict(
    n_estimators=300,
    random_state=0,
    verbose=-1,
)


_numeric = numeric_series  # 하위 호환 별칭 -- 이 모듈 밖에서 import하는 곳은 없지만 이름을 남긴다.


def build_features(df: pd.DataFrame, factors: list[ParetoFactor]) -> pd.DataFrame:
    """Build the engineered feature frame for a fixed set of screening factors.

    `factors` must come from a selector run on training data only; this
    function only reads `df` for the raw factor values, never re-derives
    `optimal_center`.

    실제 컬럼 구성은
    `src.ml.feature_builder.build_feature_frame`이 담당한다 -- 추론 경로
    (`target_hydration._screening_features`)도 같은 함수를 거치므로 두
    경로가 다시 갈라질 수 없다. 여기서는 `categories`를 넘기지 않는다
    (학습 시점에는 컬럼이 스스로 가진 범주를 그대로 쓰는 것이 정의다 --
    범주 목록은 학습 결과에서 *발견*되는 것이지, 제한할 대상이 아니다).
    """
    specs = [
        FactorFeatureSpec(
            feature=factor.feature,
            kind=factor.kind,
            relation_shape=factor.relation_shape,
            optimal_center=factor.optimal_center,
        )
        for factor in factors
    ]
    return build_feature_frame(df, specs)


@dataclass
class TargetPipelineResult:
    target: str
    factors: list[ParetoFactor]
    feature_columns: list[str]
    model: LGBMRegressor
    no_factor_available: bool = False
    baseline_value: float = 0.0


def fit_target_pipeline(
    train_df: pd.DataFrame,
    schema: Schema,
    target: str,
    *,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> TargetPipelineResult:
    """The model always uses the single strongest (by adj_r2) factor for this
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
    model = LGBMRegressor(**LGBM_PARAMS)
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


def _adjusted_r2(r2: float | None, n: int, p: int) -> float | None:
    """1 - (1 - R²)(n - 1)/(n - p - 1). `p`는 5개 타깃 모델이 쓰는 피처
    컬럼의 합집합 크기(보수적 선택 -- 타깃별 부분집합이 아니다)로,
    번들 메타데이터의 `feature_count`와 같은 수여야 한다. 자유도가
    남지 않으면(n <= p + 1) 정의되지 않으므로 None."""
    if r2 is None or not np.isfinite(r2):
        return None
    denominator = n - p - 1
    if denominator <= 0:
        return None
    return float(1.0 - (1.0 - float(r2)) * (n - 1) / denominator)


def _bottom_decile_auc(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """실제 Y가 하위 `AUC_BOTTOM_DECILE` 분위에 드는지를 양성으로 두고,
    점수는 -예측값(예측 수율이 낮을수록 위험이 크다)으로 매긴 ROC AUC.

    분위수가 경계에 몰려 라벨이 한 종류만 나오는 경우(표본이 아주
    작거나 값이 상수인 경우)에는 roc_auc_score가 ValueError를 던지므로
    None을 돌려준다 -- 화면은 "-"로 표시한다."""
    if len(actual) < 2:
        return None
    threshold = float(np.quantile(actual, AUC_BOTTOM_DECILE))
    labels = actual <= threshold
    if labels.all() or not labels.any():
        return None
    try:
        return float(roc_auc_score(labels, -predicted))
    except ValueError:
        return None


def _pooled_metrics(actual: np.ndarray, predicted: np.ndarray, *, feature_count: int) -> dict[str, float | None]:
    """전체 Y(홀드아웃 단일 집계) 전용 지표 -- `_metrics`의 회귀 지표에
    Adjusted R²와 하위 10% 식별 AUC를 더한다. 타깃별(Y1~Y5) 카드는
    모드별 근거 강도를 원인분석 화면에서 따로 보여주므로 여기에
    해당하지 않는다."""
    metrics: dict[str, float | None] = dict(_metrics(actual, predicted))
    valid = np.isfinite(actual) & np.isfinite(predicted)
    metrics["r2_adjusted"] = _adjusted_r2(metrics["r2"], int(metrics["n"]), feature_count)
    metrics["auc"] = _bottom_decile_auc(actual[valid], predicted[valid])
    return metrics


def union_feature_columns(target_results: dict[str, TargetPipelineResult]) -> list[str]:
    """5개 타깃 모델이 쓰는 피처 컬럼의 합집합 -- 번들 메타데이터의
    `feature_columns`/`feature_count`와 Adjusted R²의 `p`가 같은 수를
    가리키도록 한 곳에서만 센다(화면이 "n=X · 피처 Y"로 나란히 보여준다)."""
    return sorted(
        {
            column
            for target in FAIL_RATE_TARGETS
            for column in target_results[target].feature_columns
        }
    )


@dataclass
class PipelineEvaluation:
    target_results: dict[str, TargetPipelineResult]
    test_predictions: dict[str, np.ndarray]
    final_yield_prediction: np.ndarray
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Schema,
    *,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> PipelineEvaluation:
    target_results: dict[str, TargetPipelineResult] = {}
    test_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float | None]] = {}

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
    metrics[FINAL_YIELD_COLUMN] = _pooled_metrics(
        _numeric(test_df[FINAL_YIELD_COLUMN]).to_numpy(),
        final_yield_prediction,
        feature_count=len(union_feature_columns(target_results)),
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
                "adj_r2": None,
                "degree": None,
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
            "adj_r2": factor.adj_r2,
            "degree": factor.degree,
            "contribution_pct": factor.contribution_pct,
            "relation_shape": factor.relation_shape,
            "optimal_center": factor.optimal_center,
            "cumulative_pct": factor.cumulative_pct,
            "p_value": factor.p_value,
            "q_value": factor.q_value,
            "confidence_tier": effective_confidence_tier(factor.adj_r2, factor.p_value, under_sampled=factor.under_sampled),
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "n": metrics["n"],
            # Config 인자의 학습 시점
            # 범주 목록 -- 추론 시 새 범주/결측을 정리하는 기준이 된다.
            # R/D는 해당 없음(None).
            "categories": trained_categories_from_model(result.model) if factor.kind == "Config" else None,
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
    import lightgbm
    import sklearn

    from src.ml.hybrid import (
        HybridMultiYBundle,
        HybridTrainingResult,
        PIPELINE_VERSION,
    )

    target_models = {target: evaluation.target_results[target].model for target in FAIL_RATE_TARGETS}
    # Adjusted R²의 p와 같은 정의를 쓴다 -- union_feature_columns 참고.
    all_feature_columns = union_feature_columns(evaluation.target_results)
    bundle = HybridMultiYBundle(feature_columns=all_feature_columns, target_models=target_models)

    target_metrics = target_metrics_summary(evaluation)
    metadata = {
        "schema_version": "semicon_yield_v2",
        "pipeline_version": "screening_pareto_pipeline_v1",
        "model_version": "screening_pareto_pipeline_v1",
        "model_type": "LGBMRegressor",
        "bundle_type": "screening_pareto_pipeline",
        "model_name": "스크리닝 기반 Y1~Y5 LightGBM",
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
        "random_state": LGBM_PARAMS["random_state"],
        "lgbm_params": LGBM_PARAMS,
        "lightgbm_version": lightgbm.__version__,
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
