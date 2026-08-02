from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.ml.inference import (
    DEFAULT_DANGER_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    InferenceInputError,
    LoadedPredictionModel,
    ModelLoadError,
    PredictionResult,
    predict_dataframe,
    prepare_inference_features,
)
from src.ml.ensemble import EnsembleRegressor
from src.ml.memory_usage import log_memory_stage
from src.ml.model_io import to_json_safe
from src.preprocessing import preprocess_dataframe


logger = logging.getLogger(__name__)

DEFAULT_MAX_EXPLAIN_ROWS = 500
MAX_EXPLAIN_ROWS = 500
DEFAULT_TOP_N = 20
MAX_TOP_N = 30
DEFAULT_WAFER_TOP_N = 5
FEATURE_PATTERN = re.compile(
    r"^(?P<step>Step\d+)_(?P<type>R|D|EQ|Model|Equipment|Chamber)(?P<name>.*)$",
    re.IGNORECASE,
)


@dataclass
class ShapComputation:
    values: np.ndarray
    base_values: np.ndarray
    feature_values: np.ndarray
    feature_names: list[str]
    explanation_method: str
    is_fallback: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExplainResult:
    model_id: str
    target: str
    model_name: str
    total_rows: int
    analyzed_rows: int
    sampling_used: bool
    sampling_strategy: str
    explanation_method: str
    is_fallback: bool
    global_importance: list[dict[str, Any]]
    step_summary: list[dict[str, Any]]
    parameter_type_summary: list[dict[str, Any]]
    equipment_summary: list[dict[str, Any]]
    identifier_column: str
    wafer_explanations: list[dict[str, Any]]
    model_quality_warnings: list[str]
    warnings: list[str]
    # 전체 local SHAP는 Lot 집계에만 사용한다. API에는 상위 기여값만 노출한다.
    local_contributions: list[dict[str, Any]] = field(default_factory=list)


def parse_feature_name(feature: str) -> dict[str, str | None]:
    cleaned = feature.split("__", 1)[-1]
    indicator_match = re.search(
        r"_(missing|observed|indicator|outlier|outlier_flag)$",
        cleaned,
        re.IGNORECASE,
    )
    if indicator_match:
        base = cleaned[: indicator_match.start()]
        step_match = re.match(r"^(Step\d+)_", base, re.IGNORECASE)
        return {
            "step": step_match.group(1) if step_match else "unknown",
            "parameter_type": "Measurement Pattern",
            "parameter_name": indicator_match.group(1).lower(),
            "equipment": None,
            "original_feature_name": cleaned,
        }
    match = FEATURE_PATTERN.match(cleaned)
    if not match:
        return {
            "step": "unknown",
            "parameter_type": "unknown",
            "parameter_name": cleaned,
            "equipment": None,
            "original_feature_name": feature,
        }
    raw_type = match.group("type")
    parameter_type = raw_type.upper() if raw_type.upper() in {"R", "D", "EQ"} else raw_type.title()
    suffix = match.group("name").lstrip("_")
    canonical = cleaned
    if parameter_type in {"Model", "Equipment", "Chamber"}:
        canonical = f"{match.group('step')}_{parameter_type}"
    elif parameter_type == "EQ" and suffix:
        canonical = f"{match.group('step')}_EQ"
    return {
        "step": match.group("step"),
        "parameter_type": parameter_type,
        "parameter_name": suffix or parameter_type,
        "equipment": canonical if parameter_type in {"EQ", "Equipment"} else None,
        "original_feature_name": canonical,
    }


def _pipeline_parts(
    model: Any,
    features: pd.DataFrame,
) -> tuple[Any, np.ndarray, list[str]]:
    if not isinstance(model, Pipeline):
        return (
            model,
            features.to_numpy(dtype=np.float32, copy=False),
            list(features.columns),
        )
    if len(model.steps) < 2:
        raise ModelLoadError("설명할 수 없는 Pipeline 구조입니다.")

    transformer = model[:-1]
    estimator = model.steps[-1][1]
    try:
        transformed = transformer.transform(features)
    except Exception as exc:
        raise ModelLoadError(
            "SHAP용 feature 전처리에 실패했습니다."
        ) from exc
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_array = np.asarray(transformed, dtype=np.float32)
    try:
        names = list(transformer.get_feature_names_out())
    except Exception:
        names = list(features.columns)
    names = [str(name).split("__", 1)[-1] for name in names]
    if transformed_array.shape[1] != len(names):
        raise ModelLoadError(
            "변환된 feature 수와 feature 이름 수가 일치하지 않습니다."
        )
    return estimator, transformed_array, names


def _normalize_shap_output(
    explanation: Any,
    row_count: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != (row_count, feature_count):
        raise ValueError("SHAP 값의 행 또는 feature 수가 올바르지 않습니다.")
    base_values = np.asarray(explanation.base_values, dtype=np.float32)
    if base_values.ndim == 0:
        base_values = np.repeat(base_values.item(), row_count)
    else:
        base_values = base_values.reshape(row_count, -1)[:, 0]
    return values, base_values


def _permutation_contributions(
    estimator: Any,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    background = np.nanmedian(values, axis=0)
    background = np.where(np.isfinite(background), background, 0.0)
    baseline_predictions = np.asarray(
        estimator.predict(values),
        dtype=np.float32,
    )
    contributions = np.zeros_like(values, dtype=np.float32)
    perturbed = values.copy()
    for feature_index in range(values.shape[1]):
        original = perturbed[:, feature_index].copy()
        perturbed[:, feature_index] = background[feature_index]
        contributions[:, feature_index] = (
            baseline_predictions
            - np.asarray(estimator.predict(perturbed), dtype=np.float32)
        )
        perturbed[:, feature_index] = original
    base_values = baseline_predictions - contributions.sum(axis=1)
    return contributions, base_values


def compute_shap_values(
    model: Any,
    features: pd.DataFrame,
) -> ShapComputation:
    # The automatic Multi-Y bundle exposes the production Y model through
    # ``direct_model``.  Explain that fitted Pipeline instead of treating the
    # bundle itself as a numeric estimator.
    if not isinstance(model, Pipeline):
        model_for_target = getattr(model, "model_for_target", None)
        if callable(model_for_target):
            resolved = model_for_target("Y")
            if resolved is not None:
                model = resolved
        elif isinstance(getattr(model, "direct_model", None), Pipeline):
            model = model.direct_model
    if isinstance(model, EnsembleRegressor):
        member_results = {
            name: compute_shap_values(member, features)
            for name, member in model.models.items()
        }
        all_names = list(dict.fromkeys(
            name
            for result in member_results.values()
            for name in result.feature_names
        ))
        values = np.zeros((len(features), len(all_names)), dtype=np.float32)
        feature_values = np.zeros_like(values)
        base_values = np.zeros(len(features), dtype=np.float32)
        if model.method == "stacking" and model.meta_model is not None and hasattr(model.meta_model, "coef_"):
            coefficients = np.asarray(model.meta_model.coef_, dtype=float).reshape(-1)
            member_weights = {
                name: float(coefficients[index])
                for index, name in enumerate(model.models)
            }
            method = "ensemble_stacking_base_contribution"
        else:
            member_weights = model.weights
            method = "ensemble_weighted_shap"
        for member_name, result in member_results.items():
            weight = member_weights.get(member_name, 0.0)
            indices = [all_names.index(name) for name in result.feature_names]
            values[:, indices] += weight * result.values
            feature_values[:, indices] = result.feature_values
            base_values += weight * result.base_values
        return ShapComputation(
            values=values,
            base_values=base_values,
            feature_values=feature_values,
            feature_names=all_names,
            explanation_method=method,
            is_fallback=any(result.is_fallback for result in member_results.values()),
            warnings=list(dict.fromkeys(
                warning
                for result in member_results.values()
                for warning in result.warnings
            )),
        )
    estimator, transformed, feature_names = _pipeline_parts(
        model,
        features,
    )
    background = transformed[: min(len(transformed), 100)]
    shap_errors: list[str] = []
    try:
        import shap

        attempts: list[tuple[str, Any]] = [
            (
                "shap_auto",
                lambda: shap.Explainer(estimator, background),
            ),
            (
                "shap_tree",
                lambda: shap.TreeExplainer(estimator),
            ),
            (
                "shap_linear",
                lambda: shap.LinearExplainer(estimator, background),
            ),
        ]
        for method, factory in attempts:
            try:
                logger.info("설명기 생성 시도: %s", method)
                explainer = factory()
                explanation = explainer(transformed)
                values, base_values = _normalize_shap_output(
                    explanation,
                    len(transformed),
                    transformed.shape[1],
                )
                resolved_method = method
                explainer_name = type(explainer).__name__.lower()
                if "tree" in explainer_name:
                    resolved_method = "shap_tree"
                elif "linear" in explainer_name:
                    resolved_method = "shap_linear"
                elif "permutation" in explainer_name:
                    resolved_method = "shap_permutation"
                return ShapComputation(
                    values=values,
                    base_values=base_values,
                    feature_values=transformed,
                    feature_names=feature_names,
                    explanation_method=resolved_method,
                    is_fallback=False,
                )
            except Exception as exc:
                logger.warning(
                    "SHAP 설명기 실패: %s",
                    method,
                    exc_info=exc,
                )
                shap_errors.append(f"{method}: {type(exc).__name__}")
    except Exception as exc:
        logger.exception("SHAP 패키지를 불러오지 못했습니다.")
        shap_errors.append(f"shap_import: {type(exc).__name__}")

    try:
        values, base_values = _permutation_contributions(
            estimator,
            transformed,
        )
    except Exception as exc:
        raise ModelLoadError(
            "SHAP 및 permutation fallback 계산에 실패했습니다."
        ) from exc
    return ShapComputation(
        values=values,
        base_values=base_values,
        feature_values=transformed,
        feature_names=feature_names,
        explanation_method="permutation_contribution",
        is_fallback=True,
        warnings=[
            "현재 모델에서는 SHAP 계산을 사용할 수 없어 "
            "모델 비종속 permutation contribution을 사용했습니다.",
            "SHAP 실패 요약: " + ", ".join(shap_errors),
        ],
    )


def harmful_values(
    shap_values: np.ndarray,
    target: str,
) -> np.ndarray:
    if target == "Y":
        return np.maximum(-shap_values, 0.0)
    return np.maximum(shap_values, 0.0)


def beneficial_values(
    shap_values: np.ndarray,
    target: str,
) -> np.ndarray:
    if target == "Y":
        return np.maximum(shap_values, 0.0)
    return np.maximum(-shap_values, 0.0)


def _contribution_direction(signed_value: float, target: str) -> str:
    if target == "Y":
        return "yield_down" if signed_value < 0 else "yield_up"
    return "defect_up" if signed_value > 0 else "defect_down"


def _global_summaries(
    computation: ShapComputation,
    target: str,
    top_n: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    mean_abs = np.mean(np.abs(computation.values), axis=0)
    mean_harmful = np.mean(
        harmful_values(computation.values, target),
        axis=0,
    )
    mean_signed = np.mean(computation.values, axis=0)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, feature in enumerate(computation.feature_names):
        parsed = parse_feature_name(feature)
        rows.append(
            {
                "feature": parsed["original_feature_name"],
                "step": parsed["step"],
                "parameter_type": parsed["parameter_type"],
                "parameter_name": parsed["parameter_name"],
                "mean_abs_shap": float(mean_abs[index]),
                "mean_harmful_contribution": float(mean_harmful[index]),
                "_mean_signed_shap": float(mean_signed[index]),
            }
        )
    collapsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["feature"])
        if key not in collapsed:
            collapsed[key] = dict(row)
        else:
            collapsed[key]["mean_abs_shap"] += row["mean_abs_shap"]
            collapsed[key]["mean_harmful_contribution"] += row["mean_harmful_contribution"]
            collapsed[key]["_mean_signed_shap"] += row["_mean_signed_shap"]
    rows = list(collapsed.values())
    for row in rows:
        signed_value = float(row.pop("_mean_signed_shap"))
        row["direction"] = _contribution_direction(signed_value, target)
    rows.sort(
        key=lambda item: (
            item["mean_harmful_contribution"],
            item["mean_abs_shap"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    def aggregate(group_field: str) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            group = str(row[group_field])
            summary = groups.setdefault(
                group,
                {
                    group_field: group,
                    "mean_abs_shap": 0.0,
                    "harmful_contribution": 0.0,
                    "feature_count": 0,
                },
            )
            summary["mean_abs_shap"] += row["mean_abs_shap"]
            summary["harmful_contribution"] += row[
                "mean_harmful_contribution"
            ]
            summary["feature_count"] += 1
        aggregated = sorted(
            groups.values(),
            key=lambda item: item["harmful_contribution"],
            reverse=True,
        )
        for rank, item in enumerate(aggregated, 1):
            item["rank"] = rank
        return aggregated

    step_summary = aggregate("step")
    parameter_summary = aggregate("parameter_type")
    config_children = [
        item for item in parameter_summary
        if item["parameter_type"] in {"Model", "Equipment", "Chamber", "EQ"}
    ]
    if config_children:
        parameter_summary.append({
            "parameter_type": "Config",
            "mean_abs_shap": sum(item["mean_abs_shap"] for item in config_children),
            "harmful_contribution": sum(item["harmful_contribution"] for item in config_children),
            "feature_count": sum(item["feature_count"] for item in config_children),
            "rank": len(parameter_summary) + 1,
        })
    equipment_summary = [
        {
            "equipment": row["feature"],
            "mean_abs_shap": row["mean_abs_shap"],
            "harmful_contribution": row[
                "mean_harmful_contribution"
            ],
        }
        for row in rows
        if row["parameter_type"] in {"EQ", "Equipment"}
    ]
    for rank, item in enumerate(equipment_summary, 1):
        item["rank"] = rank
    return (
        rows[:top_n],
        step_summary,
        parameter_summary,
        equipment_summary,
        warnings,
    )


def _quality_warnings(metadata: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    metrics = metadata.get("metrics", {})
    test_r2 = metrics.get("test", {}).get("r2")
    validation_r2 = metrics.get("validation", {}).get("r2")
    if test_r2 is None or not math.isfinite(float(test_r2)):
        warnings.append(
            "Test R²가 없어 설명 결과의 신뢰도를 판단하기 어렵습니다."
        )
    elif float(test_r2) <= 0:
        warnings.append(
            "현재 모델의 Test R²가 낮아 설명 결과의 신뢰도가 제한적입니다."
        )
    if str(metadata.get("model_name")) == "DummyRegressor":
        warnings.append(
            "DummyRegressor 모델이므로 원인 기여도 해석에 적합하지 않습니다."
        )
    if (
        test_r2 is not None
        and validation_r2 is not None
        and math.isfinite(float(test_r2))
        and math.isfinite(float(validation_r2))
        and abs(float(validation_r2) - float(test_r2)) > 0.2
    ):
        warnings.append(
            "Validation과 Test R² 차이가 커 설명 결과의 일반화에 주의해야 합니다."
        )
    return warnings


def _sampling_indices(
    prediction_rows: list[dict[str, Any]],
    max_rows: int,
) -> tuple[list[int], str]:
    priorities = {"danger": 0, "warning": 1, "normal": 2, None: 3}
    ordered = sorted(
        range(len(prediction_rows)),
        key=lambda index: (
            priorities.get(prediction_rows[index].get("risk_level"), 3),
            index,
        ),
    )
    strategy = (
        "danger_warning_priority"
        if any(row.get("risk_level") for row in prediction_rows)
        else "row_order"
    )
    return ordered[:max_rows], strategy


def _collapse_local_contributions(
    rows: list[dict[str, Any]],
    target: str,
) -> list[dict[str, Any]]:
    collapsed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["feature"]),
            str(row["step"]),
            str(row["parameter_type"]),
        )
        existing = collapsed.get(key)
        if existing is None:
            collapsed[key] = dict(row)
            continue
        existing["shap_value"] += row["shap_value"]
        existing["absolute_shap"] += row["absolute_shap"]
        existing["harmful_contribution"] += row["harmful_contribution"]
        existing["beneficial_contribution"] += row["beneficial_contribution"]
        existing["value"] = None

    result = list(collapsed.values())
    for row in result:
        row["direction"] = _contribution_direction(
            float(row["shap_value"]),
            target,
        )
    return result


def explain_dataframe(
    dataframe: pd.DataFrame,
    loaded: LoadedPredictionModel,
    *,
    max_rows: int = DEFAULT_MAX_EXPLAIN_ROWS,
    top_n: int = DEFAULT_TOP_N,
    per_wafer_top_n: int = DEFAULT_WAFER_TOP_N,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    danger_threshold: float = DEFAULT_DANGER_THRESHOLD,
    prediction_result: PredictionResult | None = None,
) -> ExplainResult:
    if not 1 <= max_rows <= MAX_EXPLAIN_ROWS:
        raise InferenceInputError(
            f"max_rows는 1부터 {MAX_EXPLAIN_ROWS} 사이여야 합니다."
        )
    if not 1 <= top_n <= MAX_TOP_N:
        raise InferenceInputError(f"top_n은 1부터 {MAX_TOP_N} 사이여야 합니다.")
    if not 1 <= per_wafer_top_n <= 20:
        raise InferenceInputError(
            "per_wafer_top_n은 1부터 20 사이여야 합니다."
        )

    logger.info("설명용 예측 시작: rows=%d", len(dataframe))
    log_memory_stage(
        logger,
        "explanation_start",
        total_rows=len(dataframe),
        max_rows=max_rows,
        top_n=top_n,
    )
    predictions = prediction_result or predict_dataframe(
        dataframe,
        loaded,
        warning_threshold=warning_threshold,
        danger_threshold=danger_threshold,
        max_rows=None,
    )
    if predictions.total_rows != len(dataframe):
        raise InferenceInputError(
            "설명에 전달된 예측 결과의 행 수가 CSV와 일치하지 않습니다."
        )
    is_auto_pipeline = (
        loaded.metadata.get("pipeline_version") == "auto_multi_y_hgbr_v1"
    )
    processed, preprocessing_report = preprocess_dataframe(dataframe)
    all_features, feature_warnings = prepare_inference_features(
        dataframe if is_auto_pipeline else processed,
        list(loaded.metadata["feature_columns"]),
        allow_missing=is_auto_pipeline,
    )
    indices, sampling_strategy = _sampling_indices(
        predictions.predictions,
        max_rows,
    )
    sampled_features = all_features.iloc[indices].reset_index(drop=True)
    logger.info(
        "SHAP 분석 행 선택 완료: total=%d, analyzed=%d, strategy=%s",
        len(dataframe),
        len(sampled_features),
        sampling_strategy,
    )
    computation = compute_shap_values(
        loaded.model,
        sampled_features,
    )
    logger.info(
        "설명 값 계산 완료: method=%s",
        computation.explanation_method,
    )
    (
        global_importance,
        step_summary,
        parameter_summary,
        equipment_summary,
        parse_warnings,
    ) = _global_summaries(
        computation,
        str(loaded.metadata["target"]),
        top_n,
    )

    target = str(loaded.metadata["target"])
    harmful = harmful_values(computation.values, target)
    beneficial = beneficial_values(computation.values, target)
    wafer_explanations: list[dict[str, Any]] = []
    local_contributions: list[dict[str, Any]] = []
    for sampled_position, source_index in enumerate(indices):
        prediction_row = predictions.predictions[source_index]
        local_rows: list[dict[str, Any]] = []
        for feature_index, feature in enumerate(computation.feature_names):
            parsed = parse_feature_name(feature)
            signed_value = float(
                computation.values[
                    sampled_position,
                    feature_index,
                ]
            )
            original_value: Any
            if feature in sampled_features.columns:
                original_value = sampled_features.iloc[
                    sampled_position
                ][feature]
            else:
                original_value = computation.feature_values[
                    sampled_position,
                    feature_index,
                ]
            local_rows.append(
                {
                    "feature": parsed["original_feature_name"] or feature,
                    "value": to_json_safe(original_value),
                    "shap_value": signed_value,
                    "absolute_shap": abs(signed_value),
                    "harmful_contribution": float(
                        harmful[sampled_position, feature_index]
                    ),
                    "beneficial_contribution": float(
                        beneficial[sampled_position, feature_index]
                    ),
                    "step": parsed["step"],
                    "parameter_type": parsed["parameter_type"],
                    "direction": _contribution_direction(signed_value, target),
                }
            )
        local_rows = _collapse_local_contributions(local_rows, target)
        negative = sorted(
            local_rows,
            key=lambda item: item["harmful_contribution"],
            reverse=True,
        )
        negative = [
            row for row in negative if row["harmful_contribution"] > 0
        ][:per_wafer_top_n]
        positive = sorted(
            local_rows,
            key=lambda item: item["beneficial_contribution"],
            reverse=True,
        )
        positive = [
            row for row in positive if row["beneficial_contribution"] > 0
        ][:per_wafer_top_n]
        identifier = prediction_row.get(predictions.identifier_column)
        local_limit = min(max(top_n, per_wafer_top_n * 2), MAX_TOP_N)
        compact_local_rows = sorted(
            local_rows,
            key=lambda item: item["absolute_shap"],
            reverse=True,
        )[:local_limit]
        local_contributions.append(
            {
                "identifier": identifier,
                "lot_id": prediction_row.get("Lot_ID"),
                "wafer_id": prediction_row.get("Wafer_ID"),
                "wafer_slot": prediction_row.get("Wafer_Slot"),
                "prediction": prediction_row.get(f"predicted_{target}"),
                "risk_level": prediction_row.get("risk_level"),
                "contributions": compact_local_rows,
            }
        )
        wafer_explanations.append(
            {
                "identifier": identifier,
                "lot_id": prediction_row.get("Lot_ID"),
                "wafer_id": prediction_row.get("Wafer_ID"),
                "wafer_slot": prediction_row.get("Wafer_Slot"),
                "prediction": prediction_row.get(f"predicted_{target}"),
                "risk_level": prediction_row.get("risk_level"),
                "base_value": float(
                    computation.base_values[sampled_position]
                ),
                "top_negative_contributors": negative,
                "top_positive_contributors": positive,
            }
        )

    warnings = list(
        dict.fromkeys(
            [
                *predictions.warnings,
                *preprocessing_report["warnings"],
                *feature_warnings,
                *computation.warnings,
                *parse_warnings,
            ]
        )
    )
    result = ExplainResult(
        model_id=loaded.model_id,
        target=target,
        model_name=str(loaded.metadata["model_name"]),
        total_rows=len(dataframe),
        analyzed_rows=len(sampled_features),
        sampling_used=len(sampled_features) < len(dataframe),
        sampling_strategy=sampling_strategy,
        explanation_method=computation.explanation_method,
        is_fallback=computation.is_fallback,
        global_importance=to_json_safe(global_importance),
        step_summary=to_json_safe(step_summary),
        parameter_type_summary=to_json_safe(parameter_summary),
        equipment_summary=to_json_safe(equipment_summary),
        identifier_column=predictions.identifier_column,
        wafer_explanations=to_json_safe(wafer_explanations),
        model_quality_warnings=_quality_warnings(loaded.metadata),
        warnings=warnings,
        local_contributions=to_json_safe(local_contributions),
    )
    log_memory_stage(
        logger,
        "explanation_complete",
        analyzed_rows=result.analyzed_rows,
        top_features=len(result.global_importance),
    )
    return result
