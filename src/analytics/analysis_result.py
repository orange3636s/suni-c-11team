"""Shared source-of-truth for interactive cause analysis."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data_validation import validate_dataframe
from src.ml.explainability import ExplainResult
from src.ml.inference import LoadedPredictionModel, PredictionResult
from src.ml.model_io import to_json_safe


ANALYSIS_RESULT_VERSION = "y1_y5_cause_analysis_v2"


def dataset_fingerprint(dataframe: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, dataframe.columns)).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(dataframe, index=True).to_numpy()
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def target_descriptor(target: str) -> dict[str, str]:
    if target == "Y":
        return {"name": "Y", "label": "Final Y", "type": "final_yield", "unit": "%"}
    if target in {f"Y{index}" for index in range(1, 6)}:
        return {"name": target, "label": target, "type": "fail_rate", "unit": "%"}
    return {"name": target, "label": target, "type": "fail_bit_count", "unit": "count"}


def compose_multi_y_predictions(
    predictions: dict[str, list[float]],
    ensemble_weight: float | None,
) -> dict[str, Any]:
    """Compose final Y only from actual Y1~Y5 model outputs."""
    failure_rates = {key: predictions[key] for key in [f"Y{i}" for i in range(1, 6)] if key in predictions}
    fail_bit_counts = {key: predictions[key] for key in [f"Y{i}" for i in range(6, 11)] if key in predictions}
    row_count = len(next(iter(predictions.values()))) if predictions else 0
    if any(len(values) != row_count for values in predictions.values()):
        raise ValueError("Multi-Y 예측 결과의 행 수가 일치하지 않습니다.")
    derived: list[float] | None = None
    if len(failure_rates) == 5:
        derived = (
            100.0
            - np.sum(np.asarray([failure_rates[f"Y{i}"] for i in range(1, 6)]), axis=0)
        ).astype(float).tolist()
    del ensemble_weight  # Backward-compatible call signature; no ensemble is used.
    if derived is not None:
        derived = np.clip(np.asarray(derived, dtype=float), 0.0, 100.0).tolist()
    return to_json_safe({
        "predicted_y": derived,
        "failure_rates": failure_rates,
        "fail_bit_counts": fail_bit_counts,
    })


def _average(values: list[float] | None) -> float | None:
    return float(np.mean(values)) if values else None


def build_analysis_result(
    *,
    filename: str,
    dataframe: pd.DataFrame,
    loaded: LoadedPredictionModel,
    prediction: PredictionResult,
    explanation: ExplainResult,
    relationships: dict[str, Any],
    multi_y: dict[str, Any] | None = None,
    warning_threshold: float,
    danger_threshold: float,
    analysis_unit: str,
    created_at: datetime | None = None,
    analysis_id: str | None = None,
    lot_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = created_at or datetime.now().astimezone()
    resolved_id = analysis_id or (
        f"analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    quality = validate_dataframe(dataframe, validation_mode="analysis")
    metadata = loaded.metadata
    metrics = metadata.get("metrics", {})
    multi_y_values = multi_y or compose_multi_y_predictions(
        {prediction.target: [
            float(row[f"predicted_{prediction.target}"])
            for row in prediction.predictions
        ]},
        metadata.get("ensemble_weight"),
    )
    multi_y_summary = {
        **multi_y_values,
        "average_predicted_y": _average(multi_y_values.get("predicted_y")),
        "failure_rate_averages": {
            key: _average(value) for key, value in multi_y_values.get("failure_rates", {}).items()
        },
        "fail_bit_count_averages": {
            key: _average(value) for key, value in multi_y_values.get("fail_bit_counts", {}).items()
        },
    }
    identifiers = [
        row.get(prediction.identifier_column) for row in prediction.predictions
    ]
    multi_y_summary["wafer_results"] = [
        {
            "identifier": identifier,
            "predicted_y": (
                multi_y_values["predicted_y"][index]
                if multi_y_values.get("predicted_y") is not None else None
            ),
            "failure_rates": {
                key: values[index]
                for key, values in multi_y_values.get("failure_rates", {}).items()
            },
            "fail_bit_counts": {
                key: values[index]
                for key, values in multi_y_values.get("fail_bit_counts", {}).items()
            },
        }
        for index, identifier in enumerate(identifiers)
    ]
    warnings = [*explanation.warnings, *relationships.get("caveats", [])]
    if multi_y_summary.get("predicted_y") is None:
        warnings.append("Y1~Y5 모델이 모두 준비되지 않아 최종 Y를 계산할 수 없습니다.")
    result = {
        "analysis_id": resolved_id,
        "analysis_version": ANALYSIS_RESULT_VERSION,
        "created_at": timestamp.isoformat(),
        "model": {
            "model_id": loaded.model_id,
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version"),
            "schema_version": metadata.get("schema_version"),
            "compatibility": "compatible",
            "structure": metadata.get("model_structure", "Y1~Y5 기반 최종 Y"),
        },
        "dataset": {
            "filename": filename,
            "fingerprint": dataset_fingerprint(dataframe),
            "row_count": len(dataframe),
            "identifier_column": prediction.identifier_column,
        },
        "target": target_descriptor(prediction.target),
        "metrics": {
            **metrics,
            "evaluation_summary": metadata.get("evaluation_summary", {}),
            "train_lot_count": metadata.get("train_lot_count"),
            "validation_lot_count": metadata.get("validation_lot_count"),
            "test_lot_count": metadata.get("test_lot_count"),
            "group_cv": metadata.get("group_cv"),
            "ablation": metadata.get("missingness_sensitivity"),
        },
        "multi_y": multi_y_summary,
        "risk": {
            "warning_threshold": warning_threshold,
            "critical_threshold": danger_threshold,
            "normal_count": prediction.normal_count,
            "warning_count": prediction.warning_count,
            "critical_count": prediction.danger_count,
            "risk_probability": None,
        },
        "confidence": {
            "available": False,
            "low_confidence_count": None,
            "reason": "저장 모델에 calibration 또는 prediction interval 메타데이터가 없습니다.",
        },
        "feature_importance": {
            "priority": ["shap", "permutation", "tree_gain", "statistics"],
            "global": explanation.global_importance,
            "steps": explanation.step_summary,
            "parameter_types": explanation.parameter_type_summary,
            "permutation": metadata.get("permutation_importance"),
            "tree_gain": metadata.get("tree_importance"),
        },
        "shap": {
            "method": explanation.explanation_method,
            "is_fallback": explanation.is_fallback,
            "analyzed_rows": explanation.analyzed_rows,
        },
        "wafer_explanations": explanation.wafer_explanations,
        "relationships": relationships.get("relationship_paths", []),
        "statistics": relationships.get(
            "statistics", relationships.get("rankings", {}).get("correlation", {})
        ),
        "risk_wafers": [
            row
            for row in prediction.predictions
            if row.get("risk_level") in {"danger", "warning"}
        ][:5],
        "lot_summary": (lot_analysis or {}).get("lots", []),
        "lot_analysis": lot_analysis or {},
        "data_quality": {
            "r_measurement_coverage": quality.get("r_measurement_coverage"),
            "d_measurement_coverage": quality.get("d_measurement_coverage"),
            "config_completeness_rate": quality.get("config_completeness_rate"),
            "target_consistency_rate": quality.get("target_consistency_rate"),
            "config_parse_error_count": quality.get("config_parse_error_count"),
            "missing_indicator_used": metadata.get("missing_indicator_used"),
            "outlier_policy": metadata.get("outlier_policy"),
            "selection_bias_warnings": relationships.get("selection_bias_warnings", []),
        },
        "methodology": {
            "split_strategy": metadata.get("split_strategy", metadata.get("split_method")),
            "group_column": metadata.get("group_column"),
            "preprocessing_strategy": metadata.get("preprocessing_strategy"),
            "tuning_method": metadata.get("tuning_method"),
            "best_parameters": metadata.get("best_parameters"),
            "model_selection_metric": metadata.get("model_selection_metric"),
            "notes": [],
        },
        "warnings": list(dict.fromkeys(warnings)),
    }
    return to_json_safe(result)
