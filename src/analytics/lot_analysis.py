"""Lot-scoped aggregation of actual per-wafer model contributions."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from src.ml.explainability import ExplainResult
from src.ml.inference import PredictionResult, risk_class_confidence
from src.ml.model_io import to_json_safe


LOT_FROM_WAFER_PATTERN = re.compile(
    r"^(?P<lot>.+?)[_-]?(?:WAFER|WF|W)[_-]?(?P<slot>\d+)$",
    re.IGNORECASE,
)
STEP_PATTERN = re.compile(r"^(Step\d+)_", re.IGNORECASE)
CONFIG_PARAMETER_TYPES = {"config", "model", "equipment", "chamber", "eq"}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lot_id(row: dict[str, Any], identifier: Any) -> str | None:
    for candidate in (row.get("Lot_ID"), row.get("lot_id")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    text = str(identifier).strip() if identifier is not None else ""
    match = LOT_FROM_WAFER_PATTERN.fullmatch(text)
    return match.group("lot").rstrip("_-") if match else None


def _canonical_feature(item: dict[str, Any]) -> tuple[str, str, str]:
    feature = str(item.get("feature") or "unknown")
    parameter_type = str(item.get("parameter_type") or "unknown").strip()
    step = str(item.get("step") or "")
    if not STEP_PATTERN.match(f"{step}_"):
        match = STEP_PATTERN.match(feature.split("__")[-1])
        step = match.group(1) if match else "unknown"
    normalized_type = parameter_type.lower().replace("_", " ")
    if normalized_type in CONFIG_PARAMETER_TYPES or re.search(
        r"_(?:Config|Model|Equipment|Chamber|EQ)(?:_|$)",
        feature,
        re.IGNORECASE,
    ):
        canonical = f"{step}_Config" if step != "unknown" else feature
        return canonical, "Config", step
    if normalized_type == "r":
        return feature, "R", step
    if normalized_type == "d":
        return feature, "D", step
    # Indicator/legacy groups remain visible only through "전체".
    return feature, "Other", step


def _per_wafer_features(
    contributions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in contributions:
        feature, group, step = _canonical_feature(item)
        key = (feature, group)
        row = grouped.setdefault(
            key,
            {
                "feature": feature,
                "display_name": feature,
                "group": group,
                "step": step,
                "signed_shap": 0.0,
                "absolute_shap": 0.0,
                "adverse": 0.0,
                "improvement": 0.0,
                "source_features": [],
            },
        )
        signed = _finite(item.get("shap_value")) or 0.0
        adverse = _finite(item.get("harmful_contribution")) or 0.0
        improvement = _finite(item.get("beneficial_contribution")) or 0.0
        absolute = _finite(item.get("absolute_shap"))
        if absolute is None:
            partitioned_absolute = adverse + improvement
            absolute = (
                partitioned_absolute
                if partitioned_absolute > 0.0 or signed == 0.0
                else abs(signed)
            )
        row["signed_shap"] += signed
        row["absolute_shap"] += max(absolute, 0.0)
        row["adverse"] += adverse
        row["improvement"] += improvement
        source = str(item.get("feature") or "")
        if source and source not in row["source_features"]:
            row["source_features"].append(source)
    return list(grouped.values())


def _feature_rankings(
    wafer_rows: list[dict[str, Any]],
    total_wafer_count: int,
) -> dict[str, list[dict[str, Any]]]:
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for wafer in wafer_rows:
        for item in wafer["features"]:
            key = (item["feature"], item["group"])
            row = aggregates.setdefault(
                key,
                {
                    "feature": item["feature"],
                    "display_name": item["display_name"],
                    "group": item["group"],
                    "step": item["step"],
                    "signed_sum": 0.0,
                    "absolute_sum": 0.0,
                    "adverse_sum": 0.0,
                    "improvement_sum": 0.0,
                    "sample_count": 0,
                    "source_features": set(),
                },
            )
            row["signed_sum"] += item["signed_shap"]
            row["absolute_sum"] += item["absolute_shap"]
            row["adverse_sum"] += item["adverse"]
            row["improvement_sum"] += item["improvement"]
            row["sample_count"] += 1
            row["source_features"].update(item["source_features"])

    rankings: list[dict[str, Any]] = []
    for row in aggregates.values():
        sample_count = row["sample_count"]
        rankings.append(
            {
                "feature": row["feature"],
                "display_name": row["display_name"],
                "group": row["group"],
                "step": row["step"],
                "mean_signed_shap": row["signed_sum"] / sample_count,
                "mean_abs_shap": row["absolute_sum"] / sample_count,
                "adverse_contribution": row["adverse_sum"] / sample_count,
                "improvement_contribution": row["improvement_sum"] / sample_count,
                "sample_count": sample_count,
                "coverage": sample_count / total_wafer_count if total_wafer_count else 0.0,
                "source_features": sorted(row["source_features"]),
            }
        )
    rankings.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
    for rank, row in enumerate(rankings, 1):
        row["rank"] = rank
    return {
        "all": rankings,
        "r": [row for row in rankings if row["group"] == "R"],
        "d": [row for row in rankings if row["group"] == "D"],
        "config": [row for row in rankings if row["group"] == "Config"],
    }


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: row["adverse_contribution"], reverse=True)
    total = sum(max(float(row["adverse_contribution"]), 0.0) for row in ranked)
    cumulative = 0.0
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, 1):
        impact = max(float(row["adverse_contribution"]), 0.0)
        share = impact / total if total > 0 else 0.0
        cumulative += share
        output.append(
            {
                "rank": rank,
                "feature": row["feature"],
                "display_name": row["display_name"],
                "group": row["group"],
                "adverse_contribution": impact,
                "impact": impact,
                "share": share,
                "cumulative_share": cumulative,
                "within_threshold": (
                    cumulative <= 0.8 or cumulative - share < 0.8
                ),
                "sample_count": row["sample_count"],
                "coverage": row["coverage"],
            }
        )
    return output


def _prediction_confidence(row: dict[str, Any]) -> float | None:
    critical = _finite(row.get("critical_probability"))
    warning = _finite(row.get("warning_probability"))
    if critical is None or warning is None:
        return None
    return risk_class_confidence(critical, warning)


def _field_averages(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values = row.get(field)
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            numeric = _finite(value)
            if numeric is not None:
                totals[str(name)].append(numeric)
    return {
        name: sum(values) / len(values)
        for name, values in totals.items()
        if values
    }


def _failure_summary(
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    failure_rate_averages = _field_averages(rows, "failure_rates")
    fail_bit_count_averages = _field_averages(rows, "fail_bit_counts")
    top_failure_rate_target = (
        max(failure_rate_averages, key=failure_rate_averages.get)
        if failure_rate_averages
        else None
    )
    top_fail_bit_count_target = (
        max(fail_bit_count_averages, key=fail_bit_count_averages.get)
        if fail_bit_count_averages
        else None
    )
    # Failure rate and fail-bit count have different units. Keep both leaders
    # explicit, and retain the legacy field as a rate-first compatibility alias.
    top_failure_target = (
        top_failure_rate_target
        or top_fail_bit_count_target
        or (target if target != "Y" else None)
    )
    return {
        "failure_rate_averages": failure_rate_averages,
        "fail_bit_count_averages": fail_bit_count_averages,
        "top_failure_rate_target": top_failure_rate_target,
        "top_failure_rate_average": (
            failure_rate_averages.get(top_failure_rate_target)
            if top_failure_rate_target is not None
            else None
        ),
        "top_fail_bit_count_target": top_fail_bit_count_target,
        "top_fail_bit_count_average": (
            fail_bit_count_averages.get(top_fail_bit_count_target)
            if top_fail_bit_count_target is not None
            else None
        ),
        "top_failure_target": top_failure_target,
    }


def _top_adverse(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    top = max(rows, key=lambda row: float(row["adverse_contribution"]))
    return top if float(top["adverse_contribution"]) > 0.0 else None


def build_lot_cause_analysis(
    prediction: PredictionResult,
    explanation: ExplainResult,
) -> dict[str, Any]:
    """Aggregate actual sampled local contributions within each selected Lot."""
    predictions_by_lot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_rows = 0
    for row in prediction.predictions:
        identifier = row.get(prediction.identifier_column)
        lot = _lot_id(row, identifier)
        if lot is None:
            excluded_rows += 1
            continue
        predictions_by_lot[lot].append(row)

    locals_by_lot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    locals_by_identifier: dict[str, dict[str, Any]] = {}
    for local in explanation.local_contributions:
        identifier = local.get("identifier")
        lot = _lot_id(local, identifier)
        if lot is None:
            continue
        enriched = {
            **local,
            "features": _per_wafer_features(local.get("contributions") or []),
        }
        locals_by_lot[lot].append(enriched)
        locals_by_identifier[str(identifier)] = enriched

    lots: list[dict[str, Any]] = []
    prediction_column = f"predicted_{prediction.target}"
    for lot_id, rows in predictions_by_lot.items():
        local_rows = locals_by_lot.get(lot_id, [])
        rankings = _feature_rankings(local_rows, len(rows))
        pareto = {group: _pareto(group_rows) for group, group_rows in rankings.items()}
        predictions = [
            value for row in rows
            if (value := _finite(row.get(prediction_column))) is not None
        ]
        confidences = [
            value for row in rows
            if (value := _prediction_confidence(row)) is not None
        ]
        average_prediction = (
            sum(predictions) / len(predictions) if predictions else None
        )
        minimum_prediction = min(predictions) if predictions else None
        maximum_prediction = max(predictions) if predictions else None
        risk_extreme_prediction = (
            minimum_prediction
            if prediction.target == "Y"
            else maximum_prediction
        )
        failure_summary = _failure_summary(rows, prediction.target)
        risk_counts = {
            risk: sum(row.get("risk_level") == risk for row in rows)
            for risk in ("danger", "warning", "normal")
        }
        wafer_list: list[dict[str, Any]] = []
        for row in rows:
            identifier = row.get(prediction.identifier_column)
            prediction_value = _finite(row.get(prediction_column))
            local = locals_by_identifier.get(str(identifier))
            features = sorted(
                (local or {}).get("features") or [],
                key=lambda item: item["adverse"],
                reverse=True,
            )
            top = next(
                (item for item in features if item["adverse"] > 0.0),
                None,
            )
            top_config = next(
                (
                    item for item in features
                    if item["group"] == "Config" and item["adverse"] > 0.0
                ),
                None,
            )
            wafer_list.append(
                {
                    "identifier": identifier,
                    "lot_id": lot_id,
                    "wafer_id": row.get("Wafer_ID"),
                    "wafer_slot": row.get("Wafer_Slot"),
                    "prediction": prediction_value,
                    "predicted_value": prediction_value,
                    "predicted_yield": (
                        prediction_value if prediction.target == "Y" else None
                    ),
                    "risk_level": row.get("risk_level"),
                    "confidence": _prediction_confidence(row),
                    "top_feature": top["feature"] if top else None,
                    "top_step": top["step"] if top else None,
                    "top_config": top_config["feature"] if top_config else None,
                    "shap_available": local is not None,
                }
            )
        top_feature = _top_adverse(rankings["all"])
        top_config = _top_adverse(rankings["config"])
        lots.append(
            {
                "lot_id": lot_id,
                "wafer_count": len(rows),
                "analyzed_wafer_count": len(local_rows),
                "shap_coverage": len(local_rows) / len(rows) if rows else 0.0,
                "average_predicted_value": average_prediction,
                "average_predicted_yield": (
                    average_prediction if prediction.target == "Y" else None
                ),
                "minimum_predicted_value": minimum_prediction,
                "maximum_predicted_value": maximum_prediction,
                "risk_extreme_predicted_value": risk_extreme_prediction,
                "risk_extreme_direction": (
                    "minimum" if prediction.target == "Y" else "maximum"
                ),
                "critical_wafer_count": risk_counts["danger"],
                "warning_wafer_count": risk_counts["warning"],
                "normal_wafer_count": risk_counts["normal"],
                "average_confidence": (
                    sum(confidences) / len(confidences) if confidences else None
                ),
                **failure_summary,
                "feature_importance": rankings,
                "pareto": pareto,
                "wafer_list": wafer_list,
                "top_causes": {
                    "feature": top_feature["feature"] if top_feature else None,
                    "step": top_feature["step"] if top_feature else None,
                    "config": top_config["feature"] if top_config else None,
                    "failure_target": failure_summary["top_failure_target"],
                    "failure_rate_target": failure_summary[
                        "top_failure_rate_target"
                    ],
                    "fail_bit_count_target": failure_summary[
                        "top_fail_bit_count_target"
                    ],
                },
            }
        )

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        prediction_value = item["average_predicted_value"]
        if prediction_value is None:
            target_risk = math.inf
        else:
            target_risk = (
                prediction_value
                if prediction.target == "Y"
                else -prediction_value
            )
        return (
            -item["critical_wafer_count"],
            -item["warning_wafer_count"],
            target_risk,
            item["lot_id"],
        )

    lots.sort(key=sort_key)
    return to_json_safe(
        {
            "target": prediction.target,
            "aggregation": "selected_lot_sampled_per_wafer_shap",
            "sampling_used": explanation.sampling_used,
            "total_lot_count": len(lots),
            "excluded_row_count": excluded_rows,
            "lots": lots,
        }
    )
