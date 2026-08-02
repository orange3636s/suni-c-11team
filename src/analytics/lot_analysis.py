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
        raw_category = item.get("value")
        category = (
            str(raw_category).strip()
            if raw_category is not None and str(raw_category).strip()
            else "결측"
        )
        canonical = f"{step}_Config::{category}"
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
                "display_name": (
                    feature.split("::", 1)[1]
                    if group == "Config" and "::" in feature
                    else feature
                ),
                "group": group,
                "step": step,
                "signed_shap": 0.0,
                "absolute_shap": 0.0,
                "adverse": 0.0,
                "improvement": 0.0,
                "source_features": [],
                "observed_value": None,
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
        observed = _finite(item.get("value"))
        if observed is not None:
            row["observed_value"] = observed
    return list(grouped.values())


def _feature_rankings(
    wafer_rows: list[dict[str, Any]],
    total_wafer_count: int,
    overall_value_means: dict[tuple[str, str], float] | None = None,
    overall_yield: float | None = None,
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
                    "value_sum": 0.0,
                    "value_count": 0,
                    "source_features": set(),
                },
            )
            row["signed_sum"] += item["signed_shap"]
            row["absolute_sum"] += item["absolute_shap"]
            row["adverse_sum"] += item["adverse"]
            row["improvement_sum"] += item["improvement"]
            row["sample_count"] += 1
            comparison_value = (
                wafer.get("ranking_yield")
                if item["group"] == "Config"
                else item.get("observed_value")
            )
            if comparison_value is not None:
                row["value_sum"] += comparison_value
                row["value_count"] += 1
            row["source_features"].update(item["source_features"])

    rankings: list[dict[str, Any]] = []
    for row in aggregates.values():
        sample_count = row["sample_count"]
        lot_mean_value = (
            row["value_sum"] / row["value_count"]
            if row["value_count"]
            else None
        )
        overall_mean_value = (overall_value_means or {}).get(
            (row["feature"], row["group"])
        )
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
                "total_signed_contribution": row["signed_sum"],
                "total_absolute_contribution": row["absolute_sum"],
                "total_adverse_contribution": row["adverse_sum"],
                "total_improvement_contribution": row["improvement_sum"],
                "sample_count": sample_count,
                "coverage": sample_count / total_wafer_count if total_wafer_count else 0.0,
                "lot_mean_value": lot_mean_value,
                "overall_mean_value": overall_mean_value,
                "mean_difference": (
                    lot_mean_value - overall_mean_value
                    if lot_mean_value is not None and overall_mean_value is not None
                    else None
                ),
                "overall_yield": overall_yield,
                "source_features": sorted(row["source_features"]),
            }
        )
    rankings.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
    low_sample_config = [
        row for row in rankings
        if row["group"] == "Config" and row["sample_count"] < 5
    ]
    official = [
        row for row in rankings
        if not (row["group"] == "Config" and row["sample_count"] < 5)
    ]
    for rank, row in enumerate(official, 1):
        row["rank"] = rank
    return {
        "all": official,
        "r": [row for row in official if row["group"] == "R"],
        "d": [row for row in official if row["group"] == "D"],
        "config": [row for row in official if row["group"] == "Config"],
        "config_categories": [row for row in rankings if row["group"] == "Config"],
        "low_sample_config": low_sample_config,
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

    prediction_by_identifier = {
        str(row.get(prediction.identifier_column)): row
        for rows in predictions_by_lot.values()
        for row in rows
    }

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
        prediction_row = prediction_by_identifier.get(str(identifier), {})
        enriched["ranking_yield"] = (
            _finite(prediction_row.get("actual_Y"))
            if _finite(prediction_row.get("actual_Y")) is not None
            else _finite(prediction_row.get("predicted_Y"))
        )
        locals_by_lot[lot].append(enriched)
        locals_by_identifier[str(identifier)] = enriched

    lots: list[dict[str, Any]] = []
    prediction_column = f"predicted_{prediction.target}"
    global_value_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for local_rows in locals_by_lot.values():
        for local in local_rows:
            for feature in local["features"]:
                observed = (
                    local.get("ranking_yield")
                    if feature["group"] == "Config"
                    else feature.get("observed_value")
                )
                if observed is not None:
                    global_value_samples[(feature["feature"], feature["group"])].append(observed)
    global_value_means = {
        key: sum(values) / len(values)
        for key, values in global_value_samples.items()
        if values
    }
    all_actual_yields = [
        value for rows in predictions_by_lot.values() for row in rows
        if (value := _finite(row.get("actual_Y"))) is not None
    ]
    all_predicted_yields = [
        value for rows in predictions_by_lot.values() for row in rows
        if (value := _finite(row.get("predicted_Y"))) is not None
    ]
    overall_actual_yield = (
        sum(all_actual_yields) / len(all_actual_yields)
        if all_actual_yields else None
    )
    overall_predicted_yield = (
        sum(all_predicted_yields) / len(all_predicted_yields)
        if all_predicted_yields else None
    )
    overall_ranking_yield = (
        overall_actual_yield
        if overall_actual_yield is not None
        else overall_predicted_yield
    )
    for lot_id, rows in predictions_by_lot.items():
        local_rows = locals_by_lot.get(lot_id, [])
        rankings = _feature_rankings(
            local_rows,
            len(rows),
            global_value_means,
            overall_ranking_yield,
        )
        pareto = {
            group: _pareto(rankings[group])
            for group in ("all", "r", "d", "config")
        }
        predictions = [
            value for row in rows
            if (value := _finite(row.get(prediction_column))) is not None
        ]
        actual_yields = [
            value for row in rows
            if (value := _finite(row.get("actual_Y"))) is not None
        ]
        confidences = [
            value for row in rows
            if (value := _prediction_confidence(row)) is not None
        ]
        average_prediction = (
            sum(predictions) / len(predictions) if predictions else None
        )
        average_actual_yield = (
            sum(actual_yields) / len(actual_yields) if actual_yields else None
        )
        ranking_values = actual_yields if actual_yields else predictions
        ranking_basis = "actual_y" if actual_yields else "predicted_y"
        ranking_yield = (
            sum(ranking_values) / len(ranking_values)
            if ranking_values else None
        )
        minimum_prediction = min(predictions) if predictions else None
        maximum_prediction = max(predictions) if predictions else None
        minimum_yield = min(ranking_values) if ranking_values else None
        maximum_yield = max(ranking_values) if ranking_values else None
        yield_std = (
            math.sqrt(
                sum((value - ranking_yield) ** 2 for value in ranking_values)
                / len(ranking_values)
            )
            if ranking_values and ranking_yield is not None
            else None
        )
        risk_extreme_prediction = (
            minimum_prediction
            if prediction.target == "Y"
            else maximum_prediction
        )
        failure_summary = _failure_summary(rows, prediction.target)
        for group_rows in rankings.values():
            for feature_row in group_rows:
                feature_row["related_failure_target"] = failure_summary[
                    "top_failure_target"
                ]
                feature_row["related_fail_bit_count_target"] = failure_summary[
                    "top_fail_bit_count_target"
                ]
                feature_row["related_fail_bit_count_average"] = failure_summary[
                    "top_fail_bit_count_average"
                ]
        risk_counts = {
            risk: sum(row.get("risk_level") == risk for row in rows)
            for risk in ("danger", "warning", "normal")
        }
        wafer_list: list[dict[str, Any]] = []
        for row in rows:
            identifier = row.get(prediction.identifier_column)
            prediction_value = _finite(row.get(prediction_column))
            actual_yield = _finite(row.get("actual_Y"))
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
                    "actual_yield": actual_yield,
                    "ranking_yield": (
                        actual_yield if actual_yield is not None else prediction_value
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
                "average_actual_yield": average_actual_yield,
                "ranking_yield": ranking_yield,
                "ranking_basis": ranking_basis,
                "overall_average_yield": (
                    overall_actual_yield
                    if ranking_basis == "actual_y"
                    else overall_predicted_yield
                ),
                "difference_from_overall": (
                    ranking_yield - (
                        overall_actual_yield
                        if ranking_basis == "actual_y"
                        else overall_predicted_yield
                    )
                    if ranking_yield is not None and (
                        overall_actual_yield
                        if ranking_basis == "actual_y"
                        else overall_predicted_yield
                    ) is not None
                    else None
                ),
                "yield_loss": 100.0 - ranking_yield if ranking_yield is not None else None,
                "minimum_yield": minimum_yield,
                "maximum_yield": maximum_yield,
                "yield_standard_deviation": yield_std,
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
                "feature_importance": {
                    group: rankings[group]
                    for group in ("all", "r", "d", "config")
                },
                "config_categories": rankings["config_categories"],
                "low_sample_config": rankings["low_sample_config"],
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
        target_risk = item["ranking_yield"]
        if target_risk is None:
            target_risk = math.inf
        return (
            target_risk,
            item["lot_id"],
        )

    lots.sort(key=sort_key)
    return to_json_safe(
        {
            "target": prediction.target,
            "aggregation": "selected_lot_sampled_per_wafer_shap",
            "ranking_policy": "actual_y_if_available_else_predicted_y",
            "overall_actual_yield": overall_actual_yield,
            "overall_predicted_yield": overall_predicted_yield,
            "sampling_used": explanation.sampling_used,
            "total_lot_count": len(lots),
            "excluded_row_count": excluded_rows,
            "lots": lots,
        }
    )
