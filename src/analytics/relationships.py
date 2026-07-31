from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.column_detection import detect_feature_columns


STEP_PATTERN = re.compile(r"^Step(?P<step>\d+)_", re.IGNORECASE)
MIN_ASSOCIATION_ROWS = 3
DETAIL_POINT_LIMIT = 150


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _direction(value: float | None) -> str:
    if value is None:
        return "insufficient"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _strength(value: float | None) -> str:
    if value is None:
        return "insufficient"
    absolute = abs(value)
    if absolute >= 0.7:
        return "strong"
    if absolute >= 0.4:
        return "moderate"
    if absolute >= 0.2:
        return "weak"
    return "very_weak"


def pair_association(
    left: pd.Series,
    right: pd.Series,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {"left": _numeric(left), "right": _numeric(right)}
    )
    valid = frame.dropna()
    excluded_count = int(len(frame) - len(valid))
    if (
        len(valid) < MIN_ASSOCIATION_ROWS
        or valid["left"].nunique() < 2
        or valid["right"].nunique() < 2
    ):
        pearson = None
        spearman = None
    else:
        pearson = _finite_float(valid["left"].corr(valid["right"]))
        spearman = _finite_float(
            valid["left"].rank(method="average").corr(
                valid["right"].rank(method="average")
            )
        )
    return {
        "pearson": pearson,
        "spearman": spearman,
        "valid_count": int(len(valid)),
        "excluded_count": excluded_count,
        "direction": _direction(pearson),
        "strength": _strength(pearson),
    }


def eta_squared(
    categories: pd.Series,
    values: pd.Series,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "category": categories.astype("string"),
            "value": _numeric(values),
        }
    ).dropna()
    valid_count = int(len(frame))
    excluded_count = int(len(categories) - valid_count)
    if valid_count < MIN_ASSOCIATION_ROWS or frame["category"].nunique() < 2:
        return {
            "eta_squared": None,
            "valid_count": valid_count,
            "excluded_count": excluded_count,
            "category_count": int(frame["category"].nunique()),
        }
    grand_mean = float(frame["value"].mean())
    total = float(((frame["value"] - grand_mean) ** 2).sum())
    if total <= 0:
        effect = 0.0
    else:
        between = 0.0
        for _, group in frame.groupby("category", observed=True):
            between += len(group) * float(
                (group["value"].mean() - grand_mean) ** 2
            )
        effect = min(max(between / total, 0.0), 1.0)
    return {
        "eta_squared": effect,
        "valid_count": valid_count,
        "excluded_count": excluded_count,
        "category_count": int(frame["category"].nunique()),
    }


def calculate_pareto(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    threshold: float = 0.8,
) -> dict[str, Any]:
    cleaned = [
        {**row, "impact": max(_finite_float(row.get(score_field)) or 0.0, 0.0)}
        for row in rows
    ]
    cleaned.sort(key=lambda row: row["impact"], reverse=True)
    total = sum(row["impact"] for row in cleaned)
    cumulative = 0.0
    required_count = 0
    output: list[dict[str, Any]] = []
    for index, row in enumerate(cleaned, 1):
        share = row["impact"] / total if total > 0 else 0.0
        cumulative += share
        if required_count == 0 and cumulative >= threshold:
            required_count = index
        output.append(
            {
                **row,
                "rank": index,
                "share": share,
                "cumulative_share": cumulative,
                "within_threshold": cumulative <= threshold or index == required_count,
            }
        )
    if total <= 0:
        required_count = 0
    reached = (
        output[required_count - 1]["cumulative_share"]
        if required_count
        else 0.0
    )
    group_counts = {"R": 0, "D": 0, "EQ": 0}
    for row in output[:required_count]:
        group = str(row.get("group", ""))
        if group in group_counts:
            group_counts[group] += 1
    return {
        "threshold": threshold,
        "required_feature_count": required_count,
        "cumulative_contribution": reached,
        "total_feature_count": len(output),
        "total_impact": total,
        "group_counts": group_counts,
        "features": output,
    }


def _step_number(column: str) -> int | None:
    match = STEP_PATTERN.match(column)
    return int(match.group("step")) if match else None


def _is_categorical_equipment(series: pd.Series) -> bool:
    if not is_numeric_dtype(series):
        return True
    unique = int(series.nunique(dropna=True))
    return unique <= max(12, int(math.sqrt(max(len(series), 1))))


def _readable_name(feature: str, group: str) -> str:
    step = _step_number(feature)
    suffix = (
        "Response"
        if group == "R"
        else "Defect"
        if group == "D"
        else "Equipment"
    )
    return f"Step {step} · {suffix}" if step is not None else feature


def _correlation_ranking(
    dataframe: pd.DataFrame,
    target: str,
    method: str,
) -> list[dict[str, Any]]:
    detected = detect_feature_columns(list(dataframe.columns))
    rows: list[dict[str, Any]] = []
    target_values = dataframe[target]
    for group, columns in (
        ("R", detected["r_columns"]),
        ("D", detected["d_columns"]),
        ("EQ", detected["eq_columns"]),
    ):
        for feature in columns:
            missing_count = int(
                dataframe[[feature, target]].isna().any(axis=1).sum()
            )
            if group == "EQ" and _is_categorical_equipment(dataframe[feature]):
                effect = eta_squared(dataframe[feature], target_values)
                score = effect["eta_squared"]
                rows.append(
                    {
                        "feature": feature,
                        "display_name": _readable_name(feature, group),
                        "step": _step_number(feature),
                        "group": group,
                        "ranking_basis": "Eta squared vs target",
                        "score": score,
                        "signed_association": None,
                        "direction": "group_difference",
                        "valid_count": effect["valid_count"],
                        "missing_count": missing_count,
                        "missing_rate": (
                            missing_count / len(dataframe)
                            if len(dataframe)
                            else 0.0
                        ),
                        "category_count": effect["category_count"],
                        "is_categorical": True,
                    }
                )
                continue
            association = pair_association(dataframe[feature], target_values)
            signed = association[method]
            rows.append(
                {
                    "feature": feature,
                    "display_name": _readable_name(feature, group),
                    "step": _step_number(feature),
                    "group": group,
                    "ranking_basis": f"Absolute {method.title()} correlation",
                    "score": abs(signed) if signed is not None else None,
                    "signed_association": signed,
                    "direction": _direction(signed),
                    "valid_count": association["valid_count"],
                    "missing_count": missing_count,
                    "missing_rate": (
                        missing_count / len(dataframe)
                        if len(dataframe)
                        else 0.0
                    ),
                    "category_count": None,
                    "is_categorical": False,
                }
            )
    return sorted(
        rows,
        key=lambda row: row["score"] if row["score"] is not None else -1.0,
        reverse=True,
    )


def _shap_rankings(
    importance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in importance:
        group = str(source.get("parameter_type", "unknown")).upper()
        feature = str(source.get("feature", ""))
        rows.append(
            {
                "feature": feature,
                "display_name": _readable_name(feature.split("__")[-1], group),
                "step": _step_number(feature.split("__")[-1]),
                "group": group,
                "ranking_basis": "Mean absolute SHAP value",
                "score": _finite_float(source.get("mean_abs_shap")),
                "signed_association": None,
                "direction": str(source.get("direction", "model_contribution")),
                "valid_count": None,
                "missing_count": None,
                "missing_rate": None,
                "category_count": None,
                "is_categorical": group == "EQ",
            }
        )
    return sorted(
        rows,
        key=lambda row: row["score"] if row["score"] is not None else -1.0,
        reverse=True,
    )


def _grouped_rankings(
    rows: list[dict[str, Any]],
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "overall": rows[:top_n],
        "R": [row for row in rows if row["group"] == "R"][:top_n],
        "D": [row for row in rows if row["group"] == "D"][:top_n],
        "EQ": [row for row in rows if row["group"] == "EQ"][:top_n],
    }


def _sample_points(
    left: pd.Series,
    right: pd.Series,
) -> list[dict[str, float]]:
    frame = pd.DataFrame(
        {"x": _numeric(left), "y": _numeric(right)}
    ).dropna()
    if len(frame) > DETAIL_POINT_LIMIT:
        indices = np.linspace(
            0, len(frame) - 1, DETAIL_POINT_LIMIT, dtype=int
        )
        frame = frame.iloc[indices]
    return [
        {"x": float(row.x), "y": float(row.y)}
        for row in frame.itertuples(index=False)
    ]


def _equipment_groups(
    equipment: pd.Series,
    defect: pd.Series,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "equipment": equipment.astype("string"),
            "value": _numeric(defect),
        }
    ).dropna()
    groups: list[dict[str, Any]] = []
    for name, group in frame.groupby("equipment", observed=True):
        values = group["value"]
        groups.append(
            {
                "equipment": str(name),
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "sample_warning": len(values) < 10,
            }
        )
    groups.sort(key=lambda row: row["median"], reverse=True)
    return groups[:12]


def _shap_for_columns(
    importance: list[dict[str, Any]],
    columns: list[str],
) -> float:
    total = 0.0
    for row in importance:
        cleaned = str(row.get("feature", "")).split("__")[-1]
        if any(cleaned.startswith(column) for column in columns):
            total += _finite_float(row.get("mean_abs_shap")) or 0.0
    return total


def _confidence(valid_count: int, missing_rate: float) -> str:
    if valid_count >= 100 and missing_rate <= 0.1:
        return "sufficient"
    if valid_count >= 30 and missing_rate <= 0.3:
        return "caution"
    return "insufficient"


def _relationship_paths(
    dataframe: pd.DataFrame,
    target: str,
    shap_importance: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    detected = detect_feature_columns(list(dataframe.columns))
    by_step: dict[int, dict[str, list[str]]] = {}
    for group, columns in (
        ("R", detected["r_columns"]),
        ("D", detected["d_columns"]),
        ("EQ", detected["eq_columns"]),
    ):
        for column in columns:
            step = _step_number(column)
            if step is not None:
                by_step.setdefault(
                    step, {"R": [], "D": [], "EQ": []}
                )[group].append(column)

    paths: list[dict[str, Any]] = []
    for step, columns in sorted(by_step.items()):
        if not columns["D"] or not (columns["R"] or columns["EQ"]):
            continue
        defect = max(
            columns["D"],
            key=lambda name: abs(
                pair_association(dataframe[name], dataframe[target])[
                    "pearson"
                ]
                or 0.0
            ),
        )
        response = (
            max(
                columns["R"],
                key=lambda name: abs(
                    pair_association(dataframe[name], dataframe[defect])[
                        "pearson"
                    ]
                    or 0.0
                ),
            )
            if columns["R"]
            else None
        )
        equipment = columns["EQ"][0] if columns["EQ"] else None
        d_y = pair_association(dataframe[defect], dataframe[target])
        r_d = (
            pair_association(dataframe[response], dataframe[defect])
            if response
            else None
        )
        r_y = (
            pair_association(dataframe[response], dataframe[target])
            if response
            else None
        )
        eq_d: dict[str, Any] | None = None
        eq_y: dict[str, Any] | None = None
        eq_groups: list[dict[str, Any]] = []
        if equipment:
            if _is_categorical_equipment(dataframe[equipment]):
                eq_d = eta_squared(dataframe[equipment], dataframe[defect])
                eq_y = eta_squared(dataframe[equipment], dataframe[target])
                eq_groups = _equipment_groups(
                    dataframe[equipment], dataframe[defect]
                )
            else:
                eq_d = pair_association(
                    dataframe[equipment], dataframe[defect]
                )
                eq_y = pair_association(
                    dataframe[equipment], dataframe[target]
                )
        upstream = max(
            abs((r_d or {}).get("pearson") or 0.0),
            (eq_d or {}).get("eta_squared")
            or abs((eq_d or {}).get("pearson") or 0.0),
        )
        shap_value = _shap_for_columns(
            shap_importance,
            [name for name in (response, defect, equipment) if name],
        )
        selected_columns = [
            name for name in (response, defect, equipment, target) if name
        ]
        missing_rate = float(
            dataframe[selected_columns].isna().any(axis=1).mean()
        )
        valid_count = min(
            [
                value
                for value in (
                    d_y["valid_count"],
                    (r_d or {}).get("valid_count"),
                    (eq_d or {}).get("valid_count"),
                )
                if value is not None
            ],
            default=0,
        )
        paths.append(
            {
                "step": step,
                "response": response,
                "defect": defect,
                "equipment": equipment,
                "r_d": r_d,
                "eq_d": eq_d,
                "d_y": d_y,
                "r_y": r_y,
                "eq_y": eq_y,
                "shap_importance": shap_value,
                "valid_count": valid_count,
                "missing_rate": missing_rate,
                "confidence": _confidence(valid_count, missing_rate),
                "upstream_strength": upstream,
                "raw_path_score": upstream
                * abs(d_y["pearson"] or 0.0)
                * (shap_value if shap_value > 0 else 1.0),
                "r_vs_d": (
                    _sample_points(
                        dataframe[response], dataframe[defect]
                    )
                    if response
                    else []
                ),
                "eq_vs_d": eq_groups,
                "d_vs_y": _sample_points(
                    dataframe[defect], dataframe[target]
                ),
            }
        )
    max_score = max(
        (row["raw_path_score"] for row in paths), default=0.0
    )
    for row in paths:
        row["path_score"] = (
            row["raw_path_score"] / max_score if max_score > 0 else 0.0
        )
    paths.sort(key=lambda row: row["path_score"], reverse=True)
    for rank, row in enumerate(paths, 1):
        row["rank"] = rank
        row["path_status"] = "priority" if rank <= 3 else "reference"
        trend = row["d_y"]["direction"]
        direction_text = (
            "Final Yield 감소"
            if trend == "negative"
            else "Final Yield 증가"
            if trend == "positive"
            else "Final Yield와 뚜렷한 방향 없음"
        )
        upstream_text = (
            f"{row['response']}와 {row['defect']}가 함께 변하는 경향"
            if row["response"]
            else f"{row['equipment']} 그룹별 {row['defect']} 차이"
        )
        row["interpretation"] = (
            f"Step {row['step']}에서 {upstream_text}이 관측되었고, "
            f"{row['defect']}는 {direction_text} 방향의 연관성을 보였습니다. "
            "인과관계가 아닌 엔지니어 검토 우선순위입니다."
        )
        row.pop("raw_path_score", None)
        row.pop("upstream_strength", None)
    return paths[:top_n]


def analyze_relationships(
    dataframe: pd.DataFrame,
    *,
    target: str = "Y",
    correlation_method: str = "pearson",
    top_n: int = 10,
    shap_importance: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if target not in dataframe.columns:
        raise ValueError(f"목표 컬럼 {target}이 데이터에 없습니다.")
    if correlation_method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method는 pearson 또는 spearman이어야 합니다.")
    if not 1 <= top_n <= 20:
        raise ValueError("top_n은 1부터 20 사이여야 합니다.")
    if len(dataframe) == 0:
        raise ValueError("빈 데이터는 연관 분석할 수 없습니다.")

    correlation_rows = _correlation_ranking(
        dataframe, target, correlation_method
    )
    shap_rows = _shap_rankings(shap_importance or [])
    pareto_source = shap_rows if shap_rows else correlation_rows
    pareto_basis = (
        "Mean absolute SHAP value"
        if shap_rows
        else f"Absolute {correlation_method.title()} association"
    )
    pareto = calculate_pareto(
        pareto_source,
        score_field="score",
    )
    pareto["ranking_basis"] = pareto_basis
    pareto["caveat"] = (
        "모델 영향도 기준 누적 비율이며 실제 수율 개선량을 의미하지 않습니다."
        if shap_rows
        else "상관관계 기반 누적 설명 비율이며 실제 수율 개선량을 의미하지 않습니다."
    )
    paths = _relationship_paths(
        dataframe,
        target,
        shap_importance or [],
        top_n,
    )
    detected = detect_feature_columns(list(dataframe.columns))
    available_steps = sorted(
        {
            step
            for column in [
                *detected["r_columns"],
                *detected["d_columns"],
                *detected["eq_columns"],
            ]
            if (step := _step_number(column)) is not None
        }
    )
    return {
        "target": target,
        "correlation_method": correlation_method,
        "rankings": {
            "shap": _grouped_rankings(shap_rows, top_n),
            "correlation": _grouped_rankings(
                correlation_rows, top_n
            ),
        },
        "pareto": pareto,
        "relationship_paths": paths,
        "available_steps": available_steps,
        "confidence_criteria": {
            "sufficient": "유효 표본 100개 이상, 결측률 10% 이하",
            "caution": "유효 표본 30개 이상, 결측률 30% 이하",
            "insufficient": "그 외",
        },
        "caveats": [
            "Correlation does not imply causation.",
            "연관 경로는 엔지니어 검토를 위한 우선순위입니다.",
            "공식 공정 Spec이 아닌 데이터 기반 분석 결과입니다.",
            "표본 수가 적은 Equipment 결과는 주의가 필요합니다.",
        ],
    }
