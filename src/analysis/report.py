"""JSON analysis-report serialization: the single function backing both the
root-cause tab's "JSON 보고서 저장" button and (future) `/api/analysis/context`
for the SUNI chatbot / n8n automation. Both consumers must get the exact
same structure, so build it here once rather than assembling it in the route
or reassembling it on the frontend.

Two factor sets are deliberately different and never conflated:
  - `targets[].factors`: one narrative factor per target (the single
    strongest, eps2-ranked, from the full R+D+Config pool), included only
    if raw p < 0.05. This is what an LLM should cite as "the" driver for
    that target -- including every FDR-significant factor here would let a
    borderline second factor (e.g. Y2's Step24_R1) crowd the narrative.
  - `alarms` / `summary.alarm_wafers`: the existing, already-verified
    alarm engine (BH-FDR q<0.05 selection, see `_alarm_engine_factors`)
    used by /api/alarms and /api/alarms/summary. The report echoes that
    number rather than recomputing a different one from the narrative
    factor set, so the report and the live alarm log never disagree.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.control_range import (
    ControlRange,
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.recommendations import REPORT_TAG_TIERS, compute_recommendations
from src.analysis.rounding import round_floats
from src.analysis.screening.schema import Schema, parse_schema
from src.analysis.screening.selector import (
    ParetoFactor,
    confidence_tier,
    select_fdr_significant_factors,
    select_primary_factor,
)

REPORT_INCLUSION_P_THRESHOLD = 0.05
BINNED_PROFILE_BINS = 12

_INTERPRETATION_BY_SHAPE = {
    "monotonic_increasing": "값이 클수록 불량률이 상승하는 관계다. 값을 낮추는 방향의 조치가 유효하다.",
    "monotonic_decreasing": "값이 작을수록 불량률이 상승하는 관계다. 값을 높이는 방향의 조치가 유효하다.",
    "u_shape": (
        "최적 중심에서 양방향으로 멀어질수록 불량률이 상승한다. 값이 높은 것과 낮은 것의 조치 방향이 "
        "반대이므로 이탈 방향을 반드시 함께 기술할 것."
    ),
    "unclear": "뚜렷한 단조 또는 U자 패턴이 확인되지 않았다. 방향성을 단정하지 말고 개별 구간별 양상을 함께 검토할 것.",
}

LIMITATIONS = [
    "해당 인자가 계측된 wafer만 분석 대상이다. R은 전체의 15%, D는 5%다.",
    "평가 대상 wafer 중 상당수는 선정 인자가 하나도 계측되지 않아 판정할 수 없다.",
    "eps2는 통계적 연관성이며 인과가 아니다. 공정 순서상 선행·후행 관계나 교락 인자는 반영되지 않았다.",
    "Config는 장비당 표본이 적어 검출력이 부족할 수 있다. p<0.05를 만족하지 못한 것이 영향이 없다는 뜻은 아니다.",
    "관리한계는 '평소와 다른가'를 판정하며 '수율이 좋은가'를 보장하지 않는다.",
]


def _binned_profile(x: pd.Series, y: pd.Series, bins: int = BINNED_PROFILE_BINS) -> list[dict[str, float]]:
    try:
        q = pd.qcut(x, bins, duplicates="drop")
    except ValueError:
        return []
    frame = pd.DataFrame({"x": x, "y": y, "q": q})
    profile = []
    for _, group in frame.groupby("q", observed=True):
        profile.append({"x_center": float(group["x"].mean()), "y_mean": float(group["y"].mean()), "n": int(len(group))})
    profile.sort(key=lambda row: row["x_center"])
    return profile


def _target_stats(df: pd.DataFrame, target: str) -> dict[str, float]:
    values = pd.to_numeric(df[target], errors="coerce").dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    return {"mean": float(values.mean()), "std": float(values.std()), "q1": float(q1), "q3": float(q3)}


def _missing_pct(df: pd.DataFrame, feature: str) -> float:
    return float(df[feature].isna().mean() * 100.0)


def _control_limits_dict(control_range: ControlRange) -> dict[str, Any]:
    by_key = {line.key: line for line in control_range.reference_lines}
    sigma6_drawn = any(by_key[key].drawable for key in ("s6_lo", "s6_hi") if key in by_key)
    return {
        "lcl": control_range.lower,
        "ucl": control_range.upper,
        "one_sided": control_range.one_sided,
        "mean": control_range.mean,
        "std": control_range.std,
        "q1": control_range.q1,
        "q3": control_range.q3,
        "sigma3": [by_key["s3_lo"].value, by_key["s3_hi"].value],
        "sigma6": [by_key["s6_lo"].value, by_key["s6_hi"].value],
        "sigma6_drawn": sigma6_drawn,
    }


def _eval_result(eval_df: pd.DataFrame, control_range: ControlRange, factor: ParetoFactor) -> dict[str, Any]:
    x = pd.to_numeric(eval_df[factor.feature], errors="coerce")
    observed_mask = x.notna()
    observed = int(observed_mask.sum())

    alarms = evaluate_alarms(eval_df, control_range)
    alarm_ids = {alarm.lot_wafer_id for alarm in alarms}

    id_column = (
        eval_df["Lot_Wafer_ID"].astype(str) if "Lot_Wafer_ID" in eval_df.columns else pd.Series(eval_df.index.astype(str))
    )
    y = pd.to_numeric(eval_df[factor.target], errors="coerce")
    is_alarmed = observed_mask & id_column.isin(alarm_ids)
    is_normal = observed_mask & ~id_column.isin(alarm_ids)

    return {
        "alarms": len(alarms),
        "observed": observed,
        "mean_y_alarm": float(y[is_alarmed].mean()) if is_alarmed.any() else None,
        "mean_y_normal": float(y[is_normal].mean()) if is_normal.any() else None,
    }


def _top_factor_per_target(train_df: pd.DataFrame, schema: Schema, target: str) -> ParetoFactor | None:
    """The single strongest (highest-eps2) factor for `target` across the
    full R+D+Config pool, included only if it clears the report's own raw
    p<0.05 bar. This is the report's own narrative-inclusion rule -- a
    different, deliberately non-interchangeable concept from the alarm
    engine's factor set below (see module docstring).
    """
    factor = select_primary_factor(train_df, schema, target)
    if factor is None or factor.p_value >= REPORT_INCLUSION_P_THRESHOLD:
        return None
    return factor


def _alarm_engine_factors(train_df: pd.DataFrame, schema: Schema) -> list[ParetoFactor]:
    """The existing BH-FDR (q<0.05) factor set already used by
    /api/alarms and /api/alarms/summary -- reused as-is so the report's
    alarm numbers always agree with the live alarm log.
    """
    factors: list[ParetoFactor] = []
    for target in schema.target_cols:
        factors.extend(select_fdr_significant_factors(train_df, schema, target))
    return factors


def build_analysis_report(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    train_dataset_id: str,
    eval_dataset_id: str,
    train_meta: dict[str, Any],
    eval_meta: dict[str, Any],
    app_version: str,
    generated_at: str,
) -> dict[str, Any]:
    schema = parse_schema(train_df)

    included_factors: list[ParetoFactor] = []
    target_entries: list[dict[str, Any]] = []
    for target in schema.target_cols:
        factor = _top_factor_per_target(train_df, schema, target)
        factor_entries = []
        if factor is not None:
            included_factors.append(factor)
            control_range = compute_control_range(train_df, factor)
            factor_entries.append(
                {
                    "feature": factor.feature,
                    "kind": factor.kind,
                    "step": factor.step,
                    "rank": 1,
                    "eps2": factor.eps2,
                    "contribution_pct": factor.contribution_pct,
                    "cumulative_pct": factor.cumulative_pct,
                    "spearman_rho": factor.spearman_r,
                    "p_value": factor.p_value,
                    "q_value": factor.q_value,
                    "grade": {"strong": "강함", "moderate": "보통", "weak": "약함", "reference": "참고"}[
                        confidence_tier(factor.eps2, factor.p_value)
                    ],
                    "n_observed": factor.n_observed,
                    "n_missing_pct": _missing_pct(train_df, factor.feature),
                    "relation": {
                        "shape": factor.relation_shape,
                        "optimal_center": factor.optimal_center,
                        "interpretation": _INTERPRETATION_BY_SHAPE.get(
                            factor.relation_shape, _INTERPRETATION_BY_SHAPE["unclear"]
                        ),
                    },
                    "binned_profile": _binned_profile(
                        pd.to_numeric(train_df[factor.feature], errors="coerce"),
                        pd.to_numeric(train_df[target], errors="coerce"),
                    ),
                    "control_limits": _control_limits_dict(control_range),
                    "eval_result": _eval_result(eval_df, control_range, factor),
                }
            )
        target_entries.append(
            {"target": target, "target_stats": _target_stats(train_df, target), "factors": factor_entries}
        )

    alarm_factors = _alarm_engine_factors(train_df, schema)
    control_ranges = [compute_control_range(train_df, factor) for factor in alarm_factors]
    alarms_by_feature = {cr.feature: evaluate_alarms(eval_df, cr) for cr in control_ranges}
    verdicts = summarize_wafer_status(eval_df, control_ranges, alarms_by_feature)

    alarm_ids = [v.lot_wafer_id for v in verdicts if v.status == "alarm"]
    normal_ids = [v.lot_wafer_id for v in verdicts if v.status == "normal"]
    unmeasured_ids = [v.lot_wafer_id for v in verdicts if v.status == "unmeasured"]

    indexed = eval_df.set_index("Lot_Wafer_ID") if "Lot_Wafer_ID" in eval_df.columns else None
    mean_yield_alarm = None
    mean_yield_normal = None
    if indexed is not None and "Y" in eval_df.columns:
        if alarm_ids:
            mean_yield_alarm = float(indexed.loc[alarm_ids, "Y"].mean())
        no_alarm_group = normal_ids + unmeasured_ids
        if no_alarm_group:
            mean_yield_normal = float(indexed.loc[no_alarm_group, "Y"].mean())

    step_by_feature_target = {(factor.feature, factor.target): factor.step for factor in alarm_factors}
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    alarm_records: list[dict[str, Any]] = []
    for cr in control_ranges:
        step = step_by_feature_target.get((cr.feature, cr.target), 0)
        for alarm in alarms_by_feature.get(cr.feature, []):
            row = indexed.loc[alarm.lot_wafer_id] if indexed is not None and alarm.lot_wafer_id in indexed.index else None
            actual_y_final = float(row["Y"]) if row is not None and "Y" in row and pd.notna(row["Y"]) else None
            alarm_records.append(
                {
                    "lot_wafer_id": alarm.lot_wafer_id,
                    "lot_id": alarm.lot_id,
                    "wafer_slot": alarm.wafer_slot,
                    "step": step,
                    "feature": alarm.feature,
                    "kind": alarm.kind,
                    "target": alarm.target,
                    "value": alarm.value,
                    "normal_range": [alarm.lower, alarm.upper],
                    "deviation": alarm.deviation,
                    "direction": alarm.direction,
                    "severity": alarm.severity,
                    "actual_y_target": alarm.actual_y,
                    "actual_y_final": actual_y_final,
                }
            )
    alarm_records.sort(
        key=lambda item: (
            severity_rank.get(item["severity"], 3),
            item["lot_id"] or "",
            item["wafer_slot"] if item["wafer_slot"] is not None else -1,
        )
    )

    yield_gap_pp = (
        mean_yield_alarm - mean_yield_normal if mean_yield_alarm is not None and mean_yield_normal is not None else None
    )

    # 개선 권장 목록 (spec §3-6) -- scoped to the same "1위 인자" set as
    # targets[].factors above, not the full pool; only 강함/보통 rows are
    # exported here (참고-tagged rows still show on the live screen with
    # its own hide-by-default toggle, but don't belong in a report meant
    # to be cited as findings).
    primary_factors_by_target = {factor.target: factor for factor in included_factors}
    recommendation_rows, recommendation_summaries = compute_recommendations(
        train_df, eval_df, schema, primary_factors=primary_factors_by_target
    )
    recommendation_records = [
        {
            "lot_wafer_id": row.lot_wafer_id,
            "lot_id": row.lot_id,
            "step": row.step,
            "feature": row.feature,
            "kind": row.kind,
            "target": row.target,
            "value": row.value,
            "recommended_range": [row.recommended_lo, row.recommended_hi],
            "direction": row.direction,
            "expected_improvement_pct": row.expected_improvement_pct,
            "tag": row.tag,
        }
        for row in recommendation_rows
        if (summary := recommendation_summaries.get(row.target)) is not None and summary.grade in REPORT_TAG_TIERS
    ]
    recommendation_records.sort(key=lambda item: item["lot_wafer_id"])

    report = {
        "meta": {
            "generated_at": generated_at,
            "app_version": app_version,
            "dataset": {
                "train": {
                    "name": train_meta.get("original_filename"),
                    "rows": train_meta.get("row_count"),
                    "lots": train_meta.get("lot_count"),
                    "lot_range": _lot_range(train_meta),
                },
                "eval": {
                    "name": eval_meta.get("original_filename"),
                    "rows": eval_meta.get("row_count"),
                    "lots": eval_meta.get("lot_count"),
                    "lot_range": _lot_range(eval_meta),
                },
            },
        },
        "method": {
            "screening": "편향보정 epsilon-squared (분위수 8구간 ANOVA) + BH-FDR",
            "contribution_denominator": "전체 인자 풀(R+D+Config)의 eps2 합",
            "control_limit": "IQR 1.5배 (Q1-1.5*IQR ~ Q3+1.5*IQR), X축 기준",
            "inclusion_rule": "p < 0.05",
            "missing_policy": "대체하지 않음. pairwise deletion + _miss 태그",
        },
        "summary": {
            "targets_analyzed": len(schema.target_cols),
            "factors_included": len(included_factors),
            "excluded_low_significance": len(schema.factor_cols) - len(included_factors),
            "alarm_wafers": len(alarm_ids),
            "normal_wafers": len(normal_ids),
            "undecidable_wafers": len(unmeasured_ids),
            "mean_yield_alarm": mean_yield_alarm,
            "mean_yield_normal": mean_yield_normal,
            "yield_gap_pp": yield_gap_pp,
        },
        "targets": target_entries,
        "alarms": alarm_records,
        "recommendations": recommendation_records,
        "limitations": LIMITATIONS,
    }
    return round_floats(report)


def _lot_range(meta: dict[str, Any]) -> str | None:
    lot_min, lot_max = meta.get("lot_min"), meta.get("lot_max")
    if lot_min is None or lot_max is None:
        return None
    return f"{lot_min}~{lot_max}"
