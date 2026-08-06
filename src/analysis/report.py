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

from src.analysis.alarm_gbdt import FINAL_YIELD_COLUMN as GBDT_TARGET_COLUMN
from src.analysis.alarm_gbdt import feature_columns as gbdt_feature_columns
from src.analysis.control_range import (
    ControlRange,
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.display_text import add_display_text
from src.analysis.warning_line import compute_warning_line, fit_reference_model, observed_yield_gap
from src.analysis.llm_stats import (
    band_stability,
    chamber_interaction_p,
    config_main_effect_screening,
    judge_confidence,
    per_chamber_window,
    per_factor_measurement_bias,
    summarize_measurement_bias,
)
from src.analysis.recommendations import REPORT_TAG_TIERS, compute_factor_recommendation, compute_recommendations
from src.analysis.rounding import round_floats
from src.analysis.screening.quantile_profile import quantile_bins
from src.analysis.screening.schema import Schema, parse_schema
from src.analysis.screening.selector import (
    ParetoFactor,
    benjamini_hochberg,
    confidence_tier,
    select_fdr_significant_factors,
    select_primary_factor,
)

REPORT_INCLUSION_P_THRESHOLD = 0.05
BINNED_PROFILE_BINS = 12

# SUNI chatbot context (spec "챗봇 알람 답변 확장" A-2): safety caps so an
# uploaded dataset with many more alarms/recommendations than the bundled
# train.CSV can't blow up the LLM context payload.
ALARM_RECORD_TRUNCATE_THRESHOLD = 200
ALARM_RECORD_TRUNCATE_KEEP = 100
RECOMMENDATION_RECORD_KEEP = 50
_TAG_PRIORITY_RANK = {"priority": 2, "recommended": 1, "reference": 0}

_INTERPRETATION_BY_SHAPE = {
    "monotonic_increasing": "값이 클수록 불량률이 상승하는 관계다. 값을 낮추는 방향의 조치가 유효하다.",
    "monotonic_decreasing": "값이 작을수록 불량률이 상승하는 관계다. 값을 높이는 방향의 조치가 유효하다.",
    "u_shape": (
        "최적 중심에서 양방향으로 멀어질수록 불량률이 상승한다. 값이 높은 것과 낮은 것의 조치 방향이 "
        "반대이므로 이탈 방향을 반드시 함께 기술할 것."
    ),
    "unclear": "뚜렷한 단조 또는 U자 패턴이 확인되지 않았다. 방향성을 단정하지 말고 개별 구간별 양상을 함께 검토할 것.",
}

# Non-dataset-specific limitations -- the measurement-rate sentence
# (dataset-dependent, was hardcoded "R은 전체의 15%, D는 5%다" here, which
# is only ever true for train.CSV) is built separately by
# `_measurement_rate_limitation` and prepended per-dataset instead.
LIMITATIONS = [
    "평가 대상 wafer 중 상당수는 선정 인자가 하나도 계측되지 않아 판정할 수 없다.",
    "설명력 지표는 통계적 연관성이며 인과가 아니다. 공정 순서상 선행·후행 관계나 교락 인자는 반영되지 않았다.",
    "Config는 장비당 표본이 적어 검출력이 부족할 수 있다. p<0.05를 만족하지 못한 것이 영향이 없다는 뜻은 아니다.",
    "관리한계는 '평소와 다른가'를 판정하며 '수율이 좋은가'를 보장하지 않는다.",
    # 알람 판정 GBDT 전환 §E-2-1: "등급 기준은 절대 기준이 아니다"를 화면
    # 상세 패널뿐 아니라 JSON limitations에도 동일하게 밝힌다. 알람 등급
    # 분위수(5%/10%/15%)와 경고선 임계(0.35σ)도 같은 성격의 경험값이다.
    "알람 등급 분위수(하위 5%/10%/15%), 경고선 임계(0.35 x 표준편차), 종합 신뢰성 등급의 배점 기준은 모두 "
    "통계적으로 도출된 값이 아니라 내장 데이터셋에서 등급이 구분되도록 설정한 경험값이며 절대 기준이 아니다.",
]

# Below this measurement rate, "~만 분석 대상이다" (only the measured
# subset) reads accurately; at/above it, most wafers already have a
# reading, so the softer "~를 분석 대상으로 한다" avoids overstating how
# exclusionary the dataset actually is (spec: 문구 전수 검토 §A-1).
_HIGH_MEASUREMENT_RATE_THRESHOLD = 60.0


def _measurement_rate_limitation(df: pd.DataFrame, schema: Schema) -> str:
    """Real per-dataset R/D measurement rate (spec §A-1) -- mean of each
    column's own non-null rate, not a flat cell-count fraction, so a
    dataset with a few densely-measured columns and many sparse ones
    isn't misrepresented as "well measured" on average.
    """
    r_rate = float(df[schema.r_cols].notna().mean().mean() * 100.0) if schema.r_cols else None
    d_rate = float(df[schema.d_cols].notna().mean().mean() * 100.0) if schema.d_cols else None
    parts = []
    if r_rate is not None:
        parts.append(f"Response는 전체의 {r_rate:.1f}%")
    if d_rate is not None:
        parts.append(f"Defect는 전체의 {d_rate:.1f}%")
    if not parts:
        return "해당 인자가 계측된 wafer만 분석 대상이다."
    rates = [r for r in (r_rate, d_rate) if r is not None]
    if all(r >= _HIGH_MEASUREMENT_RATE_THRESHOLD for r in rates):
        return f"해당 인자가 계측된 wafer를 분석 대상으로 한다. {', '.join(parts)}에서 관측되었다."
    return f"해당 인자가 계측된 wafer만 분석 대상이다. {', '.join(parts)}에서 관측되었다."


def _binned_profile(x: pd.Series, y: pd.Series, bins: int = BINNED_PROFILE_BINS) -> list[dict[str, float]]:
    """Same quantile_bins grouping every other consumer (curve, window,
    optimal center) reads from -- just reshaped to this report's own
    long-standing {x_center, y_mean, n} field names rather than a second,
    separately-binned computation.
    """
    return [
        {"x_center": row["x_mean"], "y_mean": row["y_mean"], "n": row["n"]}
        for row in quantile_bins(x, y, bins=bins)
    ]


def _target_stats(df: pd.DataFrame, target: str) -> dict[str, float]:
    values = pd.to_numeric(df[target], errors="coerce").dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    return {"mean": float(values.mean()), "std": float(values.std()), "q1": float(q1), "q3": float(q3)}


def _missing_pct(df: pd.DataFrame, feature: str) -> float:
    return float(df[feature].isna().mean() * 100.0)


def _resolved_optimal_center(factor: ParetoFactor, window: Any) -> float | None:
    """Same guard as scatter.py's `_resolve_optimal_center`: an
    optimal_center outside its own recommended window is a contradiction
    (spec §3-3), never something to report as-is. `window` is the
    `FactorRecommendation` already computed for this factor (or None).
    """
    if factor.optimal_center is None or window is None:
        return factor.optimal_center
    if window.recommended_lo <= factor.optimal_center <= window.recommended_hi:
        return factor.optimal_center
    return None


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


def _warning_line_dict(reference_model: Any, train_df: pd.DataFrame, feature: str, gbdt_features: list[str]) -> dict[str, Any] | None:
    """알람 판정 GBDT 전환 §C-4: 경고선은 화면에 곡선을 그리지 않지만
    JSON 보고서에는 재현성 확인용으로 남긴다. 경고선이 없으면(이 인자
    단독으로는 예측 수율이 임계를 넘지 않음) None -- "경고선 없음"은
    정상 결과다.
    """
    if reference_model is None or feature not in train_df.columns:
        return None
    warning = compute_warning_line(reference_model, train_df, feature, gbdt_features)
    if warning is None or (warning.lower is None and warning.upper is None):
        return None
    gaps = observed_yield_gap(train_df, feature, warning)
    primary_value = warning.upper if warning.upper is not None else warning.lower
    primary_gap = gaps["upper_gap"] if warning.upper is not None else gaps["lower_gap"]
    return {
        "value": primary_value,
        "lower": warning.lower,
        "upper": warning.upper,
        "method": "partial_dependence",
        "pdp_range": warning.pdp_range,
        "observed_yield_gap": primary_gap,
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

    # 알람 판정 GBDT 전환 §C-4: 경고선을 factor마다 하나씩 계산해 보고서에
    # 싣는다. 참조 모델은 데이터셋 전체에 한 번만 학습해(§A-1과 별개 목적:
    # 신뢰구간이 아니라 부분 의존도 곡선의 모양) 아래 루프의 5개 인자가
    # 재사용한다. Y가 없거나 R+D 인자가 하나도 없으면 None -- 그 경우
    # warning_line은 모든 factor에서 생략된다(§D: 인자 선정 실패와 같은
    # 원리로, 조용히 생략하지 오류를 내지 않는다).
    gbdt_features = gbdt_feature_columns(schema)
    reference_model = (
        fit_reference_model(train_df, gbdt_features)
        if gbdt_features and GBDT_TARGET_COLUMN in train_df.columns
        else None
    )

    included_factors: list[ParetoFactor] = []
    # Two passes: chamber-interaction q-values are BH-corrected across this
    # report's own 5 primary factors (one FDR family), so every factor's
    # p-value must be collected before any q can be assigned.
    raw_targets: list[dict[str, Any]] = []
    chamber_p_values: list[float] = []
    for target in schema.target_cols:
        factor = _top_factor_per_target(train_df, schema, target)
        if factor is None:
            raw_targets.append({"target": target, "factor": None})
            continue
        included_factors.append(factor)
        control_range = compute_control_range(train_df, factor)
        one_sided = factor.relation_shape in ("monotonic_increasing", "monotonic_decreasing")
        config_col = f"Step{factor.step}_Config"
        chamber_p = chamber_interaction_p(train_df, factor.feature, target, config_col)
        chamber_p_values.append(chamber_p if chamber_p is not None else 1.0)
        raw_targets.append(
            {
                "target": target,
                "factor": factor,
                "control_range": control_range,
                "band_stability": band_stability(pd.to_numeric(train_df[factor.feature], errors="coerce")),
                "band_width": None if one_sided else control_range.band_width,
                "window": compute_factor_recommendation(train_df, factor, control_range),
                "config_col": config_col,
                "chamber_p": chamber_p,
            }
        )

    chamber_q_by_target: dict[str, float] = {}
    if chamber_p_values:
        q_values = benjamini_hochberg(chamber_p_values)
        significant_index = 0
        for entry in raw_targets:
            if entry["factor"] is None:
                continue
            chamber_q_by_target[entry["target"]] = q_values[significant_index]
            significant_index += 1

    target_entries: list[dict[str, Any]] = []
    for entry in raw_targets:
        target = entry["target"]
        factor = entry["factor"]
        factor_entries = []
        if factor is not None:
            control_range = entry["control_range"]
            chamber_p = entry["chamber_p"]
            chamber_q = chamber_q_by_target.get(target)
            chamber_significant = bool(chamber_p is not None and chamber_q is not None and chamber_q < 0.05)
            window = entry["window"]
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
                    "report_confidence": judge_confidence(
                        factor.eps2, factor.p_value, entry["band_stability"], entry["band_width"]
                    ),
                    "n_observed": factor.n_observed,
                    "n_missing_pct": _missing_pct(train_df, factor.feature),
                    "relation": {
                        "shape": factor.relation_shape,
                        "optimal_center": _resolved_optimal_center(factor, window),
                        "interpretation": _INTERPRETATION_BY_SHAPE.get(
                            factor.relation_shape, _INTERPRETATION_BY_SHAPE["unclear"]
                        ),
                    },
                    "binned_profile": _binned_profile(
                        pd.to_numeric(train_df[factor.feature], errors="coerce"),
                        pd.to_numeric(train_df[target], errors="coerce"),
                    ),
                    "control_limits": _control_limits_dict(control_range),
                    "warning_line": _warning_line_dict(reference_model, train_df, factor.feature, gbdt_features),
                    "band_stability": entry["band_stability"],
                    "band_width": entry["band_width"],
                    "window": (
                        {
                            "lo": window.recommended_lo,
                            "hi": window.recommended_hi,
                            "mean_in_window": window.mean_in_window,
                            "mean_overall": window.mean_overall,
                            "ratio": window.ratio,
                            "n_in_window": window.n_in_window,
                        }
                        if window is not None
                        else None
                    ),
                    "chamber_interaction": chamber_significant,
                    "chamber_interaction_p": chamber_p,
                    "chamber_interaction_q": chamber_q,
                    "per_chamber_window": (
                        per_chamber_window(train_df, factor.feature, target, entry["config_col"])
                        if chamber_significant
                        else None
                    ),
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
        config_col = f"Step{step}_Config"
        for alarm in alarms_by_feature.get(cr.feature, []):
            row = indexed.loc[alarm.lot_wafer_id] if indexed is not None and alarm.lot_wafer_id in indexed.index else None
            actual_y_final = float(row["Y"]) if row is not None and "Y" in row and pd.notna(row["Y"]) else None
            config_value = (
                str(row[config_col])
                if row is not None and config_col in eval_df.columns and pd.notna(row.get(config_col))
                else None
            )
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
                    "config": config_value,
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

    config_screening = config_main_effect_screening(train_df, schema)
    limitations = [_measurement_rate_limitation(train_df, schema), *LIMITATIONS]

    # 계측 편향 재검토 (spec 문구 전수 검토 §A-7): the old whole-wafer
    # aggregate test ("any R/D reading at all" vs "none") can be
    # non-significant while every individual primary factor is -- exactly
    # what happens on train.CSV, where the aggregate found p=0.74 (no
    # bias) but all 5 primary factors individually show a
    # measured-vs-unmeasured defect-rate gap at q<0.0001. Report the
    # per-factor check instead, since it's the one that actually reflects
    # what "선정 인자" narrows the analysis to.
    factor_bias = per_factor_measurement_bias(train_df, included_factors)
    bias_summary = summarize_measurement_bias(factor_bias)
    if bias_summary is None:
        pass  # too few observations to test either direction -- say nothing rather than guess
    elif bias_summary["significant_count"] == 0:
        limitations.append("계측 대상 선정에 따른 편향은 관측되지 않았다.")
    else:
        direction_word = {"low": "낮게", "high": "높게", "mixed": "다르게"}[bias_summary["direction"]]
        tested, significant = bias_summary["tested_count"], bias_summary["significant_count"]
        scope = f"선정 인자 {significant}개 모두에서" if significant == tested else f"선정 인자 {tested}개 중 {significant}개에서"
        limitations.append(
            "계측 대상이 무작위로 선정되지 않았을 가능성이 있다. "
            f"{scope} 계측된 wafer의 불량률이 미계측 wafer보다 {direction_word} 관측되었다. "
            "따라서 이 분석의 결과를 미계측 wafer로 일반화할 때는 주의가 필요하다."
        )

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
        "config_screening": {
            "n_tested": config_screening.n_tested,
            "n_significant_fdr": config_screening.n_significant_fdr,
            "max_observed_eps2": config_screening.max_observed_eps2,
            "max_observed_feature": config_screening.max_observed_feature,
            "max_observed_target": config_screening.max_observed_target,
            "mde_eps2": config_screening.mde_eps2,
            "median_n_per_group": config_screening.median_n_per_group,
        },
        "limitations": limitations,
    }
    return round_floats(report)


def _lot_range(meta: dict[str, Any]) -> str | None:
    lot_min, lot_max = meta.get("lot_min"), meta.get("lot_max")
    if lot_min is None or lot_max is None:
        return None
    return f"{lot_min}~{lot_max}"


def build_chat_context(report: dict[str, Any]) -> dict[str, Any]:
    """Reshapes build_analysis_report's flat `alarms`/`recommendations`
    lists into `{summary, records[, records_truncated, records_total]}`
    for the SUNI chatbot (spec "챗봇 알람 답변 확장" A-2) -- so a question
    about one specific wafer ("L401W07 알람이 왜 떴어?") can be answered
    from individual records, not just aggregates.

    This is a presentation-layer reshaping on top of the already-built
    report, not a second computation: `/api/analysis/report` (JSON
    download) keeps the original flat shape untouched (existing golden
    tests, existing frontend consumers); only `/api/analysis/context`
    (and the chatbot's own grounding) sees this shape. Every other key
    is passed through unchanged.
    """
    summary = report["summary"]

    # binned_profile is chart-plotting data (12-point x/y profile) that no
    # chatbot prompt section reads -- dropped here (context payload only,
    # the download report keeps it) since it's the single largest
    # per-factor field and directly trades off against the alarm/
    # recommendation record budget below.
    targets_out = [
        {
            **target_entry,
            "factors": [
                {k: v for k, v in factor.items() if k != "binned_profile"} for factor in target_entry["factors"]
            ],
        }
        for target_entry in report["targets"]
    ]

    alarm_records = report["alarms"]
    alarms_truncated = len(alarm_records) > ALARM_RECORD_TRUNCATE_THRESHOLD
    kept_alarms = alarm_records[:ALARM_RECORD_TRUNCATE_KEEP] if alarms_truncated else alarm_records
    alarms_out = {
        "summary": {
            "n_wafers": summary["alarm_wafers"],
            "n_records": len(alarm_records),
            "mean_yield_alarm": summary["mean_yield_alarm"],
            "mean_yield_normal": summary["mean_yield_normal"],
            "normal_wafers": summary["normal_wafers"],
            "undecidable_wafers": summary["undecidable_wafers"],
        },
        "records": [
            {
                "lot_wafer_id": row["lot_wafer_id"],
                "lot_id": row["lot_id"],
                "wafer_slot": row["wafer_slot"],
                "step": row["step"],
                "feature": row["feature"],
                "kind": row["kind"],
                "target": row["target"],
                "value": row["value"],
                "control_band": row["normal_range"],
                "deviation": row["deviation"],
                "direction": row["direction"],
                "severity": row["severity"],
                "actual_y_target": row["actual_y_target"],
                "actual_y_final": row["actual_y_final"],
                "config": row["config"],
            }
            for row in kept_alarms
        ],
        "records_truncated": alarms_truncated,
        "records_total": len(alarm_records),
    }

    # Already excludes 근거부족-equivalent factors (recommendation_records
    # in build_analysis_report is filtered to REPORT_TAG_TIERS = 강함/보통
    # before this function ever sees it) -- only re-sorted here (태그
    # 우선순위 -> 기대 개선 내림차순) for the chatbot's presentation; the
    # download report keeps its own lot_wafer_id order.
    recommendation_records = sorted(
        report["recommendations"],
        key=lambda item: (
            _TAG_PRIORITY_RANK.get(item["tag"], 0),
            item["expected_improvement_pct"] if item["expected_improvement_pct"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    recommendations_truncated = len(recommendation_records) > RECOMMENDATION_RECORD_KEEP
    recommendations_out = {
        "summary": {"n_records": len(recommendation_records)},
        "records": [
            {
                "lot_wafer_id": row["lot_wafer_id"],
                "lot_id": row["lot_id"],
                "step": row["step"],
                "feature": row["feature"],
                "kind": row["kind"],
                "target": row["target"],
                "value": row["value"],
                "recommended_range": row["recommended_range"],
                "direction": row["direction"],
                "expected_reduction_pct": row["expected_improvement_pct"],
                "tag": row["tag"],
            }
            for row in recommendation_records[:RECOMMENDATION_RECORD_KEEP]
        ],
        "records_truncated": recommendations_truncated,
        "records_total": len(recommendation_records),
    }

    context = {**report, "targets": targets_out, "alarms": alarms_out, "recommendations": recommendations_out}

    # `_text` companions (spec "LLM 답변·보고서 서술 다듬기" §7) -- pre-round
    # and pre-phrase every field the chat/report prompts are prone to
    # mis-render on their own (raw booleans, bracket-style ranges,
    # scientific-notation p-values, more decimals than the screen shows).
    # Both `/api/chat` modes read this same function's output (see
    # `_context_user_message` in api/routes/chat.py), so one call here
    # covers both prompts.
    return add_display_text(context)
