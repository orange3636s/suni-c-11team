"""JSON analysis-report serialization: the single function backing both the
root-cause tab's "JSON 보고서 저장" button and (future) `/api/analysis/context`
for the SUNI chatbot / n8n automation. Both consumers must get the exact
same structure, so build it here once rather than assembling it in the route
or reassembling it on the frontend.

Two factor sets are deliberately different and never conflated:
  - `targets[].factors`: one narrative factor per target (the single
    strongest, adj_r2-ranked, from the full R+D+Config pool), included only
    if raw p < 0.05. This is what an LLM should cite as "the" driver for
    that target -- including every FDR-significant factor here would let a
    borderline second factor (e.g. Y2's Step24_R1) crowd the narrative.
  - `alarms` / `summary.alarm_wafers`: a separate, already-verified
    control-limit exceedance engine (BH-FDR q<0.05 factor selection +
    IQR*1.5 control range, see `_alarm_engine_factors`) -- unrelated to the
    notification pipeline. This is the report's own standalone count, not
    echoed from any live route.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.analysis.alarm_gbdt import CONFORMAL_TARGET_COVERAGE
from src.analysis.alarm_gbdt import FINAL_YIELD_COLUMN as GBDT_TARGET_COLUMN
from src.analysis.alarm_gbdt import compute_holdout_predictions
from src.analysis.alarm_gbdt import feature_columns as gbdt_feature_columns
from src.analysis.control_range import (
    ControlRange,
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.display_text import add_display_text
from src.analysis.llm_stats import (
    band_stability,
    chamber_interaction_p,
    config_main_effect_screening,
    judge_confidence,
    per_chamber_window,
    per_factor_measurement_bias,
    summarize_measurement_bias,
)
from src.analysis.recommendations import compute_factor_recommendation
from src.analysis.rounding import round_floats
from src.analysis.screening.quantile_profile import quantile_bins
from src.analysis.screening.schema import Schema, parse_schema
from src.analysis.screening.selector import (
    ParetoFactor,
    _row_to_factor,
    benjamini_hochberg,
    effective_confidence_tier,
    select_fdr_significant_factors,
    select_primary_factor,
)

# `_top_factor_per_target`/`_alarm_engine_factors`가 이 시그니처의
# 콜백을 받으면(챗봇/보고서 라우트가 넘기는 캐시된 랭킹, 아래 참고)
# select_primary_factor/select_fdr_significant_factors의 88-인자 재계산을
# 건너뛰고 이미 나온 순위 행을 재사용한다. 콜백이 없으면(테스트 등 이
# 모듈을 단독으로 호출하는 경로) 기존처럼 직접 계산한다 -- 동작은 항상
# 동일하고, 캐시는 순수 성능 최적화다.
RankedRowsProvider = Callable[[str], list[dict[str, Any]]]

REPORT_INCLUSION_P_THRESHOLD = 0.05
BINNED_PROFILE_BINS = 12

# SUNI chatbot context: safety cap so an
# uploaded dataset with many more alarms than the bundled train.CSV can't
# blow up the LLM context payload.
ALARM_RECORD_TRUNCATE_THRESHOLD = 200
ALARM_RECORD_TRUNCATE_KEEP = 100

# `action` derivation for the SUNI chatbot's "권장 조치" answers (see
# `_build_action` and its use inside `build_chat_context` below). The LLM
# never invents action text -- it only narrates whatever the backend
# already decided to attach. Threshold from the offline validation
# referenced in
# prompts/chat_system.md's "## 검증 성적" section: recommendations below
# this expected-recovery bar had a much lower real-risk hit rate than ones
# at/above it.
ACTION_MIN_EXPECTED_GAIN_PP = 0.5

# NOTE: this report's `targets[].factors` only ever contains a factor that
# already cleared the pipeline's own measurement-sufficiency bar (raw
# p<0.05 on top of the per-kind MIN_N gates in
# screening/selector.py) -- a *low* raw R/D measurement rate is normal here
# (metrology is sampling-based, not full-lot: `n_missing_pct` sits at
# ~85-95% for every one of this report's selected factors on the bundled
# train.CSV, yet each still has n_observed in the thousands). So unlike the
# yield-ranking screen's per-wafer "core factor measured?" flag
# (`src/analysis/yield_prediction.py`'s `CoreFactorCell.measured`, narrated
# in prompts/report_yield.md), there is no meaningful population-level
# "unmeasured" gate to add here -- unmeasured/needs-more-measurement wafers
# are a per-wafer question, not a per-factor one, and this report answers
# per-factor questions.

_ACTION_SCOPE_TEMPLATE = {
    "R": "Step{step} 레시피 설정",
    "D": "Step{step} 계측 · 설비 상태",
    "Config": "Step{step} 장비 구성",
}
_ACTION_CHECKS_BY_KIND = {
    "R": ["설정 상·하한 확인", "직전 배치와 설정 차이 대조"],
    "D": ["계측 결과가 관리 대역 이탈", "해당 스텝 설비 상태 확인", "직전 스텝 산출물 확인"],
    "Config": ["동일 스텝 다른 Config 대비 불량률 격차", "해당 장비 점검 대상"],
}

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
    # 챗봇이 이 캐비어트들의 유일한 안내 경로다 -- 화면에는 대응하는
    # "해석 시 한계" 블록이 없다.
    "'근거 부족'·'관계 없음' 등급 인자는 통계적 신뢰도가 낮아 원인으로 단정할 근거가 부족하다.",
]

# Below this measurement rate, "~만 분석 대상이다" (only the measured
# subset) reads accurately; at/above it, most wafers already have a
# reading, so the softer "~를 분석 대상으로 한다" avoids overstating how
# exclusionary the dataset actually is.
_HIGH_MEASUREMENT_RATE_THRESHOLD = 60.0


def _measurement_rate_limitation(df: pd.DataFrame, schema: Schema) -> str:
    """Real per-dataset R/D measurement rate -- mean of each
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


def _interval_calibration_limitation(train_df: pd.DataFrame, features: list[str]) -> str:
    """예측 구간 캘리브레이션 한계 설명 -- conformal 여유는 판정 기준이
    아니라 참고 통계로만 언급한다(구간 폭이 ±5%p 수준이면 그것만으로
    가부를 가릴 수 있는 wafer가 6%밖에 안 된다). 화면이 실제로 보여주는
    신뢰도 신호는 수율 예측 표의 "신뢰도 n/5"(핵심 인자가 계측된 타깃 수)다.
    하드코딩된 숫자를 쓰지 않고 이 데이터셋에서 실제로 낸 conformal
    margin(q)을 매번 다시 계산한다 -- 다른 데이터셋에서는 폭이 다르므로
    고정 문구를 쓰면 거짓 정보가 된다. `compute_holdout_predictions`가
    이미 GroupKFold(5) GBDT를 한 번 적합해야 하므로(`fit_reference_model`과
    비슷한 비용), 보고서 생성 1회당 한 번만 계산한다.
    """
    coverage_pct = int(round(CONFORMAL_TARGET_COVERAGE * 100))
    reliability_note = "예측 신뢰도는 수율 예측 화면의 '신뢰도 n/5'(핵심 인자 계측 비율)로 표시되며, 웨이퍼 단위 판정을 예측 구간 폭으로 가르지 않는다."
    if not features or GBDT_TARGET_COLUMN not in train_df.columns:
        return (
            f"{reliability_note} 참고로 홀드아웃 잔차 기반 conformal 여유(목표 포함률 {coverage_pct}%)도 "
            "함께 산출하도록 설계되어 있으나, 이 데이터셋에는 예측에 쓸 R+D 인자나 최종 수율(Y)이 없어 "
            "산출하지 못했다."
        )
    holdout = compute_holdout_predictions(train_df, features)
    if holdout is None:
        return (
            f"{reliability_note} 참고로 홀드아웃 잔차 기반 conformal 여유(목표 포함률 {coverage_pct}%)도 "
            "함께 산출하도록 설계되어 있으나, 이 데이터셋은 랏 수가 부족해(GroupKFold 5-fold 구성 불가) "
            "산출하지 못했다 -- 부트스트랩 분위수로 대체되어 포함률이 보장되지 않는다."
        )
    return (
        f"{reliability_note} 참고로 홀드아웃 잔차 기반 conformal 여유는 목표 포함률 {coverage_pct}%에서 "
        f"이 데이터셋 기준 약 ±{holdout.conformal_q:.1f}%p로, 예측값 자체의 불확실성 정도를 보여주는 "
        "참고 수치일 뿐 판정 기준이 아니다."
    )


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
    -- never something to report as-is. `window` is the
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


def _top_factor_per_target(
    train_df: pd.DataFrame,
    schema: Schema,
    target: str,
    ranked_rows_provider: RankedRowsProvider | None = None,
) -> ParetoFactor | None:
    """The single strongest (highest-adj_r2) factor for `target` across the
    full R+D+Config pool, included only if it clears the report's own raw
    p<0.05 bar. This is the report's own narrative-inclusion rule -- a
    different, deliberately non-interchangeable concept from the alarm
    engine's factor set below (see module docstring).
    """
    if ranked_rows_provider is not None:
        rows = ranked_rows_provider(target)
        factor = _row_to_factor(train_df, target, rows[0]) if rows else None
    else:
        factor = select_primary_factor(train_df, schema, target)
    if factor is None or factor.p_value >= REPORT_INCLUSION_P_THRESHOLD:
        return None
    return factor


def _alarm_engine_factors(
    train_df: pd.DataFrame,
    schema: Schema,
    ranked_rows_provider: RankedRowsProvider | None = None,
) -> list[ParetoFactor]:
    """The existing BH-FDR (q<0.05) factor set already used by
    /api/alarms and /api/alarms/predictions -- reused as-is so the report's
    alarm numbers always agree with the live alarm log.
    """
    factors: list[ParetoFactor] = []
    for target in schema.target_cols:
        if ranked_rows_provider is not None:
            rows = ranked_rows_provider(target)
            factors.extend(_row_to_factor(train_df, target, row) for row in rows if row["significant"])
        else:
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
    # target -> 이미 계산된 랭킹 행을 돌려주는 콜백.
    # api/routes/analysis.py의 `_build_report_payload`가 `_cached_ranked_
    # rows_versioned`로 채워 넘긴다 -- 원인 분석 탭이 이미 계산해 둔 것과
    # 완전히 같은 순위(같은 파라미터)이므로, 여기서 select_primary_factor/
    # select_fdr_significant_factors로 88개 인자를 타깃마다 다시 채점하지
    # 않는다(0.64초 x 10회 = 6.4초/요청). 생략하면(예: 테스트가 이 함수를
    # 단독으로 부르는 경우) 기존처럼 직접 계산한다.
    ranked_rows_provider: RankedRowsProvider | None = None,
) -> dict[str, Any]:
    schema = parse_schema(train_df)

    # gbdt_features는 _interval_calibration_limitation의 conformal 캘리브레이션
    # 문구가 쓰는 R+D 인자 풀이다 (compute_holdout_predictions 참고).
    gbdt_features = gbdt_feature_columns(schema)

    included_factors: list[ParetoFactor] = []
    # Two passes: chamber-interaction q-values are BH-corrected across this
    # report's own 5 primary factors (one FDR family), so every factor's
    # p-value must be collected before any q can be assigned.
    raw_targets: list[dict[str, Any]] = []
    chamber_p_values: list[float] = []
    for target in schema.target_cols:
        factor = _top_factor_per_target(train_df, schema, target, ranked_rows_provider)
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
                "window": compute_factor_recommendation(train_df, factor, control_range, dataset_id=train_dataset_id),
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
                    "adj_r2": factor.adj_r2,
                    "degree": factor.degree,
                    "contribution_pct": factor.contribution_pct,
                    "cumulative_pct": factor.cumulative_pct,
                    "p_value": factor.p_value,
                    "q_value": factor.q_value,
                    "grade": {"strong": "강함", "moderate": "보통", "weak": "약함", "reference": "참고"}[
                        effective_confidence_tier(factor.adj_r2, factor.p_value, under_sampled=factor.under_sampled)
                    ],
                    "report_confidence": judge_confidence(
                        factor.adj_r2, factor.p_value, entry["band_stability"], entry["band_width"]
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

    alarm_factors = _alarm_engine_factors(train_df, schema, ranked_rows_provider)
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

    config_screening = config_main_effect_screening(train_df, schema)
    limitations = [
        _measurement_rate_limitation(train_df, schema),
        *LIMITATIONS,
        _interval_calibration_limitation(train_df, gbdt_features),
    ]

    # 계측 편향은 인자별로 잰다. A whole-wafer aggregate test ("any R/D
    # reading at all" vs "none") can be non-significant while every
    # individual primary factor is -- exactly what happens on train.CSV,
    # where the aggregate finds p=0.74 (no bias) but all 5 primary factors
    # individually show a measured-vs-unmeasured defect-rate gap at
    # q<0.0001. The per-factor check is the one that actually reflects what
    # "선정 인자" narrows the analysis to.
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
            "screening": "Adjusted R² (수치형: 1/2차 다항 적합, 범주형: 더미회귀) + BH-FDR",
            "contribution_denominator": "전체 인자 풀(R+D+Config)의 Adjusted R² 합",
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
        "config_screening": {
            "n_tested": config_screening.n_tested,
            "n_significant_fdr": config_screening.n_significant_fdr,
            "max_observed_adj_r2": config_screening.max_observed_adj_r2,
            "max_observed_feature": config_screening.max_observed_feature,
            "max_observed_target": config_screening.max_observed_target,
            "mde_adj_r2": config_screening.mde_adj_r2,
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


def _action_relation_note(
    relation_shape: str | None, optimal_center: float | None, window: dict[str, Any] | None
) -> str | None:
    """Which direction to watch, derived purely from the already-computed
    relation shape and recommended window -- never a new statistical
    judgment, just wording chosen by a lookup on fields report.py already
    produces."""
    if window is None:
        return None
    lo, hi = window["lo"], window["hi"]
    if relation_shape == "u_shape":
        if optimal_center is None:
            return f"양방향 이탈 모두 위험. {lo}~{hi} 복귀 대상"
        return f"양방향 이탈 모두 위험. 최적중심 {optimal_center} 기준 {lo}~{hi} 복귀 대상"
    if relation_shape == "monotonic_decreasing":
        return f"한쪽 방향만 위험. {lo} 이상 유지 대상"
    return f"한쪽 방향만 위험. {hi} 이하 유지 대상"


def _build_action(
    *,
    kind: str,
    step: int,
    relation_shape: str | None,
    optimal_center: float | None,
    window: dict[str, Any] | None,
    expected_gain_pp: float | None,
    value: float | None,
    severity: str | None,
) -> dict[str, Any] | None:
    """Derives the `action` object chat_system.md's "## 권장 조치를 물었을
    때" section narrates -- the LLM is never asked to invent a setpoint
    instruction, only to phrase whatever this function already decided.

    A value already inside the recommended window, or an expected recovery
    below the validated bar, drops the action entirely (return None) rather
    than manufacturing a low-value recommendation.
    """
    if window is not None and value is not None and window["lo"] <= value <= window["hi"]:
        return None
    if expected_gain_pp is None or expected_gain_pp < ACTION_MIN_EXPECTED_GAIN_PP:
        return None

    scope = _ACTION_SCOPE_TEMPLATE.get(kind, "Step{step} 확인").format(step=step)
    checks = list(_ACTION_CHECKS_BY_KIND.get(kind, ["확인 대상"]))
    note = _action_relation_note(relation_shape, optimal_center, window)
    if note:
        checks.append(note)

    action: dict[str, Any] = {"scope": scope, "checks": checks, "expected_gain_pp": expected_gain_pp}
    if severity is not None:
        action["urgency"] = severity
    return action


def build_chat_context(report: dict[str, Any], action_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Reshapes build_analysis_report's flat `alarms`/`recommendations`
    lists into `{summary, records[, records_truncated, records_total]}`
    for the SUNI chatbot -- so a question
    about one specific wafer ("L401W07 알람이 왜 떴어?") can be answered
    from individual records, not just aggregates.

    This is a presentation-layer reshaping on top of the already-built
    report, not a second computation: `/api/analysis/report` (JSON
    download) keeps the original flat shape untouched (existing golden
    tests, existing frontend consumers); only `/api/analysis/context`
    (and the chatbot's own grounding) sees this shape. Every other key
    is passed through unchanged.

    `action_rows` is the monitoring home's action-priority table rows
    (`api.routes.analysis._action_priority_payload("train")["rows"]`),
    threaded in by the caller so this module never imports the api layer.
    When given, every alarm record and every target's top factor gets an
    `action` field attached (see `_build_action`) -- omitted (`None`) is
    passed by callers (e.g. existing tests) that don't need `action`.
    """
    summary = report["summary"]

    # (target, feature) -> expected_recovery_pp, from the monitoring home's
    # own action-priority table (src/analysis/action_priority.py) -- reused
    # as-is (see ACTION_MIN_EXPECTED_GAIN_PP's docstring), never
    # recomputed here.
    gain_by_target_feature = {
        (row["target"], row["feature"]): row.get("expected_recovery_pp") for row in (action_rows or [])
    }

    # binned_profile is chart-plotting data (12-point x/y profile) that no
    # chatbot prompt section reads -- dropped here (context payload only,
    # the download report keeps it) since it's the single largest
    # per-factor field and directly trades off against the alarm/
    # recommendation record budget below.
    targets_out = []
    for target_entry in report["targets"]:
        target = target_entry["target"]
        factors_out = []
        for factor in target_entry["factors"]:
            factor_out = {k: v for k, v in factor.items() if k != "binned_profile"}
            action = _build_action(
                kind=factor["kind"],
                step=factor["step"],
                relation_shape=factor["relation"]["shape"],
                optimal_center=factor["relation"].get("optimal_center"),
                window=factor["window"],
                expected_gain_pp=gain_by_target_feature.get((target, factor["feature"])),
                value=None,
                severity=None,
            )
            if action is not None:
                factor_out["action"] = action
            factors_out.append(factor_out)
        targets_out.append({**target_entry, "factors": factors_out})

    # (target, feature) -> that target's top factor entry, so an alarm
    # record (which only carries feature/kind/step/value) can look up the
    # relation shape and recommended window it needs for its own `action`.
    factor_by_target_feature = {
        (t["target"], f["feature"]): f for t in targets_out for f in t["factors"]
    }

    alarm_records = report["alarms"]
    alarms_truncated = len(alarm_records) > ALARM_RECORD_TRUNCATE_THRESHOLD
    kept_alarms = alarm_records[:ALARM_RECORD_TRUNCATE_KEEP] if alarms_truncated else alarm_records
    records_out = []
    for row in kept_alarms:
        match = factor_by_target_feature.get((row["target"], row["feature"]))
        action = _build_action(
            kind=row["kind"],
            step=row["step"],
            relation_shape=match["relation"]["shape"] if match else None,
            optimal_center=match["relation"].get("optimal_center") if match else None,
            window=match["window"] if match else None,
            expected_gain_pp=gain_by_target_feature.get((row["target"], row["feature"])),
            value=row["value"],
            severity=row["severity"],
        )
        record_out = {
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
        if action is not None:
            record_out["action"] = action
        records_out.append(record_out)

    alarms_out = {
        "summary": {
            "n_wafers": summary["alarm_wafers"],
            "n_records": len(alarm_records),
            "mean_yield_alarm": summary["mean_yield_alarm"],
            "mean_yield_normal": summary["mean_yield_normal"],
            "normal_wafers": summary["normal_wafers"],
            "undecidable_wafers": summary["undecidable_wafers"],
        },
        "records": records_out,
        "records_truncated": alarms_truncated,
        "records_total": len(alarm_records),
    }

    context = {**report, "targets": targets_out, "alarms": alarms_out}

    # `_text` companions -- pre-round
    # and pre-phrase every field the chat/report prompts are prone to
    # mis-render on their own (raw booleans, bracket-style ranges,
    # scientific-notation p-values, more decimals than the screen shows).
    # Both `/api/chat` modes read this same function's output (see
    # `_context_user_message` in api/routes/chat.py), so one call here
    # covers both prompts.
    return add_display_text(context)
