"""'개선 권장 목록' -- wafers outside a factor's *recommended* range
(spec §3), as distinct from '알람 목록' (wafers outside its *control
limit*, control_range.py). Same train-derives-the-boundary /
eval-gets-judged pattern as alarms, plus:

  - the recommended range is itself derived from train (bin-profile
    threshold, see `_recommended_range_raw`) and clamped into the
    control range so a recommendation never asks someone to push a
    value past its own alarm boundary (spec §5-3)
  - "이미 알람에 잡힌 건" (same factor+target+wafer already flagged by
    control_range.evaluate_alarms) is excluded so the same deviation
    doesn't show up in both lists
  - the expected-improvement percentage is a property of the factor
    (train-derived: overall train mean vs. the train mean *within* the
    clamped range), not recomputed per evaluation run -- every
    recommendation row for a given factor carries the same figure
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.control_range import ControlRange, WaferAlarm, compute_control_range, evaluate_alarms
from src.analysis.screening.quantile_profile import quantile_bins, window_from_bins
from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import ParetoFactor, confidence_tier
from src.analysis.window_methods import MethodComparison, compare_methods

GRADE_TAG = {
    "strong": "priority",
    "moderate": "recommended",
    "weak": "reference",
    "reference": "reference",
}
TAG_LABEL = {"priority": "우선 권장", "recommended": "권장", "reference": "참고"}
# Only 강함/보통 feed the JSON report's `recommendations` array (spec §3-6,
# same rule alarms already apply elsewhere: 약함/참고 aren't confident
# enough to print as an actionable finding).
REPORT_TAG_TIERS = ("strong", "moderate")


def _recommended_range_raw(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    """The x-quantile span of the contiguous run of 12 quantile bins (the
    same profile the 구간 평균 불량률 curve is built from) whose average y
    sits at/below the factor's overall (train) mean y -- the SPC method's
    window on its own, without the F2/stability scoring machinery.
    `compute_factor_recommendation` below no longer calls this directly
    (it goes through `window_methods.compare_methods` so ML gets a fair
    shot too); kept here as the standalone SPC-only primitive that
    llm_stats.py's per-chamber breakdown still needs.
    """
    bins = quantile_bins(x, y)
    if not bins:
        return None
    return window_from_bins(bins, x, float(y.mean()))


@dataclass
class FactorRecommendation:
    """Per-factor summary: the recommended range (already clamped) and
    the expected-improvement figure every one of its wafer rows shares.
    """

    factor: ParetoFactor
    target: str
    recommended_lo: float
    recommended_hi: float
    clamped: bool
    expected_improvement_pct: float | None
    grade: str
    tag: str
    mean_in_window: float | None
    mean_overall: float
    ratio: float | None
    n_in_window: int
    methods: MethodComparison


@dataclass
class WaferRecommendation:
    lot_wafer_id: str
    lot_id: str | None
    step: int
    feature: str
    kind: str
    target: str
    value: float
    recommended_lo: float
    recommended_hi: float
    direction: str  # "up" | "down"
    expected_improvement_pct: float | None
    tag: str


def compute_factor_recommendation(
    train_df: pd.DataFrame,
    factor: ParetoFactor,
    control_range: ControlRange,
) -> FactorRecommendation | None:
    """None when the factor has no usable train-side x/y pair (e.g. a
    Config/categorical factor -- 권장 구간 only applies to numeric
    factors, same as the chart's own band)."""
    if factor.kind == "Config" or factor.feature not in train_df.columns:
        return None
    x = pd.to_numeric(train_df[factor.feature], errors="coerce")
    y = pd.to_numeric(train_df[factor.target], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    x, y = x[valid], y[valid]

    methods = compare_methods(
        x, y, control_range.lower, control_range.upper,
        cache_key=(id(train_df), factor.feature, factor.target),
    )
    winner = methods.ml if methods.adopted == "ml" else methods.spc
    if winner is None:
        # Spec §5-3: both methods' windows disappeared under clamping (or
        # couldn't be fit at all) -- no recommendation for this factor.
        return None
    clamped_lo, clamped_hi = winner.lo, winner.hi
    clamped = winner.clamped

    overall_mean = float(y.mean())
    in_range_mask = (x >= clamped_lo) & (x <= clamped_hi)
    in_range_mean = float(y[in_range_mask].mean()) if in_range_mask.any() else None
    expected_improvement_pct = (
        (overall_mean - in_range_mean) / overall_mean * 100.0
        if in_range_mean is not None and overall_mean != 0
        else None
    )
    ratio = in_range_mean / overall_mean if in_range_mean is not None and overall_mean != 0 else None

    grade = confidence_tier(factor.eps2, factor.p_value)
    return FactorRecommendation(
        factor=factor,
        target=factor.target,
        recommended_lo=clamped_lo,
        recommended_hi=clamped_hi,
        clamped=clamped,
        expected_improvement_pct=expected_improvement_pct,
        grade=grade,
        tag=GRADE_TAG[grade],
        mean_in_window=in_range_mean,
        mean_overall=overall_mean,
        ratio=ratio,
        n_in_window=int(in_range_mask.sum()),
        methods=methods,
    )


def compute_recommendations(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    schema: Schema,
    *,
    primary_factors: dict[str, ParetoFactor],
) -> tuple[list[WaferRecommendation], dict[str, FactorRecommendation]]:
    """`primary_factors` is the caller's target -> select_primary_factor()
    result (the same "1위 인자" already shown everywhere else) -- the
    recommendation list is scoped to exactly those factors, not the full
    R+D+Config pool, so it stays aligned with what the training/root-cause
    tabs call "the" factor for a target.
    """
    factor_summaries: dict[str, FactorRecommendation] = {}
    rows: list[WaferRecommendation] = []

    for target, factor in primary_factors.items():
        control_range = compute_control_range(train_df, factor)
        summary = compute_factor_recommendation(train_df, factor, control_range)
        if summary is None:
            continue
        factor_summaries[target] = summary

        already_alarmed: set[str] = {
            alarm.lot_wafer_id for alarm in evaluate_alarms(eval_df, control_range)
        }

        if factor.feature not in eval_df.columns:
            continue
        ex = pd.to_numeric(eval_df[factor.feature], errors="coerce")
        id_col = eval_df["Lot_Wafer_ID"] if "Lot_Wafer_ID" in eval_df.columns else None
        lot_col = eval_df["Lot_ID"] if "Lot_ID" in eval_df.columns else None
        y_col = eval_df[target] if target in eval_df.columns else None

        for position, value in ex.items():
            if pd.isna(value):
                continue
            if summary.recommended_lo <= value <= summary.recommended_hi:
                continue
            wafer_id = str(id_col.loc[position]) if id_col is not None else str(position)
            if wafer_id in already_alarmed:
                continue
            rows.append(
                WaferRecommendation(
                    lot_wafer_id=wafer_id,
                    lot_id=(str(lot_col.loc[position]) if lot_col is not None and pd.notna(lot_col.loc[position]) else None),
                    step=factor.step,
                    feature=factor.feature,
                    kind=factor.kind,
                    target=target,
                    value=float(value),
                    recommended_lo=summary.recommended_lo,
                    recommended_hi=summary.recommended_hi,
                    direction="down" if value > summary.recommended_hi else "up",
                    expected_improvement_pct=summary.expected_improvement_pct,
                    tag=summary.tag,
                )
            )

    return rows, factor_summaries
