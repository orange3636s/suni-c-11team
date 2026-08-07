"""인자별 권장 구간(recommended range) 계산 -- '개선 권장 목록'(per-wafer
list)은 정밀도가 무작위 수준과 다르지 않아 삭제됐다 (spec 알람 신뢰도
게이트 §B-1: train→problem 정밀도 5%). `compute_factor_recommendation`은
여전히 남아있는 두 소비처가 쓴다:

  - `src/analysis/alarm_bands.py`의 `classify_measured_bands`/
    `compute_factor_band` -- 사전 알람 로그의 "구간별 평균 수율" 카드이
    "채택된 권장구간"(SPC 또는 ML) 밖/안을 가르는 데 이 창을 그대로 쓴다
    (spec 알람 신뢰도 게이트 §C-1).
  - `src/analysis/report.py` -- JSON 보고서의 인자별 window 필드.

recommended range 자체는 train 기준(bin-profile threshold, see
`_recommended_range_raw`)으로 산출되고 control range 안쪽으로 clamp된다
(spec §5-3) -- 값을 자신의 알람 경계 밖으로 밀어내는 권장은 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.control_range import ControlRange, compute_control_range
from src.analysis.screening.quantile_profile import quantile_bins, window_from_bins
from src.analysis.screening.selector import ParetoFactor, confidence_tier
from src.analysis.window_methods import MethodComparison, compare_methods

GRADE_TAG = {
    "strong": "priority",
    "moderate": "recommended",
    "weak": "reference",
    "reference": "reference",
}


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
