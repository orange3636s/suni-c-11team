"""인자별 권장 구간(recommended range) 계산. `compute_factor_recommendation`의
소비처:

  - `src/analysis/report.py` -- JSON 보고서의 인자별 window 필드.
  - `src/analysis/scatter.py` -- 산점도의 권장 구간 표시.
  - `src/analysis/screening/fmea.py` -- FMEA 표의 권장 구간 열.
  - `src/analysis/yield_prediction.py` -- 수율 예측 권장사항의 구간 조정 제안.

recommended range 자체는 train 기준(bin-profile threshold, see
`_recommended_range_raw`)으로 산출되고 control range 안쪽으로 clamp된다 --
값을 자신의 알람 경계 밖으로 밀어내는 권장은 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.control_range import ControlRange, compute_control_range
from src.analysis.screening.quantile_profile import DEFAULT_BINS, quantile_bins, window_from_bins
from src.analysis.screening.selector import ParetoFactor, effective_confidence_tier
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
    `compute_factor_recommendation` below goes through
    `window_methods.compare_methods` (so ML gets a fair shot too) rather
    than calling this; this stays as the standalone SPC-only primitive
    that llm_stats.py's per-chamber breakdown needs.
    """
    # 권장 구간(SPC 쪽)은 자동 구간수를 쓰지 않고 12로 고정한다 -- 구간
    # 수가 바뀌면 최적 중심·권장 구간이 달라지고 이 window를 그대로 쓰는
    # 소비처(report.py/scatter.py/fmea.py/yield_prediction.py)의 판정이
    # 조용히 변한다. Sturges 자동 구간수는 히트맵/Pareto의 Adjusted R²
    # 계산(effect_size.py)에만 적용한다.
    bins = quantile_bins(x, y, bins=DEFAULT_BINS)
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
    *,
    dataset_id: str,
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
        cache_key=(dataset_id, factor.feature, factor.target),
    )
    winner = methods.ml if methods.adopted == "ml" else methods.spc
    if winner is None:
        # Both methods' windows disappeared under clamping (or couldn't be
        # fit at all) -- no recommendation for this factor.
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

    grade = effective_confidence_tier(factor.adj_r2, factor.p_value, under_sampled=factor.under_sampled)
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
