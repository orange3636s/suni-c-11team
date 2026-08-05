"""Three-way band split for the 사전 알람 로그 summary: every measured wafer
is exactly one of 관리한계 이탈 (alarm) / 권장구간 밖 / 권장구간 내, and every
selected factor's own control+recommended window can be read back on its
real value axis. Built on top of the existing alarm engine
(control_range.py) and recommendation engine (recommendations.py) rather
than recomputing anything -- a wafer already flagged "alarm" by the
BH-FDR alarm-eligible factor set never gets re-classified here, and the
227/265 split below reuses that exact same factor set's recommended
window (recommendations.compute_factor_recommendation), so the numbers
this module reports never disagree with /api/alarms or /api/recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.control_range import ControlRange
from src.analysis.recommendations import compute_factor_recommendation
from src.analysis.screening.selector import ParetoFactor

ID_COLUMN = "Lot_Wafer_ID"
FINAL_YIELD_COLUMN = "Y"


@dataclass
class BandStat:
    count: int
    mean_yield: float | None


@dataclass
class WholeWaferBands:
    out_of_recommended_ids: set[str]
    in_recommended_ids: set[str]
    alarm: BandStat
    out_of_recommended: BandStat
    in_recommended: BandStat
    unmeasured: BandStat


def classify_measured_bands(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    alarm_ids: list[str],
    normal_ids: list[str],
    unmeasured_ids: list[str],
    alarm_factors: list[ParetoFactor],
    control_ranges: list[ControlRange],
    *,
    id_column: str = ID_COLUMN,
    yield_column: str = FINAL_YIELD_COLUMN,
) -> WholeWaferBands:
    """Splits `normal_ids` (measured, not alarmed) into 권장구간 밖 / 권장구간
    내: a wafer is "밖" if it falls outside the recommended window of ANY
    alarm-eligible factor it has a reading for, "내" otherwise. The
    recommended window is the same one /api/recommendations already
    computes per factor (train-derived, clamped into the control range).
    """
    normal_id_set = set(normal_ids)
    id_series = eval_df[id_column].astype(str) if id_column in eval_df.columns else pd.Series(eval_df.index.astype(str))
    normal_mask_base = id_series.isin(normal_id_set)

    out_of_recommended_ids: set[str] = set()
    for factor, control_range in zip(alarm_factors, control_ranges):
        if factor.feature not in eval_df.columns:
            continue
        recommendation = compute_factor_recommendation(train_df, factor, control_range)
        if recommendation is None:
            continue
        x = pd.to_numeric(eval_df[factor.feature], errors="coerce")
        flagged = normal_mask_base & x.notna() & ~x.between(recommendation.recommended_lo, recommendation.recommended_hi)
        out_of_recommended_ids.update(id_series[flagged].tolist())
    in_recommended_ids = normal_id_set - out_of_recommended_ids

    y = (
        eval_df.set_index(id_series)[yield_column]
        if yield_column in eval_df.columns
        else None
    )

    def _mean_yield(ids: list[str] | set[str]) -> float | None:
        if y is None or not ids:
            return None
        values = pd.to_numeric(y.reindex(list(ids)), errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    return WholeWaferBands(
        out_of_recommended_ids=out_of_recommended_ids,
        in_recommended_ids=in_recommended_ids,
        alarm=BandStat(len(alarm_ids), _mean_yield(alarm_ids)),
        out_of_recommended=BandStat(len(out_of_recommended_ids), _mean_yield(out_of_recommended_ids)),
        in_recommended=BandStat(len(in_recommended_ids), _mean_yield(in_recommended_ids)),
        unmeasured=BandStat(len(unmeasured_ids), _mean_yield(unmeasured_ids)),
    )


@dataclass
class FactorBandPoint:
    count: int
    mean_defect_rate: float | None


@dataclass
class FactorBand:
    feature: str
    target: str
    kind: str
    x_min: float
    x_max: float
    lcl: float | None
    ucl: float | None
    recommended_lo: float | None
    recommended_hi: float | None
    out_of_control: FactorBandPoint
    out_of_recommended: FactorBandPoint
    in_recommended: FactorBandPoint


def compute_factor_band(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    factor: ParetoFactor,
    control_range: ControlRange,
) -> FactorBand | None:
    """Per-factor 3-band defect-rate breakdown on the factor's own real
    value axis -- the data behind 카드② (인자별 불량률). Unlike
    `classify_measured_bands` (whole-wafer, final yield), this reports
    the factor's *own target*'s defect rate, since a wafer can be
    "권장 구간 내" on one factor and "밖" on another -- there is no single
    per-wafer verdict to reuse here, only a per-factor one.
    """
    if factor.feature not in eval_df.columns or factor.target not in eval_df.columns:
        return None
    x = pd.to_numeric(eval_df[factor.feature], errors="coerce")
    y = pd.to_numeric(eval_df[factor.target], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    x, y = x[valid], y[valid]

    lower_bound = control_range.lower if control_range.lower is not None else float("-inf")
    upper_bound = control_range.upper if control_range.upper is not None else float("inf")
    out_of_control_mask = (x < lower_bound) | (x > upper_bound)
    in_control_mask = ~out_of_control_mask

    recommendation = compute_factor_recommendation(train_df, factor, control_range)
    if recommendation is not None:
        in_recommended_mask = in_control_mask & x.between(recommendation.recommended_lo, recommendation.recommended_hi)
        out_of_recommended_mask = in_control_mask & ~in_recommended_mask
    else:
        in_recommended_mask = in_control_mask
        out_of_recommended_mask = pd.Series(False, index=x.index)

    def _point(mask: pd.Series) -> FactorBandPoint:
        values = y[mask]
        return FactorBandPoint(count=int(mask.sum()), mean_defect_rate=float(values.mean()) if len(values) else None)

    return FactorBand(
        feature=factor.feature,
        target=factor.target,
        kind=factor.kind,
        x_min=float(x.min()),
        x_max=float(x.max()),
        lcl=control_range.lower,
        ucl=control_range.upper,
        recommended_lo=recommendation.recommended_lo if recommendation is not None else None,
        recommended_hi=recommendation.recommended_hi if recommendation is not None else None,
        out_of_control=_point(out_of_control_mask),
        out_of_recommended=_point(out_of_recommended_mask),
        in_recommended=_point(in_recommended_mask),
    )
