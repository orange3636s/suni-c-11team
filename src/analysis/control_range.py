"""Per-factor SPC control limits derived from X's own distribution (never
from Y), and alarms raised on eval data.

Earlier approaches derived the "normal range" from Y (take wafers whose
target sits in Q1..Q3, read off the X range within that subset) -- that's
P(X|Y) when what's actually needed is P(Y|X). The same X value shows up in
both low-fail and high-fail wafers, so cutting on Y and reading back X
just returns most of X's own observed range regardless of Y.

This module computes classic SPC reference lines directly from the
observed X values: mean, std, Q1/Q3, IQR*1.5 control limits (LCL/UCL --
the ones actually used for alarms), and +-3sigma / +-6sigma (reference
only, never used for alarm judgement). IQR*1.5 beats +-3sigma here because
sigma is sensitive to outliers inflating it, while IQR is quantile-based
and robust; using raw Q1-Q3 as the alarm boundary is unusable since by
definition half the data sits outside its own interquartile range. See
the module's golden test for the empirical comparison (IQR*1.5: 19 alarm
wafers / -6.14pp yield gap vs +-3sigma: 15 / -6.05pp vs raw Q1-Q3:
279 alarms / -2.39pp).

Monotonic factors (`monotonic_increasing`/`monotonic_decreasing`) only
alarm on the side that means "worse" -- a wafer with an unusually *low*
defect count is never the problem. The "good" side's line is still
computed (for reference-line display) but flagged `alarm_relevant=False`
so callers de-emphasize it instead of dropping it.

A line whose value falls outside the observed [min, max] of the factor is
marked non-drawable -- axes must never be stretched to fit a
diagnostic-only reference value like +-6sigma that lies off the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from src.analysis.screening.selector import ParetoFactor

SEVERITY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "alarm_severity.yaml"
DEFAULT_SEVERITY_THRESHOLDS = {"low_max_ratio": 0.5, "medium_max_ratio": 1.5}


@lru_cache(maxsize=1)
def _severity_thresholds() -> dict[str, float]:
    try:
        loaded = yaml.safe_load(SEVERITY_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except OSError:
        return dict(DEFAULT_SEVERITY_THRESHOLDS)
    return {**DEFAULT_SEVERITY_THRESHOLDS, **loaded}


def classify_severity(deviation: float, band_width: float) -> str:
    if band_width <= 0:
        return "high"
    ratio = abs(deviation) / band_width
    thresholds = _severity_thresholds()
    if ratio <= thresholds["low_max_ratio"]:
        return "low"
    if ratio <= thresholds["medium_max_ratio"]:
        return "medium"
    return "high"


@dataclass
class ReferenceLine:
    key: str  # "mean" | "q1" | "q3" | "iqr_lo" | "iqr_hi" | "s3_lo" | "s3_hi" | "s6_lo" | "s6_hi"
    value: float
    drawable: bool        # False if outside the factor's observed [min, max] -- never stretch the axis to fit it
    alarm_relevant: bool  # False for the "good" side of a one-sided (monotonic) factor
    formula: str          # human-readable derivation, e.g. "Q3(62.5) + 1.5 x IQR(4.3)"


@dataclass
class ControlRange:
    feature: str
    target: str
    kind: str
    relation_shape: str
    mean: float
    std: float
    q1: float
    q3: float
    lower: float | None  # IQR*1.5 LCL used for alarms; None means unbounded below (one-sided)
    upper: float | None  # IQR*1.5 UCL used for alarms; None means unbounded above (one-sided)
    one_sided: bool
    fallback_applied: bool  # always False now -- kept so the API/frontend contract doesn't need to change shape
    band_width: float       # iqr_hi - iqr_lo, used only to normalize alarm severity
    n_observed: int
    reference_lines: list[ReferenceLine] = field(default_factory=list)

    def contains(self, value: float) -> bool:
        if pd.isna(value):
            return False
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


def compute_control_range(train_df: pd.DataFrame, factor: ParetoFactor) -> ControlRange:
    """Derive factor's SPC control limits from its own observed distribution.

    `monotonic_increasing` alarms only on the upper (IQR UCL) side --
    unusually low values are never the problem. `monotonic_decreasing`
    mirrors this for the lower side. Both-sided (u_shape/unclear) factors
    alarm on either side.
    """
    feature = factor.feature
    target = factor.target
    x = pd.to_numeric(train_df[feature], errors="coerce").dropna()
    n_observed = len(x)

    mean = float(x.mean())
    std = float(x.std())
    q1_val, q3_val = (float(v) for v in x.quantile([0.25, 0.75]))
    iqr = q3_val - q1_val
    iqr_lo, iqr_hi = q1_val - 1.5 * iqr, q3_val + 1.5 * iqr
    s3_lo, s3_hi = mean - 3 * std, mean + 3 * std
    s6_lo, s6_hi = mean - 6 * std, mean + 6 * std

    shape = factor.relation_shape
    one_sided = shape in ("monotonic_increasing", "monotonic_decreasing")
    if shape == "monotonic_increasing":
        lower, upper = None, iqr_hi
    elif shape == "monotonic_decreasing":
        lower, upper = iqr_lo, None
    else:
        lower, upper = iqr_lo, iqr_hi

    x_min, x_max = float(x.min()), float(x.max())

    def _drawable(value: float) -> bool:
        return x_min <= value <= x_max

    def _alarm_relevant(is_lower_side: bool) -> bool:
        if shape == "monotonic_increasing":
            return not is_lower_side
        if shape == "monotonic_decreasing":
            return is_lower_side
        return True

    reference_lines = [
        ReferenceLine(key="mean", value=mean, drawable=_drawable(mean), alarm_relevant=False, formula="관측값 평균"),
        ReferenceLine(key="q1", value=q1_val, drawable=_drawable(q1_val), alarm_relevant=False, formula="하위 25% 지점 (1사분위수)"),
        ReferenceLine(key="q3", value=q3_val, drawable=_drawable(q3_val), alarm_relevant=False, formula="상위 25% 지점 (3사분위수)"),
        ReferenceLine(
            key="iqr_lo", value=iqr_lo, drawable=_drawable(iqr_lo), alarm_relevant=_alarm_relevant(True),
            formula=f"Q1({q1_val:.1f}) - 1.5 x IQR({iqr:.1f})",
        ),
        ReferenceLine(
            key="iqr_hi", value=iqr_hi, drawable=_drawable(iqr_hi), alarm_relevant=_alarm_relevant(False),
            formula=f"Q3({q3_val:.1f}) + 1.5 x IQR({iqr:.1f})",
        ),
        ReferenceLine(
            key="s3_lo", value=s3_lo, drawable=_drawable(s3_lo), alarm_relevant=False,
            formula=f"평균({mean:.1f}) - 3 x σ({std:.2f})",
        ),
        ReferenceLine(
            key="s3_hi", value=s3_hi, drawable=_drawable(s3_hi), alarm_relevant=False,
            formula=f"평균({mean:.1f}) + 3 x σ({std:.2f})",
        ),
        ReferenceLine(
            key="s6_lo", value=s6_lo, drawable=_drawable(s6_lo), alarm_relevant=False,
            formula=f"평균({mean:.1f}) - 6 x σ({std:.2f})",
        ),
        ReferenceLine(
            key="s6_hi", value=s6_hi, drawable=_drawable(s6_hi), alarm_relevant=False,
            formula=f"평균({mean:.1f}) + 6 x σ({std:.2f})",
        ),
    ]

    return ControlRange(
        feature=feature,
        target=target,
        kind=factor.kind,
        relation_shape=shape,
        mean=mean,
        std=std,
        q1=q1_val,
        q3=q3_val,
        lower=lower,
        upper=upper,
        one_sided=one_sided,
        fallback_applied=False,
        band_width=iqr_hi - iqr_lo,
        n_observed=n_observed,
        reference_lines=reference_lines,
    )


@dataclass
class WaferAlarm:
    lot_wafer_id: str
    lot_id: str | None
    wafer_slot: int | None
    feature: str
    kind: str
    target: str
    value: float
    lower: float | None
    upper: float | None
    deviation: float
    direction: str  # "above" | "below"
    severity: str  # "low" | "medium" | "high"
    actual_y: float | None


def evaluate_alarms(
    eval_df: pd.DataFrame,
    control_range: ControlRange,
    *,
    id_column: str = "Lot_Wafer_ID",
    lot_column: str = "Lot_ID",
    slot_column: str = "Wafer_Slot",
) -> list[WaferAlarm]:
    """Flag out-of-range wafers against the IQR*1.5 control limits.
    `actual_y` reports the control range's own target column (e.g. Y2's
    fail rate for a Y2-derived alarm) -- not the overall final yield,
    which belongs at the alarm-summary level instead.
    """
    alarms: list[WaferAlarm] = []
    if control_range.feature not in eval_df.columns:
        # eval 데이터셋에 train에서 선정된 인자 컬럼이 아예 없는 조합(예:
        # 컬럼 구성이 다른 업로드 데이터셋을 eval로 쓰는 경우)에서도 그
        # 인자는 "판정 불가"로 취급될 뿐 오류가 나면 안 된다.
        return alarms
    values = pd.to_numeric(eval_df[control_range.feature], errors="coerce")
    for position, value in values.items():
        if pd.isna(value):
            continue
        if control_range.contains(float(value)):
            continue
        row = eval_df.loc[position]
        if control_range.upper is not None and value > control_range.upper:
            direction = "above"
            deviation = float(value) - control_range.upper
        else:
            direction = "below"
            deviation = control_range.lower - float(value) if control_range.lower is not None else 0.0
        actual_y = row.get(control_range.target)
        alarms.append(
            WaferAlarm(
                lot_wafer_id=str(row.get(id_column)),
                lot_id=(str(row[lot_column]) if lot_column in eval_df.columns and pd.notna(row.get(lot_column)) else None),
                wafer_slot=(int(row[slot_column]) if slot_column in eval_df.columns and pd.notna(row.get(slot_column)) else None),
                feature=control_range.feature,
                kind=control_range.kind,
                target=control_range.target,
                value=float(value),
                lower=control_range.lower,
                upper=control_range.upper,
                deviation=float(deviation),
                direction=direction,
                severity=classify_severity(deviation, control_range.band_width),
                actual_y=(float(actual_y) if actual_y is not None and pd.notna(actual_y) else None),
            )
        )
    return alarms


@dataclass
class WaferVerdict:
    lot_wafer_id: str
    lot_id: str | None
    status: str  # "alarm" | "normal" | "unmeasured"
    alarm_count: int
    measured_factor_count: int


def summarize_wafer_status(
    eval_df: pd.DataFrame,
    control_ranges: list[ControlRange],
    alarms_by_feature: dict[str, list[WaferAlarm]],
    *,
    id_column: str = "Lot_Wafer_ID",
    lot_column: str = "Lot_ID",
) -> list[WaferVerdict]:
    alarmed_ids: dict[str, int] = {}
    for feature_alarms in alarms_by_feature.values():
        for alarm in feature_alarms:
            alarmed_ids[alarm.lot_wafer_id] = alarmed_ids.get(alarm.lot_wafer_id, 0) + 1

    verdicts: list[WaferVerdict] = []
    feature_columns = [cr.feature for cr in control_ranges]
    for _, row in eval_df.iterrows():
        wafer_id = str(row.get(id_column))
        measured = sum(1 for feature in feature_columns if pd.notna(pd.to_numeric(row.get(feature), errors="coerce")))
        alarm_count = alarmed_ids.get(wafer_id, 0)
        if measured == 0:
            status = "unmeasured"
        elif alarm_count > 0:
            status = "alarm"
        else:
            status = "normal"
        verdicts.append(
            WaferVerdict(
                lot_wafer_id=wafer_id,
                lot_id=(str(row[lot_column]) if lot_column in eval_df.columns and pd.notna(row.get(lot_column)) else None),
                status=status,
                alarm_count=alarm_count,
                measured_factor_count=measured,
            )
        )
    return verdicts
