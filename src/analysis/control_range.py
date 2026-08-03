"""Per-factor normal ranges derived from train, and alarms raised on eval data.

The core idea: take the rows whose target fail-rate sits in the "healthy"
interquartile range (Q1..Q3), and read off the range of the selected factor
within that subset. That factor range becomes the "normal range" -- a wafer
outside it, on an unseen dataset, is flagged.

Monotonic factors only get a one-sided bound (see `compute_control_range`
docstring for why the other side is intentionally left open). A too-wide
range (>=98% coverage of ALL observed values) is a sign the IQR-derived
bound isn't discriminating anything, so it falls back to the band's own
1st/99th percentile.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.screening.selector import ParetoFactor

FALLBACK_COVERAGE_THRESHOLD = 0.98
FALLBACK_LOWER_QUANTILE = 0.01
FALLBACK_UPPER_QUANTILE = 0.99

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
class ControlRange:
    feature: str
    target: str
    kind: str
    relation_shape: str
    y_q1: float
    y_q3: float
    lower: float | None  # None means unbounded below
    upper: float | None  # None means unbounded above
    one_sided: bool
    fallback_applied: bool
    band_in_ratio: float  # n_band / n_observed -- how much of the observed sample the range rests on
    n_band: int  # rows in the Q1..Q3 band used to derive the range
    n_observed: int  # rows where `feature` is observed at all (pairwise)
    band_width: float  # band's raw x extent (max - min), used to normalize alarm severity

    def contains(self, value: float) -> bool:
        if pd.isna(value):
            return False
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


def compute_control_range(train_df: pd.DataFrame, factor: ParetoFactor) -> ControlRange:
    """Derive factor's normal range from wafers whose target sits in Q1..Q3.

    `monotonic_increasing` (higher factor -> higher fail rate) only bounds
    the upper side: a wafer with an unusually *low* value is never the
    problem, so constraining the lower side would flag good wafers as
    abnormal. `monotonic_decreasing` mirrors this for the lower bound only.
    """
    feature = factor.feature
    target = factor.target
    frame = pd.DataFrame(
        {"x": pd.to_numeric(train_df[feature], errors="coerce"), "y": pd.to_numeric(train_df[target], errors="coerce")}
    ).dropna()
    n_observed = len(frame)

    y_q1, y_q3 = frame["y"].quantile([0.25, 0.75])
    band = frame[(frame["y"] >= y_q1) & (frame["y"] <= y_q3)]
    n_band = len(band)

    shape = factor.relation_shape
    if shape == "monotonic_increasing":
        lower, upper = None, float(band["x"].max())
    elif shape == "monotonic_decreasing":
        lower, upper = float(band["x"].min()), None
    else:
        lower, upper = float(band["x"].min()), float(band["x"].max())
    one_sided = shape in ("monotonic_increasing", "monotonic_decreasing")

    def _coverage(lo: float | None, hi: float | None) -> float:
        within = pd.Series(True, index=frame.index)
        if lo is not None:
            within &= frame["x"] >= lo
        if hi is not None:
            within &= frame["x"] <= hi
        return float(within.mean())

    fallback_applied = False
    if _coverage(lower, upper) >= FALLBACK_COVERAGE_THRESHOLD:
        fallback_applied = True
        if shape == "monotonic_increasing":
            upper = float(band["x"].quantile(FALLBACK_UPPER_QUANTILE))
        elif shape == "monotonic_decreasing":
            lower = float(band["x"].quantile(FALLBACK_LOWER_QUANTILE))
        else:
            lower = float(band["x"].quantile(FALLBACK_LOWER_QUANTILE))
            upper = float(band["x"].quantile(FALLBACK_UPPER_QUANTILE))

    band_in_ratio = n_band / n_observed if n_observed else 0.0
    band_width = float(band["x"].max() - band["x"].min()) if n_band else 0.0

    return ControlRange(
        feature=feature,
        target=target,
        kind=factor.kind,
        relation_shape=shape,
        y_q1=float(y_q1),
        y_q3=float(y_q3),
        lower=lower,
        upper=upper,
        one_sided=one_sided,
        fallback_applied=fallback_applied,
        band_in_ratio=band_in_ratio,
        n_band=n_band,
        n_observed=n_observed,
        band_width=band_width,
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
    """Flag out-of-range wafers. `actual_y` reports the control range's own
    target column (e.g. Y2's fail rate for a Y2-derived alarm) -- not the
    overall final yield, which belongs at the alarm-summary level instead.
    """
    alarms: list[WaferAlarm] = []
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
