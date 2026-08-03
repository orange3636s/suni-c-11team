"""Raw point-level data for the Spotfire-style scatter/box view.

The frontend does its own rendering (brushing, color-by-mode, drag-to-adjust
Q1/Q3), so this module hands back point-level rows with metadata instead of
a pre-rendered chart spec -- "서버는 데이터만 반환한다".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.control_range import ControlRange, compute_control_range
from src.analysis.screening.selector import ParetoFactor, confidence_tier

STEP_PATTERN = re.compile(r"^Step(\d+)_")
ID_COLUMN = "Lot_Wafer_ID"
LOT_COLUMN = "Lot_ID"


def _step_of(feature: str) -> int | None:
    match = STEP_PATTERN.match(feature)
    return int(match.group(1)) if match else None


def _config_column_for(feature: str, df: pd.DataFrame) -> str | None:
    step = _step_of(feature)
    if step is None:
        return None
    candidate = f"Step{step}_Config"
    return candidate if candidate in df.columns else None


def _quantile_bins(x: pd.Series, y: pd.Series, bins: int = 12) -> list[dict[str, float]]:
    try:
        q = pd.qcut(x, bins, duplicates="drop")
    except ValueError:
        return []
    frame = pd.DataFrame({"x": x, "y": y, "q": q})
    profile = []
    for _, group in frame.groupby("q", observed=True):
        n = len(group)
        y_mean = float(group["y"].mean())
        y_sem = float(group["y"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        profile.append(
            {
                "x_mean": float(group["x"].mean()),
                "y_mean": y_mean,
                "y_lo": y_mean - 1.96 * y_sem,
                "y_hi": y_mean + 1.96 * y_sem,
                "n": n,
            }
        )
    profile.sort(key=lambda row: row["x_mean"])
    return profile


def _outside_count(x: pd.Series, key: str, value: float) -> int:
    """How many of the currently-plotted points fall outside this
    reference line -- feeds the hover tooltip's "이 선 밖: N장" figure.
    Direction follows the line's own name (a "_lo"/q1 line is a floor,
    a "_hi"/q3 line is a ceiling); `mean` has no natural direction and
    isn't reported as an "outside" count.
    """
    if key in ("iqr_lo", "s3_lo", "s6_lo") or key == "q1":
        return int((x < value).sum())
    if key in ("iqr_hi", "s3_hi", "s6_hi") or key == "q3":
        return int((x > value).sum())
    return 0


@dataclass
class ScatterData:
    points: list[dict[str, Any]]
    reference_lines: list[dict[str, Any]]
    normal_range: dict[str, Any]
    bins: list[dict[str, float]]
    optimal_center: float | None
    eps2: float
    spearman_r: float | None
    p_value: float
    q_value: float
    significant: bool
    confidence_tier: str
    n: int
    axis: dict[str, str]


def build_scatter_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    factor: ParetoFactor,
) -> ScatterData:
    """`train_df` derives the SPC control limits (computed from X alone,
    never from Y); `eval_df` supplies the plotted points and the
    per-reference-line "outside count" (pass the same frame for both to
    inspect train itself).
    """
    control_range = compute_control_range(train_df, factor)

    config_column = _config_column_for(factor.feature, eval_df)
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(eval_df[factor.feature], errors="coerce"),
            "y": pd.to_numeric(eval_df[factor.target], errors="coerce"),
        }
    )
    frame["lot_wafer_id"] = eval_df[ID_COLUMN] if ID_COLUMN in eval_df.columns else None
    frame["lot_id"] = eval_df[LOT_COLUMN] if LOT_COLUMN in eval_df.columns else None
    frame["config"] = eval_df[config_column] if config_column else None
    frame = frame.dropna(subset=["x", "y"])

    points = [
        {
            "x": float(row.x),
            "y": float(row.y),
            "lot_wafer_id": (str(row.lot_wafer_id) if pd.notna(row.lot_wafer_id) else None),
            "lot_id": (str(row.lot_id) if pd.notna(row.lot_id) else None),
            "in_range": control_range.contains(row.x),
            "config": (str(row.config) if pd.notna(row.config) else None),
        }
        for row in frame.itertuples(index=False)
    ]

    reference_lines = [
        {
            "key": line.key,
            "value": line.value,
            "drawable": line.drawable,
            "alarm_relevant": line.alarm_relevant,
            "formula": line.formula,
            "outside_count": _outside_count(frame["x"], line.key, line.value),
        }
        for line in control_range.reference_lines
    ]

    return ScatterData(
        points=points,
        reference_lines=reference_lines,
        normal_range={
            "lo": control_range.lower,
            "hi": control_range.upper,
            "one_sided": control_range.one_sided,
            "fallback_applied": control_range.fallback_applied,
        },
        bins=_quantile_bins(frame["x"], frame["y"]),
        optimal_center=factor.optimal_center,
        eps2=factor.eps2,
        spearman_r=factor.spearman_r,
        p_value=factor.p_value,
        q_value=factor.q_value,
        significant=factor.significant,
        confidence_tier=confidence_tier(factor.p_value),
        n=len(frame),
        axis={
            "x_label": f"{factor.feature} (Step {factor.step} · {_kind_label(factor.kind)})",
            "y_label": f"{factor.target} 불량률 (%)",
        },
    )


def _kind_label(kind: str) -> str:
    return {"R": "계측값", "D": "결함수", "Config": "장비 설정"}.get(kind, kind)


@dataclass
class CategoricalGroup:
    category: str
    n: int
    mean: float
    median: float
    q1: float
    q3: float
    values: list[float]


@dataclass
class CategoricalScatterData:
    groups: list[CategoricalGroup]
    eps2: float
    p_value: float
    q_value: float
    significant: bool
    confidence_tier: str
    n: int
    axis: dict[str, str]


def build_categorical_data(eval_df: pd.DataFrame, factor: ParetoFactor) -> CategoricalScatterData:
    """Per-category box-plot data for a Config factor -- there is no
    numeric x, so unlike build_scatter_data this never derives a
    Q1/Q3-band normal range (a categorical value has no "range").
    """
    frame = pd.DataFrame(
        {
            "x": eval_df[factor.feature],
            "y": pd.to_numeric(eval_df[factor.target], errors="coerce"),
        }
    ).dropna()

    groups: list[CategoricalGroup] = []
    for category, group in frame.groupby("x", observed=True):
        y = group["y"]
        groups.append(
            CategoricalGroup(
                category=str(category),
                n=len(y),
                mean=float(y.mean()),
                median=float(y.median()),
                q1=float(y.quantile(0.25)),
                q3=float(y.quantile(0.75)),
                values=[float(value) for value in y.tolist()],
            )
        )
    groups.sort(key=lambda g: g.mean, reverse=True)

    return CategoricalScatterData(
        groups=groups,
        eps2=factor.eps2,
        p_value=factor.p_value,
        q_value=factor.q_value,
        significant=factor.significant,
        confidence_tier=confidence_tier(factor.p_value),
        n=len(frame),
        axis={
            "x_label": f"{factor.feature} (Step {factor.step} · 장비 설정)",
            "y_label": f"{factor.target} 불량률 (%)",
        },
    )
