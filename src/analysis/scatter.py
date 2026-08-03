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
from src.analysis.screening.selector import ParetoFactor

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


def _quantile_bins(x: pd.Series, y: pd.Series, bins: int = 8) -> list[dict[str, float]]:
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


@dataclass
class ScatterData:
    points: list[dict[str, Any]]
    y_q1: float
    y_q3: float
    band_x_min: float | None
    band_x_max: float | None
    normal_range: dict[str, Any]
    bins: list[dict[str, float]]
    optimal_center: float | None
    eps2: float
    q_value: float
    significant: bool
    n: int
    axis: dict[str, str]


def build_scatter_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    factor: ParetoFactor,
) -> ScatterData:
    """`train_df` derives Q1/Q3 and the normal range; `eval_df` supplies the
    plotted points (pass the same frame for both to inspect train itself).
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
            "in_band": bool(control_range.y_q1 <= row.y <= control_range.y_q3),
            "in_range": control_range.contains(row.x),
            "config": (str(row.config) if pd.notna(row.config) else None),
        }
        for row in frame.itertuples(index=False)
    ]

    band_frame = frame[(frame["y"] >= control_range.y_q1) & (frame["y"] <= control_range.y_q3)]
    band_x_min = float(band_frame["x"].min()) if len(band_frame) else None
    band_x_max = float(band_frame["x"].max()) if len(band_frame) else None

    return ScatterData(
        points=points,
        y_q1=control_range.y_q1,
        y_q3=control_range.y_q3,
        band_x_min=band_x_min,
        band_x_max=band_x_max,
        normal_range={
            "lo": control_range.lower,
            "hi": control_range.upper,
            "one_sided": control_range.one_sided,
            "fallback_applied": control_range.fallback_applied,
        },
        bins=_quantile_bins(frame["x"], frame["y"]),
        optimal_center=factor.optimal_center,
        eps2=factor.eps2,
        q_value=factor.q_value,
        significant=factor.significant,
        n=len(frame),
        axis={
            "x_label": f"{factor.feature} (Step {factor.step} · {_kind_label(factor.kind)})",
            "y_label": f"{factor.target} 불량률 (%)",
        },
    )


def _kind_label(kind: str) -> str:
    return {"R": "계측값", "D": "결함수", "Config": "장비 설정"}.get(kind, kind)
