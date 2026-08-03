"""Relation-shape classification: monotonic vs. U-shaped vs. unclear.

Determines what to overlay on a factor's scatter plot. A U-shaped
relationship (common for process parameters with a process-window optimum)
looks uncorrelated to Pearson/Spearman because the two tails point in
opposite directions from the center — so this module explicitly searches
for a center point ``c`` that linearizes ``|x - c|`` against ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MONOTONIC_SPEARMAN_THRESHOLD = 0.3
U_SHAPE_RATIO_THRESHOLD = 1.3
GRID_POINTS = 200


@dataclass
class ShapeResult:
    shape: str  # "monotonic_increasing" | "monotonic_decreasing" | "u_shape" | "unclear"
    optimal_center: float | None
    quantile_profile: list[tuple[float, float]]  # (bin mean x, bin mean y)


def _quantile_profile(x: pd.Series, y: pd.Series, bins: int = 8) -> list[tuple[float, float]]:
    try:
        q = pd.qcut(x, bins, duplicates="drop")
    except ValueError:
        return []
    frame = pd.DataFrame({"x": x, "y": y, "q": q})
    profile = frame.groupby("q", observed=True).agg(x_mean=("x", "mean"), y_mean=("y", "mean"))
    profile = profile.sort_values("x_mean")
    return list(zip(profile["x_mean"].tolist(), profile["y_mean"].tolist()))


def _is_monotonic(profile: list[tuple[float, float]]) -> bool:
    if len(profile) < 3:
        return False
    y_values = [p[1] for p in profile]
    diffs = np.diff(y_values)
    return bool(np.all(diffs >= 0) or np.all(diffs <= 0))


def _best_center(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    candidates = np.linspace(x.min(), x.max(), GRID_POINTS)
    best_center = float(x.median())
    best_abs_r = 0.0
    for c in candidates:
        dev = (x - c).abs()
        if dev.nunique() < 2:
            continue
        r, _ = stats.pearsonr(dev, y)
        if not np.isfinite(r):
            continue
        if abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_center = float(c)
    return best_center, best_abs_r


def classify_shape(x: pd.Series, y: pd.Series, bins: int = 8) -> ShapeResult:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    profile = _quantile_profile(frame["x"], frame["y"], bins=bins)

    spearman_r = 0.0
    if frame["x"].nunique() >= 2 and frame["y"].nunique() >= 2:
        rho, _ = stats.spearmanr(frame["x"], frame["y"])
        spearman_r = float(rho) if np.isfinite(rho) else 0.0

    if abs(spearman_r) >= MONOTONIC_SPEARMAN_THRESHOLD and _is_monotonic(profile):
        shape = "monotonic_increasing" if spearman_r > 0 else "monotonic_decreasing"
        return ShapeResult(shape=shape, optimal_center=None, quantile_profile=profile)

    pearson_r = 0.0
    if frame["x"].nunique() >= 2 and frame["y"].nunique() >= 2:
        r, _ = stats.pearsonr(frame["x"], frame["y"])
        pearson_r = float(r) if np.isfinite(r) else 0.0

    center, dev_abs_r = _best_center(frame["x"], frame["y"])
    if abs(pearson_r) < 1e-9:
        is_u_shape = dev_abs_r > 0
    else:
        is_u_shape = dev_abs_r >= U_SHAPE_RATIO_THRESHOLD * abs(pearson_r)

    if is_u_shape:
        return ShapeResult(shape="u_shape", optimal_center=center, quantile_profile=profile)

    return ShapeResult(shape="unclear", optimal_center=None, quantile_profile=profile)
