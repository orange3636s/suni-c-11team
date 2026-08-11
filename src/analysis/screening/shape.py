"""Relation-shape classification: monotonic vs. U-shaped vs. unclear.

Determines what to overlay on a factor's scatter plot. A U-shaped
relationship (common for process parameters with a process-window optimum)
looks uncorrelated to Pearson/Spearman because the two tails point in
opposite directions from the center — so this module explicitly searches
for a center point ``c`` that linearizes ``|x - c|`` against ``y`` *to
decide whether the relationship is U-shaped at all*.

That grid-search center is used only for that yes/no test, never for the
*displayed* optimal_center value: `_best_center` searches a plain linear
grid over `[x.min(), x.max()]`, which for a heavily skewed/outlier-heavy
factor can land in a region with zero observations (e.g. a few outliers
pulling x.max() far out leaves a wide empty gap the grid still spans).
The reported `optimal_center` instead always comes from
`quantile_profile.optimal_center_from_bins` -- the x_mean of the same
quantile-binned minimum-y bin the recommended window and the 구간 평균
불량률 curve are built from, so "최적 중심" can never land outside "권장
구간" or in a gap the curve itself shows as empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.screening.quantile_profile import (
    DEFAULT_BINS,
    optimal_center_from_bins,
    quantile_bins,
)

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

    # `_best_center`'s own grid-search value is discarded below -- it only
    # feeds `dev_abs_r`, the statistical test for *whether* this is a
    # U-shape at all. See module docstring for why the displayed center
    # never comes from that grid search.
    _grid_center, dev_abs_r = _best_center(frame["x"], frame["y"])
    if abs(pearson_r) < 1e-9:
        is_u_shape = dev_abs_r > 0
    else:
        is_u_shape = dev_abs_r >= U_SHAPE_RATIO_THRESHOLD * abs(pearson_r)

    if is_u_shape:
        optimal_bins = quantile_bins(frame["x"], frame["y"], bins=DEFAULT_BINS)
        center, sparse = optimal_center_from_bins(optimal_bins)
        # A minimum-y bin that's itself outlier-widened (`sparse`) isn't a
        # meaningful process-window location -- no center is better than a
        # misleading one.
        if sparse:
            center = None
        return ShapeResult(shape="u_shape", optimal_center=center, quantile_profile=profile)

    return ShapeResult(shape="unclear", optimal_center=None, quantile_profile=profile)
