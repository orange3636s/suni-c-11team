"""Single shared quantile-binning primitive.

Every caller that needs "divide x into N quantile bins, report each bin's
x/y summary" -- the 구간 평균 불량률 curve, the U-shape optimal-center
search, and the 개선 권장 window -- goes through `quantile_bins` here, so
they can never disagree about what a given bin's x actually is the way
optimal_center/recommended_range/curve used to.

This replaces the bug where a factor's "최적 중심" came from an unrelated
linear grid-search over [x.min(), x.max()] (src/analysis/screening/shape.py's
old `_best_center`) while the recommended window and the trend curve came
from quantile bins -- for a heavily right-skewed factor (a few outliers
pulling x.max() far out), the grid-search could land the center in an
empty gap the quantile bins never touch (see killing_event's Step26_R1:
grid-search center ~2150, true quantile-bin minimum at 1202, with zero
observations in between). A bin's *representative x* must always be the
mean of its own members, never the interval midpoint `(lo+hi)/2` -- for a
skewed/outlier-heavy bin the midpoint can sit in a stretch with no data
at all (exactly what happened here: bin 12 spans 1308~4039, but its
midpoint 2673 -- and the grid-search's ~2150 -- both fall in a region
with zero observations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BINS = 12
# A bin whose own [min, max] span exceeds this fraction of the factor's
# overall observed range is "sparse": outlier-widened rather than a
# genuine dense cluster, and its representative x shouldn't be trusted
# as an optimum location (spec §3-4).
SPARSE_SPAN_RATIO_THRESHOLD = 0.25


def quantile_bins(x: pd.Series, y: pd.Series, bins: int = DEFAULT_BINS) -> list[dict[str, float]]:
    """Per-bin x/y summary, sorted by `x_mean` ascending. `x_mean` (never
    the interval boundary midpoint) is every caller's definition of "where
    this bin sits" -- see module docstring for why the distinction matters.
    """
    try:
        q = pd.qcut(x, bins, duplicates="drop")
    except ValueError:
        return []
    frame = pd.DataFrame({"x": x, "y": y, "q": q})
    x_min, x_max = float(x.min()), float(x.max())
    overall_span = (x_max - x_min) or 1.0
    profile = []
    for _, group in frame.groupby("q", observed=True):
        n = len(group)
        y_mean = float(group["y"].mean())
        y_sem = float(group["y"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        x_lo = float(group["x"].min())
        x_hi = float(group["x"].max())
        bin_span_ratio = (x_hi - x_lo) / overall_span
        profile.append(
            {
                "x_mean": float(group["x"].mean()),
                "y_mean": y_mean,
                "y_lo": y_mean - 1.96 * y_sem,
                "y_hi": y_mean + 1.96 * y_sem,
                "n": n,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "bin_span_ratio": bin_span_ratio,
                "sparse": bin_span_ratio > SPARSE_SPAN_RATIO_THRESHOLD,
            }
        )
    profile.sort(key=lambda row: row["x_mean"])
    return profile


def quantile_of(sorted_values: list[float], q: float) -> float:
    """Sample quantile with linear interpolation -- matches numpy/pandas'
    default `interpolation="linear"`. `sorted_values` must already be
    sorted ascending."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]
    position = (n - 1) * q
    lower = int(position)
    upper = min(lower + 1, n - 1)
    weight = position - lower
    if weight == 0:
        return sorted_values[lower]
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def window_from_bins(bins: list[dict], x: pd.Series, y_mean_threshold: float) -> tuple[float, float] | None:
    """The contiguous run of bins (around wherever they qualify) whose
    y_mean sits at/below `y_mean_threshold`, turned into an x-range via
    quantile-interpolated bin *edges* -- not bin min/max, which an
    outlier-widened (`sparse`) bin would blow up (the same distortion
    `optimal_center_from_bins` guards against, just for a range instead
    of a point).
    """
    if not bins:
        return None
    qualifying = [i for i, row in enumerate(bins) if row["y_mean"] <= y_mean_threshold]
    if not qualifying:
        return None
    first, last = qualifying[0], qualifying[-1]
    sorted_x = sorted(float(v) for v in x.tolist())
    n_bins = len(bins)
    return quantile_of(sorted_x, first / n_bins), quantile_of(sorted_x, (last + 1) / n_bins)


def optimal_center_from_bins(bins: list[dict]) -> tuple[float | None, bool]:
    """The x_mean of the minimum-y_mean bin -- the one definition of
    "optimal center" every caller (U-shape classification, scatter chart,
    JSON report) must use instead of computing its own. Returns
    `(center, is_sparse)`: a sparse pick is returned rather than silently
    dropped so a caller that also has window info can still apply the
    window-containment check (spec §3-3); a caller without window info
    (shape classification) drops it on sparseness alone.
    """
    if not bins:
        return None, False
    k = min(range(len(bins)), key=lambda i: bins[i]["y_mean"])
    return float(bins[k]["x_mean"]), bool(bins[k]["sparse"])
