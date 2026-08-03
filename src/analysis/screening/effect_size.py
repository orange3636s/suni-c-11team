"""Bias-corrected epsilon-squared effect size, comparable across R/D/Config.

Continuous factors (R, D) are binned into quantiles and tested with a
one-way ANOVA; categorical factors (Config) use the same ANOVA directly on
their natural categories. Both paths report epsilon-squared with the
``(F-1)`` bias correction so a small-sample factor (e.g. D columns with
n~480) isn't inflated relative to a large-sample factor (R columns with
n~1500) purely because the null-hypothesis epsilon-squared for an
uncorrected eta-squared grows with (k-1)/(n-1).

All computations use pairwise deletion: only rows where both the factor and
the target are observed are used. No imputation, no clipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EffectSizeResult:
    eps2: float
    p_value: float
    n_observed: int
    pearson_r: float | None
    spearman_r: float | None
    k_groups: int


def _eps2_from_groups(groups: list[np.ndarray]) -> tuple[float, float]:
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return 0.0, 1.0
    f_stat, p_value = stats.f_oneway(*groups)
    if not np.isfinite(f_stat):
        return 0.0, 1.0
    numerator = (f_stat - 1) * (k - 1)
    denominator = numerator + (n - k)
    eps2 = numerator / denominator if denominator > 0 else 0.0
    return max(float(eps2), 0.0), float(p_value)


def _safe_corr(func, x: pd.Series, y: pd.Series) -> float | None:
    if x.nunique() < 2 or y.nunique() < 2:
        return None
    try:
        value, _ = func(x, y)
    except ValueError:
        return None
    return float(value) if np.isfinite(value) else None


def eps2_numeric(
    x: pd.Series,
    y: pd.Series,
    bins: int = 8,
    min_n: int = 100,
) -> EffectSizeResult | None:
    """Effect size for a continuous factor (R or D column) against a target."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < min_n:
        return None

    try:
        q = pd.qcut(frame["x"], bins, duplicates="drop")
    except ValueError:
        return None
    if q.nunique() < 2:
        return None

    groups = [g["y"].to_numpy() for _, g in frame.groupby(q, observed=True)]
    eps2, p_value = _eps2_from_groups(groups)

    return EffectSizeResult(
        eps2=eps2,
        p_value=p_value,
        n_observed=len(frame),
        pearson_r=_safe_corr(stats.pearsonr, frame["x"], frame["y"]),
        spearman_r=_safe_corr(stats.spearmanr, frame["x"], frame["y"]),
        k_groups=len(groups),
    )


def eps2_categorical(
    x: pd.Series,
    y: pd.Series,
    min_n: int = 20,
) -> EffectSizeResult | None:
    """Effect size for a categorical factor (Config column) against a target."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if frame.empty:
        return None

    counts = frame["x"].value_counts()
    keep = counts[counts >= min_n].index
    frame = frame[frame["x"].isin(keep)]
    if len(frame) < min_n or frame["x"].nunique() < 2:
        return None

    groups = [g["y"].to_numpy() for _, g in frame.groupby("x", observed=True)]
    eps2, p_value = _eps2_from_groups(groups)

    return EffectSizeResult(
        eps2=eps2,
        p_value=p_value,
        n_observed=len(frame),
        pearson_r=None,
        spearman_r=None,
        k_groups=len(groups),
    )
