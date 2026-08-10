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

from src.analysis.screening.bin_count import suggest_bin_count


@dataclass
class EffectSizeResult:
    eps2: float
    p_value: float
    n_observed: int
    pearson_r: float | None
    spearman_r: float | None
    k_groups: int
    # QA-2: n_observed가 hard_min_n(제외 하한) 이상이지만 min_n(종류별
    # 정상 판정 임계) 미만일 때 True -- 결과는 그대로 쓰되 호출부가 등급을
    # 한 단계 낮추고 "표본 부족" 배지를 붙이는 신호로 쓴다.
    under_sampled: bool = False


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


HARD_MIN_N = 30  # 종류 불문 이 미만은 통계적으로 무의미해 완전히 제외 (QA-2)


def eps2_numeric(
    x: pd.Series,
    y: pd.Series,
    bins: int | None = None,
    min_n: int = 100,
    hard_min_n: int = HARD_MIN_N,
) -> EffectSizeResult | None:
    """Effect size for a continuous factor (R or D column) against a target.

    `min_n`은 "정상 판정" 임계(R/D마다 다르다 -- selector.py의
    DEFAULT_MIN_N_R/DEFAULT_MIN_N_D), `hard_min_n`은 종류 불문 완전 제외
    하한이다. hard_min_n <= n < min_n 구간은 결과를 그대로 계산해
    반환하되 `under_sampled=True`로 표시한다(QA-2) -- D 인자처럼 계측률이
    낮은 종류가 min_n 미달이라는 이유만으로 화면에서 통째로 사라지는
    것을 막는다.

    TC-5: `bins`가 None이면(모든 실제 호출부가 그렇다) 표본 수에 맞춘
    Sturges 구간 수(`suggest_bin_count`)를 쓴다 -- 예전에는 8로 고정이라
    D(n~480)와 R(n~1500)이 같은 구간 폭을 강제로 나눠 가졌다. 이 함수
    하나가 히트맵(screening/heatmap.py)과 Pareto(selector.py) eps2 계산의
    유일한 공통 경로이므로, 여기 한 곳만 바꾸면 두 화면이 계속 같은 값을
    본다.
    """
    # Keep the exact pairwise-deletion/qcut/ANOVA definition, but avoid a
    # temporary two-column DataFrame plus a pandas groupby for every
    # feature-target pair. A 1,000 x 88 analysis evaluates this path 440
    # times (Y1..Y5), so those allocations dominated cold-cache latency.
    valid = x.notna() & y.notna()
    x_valid = x.loc[valid]
    y_valid = y.loc[valid]
    n_observed = len(x_valid)
    if n_observed < hard_min_n:
        return None

    effective_bins = bins if bins is not None else suggest_bin_count(n_observed)
    try:
        q = pd.qcut(x_valid, effective_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    if q.nunique() < 2:
        return None

    q_values = q.to_numpy()
    y_values = y_valid.to_numpy()
    groups = [y_values[q_values == label] for label in np.unique(q_values)]
    eps2, p_value = _eps2_from_groups(groups)

    return EffectSizeResult(
        eps2=eps2,
        p_value=p_value,
        n_observed=n_observed,
        pearson_r=_safe_corr(stats.pearsonr, x_valid, y_valid),
        spearman_r=_safe_corr(stats.spearmanr, x_valid, y_valid),
        k_groups=len(groups),
        under_sampled=n_observed < min_n,
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
