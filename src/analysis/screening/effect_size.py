"""Adjusted R² effect size, comparable across R/D/Config.

Continuous factors (R, D) are scored by the same degree-1/degree-2
polynomial fit used for the scatter/defect-rate curve overlay
(`curve_fit.fit_defect_rate_curve`) -- the R² this module reports for a
factor is the exact number the scatter chart shows, so clicking a heatmap
cell into its scatter view never disagrees with the cell. Categorical
factors (Config) use a dummy-variable regression, whose R² is algebraically
identical to eta-squared (SS_between/SS_total) from a one-way ANOVA on the
same groups.

Both paths report *Adjusted* R² -- ``1 - (1-R²)(n-1)/(n-p-1)`` -- because
raw R²/eta² is positively biased by the number of parameters (p, the
polynomial degree for numeric factors; k-1, the group count minus one, for
categorical), and that bias is not negligible at Config's per-combination
sample sizes (n~120, k up to a few dozen groups: raw eta² has a median of
~0.0035 on pure noise). Adjusted R² corrects it to ~0.

Significance uses the standard regression F-test, ``F = (R²/p) /
((1-R²)/(n-p-1))`` for numeric factors. For categorical factors this is
mathematically the same F-statistic a one-way ANOVA already produces, so
`scipy.stats.f_oneway`'s p-value is reused directly instead of running a
second, equivalent test.

All computations use pairwise deletion: only rows where both the factor and
the target are observed are used. No imputation, no clipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.curve_fit import fit_defect_rate_curve


@dataclass
class EffectSizeResult:
    adj_r2: float
    p_value: float
    n_observed: int
    # 1 or 2 for numeric (polynomial degree actually fit); None for
    # categorical (a dummy regression has no "degree" in this sense).
    degree: int | None
    k_groups: int
    # n_observed가 hard_min_n(제외 하한) 이상이지만 min_n(종류별 정상
    # 판정 임계) 미만일 때 True -- 결과는 그대로 쓰되 호출부가 등급을
    # 한 단계 낮추고 "표본 부족" 배지를 붙이는 신호로 쓴다.
    under_sampled: bool = False


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    denom = n - p - 1
    if denom <= 0:
        return 0.0
    adj = 1.0 - (1.0 - r2) * (n - 1) / denom
    return max(float(adj), 0.0)


def _regression_f_pvalue(r2: float, n: int, p: int) -> float:
    denom = n - p - 1
    if denom <= 0 or r2 <= 0:
        return 1.0
    if r2 >= 1.0:
        return 0.0
    f_stat = (r2 / p) / ((1.0 - r2) / denom)
    if not np.isfinite(f_stat) or f_stat <= 0:
        return 1.0
    return float(stats.f.sf(f_stat, p, denom))


HARD_MIN_N = 30  # 종류 불문 이 미만은 통계적으로 무의미해 완전히 제외


def adj_r2_numeric(
    x: pd.Series,
    y: pd.Series,
    min_n: int = 100,
    hard_min_n: int = HARD_MIN_N,
) -> EffectSizeResult | None:
    """Effect size for a continuous factor (R or D column) against a target.

    `min_n`은 "정상 판정" 임계(R/D마다 다르다 -- selector.py의
    DEFAULT_MIN_N_R/DEFAULT_MIN_N_D), `hard_min_n`은 종류 불문 완전 제외
    하한이다. hard_min_n <= n < min_n 구간은 결과를 그대로 계산해
    반환하되 `under_sampled=True`로 표시한다 -- D 인자처럼 계측률이 낮은
    종류가 min_n 미달이라는 이유만으로 화면에서 통째로 사라지는 것을
    막는다.

    이 함수 하나가 히트맵(screening/heatmap.py)과 Pareto(selector.py)의
    유일한 공통 경로이므로, 여기 한 곳만 바꾸면 두 화면이 계속 같은 값을
    본다. `fit_defect_rate_curve`가 자체 퇴화 처리(n<30 또는 x 고유값<5개면
    r2=0인 평균값 직선)를 갖고 있으므로 이 함수는 별도 하한 검사를
    반복하지 않는다.
    """
    valid = x.notna() & y.notna()
    x_valid = x.loc[valid].to_numpy(dtype=float)
    y_valid = y.loc[valid].to_numpy(dtype=float)
    n_observed = len(x_valid)
    if n_observed < hard_min_n:
        return None

    fit = fit_defect_rate_curve(x_valid, y_valid)
    p = fit.degree
    adj_r2 = _adjusted_r2(fit.r2, n_observed, p)
    p_value = _regression_f_pvalue(fit.r2, n_observed, p)

    return EffectSizeResult(
        adj_r2=adj_r2,
        p_value=p_value,
        n_observed=n_observed,
        degree=fit.degree,
        k_groups=0,
        under_sampled=n_observed < min_n,
    )


def adj_r2_categorical(
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
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return None

    f_stat, p_value = stats.f_oneway(*groups)
    if not np.isfinite(f_stat) or f_stat <= 0:
        adj_r2 = 0.0
        p_value = float(p_value) if np.isfinite(p_value) else 1.0
    else:
        eta2 = (f_stat * (k - 1)) / (f_stat * (k - 1) + (n - k))
        adj_r2 = _adjusted_r2(eta2, n, k - 1)
        p_value = float(p_value)

    return EffectSizeResult(
        adj_r2=adj_r2,
        p_value=p_value,
        n_observed=len(frame),
        degree=None,
        k_groups=k,
    )
