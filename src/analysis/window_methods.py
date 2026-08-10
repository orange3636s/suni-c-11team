"""SPC vs ML 권장구간 산출 방식 비교 (spec: "SPC / ML 방식 전환").

Two independent window-fitting algorithms compete for a factor's 권장
구간: SPC (the existing 12-quantile-bin rule -- same primitive the
구간 평균 불량률 curve and `optimal_center` already share, see
screening/quantile_profile.py) and ML (a shallow decision-tree
regressor's leaf boundaries). Both start from the same train-side x/y
pairs, get clamped into the factor's control range the same way, and
are scored by the same F2 x stability formula so neither is graded on
a curve.

Whichever wins ("adopted") is what everywhere else in the app -- alarm
log, 개선 권장 목록, the single canonical `optimal_center` -- uses; the
loser is kept only so the frontend can render it for side-by-side
comparison, never to drive alarms (spec §2-2).

Bootstrap resampling (60 reps) estimates how much a method's window
width jitters under resampling; a method whose boundary moves a lot
loses its `stability` multiplier, which is what keeps an ML window
that merely overfit a narrow pocket of the training data from beating
a more boring but reproducible SPC window on raw F2 alone.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.analysis.screening.quantile_profile import DEFAULT_BINS, optimal_center_from_bins, quantile_bins, window_from_bins

N_BOOTSTRAP = 60
FAIL_QUANTILE = 0.90
STABILITY_WIDTH_FRACTION = 0.15
ML_MAX_DEPTH = 3
ML_MIN_LEAF_FRACTION = 0.10
# Below this many rows a decision tree with the configured min-leaf-frac
# can't produce two distinct leaves at all -- not worth fitting.
MIN_ROWS_FOR_ML = 20

# Per-run cache for the 120-fit (60 bootstrap reps x 2 methods) comparison
# -- spec §2-5: computed once per analysis run, never recomputed just
# because a scatter chart got reopened. Keyed by (dataset_id, feature,
# target) (H-2: not id(train_df) -- see compare_methods' docstring for
# why object identity is unsafe here). Bounded + LRU-evicted so a
# long-running process doesn't grow this unboundedly across dataset
# reloads/uploads.
_CACHE_MAXSIZE = 256
_comparison_cache: "OrderedDict[tuple[str, str, str], MethodComparison]" = OrderedDict()


def _cache_get(key: tuple[str, str, str]) -> "MethodComparison | None":
    value = _comparison_cache.get(key)
    if value is not None:
        _comparison_cache.move_to_end(key)
    return value


def _cache_put(key: tuple[str, str, str], value: "MethodComparison") -> None:
    _comparison_cache[key] = value
    _comparison_cache.move_to_end(key)
    while len(_comparison_cache) > _CACHE_MAXSIZE:
        _comparison_cache.popitem(last=False)


@dataclass
class MethodWindow:
    lo: float
    hi: float
    center: float
    recall: float
    precision: float
    f2: float
    width_sd: float
    stability: float
    score: float
    clamped: bool  # True if the control range actually cut into this method's raw (unclamped) window

    def as_dict(self) -> dict[str, float | list[float] | bool]:
        return {
            "window": [self.lo, self.hi],
            "optimal_center": self.center,
            "recall": self.recall,
            "precision": self.precision,
            "f2": self.f2,
            "width_sd": self.width_sd,
            "stability": self.stability,
            "score": self.score,
            "clamped": self.clamped,
        }


@dataclass
class MethodComparison:
    spc: MethodWindow | None
    ml: MethodWindow | None
    adopted: str  # "spc" | "ml"
    adopted_reason: str


RawWindow = tuple[float, float, float]


def _spc_raw_window(x: pd.Series, y: pd.Series) -> RawWindow | None:
    """12분위 규칙: contiguous run of quantile bins at/below the overall
    mean, edges via quantile-interpolation -- identical to the
    recommendation engine's existing `_recommended_range_raw` /
    ScatterChart.tsx's `recommendedRange`, reused here rather than
    reimplemented so "SPC" never disagrees with what's already on
    screen.
    """
    # TC-5: SPC 경로는 기존 12구간 고정을 유지한다 (recommendations.py의
    # 같은 결정과 동일한 이유 -- 알람 판정에 쓰이는 window라 조용히
    # 바뀌면 안 된다).
    bins = quantile_bins(x, y, bins=DEFAULT_BINS)
    if not bins:
        return None
    window = window_from_bins(bins, x, float(y.mean()))
    if window is None:
        return None
    center, _sparse = optimal_center_from_bins(bins)
    if center is None:
        return None
    return window[0], window[1], center


def _ml_raw_window(
    x: pd.Series, y: pd.Series, *, max_depth: int = ML_MAX_DEPTH, min_leaf_frac: float = ML_MIN_LEAF_FRACTION
) -> RawWindow | None:
    """결정트리 회귀로 분할점을 학습: fit a shallow tree of x->y, group
    observations by leaf, then run the same "contiguous run at/below
    overall mean" merge SPC uses -- just over tree leaves (sorted by
    their own min-x) instead of quantile bins. `random_state=0` fixed so
    the same input always yields the same split.
    """
    n = len(x)
    min_leaf = max(1, int(n * min_leaf_frac))
    if n < MIN_ROWS_FOR_ML or n < 2 * min_leaf:
        return None
    x_arr = x.to_numpy().reshape(-1, 1)
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0)
    tree.fit(x_arr, y.to_numpy())
    leaves = tree.apply(x_arr)
    frame = pd.DataFrame({"x": x.to_numpy(), "y": y.to_numpy(), "leaf": leaves})
    grouped = (
        frame.groupby("leaf")
        .agg(m=("y", "mean"), lo=("x", "min"), hi=("x", "max"), xm=("x", "mean"))
        .sort_values("lo")
        .reset_index(drop=True)
    )
    if len(grouped) < 1:
        return None
    base = float(y.mean())
    ok = (grouped["m"] <= base).to_numpy()
    k = int(grouped["m"].to_numpy().argmin())
    a = b = k
    while a - 1 >= 0 and ok[a - 1]:
        a -= 1
    while b + 1 < len(grouped) and ok[b + 1]:
        b += 1
    return float(grouped["lo"].iloc[a]), float(grouped["hi"].iloc[b]), float(grouped["xm"].iloc[k])


def _f2(precision: float, recall: float) -> float:
    if (precision + recall) <= 0:
        return 0.0
    return 5 * precision * recall / (4 * precision + recall)


def _recall_precision(x: pd.Series, y: pd.Series, lo: float, hi: float) -> tuple[float, float]:
    """권장 구간 *밖*을 위험 신호로 본다: fail = 평가 데이터 상위 10% (구간
    밖) wafer, 재현율 = fail 중 구간 밖 비율, 정밀도 = 구간 밖 wafer 중
    fail 비율. "상위 10%"는 90th-percentile 값보다 *큰* 값만 (동률인
    threshold 자체는 fail이 아님) -- quantile(0.90) 근처에 동일값이 몰린
    이산형 인자(D 계열)에서 fail 집합이 정확히 10%를 넘지 않도록.
    """
    threshold = y.quantile(FAIL_QUANTILE)
    fail = y > threshold
    outside = (x < lo) | (x > hi)
    n_fail = int(fail.sum())
    n_outside = int(outside.sum())
    hit = int((fail & outside).sum())
    recall = hit / n_fail * 100.0 if n_fail > 0 else 0.0
    precision = hit / n_outside * 100.0 if n_outside > 0 else 0.0
    return recall, precision


def _bootstrap_width_sd(x: pd.Series, y: pd.Series, raw_fn, n_boot: int = N_BOOTSTRAP) -> float:
    frame = pd.DataFrame({"x": x.to_numpy(), "y": y.to_numpy()})
    widths: list[float] = []
    for seed in range(n_boot):
        sample = frame.sample(n=len(frame), replace=True, random_state=seed)
        window = raw_fn(sample["x"], sample["y"])
        if window is None:
            continue
        widths.append(window[1] - window[0])
    return float(np.std(widths)) if widths else 0.0


def _evaluate(x: pd.Series, y: pd.Series, raw_fn, lcl: float | None, ucl: float | None) -> MethodWindow | None:
    raw = raw_fn(x, y)
    if raw is None:
        return None
    raw_lo, raw_hi, raw_center = raw
    # A one-sided (monotonic) factor's control_range leaves the
    # non-alarming side unbounded (None) -- nothing to clamp against on
    # that side, so fall back to the method's own raw bound there,
    # mirroring the pre-existing single-method clamping behavior.
    effective_lcl = lcl if lcl is not None else raw_lo
    effective_ucl = ucl if ucl is not None else raw_hi
    lo, hi = max(raw_lo, effective_lcl), min(raw_hi, effective_ucl)
    if lo >= hi:
        # Spec: clamping collapsed the window -- no recommendation from
        # this method, not a degenerate zero-width one.
        return None
    center = min(max(raw_center, lo), hi)
    clamped = (lo, hi) != (raw_lo, raw_hi)

    recall, precision = _recall_precision(x, y, lo, hi)
    f2 = _f2(precision, recall)
    width_sd = _bootstrap_width_sd(x, y, raw_fn)
    band_width = effective_ucl - effective_lcl
    stability = min(band_width * STABILITY_WIDTH_FRACTION / width_sd, 1.0) if width_sd > 0 else 1.0
    score = f2 * stability
    return MethodWindow(
        lo=lo, hi=hi, center=center,
        recall=recall, precision=precision, f2=f2,
        width_sd=width_sd, stability=stability, score=score, clamped=clamped,
    )


def _adopted_reason(spc: MethodWindow | None, ml: MethodWindow | None, adopted: str) -> str:
    if spc is None and ml is None:
        return "권장 구간을 산출할 수 없습니다"
    if spc is None:
        return "SPC 방식은 권장 구간을 산출하지 못해 ML 방식을 사용합니다"
    if ml is None:
        return "ML 방식은 권장 구간을 산출하지 못해 SPC 방식을 사용합니다"

    winner, loser = (ml, spc) if adopted == "ml" else (spc, ml)
    if round(winner.score, 6) == round(loser.score, 6):
        return "두 방식이 같은 구간을 산출했습니다"
    # The winner's own F2 wasn't actually ahead -- it won on the
    # stability multiplier alone, i.e. the loser's F2 edge got wiped out
    # by an unstable (resample-sensitive) boundary.
    if winner.f2 <= loser.f2:
        return "F2는 근소 우위이나 구간 경계가 불안정해 감점되었습니다"

    recall_gap = winner.recall - loser.recall
    precision_gap = winner.precision - loser.precision
    no_stability_penalty = winner.stability >= 0.999
    if abs(recall_gap) >= abs(precision_gap):
        base = f"재현율이 {abs(recall_gap):.1f}%p 높습니다"
        return f"재현율이 {abs(recall_gap):.1f}%p 높고 안정성 감점이 없습니다" if no_stability_penalty else base
    base = f"정밀도가 {abs(precision_gap):.1f}%p 높습니다"
    return f"정밀도가 {abs(precision_gap):.1f}%p 높고 안정성 감점이 없습니다" if no_stability_penalty else base


def compare_methods(
    x: pd.Series,
    y: pd.Series,
    lcl: float | None,
    ucl: float | None,
    *,
    cache_key: tuple[str, str, str] | None = None,
) -> MethodComparison:
    """`x`/`y` must already be paired and NA-dropped (same contract as
    the recommendation engine's raw-window helpers); `lcl`/`ucl` are the
    factor's IQR*1.5 control limits (fixed, method-independent -- spec
    §2-3), `None` on a monotonic factor's non-alarming side.

    `cache_key`, when given, memoizes the full comparison (spec §2-5) --
    pass `(dataset_id, feature, target)` so repeat callers (a reopened
    scatter chart, the alarm/recommendation endpoints all touching the
    same factor within one analysis run) hit the cache instead of
    rerunning the 120-fit bootstrap. H-2: this used to be
    `(id(train_df), feature, target)` -- `id()` is only unique among
    currently-alive objects, so if `train_df` (itself an lru_cache'd
    DataFrame, see src/runtime/datasets.py) gets evicted and a later
    reload for a *different* dataset happens to reuse the same freed
    memory address, a stale cache entry could silently be returned for
    the wrong dataset. A string dataset_id has no such collision.
    """
    if cache_key is not None:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    spc = _evaluate(x, y, _spc_raw_window, lcl, ucl)
    ml = _evaluate(x, y, _ml_raw_window, lcl, ucl)

    if ml is not None and (spc is None or ml.score > spc.score):
        adopted = "ml"
    else:
        # Spec: 동점이면 SPC를 채택한다 (기존 방식 우선); also covers the
        # both-None case, where "spc" is just a label since callers must
        # treat a MethodComparison with spc=None,ml=None as "no
        # recommendation" regardless.
        adopted = "spc"

    result = MethodComparison(spc=spc, ml=ml, adopted=adopted, adopted_reason=_adopted_reason(spc, ml, adopted))
    if cache_key is not None:
        _cache_put(cache_key, result)
    return result
