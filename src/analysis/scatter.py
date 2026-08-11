"""Raw point-level data for the Spotfire-style scatter/box view.

The frontend does its own rendering (brushing, color-by-mode, drag-to-adjust
Q1/Q3), so this module hands back point-level rows with metadata instead of
a pre-rendered chart spec -- "서버는 데이터만 반환한다".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analysis.control_range import ControlRange, compute_control_range
from src.analysis.recommendations import compute_factor_recommendation
from src.analysis.sampling import SCATTER_POINT_MAX_ROWS
from src.analysis.screening.quantile_profile import DEFAULT_BINS, quantile_bins
from src.analysis.screening.selector import ParetoFactor, effective_confidence_tier
from src.analysis.window_methods import compare_methods

logger = logging.getLogger(__name__)

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


# Private alias for the shared definition in screening/quantile_profile.py,
# which every consumer (this module, recommendations.py, shape.py) reads
# from. Do not reintroduce a local implementation.
_quantile_bins = quantile_bins


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


def _resolve_optimal_center(
    train_df: pd.DataFrame, factor: ParetoFactor, control_range: ControlRange, *, dataset_id: str
) -> tuple[float | None, str | None]:
    """Validates the already-classified optimal_center (shape.py, itself
    now bin-mean-based -- see quantile_profile.py) against the train-side
    recommended window: both must agree, since a "최적 중심" outside its
    own "권장구간" is the exact contradiction this module exists to
    prevent. Only clamping-to-control-range can legitimately
    push a correctly-computed center outside its window; when that
    happens the center is dropped (never forced back inside) and the
    reason surfaces to the caller for the disabled-toggle tooltip.
    """
    if factor.optimal_center is None:
        return None, None
    window = compute_factor_recommendation(train_df, factor, control_range, dataset_id=dataset_id)
    if window is None:
        return factor.optimal_center, None
    lo, hi = window.recommended_lo, window.recommended_hi
    if not (lo <= factor.optimal_center <= hi):
        logger.warning(
            "optimal_center %.3f outside recommended window [%.3f, %.3f] for %s -> %s; dropping",
            factor.optimal_center, lo, hi, factor.feature, factor.target,
        )
        return None, "관리한계 조정으로 최적 지점이 권장구간을 벗어나 표시하지 않음"
    return factor.optimal_center, None


def _compute_methods(
    train_df: pd.DataFrame, factor: ParetoFactor, control_range: ControlRange, *, dataset_id: str
) -> dict[str, Any] | None:
    """SPC/ML comparison for the 방식 토글 -- train-derived, same
    x/y pair `compute_control_range`/`compute_factor_recommendation` use.
    `None` for Config factors (no numeric x to fit either method on).
    """
    if factor.kind == "Config" or factor.feature not in train_df.columns:
        return None
    x = pd.to_numeric(train_df[factor.feature], errors="coerce")
    y = pd.to_numeric(train_df[factor.target], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    comparison = compare_methods(
        x[valid], y[valid], control_range.lower, control_range.upper,
        cache_key=(dataset_id, factor.feature, factor.target),
    )
    return {
        "spc": comparison.spc.as_dict() if comparison.spc is not None else None,
        "ml": comparison.ml.as_dict() if comparison.ml is not None else None,
        "adopted": comparison.adopted,
        "adopted_reason": comparison.adopted_reason,
    }


@dataclass
class ScatterData:
    points: list[dict[str, Any]]
    reference_lines: list[dict[str, Any]]
    normal_range: dict[str, Any]
    bins: list[dict[str, float]]
    optimal_center: float | None
    optimal_center_dropped_reason: str | None
    adj_r2: float
    # 1 or 2 -- the polynomial degree `adj_r2` was fit at (scatter chart's
    # "R²=0.235 (2차)" meta line reads both from here).
    degree: int | None
    p_value: float
    q_value: float
    significant: bool
    confidence_tier: str
    under_sampled: bool
    relation_shape: str
    n: int
    axis: dict[str, str]
    methods: dict[str, Any] | None


def build_scatter_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    factor: ParetoFactor,
    *,
    dataset_id: str,
) -> ScatterData:
    """`train_df` derives the control-limit bits (`points[].in_range` /
    `normal_range`, IQR*1.5); `reference_lines` is exactly those
    control-range-derived lines and nothing else. `eval_df` supplies the
    plotted points and the per-reference-line "outside count" (pass the
    same frame for both to inspect train itself).
    """
    control_range = compute_control_range(train_df, factor)
    optimal_center, optimal_center_dropped_reason = _resolve_optimal_center(
        train_df, factor, control_range, dataset_id=dataset_id
    )
    methods = _compute_methods(train_df, factor, control_range, dataset_id=dataset_id)

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
    if len(points) > SCATTER_POINT_MAX_ROWS:
        # 그 이상은 화면에서 겹쳐 안 보인다 -- 찍는 점만 균등 간격으로
        # 줄인다. 통계량(n/adj_r2/outside_count/기준선)은 전부 위에서 이미
        # 전체 frame 기준으로 계산을 마쳤으니 이 아래로는 영향이 없다.
        stride = len(points) / SCATTER_POINT_MAX_ROWS
        points = [points[int(i * stride)] for i in range(SCATTER_POINT_MAX_ROWS)]

    # 참고선은 전부 payload에 싣되, 화면에 그릴지는 `drawable` 플래그가
    # 정한다 -- JSON을 직접 읽는 소비자가 있어 값 자체는 빼지 않는다.
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
        if line.key not in ("iqr_lo", "iqr_hi")
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
        # SPC 계열(recommendations.py/window_methods.py)과 같은 이유로
        # 12구간 고정이다 -- Sturges 자동 구간수는 히트맵/Pareto의
        # Adjusted R² 계산에만 쓴다.
        bins=_quantile_bins(frame["x"], frame["y"], bins=DEFAULT_BINS),
        optimal_center=optimal_center,
        optimal_center_dropped_reason=optimal_center_dropped_reason,
        adj_r2=factor.adj_r2,
        degree=factor.degree,
        p_value=factor.p_value,
        q_value=factor.q_value,
        significant=factor.significant,
        confidence_tier=effective_confidence_tier(factor.adj_r2, factor.p_value, under_sampled=factor.under_sampled),
        under_sampled=factor.under_sampled,
        relation_shape=factor.relation_shape,
        methods=methods,
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
    adj_r2: float
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
        adj_r2=factor.adj_r2,
        p_value=factor.p_value,
        q_value=factor.q_value,
        significant=factor.significant,
        confidence_tier=effective_confidence_tier(factor.adj_r2, factor.p_value, under_sampled=factor.under_sampled),
        n=len(frame),
        axis={
            "x_label": f"{factor.feature} (Step {factor.step} · 장비 설정)",
            "y_label": f"{factor.target} 불량률 (%)",
        },
    )
