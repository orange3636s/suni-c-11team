"""Raw point-level data for the Spotfire-style scatter/box view.

The frontend does its own rendering (brushing, color-by-mode, drag-to-adjust
Q1/Q3), so this module hands back point-level rows with metadata instead of
a pre-rendered chart spec -- "서버는 데이터만 반환한다".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.analysis.control_range import ControlRange, compute_control_range
from src.analysis.recommendations import compute_factor_recommendation
from src.analysis.screening.quantile_profile import quantile_bins
from src.analysis.screening.selector import ParetoFactor, effective_confidence_tier
from src.analysis.warning_line import compute_warning_line, observed_yield_gap
from src.analysis.window_methods import compare_methods

if TYPE_CHECKING:
    from sklearn.ensemble import HistGradientBoostingRegressor

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


# Kept as a private alias -- this module's own callers below were written
# against the underscored name before `quantile_bins` moved to
# screening/quantile_profile.py as the shared definition every consumer
# (this module, recommendations.py, shape.py) reads from.
_quantile_bins = quantile_bins


def _outside_count(x: pd.Series, key: str, value: float) -> int:
    """How many of the currently-plotted points fall outside this
    reference line -- feeds the hover tooltip's "이 선 밖: N장" figure.
    Direction follows the line's own name (a "_lo"/q1 line is a floor,
    a "_hi"/q3 line is a ceiling); `mean` has no natural direction and
    isn't reported as an "outside" count.
    """
    if key in ("iqr_lo", "s3_lo", "s6_lo", "warning_lo") or key == "q1":
        return int((x < value).sum())
    if key in ("iqr_hi", "s3_hi", "s6_hi", "warning_hi") or key == "q3":
        return int((x > value).sum())
    return 0


def _warning_reference_lines(
    train_df: pd.DataFrame,
    factor: ParetoFactor,
    reference_model: "HistGradientBoostingRegressor | None",
    gbdt_features: list[str] | None,
    frame_x: pd.Series,
) -> list[dict[str, Any]]:
    """spec 알람 판정 GBDT 전환 §C: 관리한계(IQR 1.5배) 대신 부분 의존도
    기반 경고선을 쓴다. `reference_model`이 없으면(예: GBDT용 R+D 인자가
    하나도 없는 데이터셋) 빈 리스트를 반환한다 -- 관리한계 파선/앰버 영역이
    아예 그려지지 않아야 하므로, 실패를 조용히 삼키고 "경고선 없음"과
    동일하게 취급한다.
    """
    if reference_model is None or not gbdt_features or factor.kind == "Config":
        return []
    warning = compute_warning_line(reference_model, train_df, factor.feature, gbdt_features)
    if warning is None or (warning.lower is None and warning.upper is None):
        return []
    gaps = observed_yield_gap(train_df, factor.feature, warning)

    lines: list[dict[str, Any]] = []
    if len(frame_x):
        x_min, x_max = float(frame_x.min()), float(frame_x.max())
    else:
        x_min, x_max = 0.0, 0.0
    if warning.lower is not None:
        lines.append(
            {
                "key": "warning_lo",
                "value": warning.lower,
                "drawable": x_min <= warning.lower <= x_max,
                "alarm_relevant": True,
                "formula": "예측 수율 부분 의존도 기준",
                "outside_count": _outside_count(frame_x, "warning_lo", warning.lower),
                "observed_yield_gap_pp": gaps["lower_gap"],
            }
        )
    if warning.upper is not None:
        lines.append(
            {
                "key": "warning_hi",
                "value": warning.upper,
                "drawable": x_min <= warning.upper <= x_max,
                "alarm_relevant": True,
                "formula": "예측 수율 부분 의존도 기준",
                "outside_count": _outside_count(frame_x, "warning_hi", warning.upper),
                "observed_yield_gap_pp": gaps["upper_gap"],
            }
        )
    return lines


def _resolve_optimal_center(
    train_df: pd.DataFrame, factor: ParetoFactor, control_range: ControlRange, *, dataset_id: str
) -> tuple[float | None, str | None]:
    """Validates the already-classified optimal_center (shape.py, itself
    now bin-mean-based -- see quantile_profile.py) against the train-side
    recommended window: both must agree, since a "최적 중심" outside its
    own "권장구간" is the exact contradiction this module exists to
    prevent (spec §3-3). Only clamping-to-control-range can legitimately
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
    """SPC/ML comparison for the 방식 토글 (spec §3) -- train-derived, same
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
    eps2: float
    spearman_r: float | None
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
    reference_model: "HistGradientBoostingRegressor | None" = None,
    gbdt_features: list[str] | None = None,
) -> ScatterData:
    """`train_df` derives the control-limit-derived bits still in use
    (`points[].in_range` / `normal_range`, IQR*1.5) plus, when
    `reference_model` is given, the new PDP-based 경고선 (spec 알람 판정
    GBDT 전환 §C) that replaces the LCL/UCL reference lines on screen.
    `eval_df` supplies the plotted points and the per-reference-line
    "outside count" (pass the same frame for both to inspect train itself).
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

    # 알람 판정 GBDT 전환 §C-1/§C-3: 관리한계(iqr_lo/iqr_hi)는 더 이상
    # 화면에 그리지 않는다 -- 부분 의존도 기반 경고선으로 교체됐다. 나머지
    # 참고선(mean/q1/q3/s3/s6)은 이미 화면에서 렌더되지 않던 값들이라
    # 그대로 둔다 (JSON 소비자가 있을 수 있어 굳이 제거하지 않는다).
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
    reference_lines.extend(_warning_reference_lines(train_df, factor, reference_model, gbdt_features, frame["x"]))

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
        optimal_center=optimal_center,
        optimal_center_dropped_reason=optimal_center_dropped_reason,
        eps2=factor.eps2,
        spearman_r=factor.spearman_r,
        p_value=factor.p_value,
        q_value=factor.q_value,
        significant=factor.significant,
        confidence_tier=effective_confidence_tier(factor.eps2, factor.p_value, under_sampled=factor.under_sampled),
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
        confidence_tier=effective_confidence_tier(factor.eps2, factor.p_value, under_sampled=factor.under_sampled),
        n=len(frame),
        axis={
            "x_label": f"{factor.feature} (Step {factor.step} · 장비 설정)",
            "y_label": f"{factor.target} 불량률 (%)",
        },
    )
