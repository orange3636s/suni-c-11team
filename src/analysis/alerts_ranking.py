"""RE그룹: 알림기록 판정 -- 요구 11번대로 y 합산(100 − Σ Y1~Y5) 기준
오름차순으로 수율 하위 wafer를 뽑는다. 실측 우선 예측(RC-3, y1~y5는
target_hydration이 이미 실측 우선으로 채운다)과 신뢰도 점수(RC-4)·
y1~y5 셀 색상(RC-4b)을 그대로 재사용한다 -- 여기서 다시 계산하지 않는다.

정렬은 y 오름차순 하나다(하지 말 것: 신뢰도로 정렬하거나 신뢰도
하한으로 후보를 거르지 마라). 신뢰도는 판단 재료로 표시만 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.control_range import compute_control_range
from src.analysis.reliability_score import (
    FAIL_RATE_TARGETS,
    PrimaryFactor,
    cell_color,
    compute_reliability_scores,
)
from src.analysis.screening.schema import Schema, parse_schema
from src.analysis.screening.selector import ParetoFactor, select_top_factors

ID_COLUMN = "Lot_Wafer_ID"
LOT_COLUMN = "Lot_ID"
LOW_RELIABILITY_THRESHOLD = 40


@dataclass(frozen=True)
class RangeInfo:
    lo: float | None
    hi: float | None


@dataclass(frozen=True)
class AlertCandidate:
    lot_wafer_id: str
    lot_id: str | None
    y: float
    y_components: dict[str, float]
    reliability: int
    primary_target: str
    primary_feature: str
    factor_value: float | None
    range_lo: float | None
    range_hi: float | None
    reason: str
    cells: dict[str, dict[str, object]]


@dataclass(frozen=True)
class AlertRankingSummary:
    mean_reliability: float
    min_reliability: int
    below_threshold_count: int
    zero_reliability_count: int


@dataclass(frozen=True)
class AlertRankingTable:
    candidates: list[AlertCandidate]
    summary: AlertRankingSummary
    total_wafers: int


def _reason_text(feature: str, value: float | None, range_info: RangeInfo) -> str:
    if value is None or not np.isfinite(value):
        return f"{feature} 값 없음"
    lo, hi = range_info.lo, range_info.hi
    if lo is not None and value < lo:
        return f"{feature} = {value:.1f} · 권장 하한 {lo:.1f} 대비 {lo - value:.1f} 미달"
    if hi is not None and value > hi:
        return f"{feature} = {value:.1f} · 권장 상한 {hi:.1f} 대비 +{value - hi:.1f} 초과"
    return f"{feature} = {value:.1f} (권장 {lo:.1f}~{hi:.1f})" if lo is not None and hi is not None else f"{feature} = {value:.1f}"


def _primary_factors_and_ranges(
    train_df: pd.DataFrame, schema: Schema, targets: tuple[str, ...]
) -> tuple[dict[str, PrimaryFactor | None], dict[str, RangeInfo]]:
    """타깃별 1위 인자를 한 번만 랭킹해(select_top_factors, limit=1) 신뢰도용
    PrimaryFactor와 권장 구간(compute_control_range)을 함께 뽑는다 --
    같은 전체 인자 풀 랭킹을 두 번(신뢰도용/구간용) 돌리면 88인자 x
    5타깃 스코어링이 두 배가 된다."""
    primary_factors: dict[str, PrimaryFactor | None] = {}
    ranges: dict[str, RangeInfo] = {}
    for target in targets:
        top: list[ParetoFactor] = select_top_factors(train_df, schema, target, limit=1)
        if not top:
            primary_factors[target] = None
            continue
        factor = top[0]
        primary_factors[target] = PrimaryFactor(
            feature=factor.feature,
            contribution_pct=factor.contribution_pct,
            relation_shape=factor.relation_shape,
            optimal_center=factor.optimal_center,
        )
        control_range = compute_control_range(train_df, factor)
        ranges[target] = RangeInfo(control_range.lower, control_range.upper)
    return primary_factors, ranges


def build_alert_ranking(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    hydrated_df: pd.DataFrame,
    *,
    top_n: int = 10,
) -> AlertRankingTable:
    """`hydrated_df`는 `target_hydration.hydrate_targets`가 반환한, 실측
    우선(RC-3)으로 Y1~Y5(및 Y)를 채운 프레임이어야 한다 -- 여기서는 그
    결과의 순위만 매긴다. `eval_df`는 실측/예측 판정(신뢰도·셀 색상)에
    쓰는 원본(하이드레이션 전) 프레임이다."""
    schema = parse_schema(train_df)
    primary_factors, ranges = _primary_factors_and_ranges(train_df, schema, FAIL_RATE_TARGETS)

    reliability = compute_reliability_scores(eval_df, primary_factors, FAIL_RATE_TARGETS)

    y = pd.to_numeric(hydrated_df["Y"], errors="coerce")
    order = y.sort_values(kind="mergesort").index  # stable -- y 오름차순 하나(부차 순서는 원래 행 순서 유지)

    candidates: list[AlertCandidate] = []
    for idx in order[:top_n]:
        row = hydrated_df.loc[idx]
        y_components = {t: float(row[t]) for t in FAIL_RATE_TARGETS if t in hydrated_df.columns}
        primary_target = max(y_components, key=lambda t: y_components[t]) if y_components else FAIL_RATE_TARGETS[0]
        factor = primary_factors.get(primary_target)
        feature = factor.feature if factor else None
        factor_value = float(eval_df.loc[idx, feature]) if feature and feature in eval_df.columns and pd.notna(eval_df.loc[idx, feature]) else None
        range_info = ranges.get(primary_target, RangeInfo(None, None))
        reason = _reason_text(feature or "-", factor_value, range_info) if feature else "선정 인자 없음"

        cells: dict[str, dict[str, object]] = {}
        for target in FAIL_RATE_TARGETS:
            is_measured = target in eval_df.columns and pd.notna(eval_df.loc[idx, target])
            target_factor = primary_factors.get(target)
            target_feature = target_factor.feature if target_factor else None
            target_value = (
                float(eval_df.loc[idx, target_feature])
                if target_feature and target_feature in eval_df.columns and pd.notna(eval_df.loc[idx, target_feature])
                else None
            )
            cells[target] = cell_color(target, target_value, is_measured, primary_factors)

        candidates.append(
            AlertCandidate(
                lot_wafer_id=str(row[ID_COLUMN]) if ID_COLUMN in hydrated_df.columns else str(idx),
                lot_id=str(row[LOT_COLUMN]) if LOT_COLUMN in hydrated_df.columns and pd.notna(row[LOT_COLUMN]) else None,
                y=float(row["Y"]),
                y_components=y_components,
                reliability=int(reliability.loc[idx]),
                primary_target=primary_target,
                primary_feature=feature or "-",
                factor_value=factor_value,
                range_lo=range_info.lo,
                range_hi=range_info.hi,
                reason=reason,
                cells=cells,
            )
        )

    top_reliability = reliability.loc[order[:top_n]] if len(order) else pd.Series(dtype=int)
    summary = AlertRankingSummary(
        mean_reliability=float(top_reliability.mean()) if len(top_reliability) else 0.0,
        min_reliability=int(top_reliability.min()) if len(top_reliability) else 0,
        below_threshold_count=int((top_reliability < LOW_RELIABILITY_THRESHOLD).sum()),
        zero_reliability_count=int((top_reliability == 0).sum()),
    )
    return AlertRankingTable(candidates=candidates, summary=summary, total_wafers=len(hydrated_df))
