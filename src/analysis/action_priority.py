"""모니터링 홈 블록①(조치 우선순위)·블록②(조치 가능 범위)의 공통 산출.

두 블록 모두 같은 원천에서 나온다 -- 타깃별 파레토 기여율
(CORE_FACTOR_CONTRIBUTION_MIN, 지금은 10%) 이상인 인자를 전부 남기고,
`src/analysis/screening/fmea.py`가 이미 계산하는 권장구간·구간 밖
평균 손실(`defect_rate_deviation_pct`)을 그대로 재사용한다. 다만 FMEA가
eval 데이터셋 기준인 반면 이 블록은 항상 **train.CSV 기준**이다 -- 조치
우선순위는 "이 인자를 이 구간으로 관리하면 얼마나 회수되는가"를 묻는
학습 데이터 기반의 안정적 판단이지, 지금 보고 있는 eval 배치마다
흔들리는 값이어선 안 된다.

회수 폭(recovery_width_pp) = FmeaFactor.defect_rate_deviation_pct
(구간 밖 평균 불량률 − 구간 안 평균 불량률, train 기준).
비중(share_pct) = 이 타깃의 평균 불량률 / 5개 타깃 평균 불량률 합계.
기대 회수(expected_recovery_pp) = 회수 폭 × 비중 / 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.screening.fmea import build_fmea_table
from src.analysis.yield_prediction import FAIL_RATE_TARGETS

# 기대 회수가 이 값 미만이면 "실익 낮음"으로 흐리게 표시한다(숨기지
# 않는다).
MIN_MEANINGFUL_EXPECTED_RECOVERY_PP = 0.1
# 블록② 캡션의 "10%p 늘리면" 가정. 이 값 하나만 바꾸면 캡션 문구와
# 추정치가 함께 움직인다 -- 추정은 부트스트랩이 아니라 "계측된 것 중
# 구간 밖 비율"의 단순 비례 확장이다.
COVERAGE_INCREASE_ASSUMPTION_PP = 10.0
# 이 폭 미만이면(절대값) "회수 폭이 작아서" 실익이 낮다고 설명한다.
# width_pp 자체가 없거나(권장구간 산출 불가) 비중이 낮은 경우와
# 구분되는 세 번째 사유다.
LOW_WIDTH_THRESHOLD_PP = 1.0
LOW_SHARE_THRESHOLD_PCT = 10.0


@dataclass(frozen=True)
class ActionPriorityRow:
    target: str
    feature: str
    relation_shape: str
    factor_value: float | None
    range_lo: float | None
    range_hi: float | None
    measured_count: int
    out_of_range_count: int
    total_wafers: int
    recovery_width_pp: float | None
    share_pct: float
    expected_recovery_pp: float | None
    contribution_pct: float
    dimmed: bool
    dim_reason: str | None


@dataclass(frozen=True)
class NoQualifyingTarget:
    target: str
    max_contribution_pct: float


@dataclass(frozen=True)
class ActionPriorityTable:
    rows: list[ActionPriorityRow]  # 기대 회수(expected_recovery_pp) 내림차순
    no_qualifying_factor: list[NoQualifyingTarget]
    total_wafers: int
    # 계측을 COVERAGE_INCREASE_ASSUMPTION_PP만큼 늘리면 새로
    # 드러날 것으로 추정되는 조치 대상(구간 밖) wafer 수 -- 추적 중인
    # 인자들의 평균 "계측된 것 중 구간 밖 비율"을 새로 계측될 wafer에도
    # 그대로 적용한다고 가정한 비례 추정.
    estimated_additional_action_wafers: int


def _target_loss_shares(train_df: pd.DataFrame) -> dict[str, float]:
    loss_means: dict[str, float] = {}
    for target in FAIL_RATE_TARGETS:
        if target in train_df.columns:
            series = pd.to_numeric(train_df[target], errors="coerce").dropna()
            loss_means[target] = float(series.mean()) if len(series) else 0.0
        else:
            loss_means[target] = 0.0
    total = sum(loss_means.values())
    if total <= 0:
        return {t: 0.0 for t in FAIL_RATE_TARGETS}
    return {t: loss_means[t] / total * 100.0 for t in FAIL_RATE_TARGETS}


def _dim_reason(target: str, width: float | None, share: float) -> str | None:
    if width is None:
        return "권장 구간을 산출할 수 없어 기대 회수를 계산하지 못했습니다."
    if width < LOW_WIDTH_THRESHOLD_PP:
        return "회수 폭이 작아 조치 실익이 낮음"
    if share < LOW_SHARE_THRESHOLD_PCT:
        return f"{target}는 전체 손실의 {share:.1f}%로 비중이 낮음"
    return "실익이 작아 조치 우선순위가 낮음"


def build_action_priority_table(train_df: pd.DataFrame, rows_by_target: dict[str, list[dict]]) -> ActionPriorityTable:
    total_wafers = len(train_df)
    targets = list(rows_by_target.keys())
    table = build_fmea_table(train_df, rows_by_target, targets, dataset_id="train")
    shares = _target_loss_shares(train_df)

    rows: list[ActionPriorityRow] = []
    out_ratios: list[float] = []
    for f in table.items:
        width = f.defect_rate_deviation_pct
        share = shares.get(f.target, 0.0)
        expected = (width * share / 100.0) if width is not None else None
        measured_count = round(f.measurement_rate / 100.0 * total_wafers) if total_wafers else 0
        out_count = round(f.deviation_rate_pct / 100.0 * measured_count) if measured_count else 0
        if measured_count > 0:
            out_ratios.append(out_count / measured_count)
        dimmed = expected is None or expected < MIN_MEANINGFUL_EXPECTED_RECOVERY_PP
        rows.append(
            ActionPriorityRow(
                target=f.target,
                feature=f.feature,
                relation_shape=f.relation_shape,
                factor_value=f.factor_value,
                range_lo=f.range_lo,
                range_hi=f.range_hi,
                measured_count=measured_count,
                out_of_range_count=out_count,
                total_wafers=total_wafers,
                recovery_width_pp=width,
                share_pct=share,
                expected_recovery_pp=expected,
                contribution_pct=f.contribution_pct,
                dimmed=dimmed,
                dim_reason=_dim_reason(f.target, width, share) if dimmed else None,
            )
        )
    rows.sort(key=lambda r: r.expected_recovery_pp if r.expected_recovery_pp is not None else float("-inf"), reverse=True)

    avg_out_ratio = (sum(out_ratios) / len(out_ratios)) if out_ratios else 0.0
    estimated_additional = round(total_wafers * (COVERAGE_INCREASE_ASSUMPTION_PP / 100.0) * avg_out_ratio)

    return ActionPriorityTable(
        rows=rows,
        no_qualifying_factor=[
            NoQualifyingTarget(target=n.target, max_contribution_pct=n.max_contribution_pct) for n in table.no_qualifying_factor
        ],
        total_wafers=total_wafers,
        estimated_additional_action_wafers=estimated_additional,
    )
