"""인자 선정 실패 폴백 (spec 알람 판정 GBDT 전환 §D) -- 알람 사유(§A-3)와
경고선(§C)은 "선정 인자"(타깃별 Pareto 1위 인자)를 근거로 삼는데, Y1~Y5가
결측이거나 유의 인자가 아예 없는 데이터셋에서는 그 선정 자체가 실패한다.
GBDT 알람 판정 자체는 Y(최종 수율)만 있으면 이 실패와 무관하게 동작한다
(spec: "인자 선정이 실패해도 알람은 제공된다").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.analysis.screening.schema import ALL_TARGET_COLUMNS, FINAL_YIELD_COLUMN

VALID_ROW_FRACTION = 0.5

FallbackTier = Literal["per_target", "final_yield_only", "unanalyzable"]


@dataclass
class TargetFallbackResult:
    tier: FallbackTier
    targets: list[str]  # tier="per_target"이면 Y1..Y5 중 유효한 것, "final_yield_only"면 ["Y"], "unanalyzable"이면 []
    message: str | None  # tier가 정상(§D-2 1번)이면 None


def _is_majority_valid(series: pd.Series | None, n_rows: int) -> bool:
    if series is None or n_rows == 0:
        return False
    return int(series.notna().sum()) > n_rows * VALID_ROW_FRACTION


def select_analysis_targets(df: pd.DataFrame) -> TargetFallbackResult:
    """spec §D-2 4단계 폴백 중 앞 3단계 (4단계 "분석 불가"는 targets=[]로
    표현된다):

    1) Y1~Y5 유효 값이 절반 이상  -> 타깃별 분석 (현행)
    2) Y1~Y5 결측 과다            -> Y 하나로 분석
    3) Y까지 전부 결측            -> 분석 불가
    """
    n_rows = len(df)
    valid_targets = [
        t for t in ALL_TARGET_COLUMNS if t in df.columns and _is_majority_valid(df[t], n_rows)
    ]
    if valid_targets:
        return TargetFallbackResult(tier="per_target", targets=valid_targets, message=None)

    if FINAL_YIELD_COLUMN in df.columns and _is_majority_valid(df[FINAL_YIELD_COLUMN], n_rows):
        return TargetFallbackResult(
            tier="final_yield_only",
            targets=[FINAL_YIELD_COLUMN],
            message=(
                "불량 유형별 데이터(Y1~Y5)가 없어 최종 수율(Y) 기준으로 분석했습니다.\n"
                "불량 유형별 원인 구분은 제공되지 않습니다."
            ),
        )

    return TargetFallbackResult(
        tier="unanalyzable",
        targets=[],
        message="수율 데이터(Y)가 없어 분석할 수 없습니다.",
    )


@dataclass
class NoSignificantFactorsInfo:
    n_tested: int
    n_fdr_passed: int
    max_eps2: float


def no_significant_factors_message(info: NoSignificantFactorsInfo) -> str:
    """spec §D-3: 유의 인자 0개일 때. GBDT 알람은 이 메시지와 별개로 계속
    동작한다는 점을 반드시 함께 알린다."""
    return (
        "통계적으로 유의한 인자를 찾지 못했습니다.\n\n"
        f"· 검정 {info.n_tested}건 중 FDR 통과 {info.n_fdr_passed}건, 효과 크기 조건 통과 0건\n"
        f"· 최대 설명력 {info.max_eps2:.4f}로 매우 낮습니다\n\n"
        "불량률 변동이 계측 인자로 설명되지 않습니다.\n"
        "수율 예측 기반 알람은 계속 동작합니다."
    )
