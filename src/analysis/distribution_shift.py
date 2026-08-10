"""지시서 작업 4(분포 이동 감지) -- 평가(eval) 데이터에는 실측 Y가 없는
것이 정상이다(수율 측정 전 예측이 목적이므로). 그래서 eval 자체의 실제
AUC로 신뢰도 게이트를 다시 세울 수 없다 -- 현재 게이트가 쓰는 train OOF
AUC는 eval에서의 실제 성능을 예측하지 못하는 사례가 실측으로 확인됐다
(같은 조합이 실제 test에서는 게이트를 통과하고도 성능이 무너지거나,
반대로 게이트에 걸리고도 실제로는 잘 통하는 경우가 있었다).

이 모듈은 그 공백을 메우는 게 아니라 다른 각도의 신호 하나를 더한다:
train과 eval의 인자 분포가 얼마나 다른가(공변량 이동, covariate shift).
AUC 게이트를 대체하지 않는다 -- 경고 힌트일 뿐이며, 최종 판단은 사용자가
한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ShiftLevel = Literal["low", "medium", "high", "unknown"]

# 통계적으로 도출된 값이 아니다 -- reliability.py의 배점표들과 같은 원칙
# (경험값이며 절대 기준이 아니다). 인자별 |평균 차이| / train 표준편차의
# 중앙값·최댓값을 기준으로 3단계로 나눈다.
LEVEL_MEDIAN_HIGH = 0.3
LEVEL_MEDIAN_MEDIUM = 0.15
LEVEL_MAX_HIGH = 1.0

# train과 eval 사이 결측률(notna 비율) 차이가 이 값 이상이면 별도로
# 알려준다 -- 계측률 자체가 다른 공정 조합(예: D 컬럼 계측률이 크게
# 다른 라인)을 분포 이동과 별개로 짚어내기 위함이다.
MISSING_RATE_GAP_WARNING = 0.2


@dataclass
class DistributionShiftReport:
    median: float | None
    max: float | None
    worst_feature: str | None
    level: ShiftLevel
    per_feature: dict[str, float]
    # train/eval 인자별 결측률 차이 -- 표본 부족(min_n 미만)으로 위 분포
    # 이동 계산에서 아예 제외된 인자도 여기서는 계측률 차이만으로 잡힐 수
    # 있다(계측이 거의 없는 인자일수록 오히려 계측률 차이가 신호가 된다).
    missing_rate_gap: float | None
    missing_rate_worst_feature: str | None


def compute_distribution_shift(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
    *,
    min_n: int = 30,
) -> DistributionShiftReport:
    """train 대비 eval의 인자 분포 이동. 표준화 평균 차 기준(|평균차| / train
    표준편차) -- 인자마다 단위·스케일이 달라 원시 평균차로는 서로 비교할
    수 없다.

    실측 Y가 없는 평가 데이터에서 "이 모델을 이 데이터에 써도 되는가"를
    판단하는 대리 지표다 -- AUC 게이트를 대체하지 않고 함께 표시한다.
    """
    shifts: dict[str, float] = {}
    for f in features:
        if f not in train_df.columns or f not in eval_df.columns:
            continue
        a = pd.to_numeric(train_df[f], errors="coerce").dropna()
        b = pd.to_numeric(eval_df[f], errors="coerce").dropna()
        if len(a) < min_n or len(b) < min_n:
            continue
        sd = a.std()
        if sd <= 0:
            continue
        shifts[f] = float(abs(a.mean() - b.mean()) / sd)

    missing_gaps: dict[str, float] = {}
    for f in features:
        if f not in train_df.columns or f not in eval_df.columns:
            continue
        train_rate = pd.to_numeric(train_df[f], errors="coerce").notna().mean()
        eval_rate = pd.to_numeric(eval_df[f], errors="coerce").notna().mean()
        missing_gaps[f] = float(abs(train_rate - eval_rate))
    missing_worst = max(missing_gaps, key=missing_gaps.get) if missing_gaps else None
    missing_gap = missing_gaps[missing_worst] if missing_worst is not None else None

    if not shifts:
        return DistributionShiftReport(
            median=None, max=None, worst_feature=None, level="unknown", per_feature={},
            missing_rate_gap=missing_gap, missing_rate_worst_feature=missing_worst,
        )

    med = float(np.median(list(shifts.values())))
    worst = max(shifts, key=shifts.get)
    level: ShiftLevel = (
        "high" if med > LEVEL_MEDIAN_HIGH or shifts[worst] > LEVEL_MAX_HIGH
        else ("medium" if med > LEVEL_MEDIAN_MEDIUM else "low")
    )
    return DistributionShiftReport(
        median=med,
        max=shifts[worst],
        worst_feature=worst,
        level=level,
        per_feature=shifts,
        missing_rate_gap=missing_gap,
        missing_rate_worst_feature=missing_worst,
    )
