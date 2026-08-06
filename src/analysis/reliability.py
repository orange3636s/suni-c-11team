"""종합 신뢰성 등급 (spec 알람 판정 GBDT 전환 §E) -- 5개 지표 100점 만점으로
"이 데이터셋에서 알람을 얼마나 믿을 수 있는가"를 하나의 등급으로 요약한다.

배점과 임계값은 spec이 명시한 그대로: **통계적으로 도출된 값이 아니라
내장 데이터셋에서 등급이 구분되도록 설정한 경험값**이다 (§E-2-1). 화면과
JSON 보고서 양쪽에 이 사실을 명시해야 한다 -- 감춰서 낙관적 지표처럼
보이게 하는 것 자체가 신뢰를 무너뜨린다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIGH_GRADE_MIN_SCORE = 75
MEDIUM_GRADE_MIN_SCORE = 45

GRADE_COLOR = {"높음": "#0D9668", "보통": "#D97706", "낮음": "#DC2626"}

# spec §E-1의 5개 지표 배점표를 그대로 옮긴 것 -- 절대 기준이 아니라
# 경험값이라는 점을 __doc__에 밝혔다.
AUC_BANDS = [(0.70, 40), (0.60, 25), (0.55, 10)]
N_SIG_BANDS = [(5, 25), (3, 18), (1, 10)]
MAX_EPS2_BANDS = [(0.15, 20), (0.05, 12), (0.02, 6)]
N_TRAIN_BANDS = [(5000, 10), (2000, 6)]
N_TRAIN_FLOOR_SCORE = 2
COVERAGE_BANDS = [(60.0, 5), (30.0, 3), (0.0, 1)]

MIN_HOLDOUT_SAMPLE = 30  # spec §A-4: 하위 5% 표본이 이 미만이면 AUC 추정 불안정


def _banded_score(value: float | None, bands: list[tuple[float, int]], floor: int = 0) -> int:
    if value is None:
        return 0
    for threshold, points in bands:
        if value >= threshold:
            return points
    return floor


@dataclass
class ReliabilityBreakdown:
    auc_lower_bound: float | None
    auc_score: int
    n_significant_factors: int
    n_significant_score: int
    max_eps2: float | None
    max_eps2_score: int
    n_train: int
    n_train_score: int
    coverage_pct: float | None
    coverage_score: int
    total_score: int
    grade: str  # "높음" | "보통" | "낮음"
    low_holdout_sample: bool  # spec §A-4: 하위 5% 표본 부족 경고


def grade_of_score(score: int) -> str:
    if score >= HIGH_GRADE_MIN_SCORE:
        return "높음"
    if score >= MEDIUM_GRADE_MIN_SCORE:
        return "보통"
    return "낮음"


def compute_reliability(
    *,
    fold_aucs: list[float] | None,
    n_significant_factors: int,
    max_eps2: float | None,
    n_train: int,
    coverage_pct: float | None,
    bad_sample_size: int,
) -> ReliabilityBreakdown:
    """spec §E-1/§E-2. `fold_aucs`는 alarm_gbdt.cross_validate_auc()의
    결과(5-fold 각각의 AUC) -- 평균이 아니라 **하한(5분위)**을 배점에 쓴다
    (spec §E-1: "AUC는 5-fold GroupKFold 결과의 하한(5분위)을 쓴다").
    """
    auc_lower = float(np.percentile(fold_aucs, 5)) if fold_aucs else None
    auc_score = _banded_score(auc_lower, AUC_BANDS)

    n_sig_score = _banded_score(float(n_significant_factors), N_SIG_BANDS)
    eps2_score = _banded_score(max_eps2, MAX_EPS2_BANDS)
    n_train_score = _banded_score(float(n_train), N_TRAIN_BANDS, floor=N_TRAIN_FLOOR_SCORE)
    coverage_score = _banded_score(coverage_pct, COVERAGE_BANDS)

    total = auc_score + n_sig_score + eps2_score + n_train_score + coverage_score
    return ReliabilityBreakdown(
        auc_lower_bound=auc_lower,
        auc_score=auc_score,
        n_significant_factors=n_significant_factors,
        n_significant_score=n_sig_score,
        max_eps2=max_eps2,
        max_eps2_score=eps2_score,
        n_train=n_train,
        n_train_score=n_train_score,
        coverage_pct=coverage_pct,
        coverage_score=coverage_score,
        total_score=total,
        grade=grade_of_score(total),
        low_holdout_sample=bad_sample_size < MIN_HOLDOUT_SAMPLE,
    )


def deduction_reasons(b: ReliabilityBreakdown) -> list[str]:
    """spec §E-3: "감점 사유는 코드가 생성한다. LLM에 맡기지 마라." """
    reasons: list[str] = []
    if b.auc_score < 40:
        level = "보통 수준입니다" if b.auc_score >= 25 else "낮은 수준입니다"
        reasons.append(f"알람 순위 품질이 {level}.")
    if b.n_significant_score < 25:
        reasons.append(f"통계적으로 유의한 인자가 {b.n_significant_factors}개로 적습니다.")
    if b.max_eps2_score < 20:
        eps2_text = f"{b.max_eps2:.4f}" if b.max_eps2 is not None else "0"
        reasons.append(f"가장 강한 인자의 설명력이 {eps2_text}로 낮습니다.")
    if b.n_train_score < 10:
        reasons.append(f"학습 표본이 {b.n_train:,}행으로 적습니다.")
    if b.coverage_score < 5:
        coverage_text = f"{b.coverage_pct:.1f}%" if b.coverage_pct is not None else "0%"
        reasons.append(f"판정 커버리지가 {coverage_text}로 낮습니다.")
    return reasons
