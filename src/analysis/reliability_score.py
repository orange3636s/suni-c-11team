"""RC-4/RC-4b: 웨이퍼별 신뢰도 점수(0~100)와 y1~y5 셀 색상 -- "요구 11번"
(알림 목록이 y 합산 기준 순위를 쓴다)이 도입되면서 필요해졌다. 실측이
하나도 없는 wafer는 다섯 모드가 전부 모델 추정값이고, 핵심 인자마저
미계측이면 사실상 평균값의 합이다 -- 그런 y가 "수율 낮은 10개"에 들어가면
근거 없는 알림이 된다.

산식(RC-4): 모드 신뢰_k = 1.0(실측) | 파레토 기여율(예측+인자 계측) |
0.0(예측+인자 미계측). 원값 = Σ 모드 신뢰_k (0~5). 신뢰도 =
round(원값 × 20).

기여율은 히트맵·Pareto가 이미 쓰는 값(selector.py의 contribution_pct)을
그대로 재사용한다 -- 여기서 다시 계산하지 않는다. 손실 기여율(S 점수)과
혼동하지 않도록 별도 가중을 얹지 않는다("하지 말 것": 신뢰도에 손실
기여율 가중을 적용하지 마라).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import ParetoFactor, select_top_factors

FAIL_RATE_TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")

# RC-4b 농도 구간 -- 기여율(0~100 스케일)로 4단계.
SHADE_DARK_MIN = 60.0
SHADE_MEDIUM_MIN = 20.0
SHADE_LIGHT_MIN = 5.0


@dataclass(frozen=True)
class PrimaryFactor:
    feature: str
    contribution_pct: float
    relation_shape: str
    optimal_center: float | None


def primary_factors_by_target(train_df: pd.DataFrame, schema: Schema, targets: tuple[str, ...] = FAIL_RATE_TARGETS) -> dict[str, PrimaryFactor | None]:
    """타깃별 1위(최대 ε²) 인자 -- train.CSV 기준으로 한 번만 계산해
    재사용한다(RC-4/RC-4b 둘 다 이 값을 쓴다). 후보가 아예 없는 타깃은
    None(모든 모드 신뢰가 0.0이 된다, 인자 자체가 없으니 계측 여부를
    따질 수 없다)."""
    result: dict[str, PrimaryFactor | None] = {}
    for target in targets:
        top: list[ParetoFactor] = select_top_factors(train_df, schema, target, limit=1)
        if not top:
            result[target] = None
            continue
        factor = top[0]
        result[target] = PrimaryFactor(
            feature=factor.feature,
            contribution_pct=factor.contribution_pct,
            relation_shape=factor.relation_shape,
            optimal_center=factor.optimal_center,
        )
    return result


def compute_reliability_scores(
    eval_df: pd.DataFrame,
    primary_factors: dict[str, PrimaryFactor | None],
    targets: tuple[str, ...] = FAIL_RATE_TARGETS,
) -> pd.Series:
    """행별 신뢰도(0~100 정수) -- eval_df와 같은 인덱스로 반환한다."""
    raw = pd.Series(0.0, index=eval_df.index)
    for target in targets:
        measured = pd.to_numeric(eval_df[target], errors="coerce").notna() if target in eval_df.columns else pd.Series(False, index=eval_df.index)
        raw = raw + measured.astype(float)

        factor = primary_factors.get(target)
        if factor is None:
            continue
        if factor.feature not in eval_df.columns:
            continue
        factor_measured = pd.to_numeric(eval_df[factor.feature], errors="coerce").notna()
        # 예측인데(미실측) 인자는 계측된 행 -- 기여율만큼만 인정한다.
        credit = (~measured) & factor_measured
        raw = raw + credit.astype(float) * (factor.contribution_pct / 100.0)
    return (raw * 20.0).round().astype(int).clip(0, 100)


def cell_direction(factor: PrimaryFactor | None, factor_value: float | None) -> str | None:
    """RC-4b: 방향(악화/개선) 판정 -- 전체 상관계수가 아니라 형태별
    규칙으로 판정한다("하지 말 것"). U자형은 꼭짓점 대비 웨이퍼 위치로,
    1차 관계는 기울기 부호로 고정 판정한다. 형태가 불분명(unclear)하면
    방향을 매기지 않는다(None -- 회색으로 표시)."""
    if factor is None or factor_value is None or not np.isfinite(factor_value):
        return None
    if factor.relation_shape == "u_shape" and factor.optimal_center is not None:
        return "red" if factor_value > factor.optimal_center else "blue"
    if factor.relation_shape == "monotonic_increasing":
        return "red"
    if factor.relation_shape == "monotonic_decreasing":
        return "blue"
    return None


def shade_bucket(contribution_pct: float) -> str:
    """RC-4b 농도 구간 -- 기여율(0~100)이 클수록 진하다."""
    if contribution_pct >= SHADE_DARK_MIN:
        return "dark"
    if contribution_pct >= SHADE_MEDIUM_MIN:
        return "medium"
    if contribution_pct >= SHADE_LIGHT_MIN:
        return "light"
    return "gray"


def cell_color(
    target: str,
    factor_value: float | None,
    is_measured: bool,
    primary_factors: dict[str, "PrimaryFactor | None"],
) -> dict[str, object]:
    """RC-4b: 한 (wafer, target) 셀의 색상 메타데이터. 실측값은 색을 쓰지
    않는다("하지 말 것: 실측값 셀에 색을 입히지 마라") -- direction=None,
    shade="measured"로 반환해 프런트가 무채색으로 렌더하게 한다.
    factor_value/optimal_center은 툴팁(RC-4b "파레토 기여율 68.2% ·
    꼭짓점 55.8 오른쪽 → 증가 시 악화")을 프런트에서 그대로 조립하는 데
    쓴다."""
    factor = primary_factors.get(target)
    if is_measured:
        return {
            "direction": None, "shade": "measured",
            "feature": factor.feature if factor else None, "contribution_pct": None,
            "factor_value": factor_value, "optimal_center": None,
        }
    if factor is None:
        return {"direction": None, "shade": "gray", "feature": None, "contribution_pct": None, "factor_value": None, "optimal_center": None}
    return {
        "direction": cell_direction(factor, factor_value),
        "shade": shade_bucket(factor.contribution_pct),
        "feature": factor.feature,
        "contribution_pct": factor.contribution_pct,
        "factor_value": factor_value,
        "optimal_center": factor.optimal_center,
    }
