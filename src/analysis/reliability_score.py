"""y1~y5 셀 색상(방향/농도) 계산. `cell_color`가 (wafer, target) 셀의
방향(악화/개선)과 농도 구간을 반환해, 프런트가 예측값 셀을 무채색이
아닌 색으로 렌더할 수 있게 한다.

기여율은 히트맵·Pareto가 이미 쓰는 값(selector.py의 contribution_pct)을
그대로 재사용한다 -- 여기서 다시 계산하지 않는다.
"""

from __future__ import annotations

import numpy as np

from src.analysis.screening.selector import ParetoFactor

FAIL_RATE_TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")

# 농도 구간 -- 기여율(0~100 스케일)로 4단계.
SHADE_DARK_MIN = 60.0
SHADE_MEDIUM_MIN = 20.0
SHADE_LIGHT_MIN = 5.0


def cell_direction(factor: ParetoFactor | None, factor_value: float | None) -> str | None:
    """방향(악화/개선) 판정 -- 전체 상관계수가 아니라 형태별 규칙으로
    판정한다. U자형은 꼭짓점 대비 웨이퍼 위치로,
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
    """농도 구간 -- 기여율(0~100)이 클수록 진하다."""
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
    primary_factors: dict[str, "ParetoFactor | None"],
) -> dict[str, object]:
    """한 (wafer, target) 셀의 색상 메타데이터. 실측값 셀에는 색을 입히지
    않는다 -- direction=None, shade="measured"로 반환해 프런트가 무채색으로
    렌더하게 한다.
    factor_value/optimal_center은 툴팁("파레토 기여율 68.2% ·
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
