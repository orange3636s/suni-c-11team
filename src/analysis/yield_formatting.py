"""화면(수율 예측 표)과 발송 메시지가 같은 반올림 규칙을 쓰게
하는 단일 소스 -- 각자 따로 반올림하면 예를 들어 화면은 "85.00%",
메시지는 "85.0%"처럼 어긋나 사용자가 불일치로 인식한다.

수율·불량률   소수 2자리   85.00%
기여율        소수 1자리   82.5%
신뢰도        정수 분수    3/5
감소량        소수 1자리   0.8%p
"""

from __future__ import annotations


def format_yield_pct(value: float) -> str:
    return f"{value:.2f}%"


def format_contribution_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_reliability_fraction(count: int, total: int = 5) -> str:
    return f"{count}/{total}"


def format_decrease_pct(value: float) -> str:
    return f"{value:.1f}%p"


def format_factor_value(value: float) -> str:
    return f"{value:.1f}"
