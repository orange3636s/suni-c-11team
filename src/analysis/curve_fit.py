"""불량률 2차 곡선 f(x) -- frontend/lib/defectRateCurve.ts의 Python 이식.

수율 예측 표의 "감소량"(현재 인자값과 권장 구간 중심 사이의 곡선상 차이)은
백엔드에서 응답을 만들 때 이미 계산해 내려보낸다(산점도 차트처럼 프런트가
매번 다시 적합할 필요가 없다) -- 그래서 프런트 전용이던 적합 로직을 여기서
그대로 재현한다. 채택 규칙(F-검정 p<0.01 AND 2차 계수 c>0일 때만 2차 채택,
n<30이거나 x 고유값<5개면 평균값 직선으로 퇴화)을 한 글자도 바꾸지
않는다 -- 두 화면(산점도 오버레이, 수율 예측 감소량)이 같은 데이터를 놓고
다른 곡선을 보여주면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

MIN_N = 30
MIN_DISTINCT_X = 5
QUADRATIC_P_THRESHOLD = 0.01


@dataclass(frozen=True)
class CurveFit:
    degree: int  # 1 | 2
    coeffs: tuple[float, ...]  # (a, b) for degree 1 (y = a + bx), (a, b, c) for degree 2
    r2: float
    domain: tuple[float, float]  # observed x [min, max] -- evaluate_curve clamps to this, never extrapolates


def _degenerate(x: np.ndarray, y: np.ndarray, domain: tuple[float, float]) -> CurveFit:
    mean_y = float(np.mean(y)) if len(y) else 0.0
    return CurveFit(degree=1, coeffs=(mean_y, 0.0), r2=0.0, domain=domain)


def fit_defect_rate_curve(x: np.ndarray, y: np.ndarray) -> CurveFit:
    """OLS-fits a 1st and (if valid) 2nd degree polynomial to (x, y), picks
    the degree via an F-test comparing the nested models -- mirrors
    `frontend/lib/defectRateCurve.ts::fitDefectRateCurve` exactly."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    domain = (float(np.min(x)), float(np.max(x))) if n > 0 else (0.0, 0.0)

    if n < MIN_N or len(np.unique(x)) < MIN_DISTINCT_X:
        return _degenerate(x, y, domain)

    try:
        b_lin, a_lin = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return _degenerate(x, y, domain)

    y_hat_lin = a_lin + b_lin * x
    mean_y = float(np.mean(y))
    rss_linear = float(np.sum((y - y_hat_lin) ** 2))
    tss = float(np.sum((y - mean_y) ** 2))
    r2_linear = 1.0 - rss_linear / tss if tss > 0 else 0.0

    degree = 1
    coeffs: tuple[float, ...] = (a_lin, b_lin)
    r2 = r2_linear

    try:
        c_q, b_q, a_q = np.polyfit(x, y, 2)
    except (np.linalg.LinAlgError, ValueError):
        c_q = b_q = a_q = None

    if c_q is not None:
        y_hat_quad = a_q + b_q * x + c_q * x * x
        rss_quad = float(np.sum((y - y_hat_quad) ** 2))
        r2_quad = 1.0 - rss_quad / tss if tss > 0 else 0.0
        df2 = n - 3
        if df2 > 0:
            if rss_quad <= 1e-12:
                p_value = 0.0
            else:
                f_stat = ((rss_linear - rss_quad) / 1) / (rss_quad / df2)
                p_value = float(stats.f.sf(f_stat, 1, df2)) if f_stat > 0 else 1.0
            # p < 0.01 AND 2차 계수(c) > 0(위로 볼록, U자)일 때만
            # 2차를 채택한다 -- "가운데가 최악"인 c<=0은 버린다.
            if p_value < QUADRATIC_P_THRESHOLD and c_q > 0:
                degree = 2
                coeffs = (a_q, b_q, c_q)
                r2 = r2_quad

    return CurveFit(degree=degree, coeffs=coeffs, r2=r2, domain=domain)


def evaluate_curve(fit: CurveFit, x: float) -> float:
    """Evaluates the fitted curve at `x`, clamped to the observed domain --
    never extrapolates beyond where data was actually seen."""
    lo, hi = fit.domain
    clamped = min(max(x, lo), hi)
    if fit.degree == 2:
        a, b, c = fit.coeffs
        return a + b * clamped + c * clamped * clamped
    a, b = fit.coeffs
    return a + b * clamped
