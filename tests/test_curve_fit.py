"""Tests for src/analysis/curve_fit.py (VD-3) -- Python 이식이
frontend/lib/defectRateCurve.ts와 같은 채택 규칙(F-검정 p<0.01 AND
2차 계수 c>0)을 지키는지 확인한다."""

from __future__ import annotations

import numpy as np

from src.analysis.curve_fit import evaluate_curve, fit_defect_rate_curve


def test_degenerate_for_small_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10)
    y = rng.normal(size=10)
    fit = fit_defect_rate_curve(x, y)
    assert fit.degree == 1
    assert fit.coeffs[1] == 0.0  # 평평한 평균값 직선


def test_degenerate_for_few_distinct_x():
    rng = np.random.default_rng(0)
    x = rng.choice([1.0, 2.0, 3.0], size=50)
    y = rng.normal(size=50)
    fit = fit_defect_rate_curve(x, y)
    assert fit.degree == 1
    assert fit.coeffs[1] == 0.0


def test_quadratic_adopted_for_clear_u_shape():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, size=300)
    y = 5.0 + 0.01 * (x - 50) ** 2 + rng.normal(scale=0.2, size=300)
    fit = fit_defect_rate_curve(x, y)
    assert fit.degree == 2
    assert fit.coeffs[2] > 0
    # 꼭짓점(x=50) 근처가 최소값이어야 한다.
    assert evaluate_curve(fit, 50.0) < evaluate_curve(fit, 0.0)
    assert evaluate_curve(fit, 50.0) < evaluate_curve(fit, 100.0)


def test_downward_convex_quadratic_rejected():
    """가운데가 최악(c<=0)인 2차는 통계적으로 유의해도 채택하지 않는다 --
    "권장 구간" 개념과 모순되기 때문이다."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 100, size=300)
    y = 5.0 - 0.01 * (x - 50) ** 2 + rng.normal(scale=0.2, size=300)
    fit = fit_defect_rate_curve(x, y)
    assert fit.degree == 1


def test_evaluate_curve_clamps_to_domain():
    rng = np.random.default_rng(3)
    x = rng.uniform(10, 20, size=200)
    y = 2.0 * x + rng.normal(scale=0.1, size=200)
    fit = fit_defect_rate_curve(x, y)
    assert evaluate_curve(fit, 1000.0) == evaluate_curve(fit, fit.domain[1])
    assert evaluate_curve(fit, -1000.0) == evaluate_curve(fit, fit.domain[0])
