"""Tests for src/analysis/reliability.py -- 종합 신뢰성 등급
(spec: 알람 판정 GBDT 전환 §E)."""

from __future__ import annotations

from src.analysis.reliability import compute_reliability, deduction_reasons, grade_of_score


def test_high_reliability_dataset_scores_high():
    b = compute_reliability(
        fold_aucs=[0.75, 0.78, 0.80, 0.72, 0.77],
        n_significant_factors=6,
        max_eps2=0.20,
        n_train=8000,
        coverage_pct=70.0,
        bad_sample_size=100,
    )
    assert b.grade == "높음"
    assert b.total_score >= 75


def test_low_reliability_dataset_scores_low():
    """killing_event류: AUC가 무작위 수준, 유의 인자 거의 없음, 설명력 낮음."""
    b = compute_reliability(
        fold_aucs=[0.50, 0.48, 0.52, 0.49, 0.51],
        n_significant_factors=0,
        max_eps2=0.005,
        n_train=10000,
        coverage_pct=5.0,
        bad_sample_size=15,
    )
    assert b.grade == "낮음"
    assert b.total_score <= 44
    assert b.low_holdout_sample is True  # bad_sample_size < 30


def test_auc_score_uses_lower_bound_not_mean():
    """spec §E-1 핵심: 평균이 아니라 5분위(하한)를 쓴다 -- 변동이 큰 폴드
    구성에서는 평균과 하한의 등급 배점이 달라져야 한다."""
    volatile = compute_reliability(
        fold_aucs=[0.95, 0.95, 0.95, 0.95, 0.40],  # mean ~0.84, 5th pct ~ near 0.40
        n_significant_factors=5,
        max_eps2=0.15,
        n_train=5000,
        coverage_pct=60.0,
        bad_sample_size=50,
    )
    stable = compute_reliability(
        fold_aucs=[0.84, 0.84, 0.84, 0.84, 0.84],  # same mean, no volatility
        n_significant_factors=5,
        max_eps2=0.15,
        n_train=5000,
        coverage_pct=60.0,
        bad_sample_size=50,
    )
    assert volatile.auc_score < stable.auc_score


def test_grade_boundaries():
    assert grade_of_score(75) == "높음"
    assert grade_of_score(74) == "보통"
    assert grade_of_score(45) == "보통"
    assert grade_of_score(44) == "낮음"


def test_deduction_reasons_are_generated_from_scores_not_llm():
    b = compute_reliability(
        fold_aucs=[0.50, 0.51, 0.49, 0.52, 0.50],
        n_significant_factors=0,
        max_eps2=0.01,
        n_train=500,
        coverage_pct=10.0,
        bad_sample_size=10,
    )
    reasons = deduction_reasons(b)
    assert len(reasons) > 0
    assert all(isinstance(r, str) and r for r in reasons)


def test_no_auc_data_scores_zero_for_that_metric():
    b = compute_reliability(
        fold_aucs=None,
        n_significant_factors=3,
        max_eps2=0.10,
        n_train=3000,
        coverage_pct=40.0,
        bad_sample_size=50,
    )
    assert b.auc_score == 0
    assert b.auc_lower_bound is None
