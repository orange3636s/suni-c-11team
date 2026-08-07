"""Tests for src/analysis/alarm_gbdt.py -- GBDT 부트스트랩 앙상블 알람 판정
(spec: 알람 판정 GBDT 전환 §A). Synthetic frames only so these always run
in CI regardless of whether data/raw/*.CSV is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.alarm_gbdt import (
    classify_offset,
    classify_wafer,
    compute_holdout_predictions,
    cross_validate_auc,
    cross_validate_transfer_auc,
    fit_bootstrap_ensemble,
    prepare_feature_matrix,
    score_wafers,
)


def _synthetic_df(n: int = 400, n_lots: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r1 = rng.normal(50, 10, n)
    r2 = rng.normal(20, 5, n)
    noise = rng.normal(0, 2, n)
    y = 90 - 0.3 * np.abs(r1 - 50) + 0.1 * r2 + noise
    lot_ids = [f"LOT{i % n_lots:03d}" for i in range(n)]
    return pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"W{i}" for i in range(n)],
            "Lot_ID": lot_ids,
            "Step1_R1": r1,
            "Step2_R1": r2,
            "Y": y,
        }
    )


def test_prepare_feature_matrix_fills_missing_columns_with_nan():
    df = pd.DataFrame({"Step1_R1": [1.0, 2.0, np.nan]})
    x = prepare_feature_matrix(df, ["Step1_R1", "Step9_R1"])
    assert x["Step9_R1"].isna().all()
    assert x["Step1_R1"].tolist() == [1.0, 2.0, None] or x["Step1_R1"].isna().iloc[2]


def test_fit_bootstrap_ensemble_is_deterministic():
    df = _synthetic_df()
    train_df, eval_df = df.iloc[:300], df.iloc[300:]
    features = ["Step1_R1", "Step2_R1"]
    first = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    second = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    np.testing.assert_allclose(first.pred_mean, second.pred_mean)
    np.testing.assert_allclose(first.pred_hi, second.pred_hi)
    assert (first.pred_lo <= first.pred_mean).all()
    assert (first.pred_mean <= first.pred_hi).all()


def test_classify_offset_maps_sensitivity_to_sigma_multiplier():
    """spec 사전 알람 로그 전면 개편 §A-3: s=0(오경보 최소)이 가장 보수적
    (+0.6σ), s=1(미탐 최소)이 가장 민감(-0.2σ)해야 한다."""
    assert classify_offset(0.0) == pytest.approx(0.6)
    assert classify_offset(1.0) == pytest.approx(-0.2)
    assert classify_offset(0.5) == pytest.approx(0.2)


def test_classify_wafer_uses_pred_hi_not_pred_mean():
    """spec §B-1 핵심: 알람 판정은 pred_hi(상한) 기준 -- 예측이 흔들리는
    (구간이 넓은) wafer는 상한이 높아 알람에서 자동 제외되어야 한다."""
    # target=85, sensitivity=0.5 -> off=0.2, sigma=10 이면 심각 임계는
    # 85-6=79, 위험 81, 주의 83.
    grade = classify_wafer(pred_hi=90.0, pred_lo=80.0, target=85.0, sensitivity=0.5, sigma=10.0)
    assert grade is None  # 구간이 목표를 가로지름 -- 판별불가
    assert classify_wafer(pred_hi=78.0, pred_lo=70.0, target=85.0, sensitivity=0.5, sigma=10.0) == "심각"


def test_classify_wafer_five_classes_do_not_overlap():
    """spec §B-1 핵심: 심각 -> 위험 -> 주의 -> 정상 -> 판별불가는 서로
    배타적이다."""
    target, sensitivity, sigma = 85.0, 0.5, 10.0
    off = classify_offset(sensitivity)
    severe_edge = target - (off + 0.4) * sigma
    danger_edge = target - (off + 0.2) * sigma
    caution_edge = target - off * sigma
    assert classify_wafer(pred_hi=severe_edge, pred_lo=severe_edge - 1, target=target, sensitivity=sensitivity, sigma=sigma) == "심각"
    assert classify_wafer(pred_hi=danger_edge, pred_lo=danger_edge - 1, target=target, sensitivity=sensitivity, sigma=sigma) == "위험"
    assert classify_wafer(pred_hi=caution_edge, pred_lo=caution_edge - 1, target=target, sensitivity=sensitivity, sigma=sigma) == "주의"
    assert classify_wafer(pred_hi=target + 5, pred_lo=target + 1, target=target, sensitivity=sensitivity, sigma=sigma) == "정상"
    assert classify_wafer(pred_hi=target + 5, pred_lo=target - 1, target=target, sensitivity=sensitivity, sigma=sigma) is None


def test_classify_wafer_gate_failure_suppresses_alarm_tiers_only():
    """spec §B-4 핵심: 신뢰도 게이트 미달이면 심각/위험/주의는 안 나오지만
    정상/판별불가는 그대로 계산된다."""
    target, sensitivity, sigma = 85.0, 0.5, 10.0
    # 심각 조건을 만족하는 wafer라도 gate_passed=False면 알람이 아니다.
    grade = classify_wafer(pred_hi=50.0, pred_lo=40.0, target=target, sensitivity=sensitivity, sigma=sigma, gate_passed=False)
    assert grade is None
    grade_normal = classify_wafer(pred_hi=90.0, pred_lo=86.0, target=target, sensitivity=sensitivity, sigma=sigma, gate_passed=False)
    assert grade_normal == "정상"


def test_score_wafers_does_not_fix_a_count_and_sums_to_all_wafers():
    """spec §B-1 핵심: 알람 개수를 고정하지 않는다. 다섯 분류(None 포함)의
    합은 언제나 평가 wafer 수와 같다."""
    df = _synthetic_df(n=200, seed=1)
    train_df, eval_df = df.iloc[:150], df.iloc[150:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    sigma = float(train_df["Y"].std())
    scored = score_wafers(eval_df, pred, target=85.0, sensitivity=0.5, sigma=sigma)
    assert len(scored) == len(eval_df)
    assert all(s.grade in ("심각", "위험", "주의", "정상", None) for s in scored)
    assert all(0.0 <= s.risk_percentile <= 100.0 for s in scored)


def test_score_wafers_marks_unmeasured_wafers_as_ungraded():
    """spec §B-2 핵심: measured_ids에 없는 wafer는 measured=False이고
    grade는 항상 None(판별불가-미계측)이어야 한다."""
    df = _synthetic_df(n=100, seed=2)
    train_df, eval_df = df.iloc[:60], df.iloc[60:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    sigma = float(train_df["Y"].std())
    scored = score_wafers(eval_df, pred, target=200.0, sensitivity=1.0, sigma=sigma, measured_ids=set())
    assert all(not s.measured for s in scored)
    assert all(s.grade is None for s in scored)


def test_compute_holdout_predictions_covers_full_train_out_of_fold():
    df = _synthetic_df(n=300, n_lots=60, seed=3)
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    assert len(holdout.actual_y) == len(holdout.pred_point) == len(df)
    assert holdout.residual_std >= 0


def test_compute_holdout_predictions_returns_none_when_too_few_lots():
    df = _synthetic_df(n=50, n_lots=2, seed=2)
    features = ["Step1_R1", "Step2_R1"]
    assert compute_holdout_predictions(df, features, n_splits=5) is None


def test_cross_validate_auc_returns_none_when_too_few_lots():
    df = _synthetic_df(n=50, n_lots=2, seed=2)  # fewer distinct lots than n_splits
    result = cross_validate_auc(df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert result is None


def test_cross_validate_auc_ranks_better_than_chance_on_clear_signal():
    df = _synthetic_df(n=600, n_lots=60, seed=3)
    aucs = cross_validate_auc(df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert aucs is not None
    assert len(aucs) <= 5
    assert all(0.0 <= a <= 1.0 for a in aucs)
    # Step1_R1 is a strong, clean signal for Y by construction -- expect
    # meaningfully-better-than-random ranking on average.
    assert float(np.mean(aucs)) > 0.6


def test_cross_validate_transfer_auc_stays_high_when_eval_shares_train_distribution():
    """알람 신뢰도 게이트 §A-1 -- train과 같은 생성 과정을 따르는 eval이면
    (분포가 같은 정상 조합) 전이 AUC도 self-CV와 비슷하게 높아야 한다."""
    df = _synthetic_df(n=600, n_lots=60, seed=3)
    train_df, eval_df = df.iloc[:400], df.iloc[400:]
    aucs = cross_validate_transfer_auc(train_df, eval_df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert aucs is not None
    assert len(aucs) == 5
    assert float(np.percentile(aucs, 5)) > 0.6


def test_cross_validate_transfer_auc_collapses_when_eval_distribution_shifts():
    """알람 신뢰도 게이트 §A-1 핵심: eval의 Y가 train의 인자와 무관한
    분포로 바뀌면(전형적인 "문제" 데이터셋) 전이 AUC가 무작위 수준으로
    떨어져야 한다 -- 이게 바로 게이트가 잡아야 하는 상황이다."""
    train_df = _synthetic_df(n=600, n_lots=60, seed=3)
    rng = np.random.default_rng(99)
    n = 300
    shifted_eval = pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"SW{i}" for i in range(n)],
            "Lot_ID": [f"SLOT{i % 30:03d}" for i in range(n)],
            "Step1_R1": rng.normal(50, 10, n),
            "Step2_R1": rng.normal(20, 5, n),
            # Y is pure noise, unrelated to the features -- no model can
            # transfer discrimination onto this distribution.
            "Y": rng.normal(60, 15, n),
        }
    )
    aucs = cross_validate_transfer_auc(train_df, shifted_eval, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert aucs is not None
    auc_lo = float(np.percentile(aucs, 5))
    assert auc_lo < 0.65  # falls under the AUC_GATE threshold


def test_cross_validate_transfer_auc_returns_none_when_too_few_lots():
    train_df = _synthetic_df(n=50, n_lots=2, seed=2)
    eval_df = _synthetic_df(n=50, n_lots=10, seed=4)
    result = cross_validate_transfer_auc(train_df, eval_df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert result is None
