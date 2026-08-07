"""Tests for src/analysis/alarm_gbdt.py -- GBDT 부트스트랩 앙상블 알람 판정
(spec: 알람 판정 GBDT 전환 §A). Synthetic frames only so these always run
in CI regardless of whether data/raw/*.CSV is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.alarm_gbdt import (
    BAD_LABEL_QUANTILE,
    compute_grade_thresholds,
    cross_validate_auc,
    cross_validate_transfer_auc,
    fit_bootstrap_ensemble,
    grade_of,
    prepare_feature_matrix,
    score_alarms,
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


def test_grade_of_uses_pred_hi_not_pred_mean():
    """spec §A-2 핵심: 알람 판정은 pred_hi(상한) 기준 -- 예측이 흔들리는
    (구간이 넓은) wafer는 상한이 높아 알람에서 자동 제외되어야 한다."""
    from src.analysis.alarm_gbdt import GradeThresholds

    thresholds = GradeThresholds(severe=80.0, danger=82.0, caution=84.0)
    # pred_hi가 85(주의 기준보다도 높음)면 "심각/위험/주의" 알람이 아니어야
    # 한다 -- 예측이 불안정하다는 뜻이다. ("개선 권고" 등급은 삭제됐다 --
    # spec 알람 신뢰도 게이트 §B-1: 정밀도가 무작위 수준과 다르지 않았다.)
    grade = grade_of(pred_hi=85.0, thresholds=thresholds)
    assert grade is None
    assert grade_of(pred_hi=79.0, thresholds=thresholds) == "심각"


def test_grade_thresholds_use_quantiles_not_std():
    """spec §A-2 핵심: Y 분포가 정규가 아닌 데이터셋(왜도가 큰 분포)에서도
    임계가 음수가 되지 않아야 한다 (표준편차 배수 방식이었다면 무너진다)."""
    skewed_y = pd.Series([*([50.0] * 90), *([1.0] * 5), *([99.0] * 5)])
    df = pd.DataFrame({"Y": skewed_y})
    thresholds = compute_grade_thresholds(df)
    assert thresholds.severe > 0
    assert thresholds.severe <= thresholds.danger <= thresholds.caution


def test_score_alarms_does_not_fix_a_count():
    """spec §A-2 핵심: 알람 개수를 고정하지 않는다 -- 등급 조건을 만족하는
    wafer 수만큼만 나온다."""
    df = _synthetic_df(n=200, seed=1)
    train_df, eval_df = df.iloc[:150], df.iloc[150:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    thresholds = compute_grade_thresholds(train_df)
    alarms = score_alarms(eval_df, pred, thresholds)
    assert len(alarms) <= len(eval_df)
    assert all(a.grade in ("심각", "위험", "주의") for a in alarms)
    assert all(0.0 <= a.risk_percentile <= 100.0 for a in alarms)


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
