"""Tests for src/analysis/alarm_gbdt.py -- 남은 것은 GBDT 특징 준비
유틸(feature_columns/step_of/prepare_feature_matrix)과 예측 구간 conformal
캘리브레이션(compute_holdout_predictions)뿐이다. 옛 알람 등급(심각/위험/
주의)·AUC 신뢰도 게이트·부트스트랩 앙상블은 전부 폐기됐다 -- 알림은
수율 예측 갱신 파이프라인(yield_update_dispatch)이 전담한다.

Synthetic frames only so these always run in CI regardless of whether
data/raw/*.CSV is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.alarm_gbdt import (
    CONFORMAL_TARGET_COVERAGE,
    compute_holdout_predictions,
    prepare_feature_matrix,
    step_of,
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


def test_step_of_parses_step_prefix_and_ignores_others():
    """지시서 작업 2: "Step12_R3" -> 12, 패턴에 안 맞는 인자는 None."""
    assert step_of("Step12_R3") == 12
    assert step_of("Step1_D1") == 1
    assert step_of("Config1") is None


def test_prepare_feature_matrix_masks_features_past_max_step():
    """지시서 작업 2 핵심: max_step보다 뒤 스텝의 인자는 값이 있어도 전부
    NaN으로 가려야 한다. 스텝 패턴이 없는 인자는 절대 가려지지 않는다."""
    df = pd.DataFrame({"Step1_R1": [1.0], "Step9_R1": [2.0], "Config1": [3.0]})
    x = prepare_feature_matrix(df, ["Step1_R1", "Step9_R1", "Config1"], max_step=5)
    assert x["Step1_R1"].iloc[0] == 1.0
    assert x["Step9_R1"].isna().all()
    assert x["Config1"].iloc[0] == 3.0


def test_prepare_feature_matrix_max_step_none_is_unmasked():
    df = pd.DataFrame({"Step1_R1": [1.0], "Step30_R1": [2.0]})
    x = prepare_feature_matrix(df, ["Step1_R1", "Step30_R1"], max_step=None)
    assert x["Step30_R1"].iloc[0] == 2.0


def test_compute_holdout_predictions_covers_full_train_out_of_fold():
    df = _synthetic_df(n=300, n_lots=60, seed=3)
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    assert len(holdout.actual_y) == len(holdout.pred_point) == len(df)
    assert holdout.residual_std >= 0
    assert holdout.n_holdout == len(df)


def test_compute_holdout_predictions_returns_none_when_too_few_lots():
    df = _synthetic_df(n=50, n_lots=2, seed=2)
    features = ["Step1_R1", "Step2_R1"]
    assert compute_holdout_predictions(df, features, n_splits=5) is None


def test_compute_holdout_predictions_conformal_q_is_nonnegative_quantile_of_abs_residual():
    """spec §BA-1 핵심: q는 |실제 - OOF 예측|의 목표 포함률 분위수다 --
    ±1.645*residual_std 근사(정규성 가정)가 아니라 분포 가정 없는
    분위수여야 한다."""
    df = _synthetic_df(n=300, n_lots=60, seed=3)
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    assert holdout.coverage == pytest.approx(CONFORMAL_TARGET_COVERAGE)
    residuals = np.abs(holdout.actual_y - holdout.pred_point)
    assert holdout.conformal_q == pytest.approx(float(np.percentile(residuals, CONFORMAL_TARGET_COVERAGE * 100.0)))
    assert holdout.conformal_q >= 0


def test_compute_holdout_predictions_higher_coverage_yields_wider_margin():
    """spec §BA-3 핵심: 목표 포함률을 올리면(보수적으로) q가 커져야
    한다(더 넓은 구간 -> 알람 감소 방향)."""
    df = _synthetic_df(n=300, n_lots=60, seed=3)
    features = ["Step1_R1", "Step2_R1"]
    q_90 = compute_holdout_predictions(df, features, n_splits=5, coverage=0.90).conformal_q
    q_95 = compute_holdout_predictions(df, features, n_splits=5, coverage=0.95).conformal_q
    assert q_95 >= q_90
