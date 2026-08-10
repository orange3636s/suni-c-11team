"""Tests for src/analysis/distribution_shift.py -- 지시서 작업 4(분포 이동
감지). Synthetic frames only so these always run in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.distribution_shift import (
    MISSING_RATE_GAP_WARNING,
    compute_distribution_shift,
)


def _df(values: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(values)


def test_returns_unknown_when_no_feature_has_enough_samples():
    train_df = _df({"Step1_R1": np.arange(5, dtype=float)})
    eval_df = _df({"Step1_R1": np.arange(5, dtype=float)})
    report = compute_distribution_shift(train_df, eval_df, ["Step1_R1"], min_n=30)
    assert report.level == "unknown"
    assert report.median is None
    assert report.max is None
    assert report.worst_feature is None
    assert report.per_feature == {}


def test_identical_distributions_are_low_shift():
    rng = np.random.default_rng(0)
    values = rng.normal(50, 10, 200)
    train_df = _df({"Step1_R1": values})
    eval_df = _df({"Step1_R1": values.copy()})
    report = compute_distribution_shift(train_df, eval_df, ["Step1_R1"], min_n=30)
    assert report.level == "low"
    assert report.median == pytest.approx(0.0, abs=1e-9)


def test_large_mean_shift_is_flagged_high():
    rng = np.random.default_rng(1)
    train_df = _df({"Step1_R1": rng.normal(50, 5, 200)})
    # eval의 평균이 train 표준편차의 2배 이상 떨어져 있다 -- 명백한 이동.
    eval_df = _df({"Step1_R1": rng.normal(70, 5, 200)})
    report = compute_distribution_shift(train_df, eval_df, ["Step1_R1"], min_n=30)
    assert report.level == "high"
    assert report.worst_feature == "Step1_R1"
    assert report.max > 1.0


def test_zero_variance_train_feature_is_skipped():
    """표준편차가 0이면(train에서 값이 전부 같음) 나눗셈이 정의되지 않아
    그 인자는 계산에서 제외돼야 한다."""
    train_df = _df({"Const": np.full(40, 5.0), "Step1_R1": np.arange(40, dtype=float)})
    eval_df = _df({"Const": np.full(40, 9.0), "Step1_R1": np.arange(40, dtype=float) + 1})
    report = compute_distribution_shift(train_df, eval_df, ["Const", "Step1_R1"], min_n=30)
    assert "Const" not in report.per_feature
    assert "Step1_R1" in report.per_feature


def test_missing_rate_gap_flags_large_measurement_rate_difference():
    n = 100
    rng = np.random.default_rng(2)
    train_col = rng.normal(0, 1, n)
    eval_col = rng.normal(0, 1, n)
    eval_col[: int(n * 0.5)] = np.nan  # eval의 50%만 계측
    train_df = _df({"Step1_D1": train_col})
    eval_df = _df({"Step1_D1": eval_col})
    report = compute_distribution_shift(train_df, eval_df, ["Step1_D1"], min_n=10)
    assert report.missing_rate_worst_feature == "Step1_D1"
    assert report.missing_rate_gap == pytest.approx(0.5, abs=0.05)
    assert report.missing_rate_gap >= MISSING_RATE_GAP_WARNING


def test_missing_columns_are_skipped_without_error():
    train_df = _df({"Step1_R1": np.arange(40, dtype=float)})
    eval_df = _df({"Step1_R1": np.arange(40, dtype=float)})
    report = compute_distribution_shift(train_df, eval_df, ["Step1_R1", "Step2_R1"], min_n=30)
    assert "Step2_R1" not in report.per_feature
    assert report.missing_rate_worst_feature != "Step2_R1"
