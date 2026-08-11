"""Tests for src/analysis/data_limitations.py (작업 지시서 WL) -- MNAR
계측 편향 배수와 랏 분산 분해를 합성 데이터로 검증한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.data_limitations import (
    build_mnar_rate_report,
    compute_mode_variance_share,
    compute_variance_decomposition,
    overall_measurement_rate,
    worst_decile_measurement_rate,
)

BUNDLED_TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV"


def test_worst_decile_measurement_rate_detects_post_hoc_measurement():
    """계측이 최악 10%(가장 나쁜 Y)에 몰리도록 만든 합성 데이터에서, 최악
    10% 계측률이 전체 계측률보다 훨씬 높게 나와야 한다."""
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.normal(loc=85, scale=5, size=n)
    # 최악 20장(Y가 가장 낮다고 가정 -- 여기선 값이 클수록 나쁘다는 실제
    # 스키마와 반대로도 검증되게 낮은 쪽을 "나쁨"으로 뒤집어 만든다.
    worst_mask = y < np.percentile(y, 10)
    measured = np.zeros(n, dtype=bool)
    measured[worst_mask] = rng.random(worst_mask.sum()) < 0.9  # 최악 10%는 90% 계측
    measured[~worst_mask] = rng.random((~worst_mask).sum()) < 0.05  # 나머지는 5%
    df = pd.DataFrame({"feature": np.where(measured, 1.0, np.nan), "Y1": -y})  # Y1: 값이 클수록 나쁘다는 스키마에 맞춰 부호 반전

    overall = overall_measurement_rate(df, "feature")
    worst = worst_decile_measurement_rate(df, "feature", "Y1")
    assert overall is not None and worst is not None
    assert worst > overall * 3  # 뚜렷한 MNAR 신호


def test_worst_decile_measurement_rate_none_when_columns_missing():
    df = pd.DataFrame({"Y1": [1.0, 2.0]})
    assert worst_decile_measurement_rate(df, "missing_feature", "Y1") is None


def test_build_mnar_rate_report_sorted_by_ratio_descending():
    rng = np.random.default_rng(1)
    n = 500
    y1 = rng.normal(size=n)
    df = pd.DataFrame({"Y1": y1})
    # feature_a: 강한 MNAR (최악 10%에서 계측 몰림), feature_b: 거의 무작위.
    # worst_decile_measurement_rate는 값이 클수록 나쁘다고 본다(Y1~Y5
    # 불량률 스키마) -- 상위 10%(가장 큰 y1)를 "최악"으로 잡는다.
    worst_idx = np.argsort(y1)[-(n // 10):]
    a = np.zeros(n)
    a[:] = np.nan
    a[worst_idx] = 1.0
    extra = rng.choice(n, size=int(n * 0.05), replace=False)
    a[extra] = 1.0
    df["feature_a"] = a
    df["feature_b"] = np.where(rng.random(n) < 0.15, 1.0, np.nan)

    report = build_mnar_rate_report(df, [("Y1", "feature_a"), ("Y1", "feature_b")])
    assert [r.feature for r in report] == ["feature_a", "feature_b"]
    assert report[0].ratio > report[1].ratio


def test_compute_variance_decomposition_matches_naive_ratio_and_no_effect_line():
    """between_lot_pct는 편향 보정 없는 단순 var(랏평균)/var(Y)여야 하고,
    no_effect_expected_pct는 1/랏당wafer수와 같아야 한다(WL-2)."""
    rng = np.random.default_rng(2)
    n_lots = 40
    wafers_per_lot = 25
    lot_ids = np.repeat([f"L{i}" for i in range(n_lots)], wafers_per_lot)
    # 랏 효과를 전혀 넣지 않는다 -- 순수 wafer 간 노이즈만.
    y = rng.normal(loc=90, scale=3, size=n_lots * wafers_per_lot)
    df = pd.DataFrame({"Lot_ID": lot_ids, "Y": y})

    result = compute_variance_decomposition(df)
    assert result is not None
    assert result.lot_count == n_lots
    assert result.wafers_per_lot == wafers_per_lot
    assert result.no_effect_expected_pct == 100.0 / wafers_per_lot  # = 4.0

    # 직접 계산한 단순 비율과 정확히 일치해야 한다(반올림 오차만 허용).
    total_var = df["Y"].var(ddof=1)
    lot_means = df.groupby("Lot_ID")["Y"].mean()
    expected_between_pct = lot_means.var(ddof=1) / total_var * 100.0
    assert result.between_lot_pct == pytest.approx(expected_between_pct, rel=1e-6)
    assert result.within_lot_pct == pytest.approx(100.0 - expected_between_pct, rel=1e-6)

    # 랏 효과가 없으므로(순수 노이즈), 관측값이 무효과 기대값과 크게
    # 벗어나지 않아야 한다 -- 느슨한 허용 범위(랏 수가 많지 않아 표본
    # 변동이 있다).
    assert abs(result.between_lot_pct - result.no_effect_expected_pct) < 5.0


def test_compute_variance_decomposition_none_when_too_few_lots():
    df = pd.DataFrame({"Lot_ID": ["L1"] * 5, "Y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert compute_variance_decomposition(df) is None


def _synthetic_mode_share_df(n: int = 500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Y1": rng.normal(loc=2.0, scale=0.5, size=n),
            "Y2": rng.normal(loc=3.5, scale=1.5, size=n),
            "Y3": rng.normal(loc=1.5, scale=0.8, size=n),
            "Y4": rng.normal(loc=2.5, scale=1.0, size=n),
            "Y5": rng.normal(loc=0.5, scale=0.2, size=n),
        }
    )


def test_mode_variance_share_sums_to_100():
    df = _synthetic_mode_share_df()
    rows = compute_mode_variance_share(df)
    assert rows is not None
    assert sum(r.variance_share_pct for r in rows) == pytest.approx(100.0, abs=1e-6)


def test_mode_variance_share_sorted_desc():
    df = _synthetic_mode_share_df()
    rows = compute_mode_variance_share(df)
    assert rows is not None
    shares = [r.variance_share_pct for r in rows]
    assert shares == sorted(shares, reverse=True)


def test_mode_variance_share_none_when_target_missing():
    df = _synthetic_mode_share_df().drop(columns=["Y3"])
    assert compute_mode_variance_share(df) is None


@pytest.mark.skipif(
    not BUNDLED_TRAIN_CSV_PATH.exists(),
    reason="data/bundled/train.CSV not present",
)
def test_mode_variance_share_matches_train_reference():
    df = pd.read_csv(BUNDLED_TRAIN_CSV_PATH)
    rows = compute_mode_variance_share(df)
    assert rows is not None
    shares = {r.target: r.variance_share_pct for r in rows}
    expected = {"Y2": 44.6, "Y3": 22.5, "Y1": 17.7, "Y4": 13.6, "Y5": 1.5}
    for target, expected_pct in expected.items():
        assert shares[target] == pytest.approx(expected_pct, abs=0.1)
    assert [r.target for r in rows] == ["Y2", "Y3", "Y1", "Y4", "Y5"]
