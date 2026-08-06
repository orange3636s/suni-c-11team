"""Tests for src/analysis/warning_line.py -- PDP 기반 수율 경고선
(spec: 알람 판정 GBDT 전환 §C). Synthetic frames only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.warning_line import (
    build_alarm_reason,
    compute_all_warning_lines,
    compute_warning_line,
    fit_reference_model,
    observed_yield_gap,
)


def _monotonic_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Step1_D1 값이 커질수록 Y(수율)가 뚜렷하게 낮아지는 단조 관계 --
    Step1_D1=14.0 부근에서 경고선이 나와야 하는 spec 예시(Step1_D1 vs Y3,
    경고선 11.8)와 같은 형태."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(2, 20, n)
    noise = rng.normal(0, 0.5, n)
    y = 95 - 0.5 * x + noise
    flat = rng.normal(50, 10, n)  # 두 번째 인자는 Y와 거의 무관 (경고선 없음 기대)
    return pd.DataFrame({"Step1_D1": x, "Step2_R1": flat, "Y": y})


def test_monotonic_factor_gets_one_sided_warning_line():
    df = _monotonic_df()
    features = ["Step1_D1", "Step2_R1"]
    model = fit_reference_model(df, features)
    wl = compute_warning_line(model, df, "Step1_D1", features)
    assert wl is not None
    # 단조 감소(Y가 x에 반비례)이므로 상한만 나오고 하한은 없어야 한다.
    assert wl.upper is not None
    assert wl.lower is None
    assert wl.pdp_range > 0


def test_flat_factor_gets_no_warning_line():
    df = _monotonic_df()
    features = ["Step1_D1", "Step2_R1"]
    model = fit_reference_model(df, features)
    wl = compute_warning_line(model, df, "Step2_R1", features)
    # Step2_R1은 Y와 거의 무관하므로 부분 의존도 변동폭이 작아 임계를
    # 넘지 않아야 한다 (spec: "곡선이 임계에 닿지 않으면 경고선이 없다").
    assert wl is not None
    assert wl.upper is None
    assert wl.lower is None


def test_observed_yield_gap_uses_real_data_not_predictions():
    """spec §C-4-1 핵심: 범례에 쓰는 수율 차이는 예측값이 아니라 관측값이다."""
    df = _monotonic_df(n=1000)
    features = ["Step1_D1", "Step2_R1"]
    model = fit_reference_model(df, features)
    wl = compute_warning_line(model, df, "Step1_D1", features)
    assert wl is not None and wl.upper is not None
    gaps = observed_yield_gap(df, "Step1_D1", wl)
    # 경고선 밖(값이 큼)은 Y가 낮으므로 gap은 음수여야 한다.
    assert gaps["upper_gap"] is not None
    assert gaps["upper_gap"] < 0
    # 직접 pandas로 재계산해 일치하는지 확인 (모델 예측을 쓰지 않았는지 검증).
    out = df["Step1_D1"] >= wl.upper
    expected = df.loc[out, "Y"].mean() - df.loc[~out, "Y"].mean()
    assert gaps["upper_gap"] == expected


def test_observed_yield_gap_omitted_below_min_sample():
    df = _monotonic_df(n=200)
    features = ["Step1_D1"]
    model = fit_reference_model(df, features)
    # A warning line so far out that fewer than 30 rows exceed it.
    from src.analysis.warning_line import WarningLine

    extreme = WarningLine(lower=None, upper=float(df["Step1_D1"].max() + 100), pdp_range=1.0)
    gaps = observed_yield_gap(df, "Step1_D1", extreme)
    assert gaps["upper_gap"] is None


def test_build_alarm_reason_names_the_exceeded_factor():
    df = _monotonic_df()
    features = ["Step1_D1", "Step2_R1"]
    model = fit_reference_model(df, features)
    warning_lines = compute_all_warning_lines(model, df, features)
    assert "Step1_D1" in warning_lines

    row = pd.Series({"Step1_D1": warning_lines["Step1_D1"].upper + 5, "Step2_R1": 50.0})
    reason = build_alarm_reason(row, warning_lines)
    assert "Step1_D1" in reason
    assert "초과" in reason


def test_build_alarm_reason_falls_back_to_combination_message_when_nothing_exceeds():
    df = _monotonic_df()
    features = ["Step1_D1", "Step2_R1"]
    model = fit_reference_model(df, features)
    warning_lines = compute_all_warning_lines(model, df, features)

    row = pd.Series({"Step1_D1": df["Step1_D1"].median(), "Step2_R1": df["Step2_R1"].median()})
    reason = build_alarm_reason(row, warning_lines)
    assert reason == "개별 인자는 정상 범위이나 조합이 위험 패턴에 해당"


def test_build_alarm_reason_picks_largest_overage_and_counts_the_rest():
    from src.analysis.warning_line import WarningLine

    warning_lines = {
        "A": WarningLine(lower=None, upper=10.0, pdp_range=1.0),
        "B": WarningLine(lower=None, upper=10.0, pdp_range=1.0),
    }
    # A overshoots by 10% (11/10), B overshoots by 100% (20/10) -- B should win.
    row = pd.Series({"A": 11.0, "B": 20.0})
    reason = build_alarm_reason(row, warning_lines)
    assert reason.startswith("B")
    assert "외 1건" in reason
