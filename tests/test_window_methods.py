"""Tests for src/analysis/window_methods.py -- the SPC/ML 권장구간 비교
(spec: "SPC / ML 방식 전환").

The golden-value checks skip gracefully when data/raw/train.CSV is absent
(same convention as test_control_range_golden.py); the rest run against
small synthetic frames so they always execute in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.window_methods import compare_methods

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

# spec's worked examples (train.CSV) -- windows should land close to these;
# exact reproduction isn't required (§1-3: "정확히 일치하지 않아도 되나
# 크게 벗어나면 구현을 점검한다"), so tolerance is generous.
GOLDEN_WINDOWS = {
    "Y1": {"feature": "Step28_R1", "spc": (54.7, 61.5), "ml": (55.2, 60.7)},
    "Y2": {"feature": "Step16_R1", "spc": (55.7, 61.7), "ml": (56.1, 62.0)},
    "Y3": {"feature": "Step1_D1", "spc": (3.0, 9.0), "ml": (3.0, 9.0)},
    "Y4": {"feature": "Step24_R1", "spc": (53.2, 61.5), "ml": (54.1, 62.5)},
    "Y5": {"feature": "Step18_R1", "spc": (52.5, 59.7), "ml": (52.9, 58.8)},
}
TOLERANCE = 0.5

pytestmark_golden = pytest.mark.skipif(
    not TRAIN_CSV_PATH.exists(),
    reason="data/raw/train.CSV is not tracked in git; place it locally to run golden checks.",
)


def _control_limits(x: pd.Series) -> tuple[float, float]:
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_csv(TRAIN_CSV_PATH)


@pytest.mark.skipif(not TRAIN_CSV_PATH.exists(), reason="data/raw/train.CSV not present")
@pytest.mark.parametrize("target", list(GOLDEN_WINDOWS))
def test_golden_windows_close_to_spec_examples(train_df, target):
    expected = GOLDEN_WINDOWS[target]
    feature = expected["feature"]
    x = pd.to_numeric(train_df[feature], errors="coerce")
    y = pd.to_numeric(train_df[target], errors="coerce")
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    lcl, ucl = _control_limits(x)

    comparison = compare_methods(x, y, lcl, ucl)
    assert comparison.spc is not None
    assert comparison.ml is not None
    assert comparison.spc.lo == pytest.approx(expected["spc"][0], abs=TOLERANCE)
    assert comparison.spc.hi == pytest.approx(expected["spc"][1], abs=TOLERANCE)
    assert comparison.ml.lo == pytest.approx(expected["ml"][0], abs=TOLERANCE)
    assert comparison.ml.hi == pytest.approx(expected["ml"][1], abs=TOLERANCE)


@pytest.mark.skipif(not TRAIN_CSV_PATH.exists(), reason="data/raw/train.CSV not present")
def test_step1_d1_y3_is_a_genuine_tie_resolved_to_spc(train_df):
    """Step1_D1 (discrete defect count) -- SPC and ML land on the exact
    same window here, so the tie-break rule (spec: "동점이면 SPC를
    채택한다") must pick SPC, not whichever happened to be evaluated
    second.
    """
    x = pd.to_numeric(train_df["Step1_D1"], errors="coerce")
    y = pd.to_numeric(train_df["Y3"], errors="coerce")
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    lcl, ucl = _control_limits(x)

    comparison = compare_methods(x, y, lcl, ucl)
    assert comparison.spc is not None and comparison.ml is not None
    assert (comparison.spc.lo, comparison.spc.hi) == pytest.approx((comparison.ml.lo, comparison.ml.hi), abs=1e-6)
    assert comparison.adopted == "spc"
    assert comparison.adopted_reason == "두 방식이 같은 구간을 산출했습니다"


@pytest.mark.skipif(not TRAIN_CSV_PATH.exists(), reason="data/raw/train.CSV not present")
@pytest.mark.parametrize("target", list(GOLDEN_WINDOWS))
def test_optimal_center_always_within_window_both_methods(train_df, target):
    feature = GOLDEN_WINDOWS[target]["feature"]
    x = pd.to_numeric(train_df[feature], errors="coerce")
    y = pd.to_numeric(train_df[target], errors="coerce")
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    lcl, ucl = _control_limits(x)

    comparison = compare_methods(x, y, lcl, ucl)
    for method in (comparison.spc, comparison.ml):
        assert method is not None
        assert method.lo <= method.center <= method.hi


def _synthetic_u_shape(n: int = 400, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.uniform(0, 100, size=n))
    y = pd.Series(((x - 50) / 10) ** 2 + rng.normal(0, 1, size=n))
    return x, y


def test_ml_random_state_is_deterministic():
    """Same input must always yield the same ML window/center -- spec
    §1-1: "random_state=0 고정. 같은 입력이면 항상 같은 결과가 나와야
    한다"."""
    x, y = _synthetic_u_shape()
    lcl, ucl = _control_limits(x)
    first = compare_methods(x, y, lcl, ucl)
    second = compare_methods(x, y, lcl, ucl)
    assert first.ml is not None and second.ml is not None
    assert (first.ml.lo, first.ml.hi, first.ml.center) == (second.ml.lo, second.ml.hi, second.ml.center)
    assert first.ml.score == second.ml.score


def test_windows_clamped_into_control_range():
    x, y = _synthetic_u_shape(seed=2)
    # Deliberately tight control range so clamping actually engages.
    lcl, ucl = 20.0, 80.0
    comparison = compare_methods(x, y, lcl, ucl)
    for method in (comparison.spc, comparison.ml):
        if method is None:
            continue
        assert method.lo >= lcl - 1e-9
        assert method.hi <= ucl + 1e-9


def test_compare_methods_cache_hits_avoid_recompute():
    """spec §2-5: bootstrap runs once per analysis run and is cached, not
    recomputed on every scatter open."""
    x, y = _synthetic_u_shape(seed=3)
    lcl, ucl = _control_limits(x)
    key = (id(x), "unit-test-feature", "unit-test-target")
    first = compare_methods(x, y, lcl, ucl, cache_key=key)
    second = compare_methods(x, y, lcl, ucl, cache_key=key)
    assert first is second


def test_too_few_rows_for_ml_falls_back_to_spc_only():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.uniform(0, 10, size=10))
    y = pd.Series(rng.uniform(0, 1, size=10))
    lcl, ucl = _control_limits(x)
    comparison = compare_methods(x, y, lcl, ucl)
    assert comparison.ml is None
    assert comparison.adopted == "spc"
