"""Tests for src/analysis/screening/fmea.py -- the FMEA(고장모드영향분석)
분석표 (모니터링 홈, 작업 지시서 WE). Synthetic frames only, so these
always run in CI regardless of whether data/raw/*.CSV is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.screening.fmea import (
    MIN_CONTRIBUTION_PCT,
    _mnar_gap_pp,
    _score_factor,
    build_fmea_table,
)
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import _ranked_rows_with_contribution, _row_to_factor


def _synthetic_df(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Two targets (Y1/Y2), three candidate factors:

    - Step1_R1 -> Y1: real signal, low measurement rate (~15%). Its measured
      values are mostly a tight "core" cluster (stays inside the recommended
      range) plus a small far-outlier "tail" (falls outside the range)
      whose Y1 is deliberately much higher (worse) -- so "out-of-range mean
      Y1" is well above "in-range mean Y1", giving a large positive
      deviation.
    - Step2_R1 -> Y2: well-measured (~90%) but pure noise -- since Y2 has no
      other candidate, this still captures most of the contribution pool
      under the new contribution-based selection rule (WE-2 doesn't gate on
      significance, only on relative contribution share).
    - Step3_Config -> Y1: categorical, fully measured -- must never appear
      in the FMEA table regardless of its score (Config is out of scope).

    Y3/Y4/Y5 are present (constant-ish) so every target in TARGETS resolves,
    matching the shape of a real 5-target dataset.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Lot_Wafer_ID": [f"W{i:04d}" for i in range(n)]})

    measured1 = rng.random(n) < 0.15
    is_tail = measured1 & (rng.random(n) < 0.10)  # ~10% of the measured subset
    is_core = measured1 & ~is_tail

    x1 = np.full(n, np.nan)
    y1 = np.full(n, np.nan)
    # Y1 behaves like the real Y1~Y5 columns: a "불량률" where higher is
    # worse (scatter.py's axis label). The core cluster (inside the
    # recommended range) gets a LOW (good) Y1; the far outlier tail
    # (outside the range) gets a HIGH (bad) Y1 -- so "out-of-range mean −
    # in-range mean" (the benefit of staying in range) comes out positive.
    x1[is_core] = rng.normal(loc=50, scale=5, size=int(is_core.sum()))
    y1[is_core] = rng.normal(loc=10, scale=2, size=int(is_core.sum()))
    x1[is_tail] = rng.normal(loc=200, scale=5, size=int(is_tail.sum()))  # far outside any IQR bound
    y1[is_tail] = rng.normal(loc=40, scale=2, size=int(is_tail.sum()))  # much higher (worse) Y1
    df["Step1_R1"] = x1
    df["Y1"] = np.where(measured1, y1, rng.normal(loc=15, scale=5, size=n))

    x2 = rng.normal(loc=30, scale=5, size=n)
    measured2 = rng.random(n) < 0.9
    df["Step2_R1"] = np.where(measured2, x2, np.nan)
    df["Y2"] = rng.normal(loc=15, scale=3, size=n)  # independent of Step2_R1

    df["Step3_Config"] = rng.choice(["EQ1-CH1", "EQ2-CH2"], size=n)

    df["Y3"] = rng.normal(loc=10, scale=1, size=n)
    df["Y4"] = rng.normal(loc=5, scale=1, size=n)
    df["Y5"] = rng.normal(loc=8, scale=1, size=n)

    # Final yield column, shifted between Step1_R1's measured/unmeasured
    # groups -- exercises the MNAR gap.
    df["Y"] = np.where(measured1, rng.normal(loc=80, scale=2, size=n), rng.normal(loc=90, scale=2, size=n))
    return df


DATASET_ID = "synthetic-test"


def _rows_by_target(df: pd.DataFrame, targets: list[str]) -> dict[str, list[dict]]:
    schema = parse_schema(df)
    return {t: _ranked_rows_with_contribution(df, schema, t, 0.05, 100, 40, 20) for t in targets}


def test_score_factor_excludes_config():
    df = _synthetic_df()
    schema = parse_schema(df)
    rows = _ranked_rows_with_contribution(df, schema, "Y1", 0.05, 100, 40, 20)
    config_row = next(r for r in rows if r["kind"] == "Config")
    factor = _row_to_factor(df, "Y1", config_row)
    assert _score_factor(df, factor, config_row["contribution_pct"], len(df), dataset_id=DATASET_ID) is None


def test_score_factor_populates_contribution_and_deviation_for_real_signal():
    df = _synthetic_df()
    schema = parse_schema(df)
    rows = _ranked_rows_with_contribution(df, schema, "Y1", 0.05, 100, 40, 20)
    row = next(r for r in rows if r["feature"] == "Step1_R1")
    factor = _row_to_factor(df, "Y1", row)
    scored = _score_factor(df, factor, row["contribution_pct"], len(df), dataset_id=DATASET_ID)
    assert scored is not None
    assert scored.contribution_pct == row["contribution_pct"]
    assert 0.0 <= scored.deviation_rate_pct <= 100.0
    # the recommended range genuinely separates Y1 -- expect a real positive gap
    assert scored.defect_rate_deviation_pct is not None
    assert scored.defect_rate_deviation_pct > 0
    # low measurement rate (~15%) -> worst-decile rate should differ from
    # overall (MNAR-style signal), and be a valid percentage.
    assert scored.worst_decile_measurement_rate_pct is not None
    assert 0.0 <= scored.worst_decile_measurement_rate_pct <= 100.0
    # KA-1: expected_defect_rate_pct is target-column basis (~10, the core
    # cluster's Y1 mean) -- expected_yield_pct is a *different* question,
    # the final-Y-column basis (~80, since Y is independent of the x1
    # core/tail split and only depends on measured1).
    assert scored.expected_defect_rate_pct is not None
    assert 5.0 <= scored.expected_defect_rate_pct <= 15.0
    assert scored.expected_yield_pct is not None
    assert 75.0 <= scored.expected_yield_pct <= 85.0


def test_build_fmea_table_selects_by_contribution_threshold_and_excludes_config():
    df = _synthetic_df()
    targets = ["Y1", "Y2", "Y3", "Y4", "Y5"]
    rows_by_target = _rows_by_target(df, targets)

    table = build_fmea_table(df, rows_by_target, targets, dataset_id=DATASET_ID)

    features = [item.feature for item in table.items]
    assert "Step1_R1" in features  # Y1's dominant factor (100% contribution)
    assert "Step3_Config" not in features  # Config never appears regardless of score

    # Every surviving row cleared the WE-2 contribution threshold for its own target.
    for item in table.items:
        assert item.contribution_pct >= MIN_CONTRIBUTION_PCT

    # Sorted by defect-rate deviation descending (WE-4), not by any RPN concept.
    deviations = [item.defect_rate_deviation_pct for item in table.items]
    assert deviations == sorted(deviations, reverse=True)


def test_build_fmea_table_records_no_qualifying_factor_reason():
    """A target whose best candidate never clears the contribution
    threshold gets a reason row instead of a table row."""
    df = _synthetic_df()
    targets = ["Y1", "Y2", "Y3", "Y4", "Y5"]
    rows_by_target = _rows_by_target(df, targets)
    # Force Y3 to have no candidate at all -> its best contribution is 0.
    rows_by_target["Y3"] = []

    table = build_fmea_table(df, rows_by_target, targets, dataset_id=DATASET_ID)

    assert any(n.target == "Y3" and n.max_contribution_pct == 0.0 for n in table.no_qualifying_factor)
    assert all(item.target != "Y3" for item in table.items)


def test_mnar_gap_pp_detects_shifted_final_yield():
    df = _synthetic_df()
    gap = _mnar_gap_pp(df, "Step1_R1")
    assert gap is not None
    # measured group centered at Y=80, unmeasured at Y=90 -> gap is negative
    assert gap < -5.0


def test_mnar_gap_pp_none_when_sample_too_small():
    df = pd.DataFrame(
        {
            "Step1_R1": [1.0] * 5 + [np.nan] * 200,
            "Y": [80.0] * 5 + [90.0] * 200,
        }
    )
    assert _mnar_gap_pp(df, "Step1_R1") is None


def test_score_factor_expected_yield_pct_none_without_final_y_column():
    """KA-1: '진짜 수율' 열은 최종 Y 컬럼이 있을 때만 채워진다 -- 없으면
    조용히 None이지, 불량률 값으로 대체하지 않는다."""
    df = _synthetic_df()
    df = df.drop(columns=["Y"])
    schema = parse_schema(df)
    rows = _ranked_rows_with_contribution(df, schema, "Y1", 0.05, 100, 40, 20)
    row = next(r for r in rows if r["feature"] == "Step1_R1")
    factor = _row_to_factor(df, "Y1", row)
    scored = _score_factor(df, factor, row["contribution_pct"], len(df), dataset_id=DATASET_ID)
    assert scored is not None
    assert scored.expected_yield_pct is None
    # defect-rate fields are unaffected by the missing final-Y column.
    assert scored.expected_defect_rate_pct is not None


def test_build_fmea_table_empty_when_no_targets():
    df = _synthetic_df()
    table = build_fmea_table(df, {}, [], dataset_id=DATASET_ID)
    assert table.items == []
    assert table.no_qualifying_factor == []
    assert table.correlation_shortage_wafers == 0
    # No candidate factor at all -> nobody is measured on "the qualifying
    # set", so every wafer trivially counts as a measurement-shortage wafer.
    assert table.measurement_shortage_wafers == len(df)
