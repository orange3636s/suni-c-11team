"""Tests for src/analysis/measurement_expansion.py -- the '계측 확대 권고'
카드 (spec: 문구 전수 검토 PART B). Synthetic frames only, so these always
run in CI regardless of whether data/raw/*.CSV is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.measurement_expansion import (
    CV_HIGH_THRESHOLD,
    CV_MODERATE_THRESHOLD,
    RATE_LOW_THRESHOLD,
    _measured_any_mask,
    _recommend,
    _simulate_additional_judged,
    _simulate_single_factor_judged,
    compute_measurement_expansion,
    project_new_factor_discoveries,
)
from src.analysis.alarm_bands import BandStat, WholeWaferBands
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import score_all_factors


def _synthetic_eval_df(n: int = 1000, measured_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Lot_Wafer_ID": [f"W{i}" for i in range(n)]})
    for feature in ("Step1_R1", "Step2_R1", "Step3_R1"):
        values = rng.normal(size=n)
        mask = rng.random(n) < measured_frac
        df[feature] = np.where(mask, values, np.nan)
    return df


def test_recommend_thresholds():
    assert _recommend(0.05, 0.05) == ("+10%p", "계측률이 가장 낮아 판정 공백이 큽니다")
    assert _recommend(0.20, CV_HIGH_THRESHOLD + 0.01) == ("+15%p", "추정이 흔들려 권장구간 신뢰도가 낮습니다")
    assert _recommend(0.20, CV_MODERATE_THRESHOLD + 0.01) == ("+10%p", "추정 안정화로 권장구간이 정밀해집니다")
    assert _recommend(0.20, 0.01) == ("유지", "현재 추정이 충분히 안정적입니다")
    # rate exactly at threshold does not trigger the low-rate branch
    assert _recommend(RATE_LOW_THRESHOLD, 0.01)[0] == "유지"


def test_simulate_additional_judged_is_deterministic():
    df = _synthetic_eval_df()
    original_any = _measured_any_mask(df, ["Step1_R1", "Step2_R1", "Step3_R1"])
    first = _simulate_additional_judged(df, ["Step1_R1", "Step2_R1", "Step3_R1"], original_any)
    second = _simulate_additional_judged(df, ["Step1_R1", "Step2_R1", "Step3_R1"], original_any)
    assert first == second
    assert first > 0


def test_single_factor_simulation_differs_by_unmeasured_pool_size():
    """A single-factor simulation must be scoped to wafers that were
    genuinely blocked (all factors missing) beforehand -- otherwise every
    factor trivially reports the same figure (the pick count itself),
    which was the exact bug this test guards against."""
    df = _synthetic_eval_df(n=1000, measured_frac=0.10, seed=1)
    # a second factor with much higher baseline measurement, so it overlaps
    # differently with the "originally blocked" set than Step1_R1 does.
    rng = np.random.default_rng(2)
    df["Step9_R1"] = np.where(rng.random(len(df)) < 0.90, rng.normal(size=len(df)), np.nan)

    features = ["Step1_R1", "Step9_R1"]
    original_any = _measured_any_mask(df, features)
    gained_rare_factor = _simulate_single_factor_judged(df, "Step1_R1", original_any)
    gained_common_factor = _simulate_single_factor_judged(df, "Step9_R1", original_any)
    # Step9_R1 is already measured for ~90% of wafers, so its small
    # remaining-unmeasured pool is almost entirely wafers where Step1_R1 is
    # ALSO unmeasured (genuinely "all missing") -- nearly every pick counts.
    # Step1_R1's much larger unmeasured pool mostly overlaps with wafers
    # Step9_R1 already measured, so most of its picks were already
    # judgeable and don't count as newly unblocked.
    assert gained_common_factor > gained_rare_factor


def test_project_new_factor_discoveries_no_crash_on_weak_signal():
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {
            "Step1_R1": rng.normal(size=n),
            "Step2_R1": rng.normal(size=n),
            "Y1": rng.normal(size=n),
        }
    )
    schema = parse_schema(df)
    rows_by_target = {"Y1": score_all_factors(df, schema, "Y1")}
    discoveries = project_new_factor_discoveries(rows_by_target)
    assert isinstance(discoveries, list)
    assert len(discoveries) <= 3


def test_compute_measurement_expansion_collapses_when_blocked_share_is_low():
    eval_df = _synthetic_eval_df(n=1000, measured_frac=0.95, seed=4)
    bands = WholeWaferBands(
        out_of_recommended_ids=set(),
        in_recommended_ids=set(),
        alarm=BandStat(0, None),
        out_of_recommended=BandStat(100, 85.0),
        in_recommended=BandStat(900, 90.0),
        unmeasured=BandStat(5, None),
    )
    train_df = eval_df.copy()
    summary = compute_measurement_expansion(
        train_df, eval_df, {}, {}, {}, bands, ["Step1_R1", "Step2_R1", "Step3_R1"], total_wafers=1000
    )
    assert summary.show_full_card is False
    assert summary.action_blocked_wafers == 5
    assert summary.priorities == []


def test_compute_measurement_expansion_shows_full_card_when_blocked_share_is_high():
    eval_df = _synthetic_eval_df(n=1000, measured_frac=0.10, seed=5)
    bands = WholeWaferBands(
        out_of_recommended_ids=set(),
        in_recommended_ids=set(),
        alarm=BandStat(10, 80.0),
        out_of_recommended=BandStat(200, 85.0),
        in_recommended=BandStat(200, 90.0),
        unmeasured=BandStat(590, None),
    )
    train_df = eval_df.copy()
    summary = compute_measurement_expansion(
        train_df, eval_df, {}, {}, {}, bands, ["Step1_R1", "Step2_R1", "Step3_R1"], total_wafers=1000
    )
    assert summary.show_full_card is True
    assert summary.additional_judged >= 0
