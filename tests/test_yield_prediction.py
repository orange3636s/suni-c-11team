"""Tests for src/analysis/yield_prediction.py (VA~VD) -- 인자 폴백,
신뢰도 n/5, 권장사항 두 갈래를 합성 데이터로 검증한다(실제 train/eval
로딩은 route-level 테스트가 다룬다). 파레토 기여율 실측 대조(VA-2)와
1위 계측 분포(VA-3)는 real_data 테스트가 번들 CSV로 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.yield_prediction import CONTRIBUTION_THRESHOLD, build_yield_prediction_table

FAIL_TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")
N = 400


def _synthetic_frames(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """모든 타깃이 단일 인자(Step1_R{k})로 설명되고, train==eval이라
    전부 실측인 기본 시나리오 -- 정렬/기본 신뢰도(실측 override)를
    검증하는 데 쓴다."""
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({"Lot_Wafer_ID": [f"L{i:03d}W01" for i in range(N)], "Lot_ID": [f"L{i:03d}" for i in range(N)]})
    x = rng.normal(loc=50, scale=10, size=N)
    for k, (target, coef) in enumerate(zip(FAIL_TARGETS, (1.0, 0.8, 0.6, 0.4, 0.2)), start=1):
        train[f"Step1_R{k}"] = x + rng.normal(scale=2, size=N)
        train[target] = np.clip(coef * (x - 50) / 10 + 5 + rng.normal(scale=0.5, size=N), 0, 20)
    train["Y"] = 100 - train[list(FAIL_TARGETS)].sum(axis=1)
    eval_df = train.copy()
    return train, eval_df


def _fallback_frame(seed: int = 7) -> pd.DataFrame:
    """Y1 하나에 3개의 후보 인자를 둔다: Step1_R1(rank1, ~60%),
    Step2_R1(rank2, ~40% -- 20% 임계를 넘는 "폴백해도 유효한" 케이스),
    Step3_R1(rank3, ~0% -- 임계 미달, 폴백돼도 계측 추가 대상으로 남는
    케이스). Y2~Y5는 스키마상 후보 인자가 아예 없어(select_top_factors가
    빈 리스트 반환) 이 테스트의 관심사에서 제외된다."""
    rng = np.random.default_rng(seed)
    n = 600
    df = pd.DataFrame({"Lot_Wafer_ID": [f"F{i:04d}W01" for i in range(n)], "Lot_ID": [f"F{i:04d}" for i in range(n)]})
    x = rng.normal(loc=50, scale=10, size=n)
    df["Step1_R1"] = x + rng.normal(scale=1.0, size=n)
    df["Step2_R1"] = 0.6 * x + rng.normal(scale=3.0, size=n)
    df["Step3_R1"] = rng.normal(loc=50, scale=10, size=n)
    df["Y1"] = np.clip(3.0 + 0.02 * (df["Step1_R1"] - 50) ** 2 + 0.15 * (df["Step2_R1"] - 50) + rng.normal(scale=0.5, size=n), 0, 30)
    for target in ("Y2", "Y3", "Y4", "Y5"):
        df[target] = 5.0  # 상수 -- select_top_factors가 후보를 찾지 못한다(분산 0).
    df["Y"] = 100 - df[list(FAIL_TARGETS)].sum(axis=1)
    return df


def test_candidates_sorted_by_y_ascending():
    train_df, eval_df = _synthetic_frames()
    table = build_yield_prediction_table(train_df, eval_df, eval_df)
    ys = [c.y for c in table.candidates]
    assert ys == sorted(ys)


def test_fully_measured_targets_get_max_reliability():
    """VC-1: 실측이 있는 타깃은 계측으로 센다 -- train==eval(전부 실측)이면
    모든 후보가 5/5여야 한다."""
    train_df, eval_df = _synthetic_frames()
    table = build_yield_prediction_table(train_df, eval_df, eval_df)
    assert table.candidates
    assert all(c.reliability.count == 5 for c in table.candidates)
    assert table.unmeasured_wafer_ids == []


def test_wafer_with_zero_reliability_is_excluded_and_counted_separately():
    train_df, eval_df = _synthetic_frames()
    eval_missing = eval_df.copy()
    cols = [*FAIL_TARGETS, "Y", *[f"Step1_R{k}" for k in range(1, 6)]]
    eval_missing.loc[:, cols] = np.nan
    table = build_yield_prediction_table(train_df, eval_missing, train_df)
    assert table.candidates == []
    assert len(table.unmeasured_wafer_ids) == N
    assert table.total_wafers == N


def test_fallback_to_rank2_when_rank1_unmeasured():
    train_df = _fallback_frame()
    eval_df = train_df.copy()
    eval_df.loc[eval_df.index[:200], "Step1_R1"] = np.nan  # rank1 미계측 -> rank2로 폴백
    table = build_yield_prediction_table(train_df, eval_df, train_df)

    fallback_wafer = eval_df.iloc[0]["Lot_Wafer_ID"]
    candidate = next(c for c in table.candidates if c.lot_wafer_id == fallback_wafer)
    y1_cell = candidate.core_factors["Y1"]
    assert y1_cell.rank_used == 2
    assert y1_cell.feature == "Step2_R1"
    # VA-4: 폴백된 인자는 자기 자신의(낮은) 기여율로 표시된다.
    assert y1_cell.contribution_pct is not None and y1_cell.contribution_pct < 60
    # 이 데이터셋에서는 rank2도 20%를 넘으므로 여전히 신뢰도에 카운트된다.
    assert y1_cell.contribution_pct >= CONTRIBUTION_THRESHOLD
    assert ("Y1", "Step2_R1") in candidate.reliability.measured


def test_fallback_below_threshold_counts_as_measurement_gap_not_reliable():
    """VA-3/VC-1/VD-4: Y1 자체가 예측(미실측)이고 rank1·rank2 인자도 모두
    미계측이라 rank3(기여율<20%)로 폴백되면 -- 표시는 rank3로 되지만
    (VA-4), 신뢰도에는 카운트되지 않고(VC-1), 계측 추가 제안은 rank1
    이름(Step1_R1)을 쓴다(VD-4: "폴백으로 하위 인자를 쓰고 있어도 1위가
    없으면 제안 대상")."""
    train_df = _fallback_frame()
    eval_df = train_df.copy()
    eval_df.loc[eval_df.index[:200], ["Step1_R1", "Step2_R1", "Y1"]] = np.nan
    table = build_yield_prediction_table(train_df, eval_df, train_df)

    fallback_wafer = eval_df.iloc[0]["Lot_Wafer_ID"]
    candidate = next(c for c in table.candidates if c.lot_wafer_id == fallback_wafer)
    y1_cell = candidate.core_factors["Y1"]
    assert y1_cell.rank_used == 3
    assert y1_cell.feature == "Step3_R1"
    assert y1_cell.contribution_pct is not None and y1_cell.contribution_pct < CONTRIBUTION_THRESHOLD

    assert ("Y1", "Step1_R1") in candidate.reliability.unmeasured
    assert "Y1" in candidate.recommendation.measurement_gap_targets
    assert "Step1_R1" in candidate.recommendation.text
    assert "Y1" not in candidate.recommendation.adjustable_targets


def test_all_ranks_unmeasured_gives_no_factor():
    """VA-3: "전부 미계측"이면 해당 (웨이퍼, 타깃)의 핵심 인자는 None이다."""
    train_df = _fallback_frame()
    eval_df = train_df.copy()
    eval_df.loc[eval_df.index[:5], ["Step1_R1", "Step2_R1", "Step3_R1", "Y1"]] = np.nan
    table = build_yield_prediction_table(train_df, eval_df, train_df)

    wafer = eval_df.iloc[0]["Lot_Wafer_ID"]
    candidate = next(c for c in table.candidates if c.lot_wafer_id == wafer)
    y1_cell = candidate.core_factors["Y1"]
    assert y1_cell.feature is None
    assert y1_cell.rank_used is None
    assert y1_cell.contribution_pct is None
    assert ("Y1", "Step1_R1") in candidate.reliability.unmeasured


def test_adjustable_recommendation_includes_current_value_and_range():
    train_df = _fallback_frame()
    eval_df = train_df.copy()
    table = build_yield_prediction_table(train_df, eval_df, train_df)
    with_adjustment = [c for c in table.candidates if "Y1" in c.recommendation.adjustable_targets]
    assert with_adjustment
    sample = with_adjustment[0]
    assert "Step1_R1" in sample.recommendation.text or "Step2_R1" in sample.recommendation.text
    # YC-3: 화살표 축약형("Step1_R1 12.3 → 4.0~10.0 · Y1 −2.1%p")으로
    # 바뀌었다 -- 화살표와 %p 부호가 문구에 있는지로 형식을 확인한다.
    assert "→" in sample.recommendation.text
    assert "%p" in sample.recommendation.text


TRAIN_PATH = Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV"
TEST_PATH = Path(__file__).resolve().parents[1] / "data" / "bundled" / "test_remove_y.CSV"


@pytest.mark.skipif(not (TRAIN_PATH.exists() and TEST_PATH.exists()), reason="bundled train/test CSV not present")
def test_real_data_only_rank1_clears_contribution_threshold():
    """VA-2: "기여율 20% 이상인 인자는 타깃당 1위 하나뿐이다" -- 번들
    train.CSV로 재확인한다(재사용한 select_top_factors 계산 그대로)."""
    train_df = pd.read_csv(TRAIN_PATH)
    table = build_yield_prediction_table(train_df, train_df, train_df, dataset_id="train")
    for target in FAIL_TARGETS:
        factor = table.primary_factors[target]
        assert factor is not None
        assert factor.contribution_pct >= CONTRIBUTION_THRESHOLD


@pytest.mark.skipif(not (TRAIN_PATH.exists() and TEST_PATH.exists()), reason="bundled train/test CSV not present")
def test_real_data_fallback_distribution_totals_match_combinations():
    """VA-3: 웨이퍼 x 타깃 조합 수(test.CSV 1,000장 x 5타깃 = 5,000)와
    폴백 순위 분포 합이 일치해야 한다."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    table = build_yield_prediction_table(train_df, test_df, test_df, dataset_id="test")
    assert table.fallback_summary.total_combinations == len(test_df) * len(FAIL_TARGETS)
    assert table.fallback_summary.rank_counts[1] > 0
