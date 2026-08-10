"""Tests for src/analysis/alerts_ranking.py (RE-1) -- y 오름차순 순위,
top_n 제한, 신뢰도 요약. 합성 데이터로 순수 로직만 검증한다(실제
train/eval 로딩은 route-level 테스트에서 다룬다).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.alerts_ranking import build_alert_ranking

N = 400


def _synthetic_frames(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({"Lot_Wafer_ID": [f"L{i:03d}W01" for i in range(N)], "Lot_ID": [f"L{i:03d}" for i in range(N)]})
    x = rng.normal(loc=50, scale=10, size=N)
    for target, coef in zip(("Y1", "Y2", "Y3", "Y4", "Y5"), (1.0, 0.8, 0.6, 0.4, 0.2)):
        train[f"Step1_R{list('12345').index(target[-1]) + 1}"] = x + rng.normal(scale=2, size=N)
        train[target] = np.clip(coef * (x - 50) / 10 + 5 + rng.normal(scale=0.5, size=N), 0, 20)
    train["Y"] = 100 - train[["Y1", "Y2", "Y3", "Y4", "Y5"]].sum(axis=1)
    eval_df = train.copy()
    return train, eval_df


def test_candidates_sorted_by_y_ascending():
    train_df, eval_df = _synthetic_frames()
    table = build_alert_ranking(train_df, eval_df, eval_df, top_n=10)
    ys = [c.y for c in table.candidates]
    assert ys == sorted(ys)


def test_top_n_limits_candidate_count():
    train_df, eval_df = _synthetic_frames()
    table = build_alert_ranking(train_df, eval_df, eval_df, top_n=5)
    assert len(table.candidates) == 5


def test_fully_measured_candidates_get_reliability_100():
    train_df, eval_df = _synthetic_frames()
    table = build_alert_ranking(train_df, eval_df, eval_df, top_n=10)
    assert all(c.reliability == 100 for c in table.candidates)
    assert table.summary.mean_reliability == 100.0
    assert table.summary.below_threshold_count == 0


def test_predicted_candidates_get_lower_reliability():
    train_df, eval_df = _synthetic_frames()
    eval_missing = eval_df.copy()
    eval_missing.loc[:, ["Y1", "Y2", "Y3", "Y4", "Y5", "Y"]] = np.nan
    # 인자(Step1_R*)까지 전부 미계측이면 신뢰도는 0이어야 한다.
    eval_missing.loc[:, ["Step1_R1", "Step1_R2", "Step1_R3", "Step1_R4", "Step1_R5"]] = np.nan
    table = build_alert_ranking(train_df, eval_missing, train_df, top_n=10)
    assert all(c.reliability == 0 for c in table.candidates)
    assert table.summary.zero_reliability_count == 10


def test_reason_mentions_primary_feature_and_range():
    train_df, eval_df = _synthetic_frames()
    table = build_alert_ranking(train_df, eval_df, eval_df, top_n=1)
    candidate = table.candidates[0]
    assert candidate.primary_feature in candidate.reason
