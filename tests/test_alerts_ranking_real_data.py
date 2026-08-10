"""Integration-lite check for src/analysis/alerts_ranking.py against the
real bundled train.CSV (fully measured, so no model/store is needed --
this validates the pipeline against real column shapes/Config columns/
step counts, complementing test_alerts_ranking.py's synthetic-data unit
tests of the formula itself).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.alerts_ranking import build_alert_ranking

TRAIN_PATH = Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV"


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="data/bundled/train.CSV not present")
def test_build_alert_ranking_against_real_train_csv():
    train_df = pd.read_csv(TRAIN_PATH)
    table = build_alert_ranking(train_df, train_df, train_df, top_n=10)

    assert len(table.candidates) == 10
    ys = [c.y for c in table.candidates]
    assert ys == sorted(ys)
    # train.CSV is fully measured -- every candidate should score 100.
    assert all(c.reliability == 100 for c in table.candidates)
    assert table.summary.mean_reliability == 100.0
    assert table.total_wafers == len(train_df)
    for candidate in table.candidates:
        assert candidate.primary_target in {"Y1", "Y2", "Y3", "Y4", "Y5"}
        assert candidate.lot_wafer_id
