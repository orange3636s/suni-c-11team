"""계측 마스크 유틸. `_measured_any_mask`("이 인자들 중 하나라도 계측된
wafer")를 `src/analysis/screening/fmea.py`의 측정/상관 부족 집계가 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _measured_bool_columns(eval_df: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    """`features`별 계측 여부를 순수 numpy 불리언 배열(위치 기반)로 미리
    한 번씩만 계산한다 -- pandas `.loc`/`.reindex`는 라벨 정렬 비용이 붙는다."""
    return {
        f: pd.to_numeric(eval_df[f], errors="coerce").notna().to_numpy()
        for f in features
        if f in eval_df.columns
    }


def _measured_any_mask(eval_df: pd.DataFrame, features: list[str]) -> pd.Series:
    """wafer별로 `features` 중 하나라도 계측되어 있는지."""
    columns = _measured_bool_columns(eval_df, features)
    if not columns:
        return pd.Series(False, index=eval_df.index)
    any_measured = np.zeros(len(eval_df), dtype=bool)
    for values in columns.values():
        any_measured |= values
    return pd.Series(any_measured, index=eval_df.index)
