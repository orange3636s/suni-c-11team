"""MA-3: '계측 확대 권고' 카드(요약 카드 3개 + 인자별 우선순위 표 +
새 인자 발견 카드)는 모니터링 홈 재설계로 화면에서 사라졌다 -- 그
카드 하나만을 위한 시뮬레이션(`compute_measurement_expansion` 등)도
함께 지웠다. `_measured_any_mask`만 `src/analysis/screening/fmea.py`가
여전히 쓴다("이 인자들 중 하나라도 계측된 wafer" 마스크는 FMEA의
측정/상관 부족 집계에도 필요한 범용 유틸이라 남긴다).
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
