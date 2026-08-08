from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.routes.datasets import get_dataset_registry
from api.schemas.monitoring import ConfigTreemapResponse
from src.analysis.screening.effect_size import eps2_categorical
from src.analysis.screening.schema import FINAL_YIELD_COLUMN, Schema, parse_schema
from src.analysis.screening.selector import DEFAULT_FDR_ALPHA, DEFAULT_MIN_N_CATEGORICAL, benjamini_hochberg
from src.runtime.datasets import DatasetNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


def _dataframe_or_404(dataset_id: str):
    registry = get_dataset_registry()
    try:
        return registry.get_dataframe(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.") from exc


@router.get("/config-treemap", response_model=ConfigTreemapResponse)
def get_config_treemap(dataset: str, step: int) -> dict[str, Any]:
    """Step별 Config(장비 조합) x 최종 수율(Y) 집계 -- 모니터링 홈 트리맵
    전용. 기존 /api/screening/scatter/categorical은 target이 Y1~Y5로
    제한돼 있어(schema.target_cols) 최종 수율 Y를 대상으로 쓸 수 없다 --
    그래서 별도 집계 엔드포인트를 둔다. Config 문자열은 여기서 Model/EQ/
    Chamber로 분해하지 않는다(src/config_parser.py와 같은 원칙) -- 원문
    그대로 돌려주고 프론트가 표시용으로 쪼갠다.
    """
    df = _dataframe_or_404(dataset)
    schema = parse_schema(df)
    config_column = f"Step{step}_Config"
    if config_column not in schema.config_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{config_column}' 인자가 없습니다.")
    if FINAL_YIELD_COLUMN not in df.columns:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="이 데이터셋에는 최종 수율(Y) 컬럼이 없습니다.")

    valid = df[[config_column, FINAL_YIELD_COLUMN]].dropna()
    overall_mean = float(valid[FINAL_YIELD_COLUMN].mean()) if len(valid) > 0 else 0.0

    groups = [
        {
            "config": str(config_value),
            "n": int(len(rows)),
            "mean": float(rows[FINAL_YIELD_COLUMN].mean()),
            "median": float(rows[FINAL_YIELD_COLUMN].median()),
            "p5": float(rows[FINAL_YIELD_COLUMN].quantile(0.05)),
            "p95": float(rows[FINAL_YIELD_COLUMN].quantile(0.95)),
        }
        for config_value, rows in valid.groupby(config_column, sort=True)
    ]

    return {
        "dataset_id": dataset,
        "step": step,
        "overall_mean": overall_mean,
        "significant": _is_config_significant(df, schema, config_column),
        "groups": groups,
    }


def _is_config_significant(df, schema: Schema, config_column: str) -> bool:
    """C-3: 이 스텝의 Config가 최종 수율(Y) 차이를 설명하는지 -- 범주형
    히트맵과 같은 규칙(ANOVA eps2 + BH-FDR)이다. FDR 가족은 이
    데이터셋의 모든 스텝 Config x 최종 수율 검정 전체다 -- 요청받은
    스텝 하나만 검정하면 다중 비교 보정이 무의미해진다.
    """
    p_values: list[float] = []
    tested_columns: list[str] = []
    for column in schema.config_cols:
        if column not in df.columns:
            continue
        result = eps2_categorical(df[column], df[FINAL_YIELD_COLUMN], min_n=DEFAULT_MIN_N_CATEGORICAL)
        if result is None:
            continue
        p_values.append(result.p_value)
        tested_columns.append(column)
    if config_column not in tested_columns:
        return False
    q_values = benjamini_hochberg(p_values)
    return bool(q_values[tested_columns.index(config_column)] < DEFAULT_FDR_ALPHA)
