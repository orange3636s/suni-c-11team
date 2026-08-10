from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from api.routes.analysis import _hydrated_targets_or_409
from api.schemas.monitoring import ConfigTreemapResponse
from src.analysis.screening.effect_size import eps2_categorical
from src.analysis.screening.schema import FINAL_YIELD_COLUMN, Schema, parse_schema
from src.analysis.screening.selector import DEFAULT_FDR_ALPHA, DEFAULT_MIN_N_CATEGORICAL, benjamini_hochberg
from src.config_parser import parse_config_hierarchy_value

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/config-treemap", response_model=ConfigTreemapResponse)
def get_config_treemap(dataset: str, step: int = Query(..., ge=1), target: str = "Y1") -> dict[str, Any]:
    """Selected defect rate grouped by the server-owned Config hierarchy."""
    if target not in {"Y", "Y1", "Y2", "Y3", "Y4", "Y5"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="target은 Y1~Y5 중 하나여야 합니다.")

    hydrated = _hydrated_targets_or_409(dataset)
    df = hydrated.dataframe
    schema = parse_schema(df)
    config_column = f"Step{step}_Config"
    if config_column not in schema.config_cols:
        return {
            "dataset_id": dataset,
            "step": step,
            "target": target,
            "target_label": "최종 수율" if target == "Y" else f"{target} 평균 불량률",
            "deprecated_target": target == "Y",
            "overall_mean": 0.0,
            "significant": False,
            "groups": [],
            "empty_reason": f"{config_column} 컬럼이 없어 설비 구성을 집계할 수 없습니다.",
            "target_provenance": hydrated.provenance.as_dict(),
        }

    valid = df[[config_column, target]].dropna()
    overall_mean = float(valid[target].mean()) if len(valid) > 0 else 0.0

    groups = []
    for config_value, rows in valid.groupby(config_column, sort=True):
        hierarchy = parse_config_hierarchy_value(config_column, config_value)
        groups.append(
            {
                "config": str(config_value),
                "model": hierarchy["model"],
                "equipment": hierarchy["equipment"],
                "chamber": hierarchy["chamber"],
                "n": int(len(rows)),
                "mean": float(rows[target].mean()),
                "median": float(rows[target].median()),
                "p5": float(rows[target].quantile(0.05)),
                "p95": float(rows[target].quantile(0.95)),
            }
        )

    return {
        "dataset_id": dataset,
        "step": step,
        "target": target,
        "target_label": "최종 수율" if target == "Y" else f"{target} 평균 불량률",
        "deprecated_target": target == "Y",
        "overall_mean": overall_mean,
        "significant": _is_config_significant(df, schema, config_column, target),
        "groups": groups,
        "empty_reason": None if groups else f"{config_column}에 집계 가능한 값이 없습니다.",
        "target_provenance": hydrated.provenance.as_dict(),
    }


def _is_config_significant(df, schema: Schema, config_column: str, target: str = FINAL_YIELD_COLUMN) -> bool:
    """Whether this Config explains the selected target (ANOVA + BH-FDR).

    The correction family is every Config step against the same target.
    """
    p_values: list[float] = []
    tested_columns: list[str] = []
    for column in schema.config_cols:
        if column not in df.columns:
            continue
        if target not in df.columns:
            continue
        result = eps2_categorical(df[column], df[target], min_n=DEFAULT_MIN_N_CATEGORICAL)
        if result is None:
            continue
        p_values.append(result.p_value)
        tested_columns.append(column)
    if config_column not in tested_columns:
        return False
    q_values = benjamini_hochberg(p_values)
    return bool(q_values[tested_columns.index(config_column)] < DEFAULT_FDR_ALPHA)
