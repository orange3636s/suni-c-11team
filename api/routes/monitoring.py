from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from api.routes.analysis import _hydrated_targets_or_409
from api.schemas.monitoring import ConfigTreemapResponse
from api.settings import settings
from src.analysis.sampling import ANALYSIS_SAMPLE_MAX_ROWS, stratified_sample
from src.analysis.screening.effect_size import adj_r2_categorical
from src.analysis.screening.schema import FINAL_YIELD_COLUMN, Schema, parse_schema
from src.analysis.screening.selector import DEFAULT_FDR_ALPHA, DEFAULT_MIN_N_CATEGORICAL, benjamini_hochberg
from src.config_parser import parse_config_hierarchy_value
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/config-treemap", response_model=ConfigTreemapResponse)
def get_config_treemap(dataset: str, step: int = Query(..., ge=1), target: str = "Y1") -> dict[str, Any]:
    """Selected defect rate grouped by the server-owned Config hierarchy."""
    if target not in {"Y", "Y1", "Y2", "Y3", "Y4", "Y5"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="target은 Y1~Y5 중 하나여야 합니다.")

    hydrated = _hydrated_targets_or_409(dataset)
    # 그룹 평균/분위수만 필요하므로 20,000행 초과 데이터셋은 로트
    # 단위 표본을 쓴다 -- schema 파싱은 컬럼 이름만 보므로 표본 여부와
    # 무관하다(전체 df로 그대로 판단).
    schema = parse_schema(hydrated.dataframe)
    df, sample_info = stratified_sample(
        hydrated.dataframe, max_rows=ANALYSIS_SAMPLE_MAX_ROWS, dataset_version=hydrated.provenance.dataset_version
    )
    sample_info_dict = sample_info.as_dict() if sample_info.is_sampled else None
    analysis_id = _current_analysis_id()
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
            "analysis_id": analysis_id,
            "sample_info": sample_info_dict,
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
        "analysis_id": analysis_id,
        "sample_info": sample_info_dict,
    }


def _current_analysis_id() -> str | None:
    """모델 분석 파이프라인이 마지막으로 저장한 스냅샷의
    analysis_id -- Config별 트리맵은 그 자체를 캐시하지 않고 항상
    즉석 계산하지만, 이 값을 함께 실어 보내면 프런트가 모니터링/원인
    분석/수율 예측과 같은 분석 회차를 보고 있는지 표시할 수 있다."""
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    snapshot = store.get_refresh_snapshot_status()["snapshot"]
    return (snapshot or {}).get("analysis_id")


def _is_config_significant(df, schema: Schema, config_column: str, target: str = FINAL_YIELD_COLUMN) -> bool:
    """Whether this Config explains the selected target (dummy-regression
    Adjusted R²'s one-way-ANOVA F-test + BH-FDR).

    Only the significance badge comes from here -- each tile's *color* is a
    separate mean-based calculation on the frontend (`colorForMean`) and is
    unaffected by which effect-size statistic backs this test.

    The correction family is every Config step against the same target.
    """
    p_values: list[float] = []
    tested_columns: list[str] = []
    for column in schema.config_cols:
        if column not in df.columns:
            continue
        if target not in df.columns:
            continue
        result = adj_r2_categorical(df[column], df[target], min_n=DEFAULT_MIN_N_CATEGORICAL)
        if result is None:
            continue
        p_values.append(result.p_value)
        tested_columns.append(column)
    if config_column not in tested_columns:
        return False
    q_values = benjamini_hochberg(p_values)
    return bool(q_values[tested_columns.index(config_column)] < DEFAULT_FDR_ALPHA)
