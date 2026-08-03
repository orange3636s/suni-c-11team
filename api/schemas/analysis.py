from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParetoFactorSchema(BaseModel):
    target: str
    feature: str
    kind: str
    step: int
    eps2: float
    p_value: float
    q_value: float
    pearson_r: float | None
    spearman_r: float | None
    n_observed: int
    contribution_pct: float
    cumulative_pct: float
    significant: bool
    relation_shape: str
    optimal_center: float | None


class TargetScreeningResultSchema(BaseModel):
    target: str
    factors: list[ParetoFactorSchema] = Field(default_factory=list)
    reference_only: list[ParetoFactorSchema] = Field(default_factory=list)
    excluded_count: int
    no_significant_factor: bool


class ScreeningResponse(BaseModel):
    dataset_id: str
    targets: list[TargetScreeningResultSchema]
    schema_warnings: list[str] = Field(default_factory=list)


class ScatterPointSchema(BaseModel):
    x: float
    y: float
    lot_wafer_id: str | None
    lot_id: str | None
    in_band: bool
    in_range: bool
    config: str | None


class NormalRangeSchema(BaseModel):
    lo: float | None
    hi: float | None
    one_sided: bool
    fallback_applied: bool


class ScreeningScatterResponse(BaseModel):
    points: list[ScatterPointSchema]
    y_q1: float
    y_q3: float
    band_x_min: float | None
    band_x_max: float | None
    normal_range: NormalRangeSchema
    bins: list[dict[str, float]]
    optimal_center: float | None
    eps2: float
    q_value: float
    n: int
    axis: dict[str, str]


class HeatmapScaleSchema(BaseModel):
    min: float
    max: float


class HeatmapResponse(BaseModel):
    dataset_id: str
    metric: str
    features: list[str]
    targets: list[str]
    values: list[list[float | None]]
    n: list[list[int]]
    q: list[list[float | None]]
    significant: list[list[bool]]
    scale: HeatmapScaleSchema
    excluded_configs: int


class ControlRangeSchema(BaseModel):
    feature: str
    target: str
    kind: str
    relation_shape: str
    y_q1: float
    y_q3: float
    lower: float | None
    upper: float | None
    one_sided: bool
    fallback_applied: bool
    band_in_ratio: float
    n_band: int
    n_observed: int


class ControlRangeListResponse(BaseModel):
    train_dataset_id: str
    items: list[ControlRangeSchema]
    no_significant_factor_targets: list[str] = Field(default_factory=list)


class AlarmItemSchema(BaseModel):
    lot_wafer_id: str
    lot_id: str | None
    wafer_slot: int | None
    step: int
    feature: str
    kind: str
    target: str
    value: float
    normal_range: list[float | None]
    deviation: float
    direction: str
    severity: str
    actual_y: float | None


class AlarmListResponse(BaseModel):
    train_dataset_id: str
    eval_dataset_id: str
    items: list[AlarmItemSchema]
    total: int


class WaferStatusCounts(BaseModel):
    alarm: int
    normal: int
    unmeasured: int


class LotAlarmCount(BaseModel):
    lot_id: str
    alarm_count: int


class AlarmSummaryResponse(BaseModel):
    train_dataset_id: str
    eval_dataset_id: str
    counts: WaferStatusCounts
    alarm_group_yield_avg: float | None
    no_alarm_group_yield_avg: float | None
    yield_gap: float | None
    top_lots: list[LotAlarmCount] = Field(default_factory=list)


class TargetPerformanceSchema(BaseModel):
    target: str
    no_significant_factor: bool
    feature: str | None
    kind: str | None
    eps2: float | None
    relation_shape: str | None
    optimal_center: float | None
    r2: float | None
    rmse: float | None
    mae: float | None
    n: int | None


class ModelPerformanceResponse(BaseModel):
    model_id: str | None
    trained_at: str | None
    source_filename: str | None
    targets: list[TargetPerformanceSchema]
    final_yield: TargetPerformanceSchema | None
