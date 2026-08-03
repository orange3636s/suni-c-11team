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


class ParetoRankingItemSchema(BaseModel):
    feature: str
    kind: str
    step: int
    eps2: float
    p_value: float
    q_value: float
    significant: bool
    confidence_tier: str
    n_observed: int
    contribution_pct: float
    cumulative_pct: float


class ParetoRankingResponse(BaseModel):
    dataset_id: str
    target: str
    kind: str
    total_factor_count: int
    items: list[ParetoRankingItemSchema]


class ScatterPointSchema(BaseModel):
    x: float
    y: float
    lot_wafer_id: str | None
    lot_id: str | None
    in_range: bool
    config: str | None


class NormalRangeSchema(BaseModel):
    lo: float | None
    hi: float | None
    one_sided: bool
    fallback_applied: bool


class ReferenceLineSchema(BaseModel):
    key: str
    value: float
    drawable: bool
    alarm_relevant: bool
    formula: str
    outside_count: int


class ScreeningScatterResponse(BaseModel):
    points: list[ScatterPointSchema]
    reference_lines: list[ReferenceLineSchema]
    normal_range: NormalRangeSchema
    bins: list[dict[str, float]]
    optimal_center: float | None
    eps2: float
    p_value: float
    q_value: float
    significant: bool
    confidence_tier: str
    n: int
    axis: dict[str, str]


class CategoricalGroupSchema(BaseModel):
    category: str
    n: int
    mean: float
    median: float
    q1: float
    q3: float
    values: list[float]


class CategoricalScatterResponse(BaseModel):
    groups: list[CategoricalGroupSchema]
    eps2: float
    p_value: float
    q_value: float
    significant: bool
    confidence_tier: str
    n: int
    axis: dict[str, str]


class HeatmapScaleSchema(BaseModel):
    min: float
    max: float


class HeatmapResponse(BaseModel):
    dataset_id: str
    metric: str
    kind: str
    features: list[str]
    targets: list[str]
    values: list[list[float | None]]
    n: list[list[int]]
    q: list[list[float | None]]
    significant: list[list[bool]]
    tier: list[list[str | None]]
    scale: HeatmapScaleSchema
    excluded_configs: int


class ControlRangeSchema(BaseModel):
    feature: str
    target: str
    kind: str
    relation_shape: str
    mean: float
    std: float
    q1: float
    q3: float
    lower: float | None
    upper: float | None
    one_sided: bool
    fallback_applied: bool
    band_width: float
    n_observed: int
    reference_lines: list[ReferenceLineSchema] = Field(default_factory=list)


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


class ReportMethodSchema(BaseModel):
    screening: str
    contribution_denominator: str
    control_limit: str
    inclusion_rule: str
    missing_policy: str


class ReportSummarySchema(BaseModel):
    targets_analyzed: int
    factors_included: int
    excluded_low_significance: int
    alarm_wafers: int
    normal_wafers: int
    undecidable_wafers: int
    mean_yield_alarm: float | None
    mean_yield_normal: float | None
    yield_gap_pp: float | None


class ReportRelationSchema(BaseModel):
    shape: str
    optimal_center: float | None
    interpretation: str


class ReportControlLimitsSchema(BaseModel):
    lcl: float | None
    ucl: float | None
    one_sided: bool
    mean: float
    std: float
    q1: float
    q3: float
    sigma3: list[float | None]
    sigma6: list[float | None]
    sigma6_drawn: bool


class ReportEvalResultSchema(BaseModel):
    alarms: int
    observed: int
    mean_y_alarm: float | None
    mean_y_normal: float | None


class ReportFactorSchema(BaseModel):
    feature: str
    kind: str
    step: int
    rank: int
    eps2: float
    contribution_pct: float
    cumulative_pct: float
    spearman_rho: float | None
    p_value: float
    q_value: float
    grade: str
    n_observed: int
    n_missing_pct: float
    relation: ReportRelationSchema
    binned_profile: list[dict[str, float]]
    control_limits: ReportControlLimitsSchema
    eval_result: ReportEvalResultSchema


class ReportTargetStatsSchema(BaseModel):
    mean: float
    std: float
    q1: float
    q3: float


class ReportTargetEntrySchema(BaseModel):
    target: str
    target_stats: ReportTargetStatsSchema
    factors: list[ReportFactorSchema] = Field(default_factory=list)


class ReportAlarmRecordSchema(BaseModel):
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
    actual_y_target: float | None
    actual_y_final: float | None


class AnalysisReportResponse(BaseModel):
    meta: dict[str, Any]
    method: ReportMethodSchema
    summary: ReportSummarySchema
    targets: list[ReportTargetEntrySchema]
    alarms: list[ReportAlarmRecordSchema]
    limitations: list[str]
