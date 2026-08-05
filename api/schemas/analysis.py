from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
    total_factor_count: int
    n80: int | None
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


class BinProfileSchema(BaseModel):
    x_mean: float
    y_mean: float
    y_lo: float
    y_hi: float
    n: int
    x_lo: float
    x_hi: float
    bin_span_ratio: float
    # An outlier-widened bin (its own [x_lo, x_hi] spans an outsized share
    # of the factor's overall range) -- the frontend draws this bin's
    # curve segment dashed rather than solid (spec §3-4).
    sparse: bool


class ScreeningScatterResponse(BaseModel):
    points: list[ScatterPointSchema]
    reference_lines: list[ReferenceLineSchema]
    normal_range: NormalRangeSchema
    bins: list[BinProfileSchema]
    optimal_center: float | None
    # Set only when a classified optimal_center existed but was dropped
    # (fell outside its own recommended window after control-range
    # clamping, spec §3-3) -- the frontend disables the 최적 중심 toggle
    # and shows this as the tooltip reason instead of "단조 관계라...".
    optimal_center_dropped_reason: str | None
    eps2: float
    spearman_r: float | None
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
    out_of_recommended: int
    in_recommended: int
    unmeasured: int


class LotAlarmCount(BaseModel):
    lot_id: str
    alarm_count: int


class BandYieldSchema(BaseModel):
    alarm: float | None
    out_of_recommended: float | None
    in_recommended: float | None
    unmeasured: float | None


class FactorBandPointSchema(BaseModel):
    count: int
    mean_defect_rate: float | None


class FactorBandSchema(BaseModel):
    feature: str
    target: str
    kind: str
    x_min: float
    x_max: float
    lcl: float | None
    ucl: float | None
    recommended_lo: float | None
    recommended_hi: float | None
    out_of_control: FactorBandPointSchema
    out_of_recommended: FactorBandPointSchema
    in_recommended: FactorBandPointSchema


class AlarmSummaryResponse(BaseModel):
    train_dataset_id: str
    eval_dataset_id: str
    total_wafers: int
    measured_wafers: int
    counts: WaferStatusCounts
    band_yield: BandYieldSchema
    top_lots: list[LotAlarmCount] = Field(default_factory=list)
    measurement_bias_p: float | None
    factor_bands: list[FactorBandSchema] = Field(default_factory=list)


class RecommendationItemSchema(BaseModel):
    lot_wafer_id: str
    lot_id: str | None
    step: int
    feature: str
    kind: str
    target: str
    value: float
    recommended_range: list[float]
    direction: str
    expected_improvement_pct: float | None
    tag: str


class RecommendationListResponse(BaseModel):
    train_dataset_id: str
    eval_dataset_id: str
    items: list[RecommendationItemSchema]
    total: int
    excluded_alarm_count: int


class TargetPerformanceSchema(BaseModel):
    target: str
    no_factor_available: bool
    feature: str | None
    kind: str | None
    eps2: float | None
    contribution_pct: float | None
    relation_shape: str | None
    optimal_center: float | None
    p_value: float | None
    confidence_tier: str | None
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


class ReportWindowSchema(BaseModel):
    lo: float
    hi: float
    mean_in_window: float | None
    mean_overall: float
    ratio: float | None
    n_in_window: int


class ReportPerChamberWindowSchema(BaseModel):
    lo: float
    hi: float
    ratio: float | None
    n: int


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
    report_confidence: str
    n_observed: int
    n_missing_pct: float
    relation: ReportRelationSchema
    binned_profile: list[dict[str, float]]
    control_limits: ReportControlLimitsSchema
    band_stability: float
    band_width: float | None
    window: ReportWindowSchema | None
    chamber_interaction: bool
    chamber_interaction_p: float | None
    chamber_interaction_q: float | None
    per_chamber_window: dict[str, ReportPerChamberWindowSchema] | None
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


# ContextFactorSchema mirrors ReportFactorSchema minus `binned_profile` --
# build_chat_context drops that field (chart-plotting data no chatbot
# prompt reads) to make room in the context's size budget for individual
# alarm/recommendation records instead.
class ContextFactorSchema(BaseModel):
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
    report_confidence: str
    n_observed: int
    n_missing_pct: float
    relation: ReportRelationSchema
    control_limits: ReportControlLimitsSchema
    band_stability: float
    band_width: float | None
    window: ReportWindowSchema | None
    chamber_interaction: bool
    chamber_interaction_p: float | None
    chamber_interaction_q: float | None
    per_chamber_window: dict[str, ReportPerChamberWindowSchema] | None
    eval_result: ReportEvalResultSchema


class ContextTargetEntrySchema(BaseModel):
    target: str
    target_stats: ReportTargetStatsSchema
    factors: list[ContextFactorSchema] = Field(default_factory=list)


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


class ReportConfigScreeningSchema(BaseModel):
    n_tested: int
    n_significant_fdr: int
    max_observed_eps2: float | None
    max_observed_feature: str | None
    max_observed_target: str | None
    mde_eps2: float | None
    median_n_per_group: int | None


class AnalysisReportResponse(BaseModel):
    meta: dict[str, Any]
    method: ReportMethodSchema
    summary: ReportSummarySchema
    targets: list[ReportTargetEntrySchema]
    alarms: list[ReportAlarmRecordSchema]
    recommendations: list[RecommendationItemSchema] = Field(default_factory=list)
    config_screening: ReportConfigScreeningSchema
    limitations: list[str]


# ---------------------------------------------------------------------------
# SUNI chatbot context (/api/analysis/context): same report, but `alarms`/
# `recommendations` are grouped into {summary, records[, records_truncated,
# records_total]} instead of a flat list, so the LLM can cite an individual
# wafer's alarm/recommendation record, not just the aggregate counts.
# ---------------------------------------------------------------------------


class ContextAlarmRecordSchema(BaseModel):
    lot_wafer_id: str
    lot_id: str | None
    wafer_slot: int | None
    step: int
    feature: str
    kind: str
    target: str
    value: float
    control_band: list[float | None]
    deviation: float
    direction: str
    severity: str
    actual_y_target: float | None
    actual_y_final: float | None
    config: str | None


class ContextAlarmsSummarySchema(BaseModel):
    n_wafers: int
    n_records: int
    mean_yield_alarm: float | None
    mean_yield_normal: float | None
    normal_wafers: int
    undecidable_wafers: int


class ContextAlarmsSchema(BaseModel):
    summary: ContextAlarmsSummarySchema
    records: list[ContextAlarmRecordSchema]
    records_truncated: bool
    records_total: int


class ContextRecommendationRecordSchema(BaseModel):
    lot_wafer_id: str
    lot_id: str | None
    step: int
    feature: str
    kind: str
    target: str
    value: float
    recommended_range: list[float]
    direction: str
    expected_reduction_pct: float | None
    tag: str


class ContextRecommendationsSummarySchema(BaseModel):
    n_records: int


class ContextRecommendationsSchema(BaseModel):
    summary: ContextRecommendationsSummarySchema
    records: list[ContextRecommendationRecordSchema]
    records_truncated: bool
    records_total: int


class AnalysisContextResponse(BaseModel):
    meta: dict[str, Any]
    method: ReportMethodSchema
    summary: ReportSummarySchema
    targets: list[ContextTargetEntrySchema]
    alarms: ContextAlarmsSchema
    recommendations: ContextRecommendationsSchema
    config_screening: ReportConfigScreeningSchema
    limitations: list[str]
