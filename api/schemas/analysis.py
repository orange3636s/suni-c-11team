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
    # 전체 후보 풀(top-5로 잘리기 전) 기준 집계 -- 차트 표시 규칙(spec §B)의
    # 0개-타깃 안내 문구("검정 58건 · FDR 통과 0건 · 효과 크기 조건 통과 0건")가 쓴다.
    fdr_pass_count: int
    effect_size_pass_count: int
    max_eps2: float | None
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
    # key가 "warning_lo"/"warning_hi"일 때만 채워진다 (spec 알람 판정 GBDT
    # 전환 §C-4-1) -- 범례에 쓰는 실측 수율 차이(%p), 표본 30장 미만이면
    # None.
    observed_yield_gap_pp: float | None = None


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


class MethodWindowSchema(BaseModel):
    window: list[float]  # [lo, hi]
    optimal_center: float
    recall: float
    precision: float
    f2: float
    width_sd: float
    stability: float
    score: float
    clamped: bool


class MethodComparisonSchema(BaseModel):
    spc: MethodWindowSchema | None
    ml: MethodWindowSchema | None
    # "spc" | "ml" -- which method's window backs `normal_range`-adjacent
    # consumers everywhere else in the app (alarm log, 개선 권장 목록); the
    # SPC/ML toggle only changes which method is *displayed*, never this.
    adopted: str
    adopted_reason: str


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
    relation_shape: str
    n: int
    axis: dict[str, str]
    # SPC vs ML 권장구간 비교 (spec: "SPC/ML 방식 전환") -- None for Config
    # factors, which have no numeric x to fit either method on.
    methods: MethodComparisonSchema | None


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
    """알람 판정 GBDT 전환 (spec §A-3) -- 관리한계 이탈량이 아니라 부트스트랩
    앙상블 예측 수율의 신뢰구간 상한(pred_hi) 기준. 예측 수율 절대값과
    신뢰구간은 화면에 노출하지 않는다(spec §A-3: 오차가 Y 표준편차의
    72~80%라 잘못된 확신을 준다) -- `risk_percentile`(순위)만 내보낸다.
    """

    lot_wafer_id: str
    lot_id: str | None
    grade: str  # "심각" | "위험" | "주의" | "개선 권고"
    risk_percentile: float  # 0-100, 낮을수록 위험
    reason: str


class AlarmListResponse(BaseModel):
    train_dataset_id: str
    eval_dataset_id: str
    items: list[AlarmItemSchema]
    total: int
    # 심각+위험+주의 합계 (개선 권고 제외) -- spec §A-2 "알람이 평가 대상의
    # 10%를 넘을 때 경고" 판단에 쓴다.
    alarm_total: int
    improvement_total: int
    evaluated_total: int
    alarm_share_warning: bool


class WaferStatusCounts(BaseModel):
    alarm: int
    # 필드명은 그대로 두지만(하위 호환), 이제 각각 개선 권고/정상 구간의
    # 개수다 (spec §E-2: 예측 수율 GBDT 등급 기준으로 재정의).
    out_of_recommended: int
    in_recommended: int
    unmeasured: int


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
    # 강함/보통 (spec §E-2: 드롭다운에 강함·보통 등급 인자 전부) -- 화면
    # 배지 표시용.
    confidence_tier: str
    x_min: float
    x_max: float
    lcl: float | None
    ucl: float | None
    recommended_lo: float | None
    recommended_hi: float | None
    out_of_control: FactorBandPointSchema
    out_of_recommended: FactorBandPointSchema
    in_recommended: FactorBandPointSchema


class MeasurementBiasSummary(BaseModel):
    tested_count: int
    significant_count: int
    # "low" | "high" | "mixed" -- only meaningful when significant_count > 0
    direction: str | None


class AlarmSummaryResponse(BaseModel):
    train_dataset_id: str
    eval_dataset_id: str
    total_wafers: int
    measured_wafers: int
    counts: WaferStatusCounts
    band_yield: BandYieldSchema
    measurement_bias: MeasurementBiasSummary | None
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


class ReportWarningLineSchema(BaseModel):
    """알람 판정 GBDT 전환 §C-4 -- 화면에는 위치만 쓰고(경고선), 곡선/PDP
    수치는 표시하지 않는다. JSON 보고서에는 재현성 확인용으로 남긴다."""

    value: float
    lower: float | None
    upper: float | None
    method: str
    pdp_range: float
    observed_yield_gap: float | None


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
    warning_line: ReportWarningLineSchema | None = None
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
    warning_line: ReportWarningLineSchema | None = None
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


class FactorPrioritySchema(BaseModel):
    feature: str
    target: str
    measurement_rate: float
    recommendation: str  # "+10%p" | "+15%p" | "유지"
    reason: str
    additional_judged: int
    yield_contribution_pp: float | None


class NewFactorDiscoverySchema(BaseModel):
    feature: str
    target: str
    kind: str


class MeasurementExpansionResponse(BaseModel):
    train_dataset_id: str
    eval_dataset_id: str
    action_blocked_wafers: int
    total_wafers: int
    additional_judged: int
    action_target: int
    expected_yield_gain_pp: float | None
    show_full_card: bool
    priorities: list[FactorPrioritySchema] = Field(default_factory=list)
    new_factor_discoveries: list[NewFactorDiscoverySchema] = Field(default_factory=list)


RELIABILITY_THRESHOLDS_DISCLAIMER = (
    "등급 기준은 내장 데이터셋에서 구분이 되도록 설정한 경험값이며 절대 기준이 아닙니다."
)


class ReliabilityResponse(BaseModel):
    """종합 신뢰성 등급 (spec 알람 판정 GBDT 전환 §E)."""

    dataset_id: str
    grade: str  # "높음" | "보통" | "낮음"
    total_score: int
    auc_lower_bound: float | None
    auc_score: int
    n_significant_factors: int
    n_significant_score: int
    max_eps2: float | None
    max_eps2_score: int
    n_train: int
    n_train_score: int
    coverage_pct: float | None
    coverage_score: int
    deduction_reasons: list[str] = Field(default_factory=list)
    low_holdout_sample: bool
    thresholds_disclaimer: str = RELIABILITY_THRESHOLDS_DISCLAIMER
    target_fallback_tier: str  # "per_target" | "final_yield_only" | "unanalyzable"
    target_fallback_message: str | None


class PreprocessingModeResultSchema(BaseModel):
    mode: str  # "A" | "B" | "C"
    label: str
    r2: float
    adopted: bool


class PreprocessingComparisonResponse(BaseModel):
    """전처리 방식 A/B/C 실시간 비교 (spec 설정 패널 신설 §E) -- 데이터셋마다
    재계산된다. 실제 파이프라인이 채택하는 방식(B)과 이 표의 1위가 다를 수
    있다 (§E-5-1)."""

    dataset_id: str
    dataset_label: str
    results: list[PreprocessingModeResultSchema] = Field(default_factory=list)
    winner: str
    b_equals_c: bool
    holdout_note: str
    winner_note: str | None
