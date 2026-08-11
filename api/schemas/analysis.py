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
    # QA-2: 하한(30) 이상이지만 종류별 정상 판정 임계 미만 -- 배제 대신
    # "표본 부족" 배지로 표시하고 confidence_tier는 이미 한 단계 낮춰
    # 내려온다.
    under_sampled: bool = False


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
    analyzable_target_samples: int = 0
    model_available: bool = False
    factor_measurement_insufficient: bool = False
    target_provenance: dict[str, Any] | None = None


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
    under_sampled: bool = False
    relation_shape: str
    n: int
    axis: dict[str, str]
    # SPC vs ML 권장구간 비교 (spec: "SPC/ML 방식 전환") -- None for Config
    # factors, which have no numeric x to fit either method on.
    methods: MethodComparisonSchema | None
    target_provenance: dict[str, Any] | None = None


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
    target_provenance: dict[str, Any] | None = None


class HeatmapScaleSchema(BaseModel):
    min: float
    max: float


class HeatmapResponse(BaseModel):
    dataset_id: str
    metric: str
    # "numeric" | "categorical" -- 프론트가 어느 그리드를 받았는지
    # 캐시/렌더 분기에 쓴다 (spec E).
    kind: str = "numeric"
    features: list[str]
    targets: list[str]
    values: list[list[float | None]]
    # TC-4: numeric 보기는 항상 ε²(설명력, 부호 없음)를 values/셀 농도로
    # 쓰고, rho(부호 있는 스피어만 상관)는 색상 방향에만 쓴다 -- U자
    # 관계도 진하게 표시되면서 방향은 색으로 읽힌다. categorical 보기는
    # 정의상 방향이 없어 이 그리드가 전부 None이다.
    rho: list[list[float | None]] = Field(default_factory=list)
    n: list[list[int]]
    q: list[list[float | None]]
    significant: list[list[bool]]
    tier: list[list[str | None]]
    # QA-3: 상관계수는 그려지지만(n>=30) 유의 인자 판정에서는 종류별
    # 표본 게이트(R>=100/D>=40) 미달로 빠지는 셀 -- 히트맵과 유의 인자
    # 목록이 어긋나 보이지 않도록 별도 표시(사선 등)에 쓴다.
    gate_excluded: list[list[bool]] = Field(default_factory=list)
    scale: HeatmapScaleSchema
    excluded_configs: int
    target_provenance: dict[str, Any] | None = None


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


class AlertCellColorSchema(BaseModel):
    """RC-4b: y1~y5 셀 하나의 색상 메타데이터. direction은 "red"|"blue"|
    None(방향 불분명), shade는 "dark"|"medium"|"light"|"gray"|"measured"
    (실측값은 색을 쓰지 않는다 -- 프런트가 무채색으로 렌더한다)."""

    direction: str | None
    shade: str
    feature: str | None
    contribution_pct: float | None
    factor_value: float | None
    optimal_center: float | None


class YieldCoreFactorCellSchema(BaseModel):
    """VA-3/VA-4: 웨이퍼·타깃별로 실제 쓰인(폴백 포함) 핵심 인자."""

    feature: str | None
    contribution_pct: float | None
    rank_used: int | None
    factor_value: float | None


class YieldReliabilityDetailItemSchema(BaseModel):
    target: str
    feature: str


class YieldReliabilityInfoSchema(BaseModel):
    """VC-1/VC-2: n/5 신뢰도와 툴팁용 계측/미계측 타깃 상세."""

    count: int
    measured: list[YieldReliabilityDetailItemSchema]
    unmeasured: list[YieldReliabilityDetailItemSchema]


class YieldRecommendationSchema(BaseModel):
    """VD-2: 두 갈래(구간 조정/계측 추가)로 조립된 권장사항 문장."""

    text: str
    adjustable_targets: list[str]
    measurement_gap_targets: list[str]


class YieldCandidateSchema(BaseModel):
    lot_wafer_id: str
    lot_id: str | None
    y: float
    y_components: dict[str, float]
    cells: dict[str, AlertCellColorSchema]
    core_factors: dict[str, YieldCoreFactorCellSchema]
    reliability: YieldReliabilityInfoSchema
    recommendation: YieldRecommendationSchema


class YieldFallbackSummarySchema(BaseModel):
    """VA-3: 폴백 순위 분포 -- "58%가 전부 미계측이다" 같은 통계를 화면에
    드러내는 데 쓴다. `rank_counts`의 키는 "1".."5"(JSON은 정수 키를
    지원하지 않는다)."""

    rank_counts: dict[str, int]
    none_count: int
    total_combinations: int


class YieldPredictionResponse(BaseModel):
    """VA~VD: 수율 예측 순위 목록. 정렬 기본값은 y(=100 − Σ Y1~Y5)
    오름차순이며, 그 밖의 정렬·검색·상위 10/전체 보기 전환은 프런트가
    이 전체 목록(신뢰도==0인 웨이퍼는 제외, `unmeasured_*`로 별도 제공) 위에서 수행한다."""

    train_dataset_id: str
    eval_dataset_id: str
    total_wafers: int
    candidates: list[YieldCandidateSchema]
    unmeasured_wafer_ids: list[str]
    unmeasured_count: int
    fallback_summary: YieldFallbackSummarySchema
    target_provenance: dict[str, Any] | None = None
    # SC그룹: "모델 분석" 파이프라인이 저장한 스냅샷 캐시에서 왔으면 그
    # 스냅샷의 analysis_id, 즉석 계산이면 None -- 네 화면이 같은 분석
    # 회차를 공유하는지 프런트가 구분할 수 있게 한다.
    analysis_id: str | None = None


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
    # 지시서 I-2: 모델 학습 팝업의 "데이터 크기" 표시줄용.
    row_count: int | None = None
    feature_count: int | None = None


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
    config_screening: ReportConfigScreeningSchema
    limitations: list[str]


# ---------------------------------------------------------------------------
# SUNI chatbot context (/api/analysis/context): same report, but `alarms`
# is grouped into {summary, records[, records_truncated, records_total]}
# instead of a flat list, so the LLM can cite an individual wafer's alarm
# record, not just the aggregate counts.
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


class AnalysisContextResponse(BaseModel):
    meta: dict[str, Any]
    method: ReportMethodSchema
    summary: ReportSummarySchema
    targets: list[ContextTargetEntrySchema]
    alarms: ContextAlarmsSchema
    config_screening: ReportConfigScreeningSchema
    limitations: list[str]


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
