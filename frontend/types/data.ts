export type ColumnDetectionResult = {
  id: string[];
  r: string[];
  d: string[];
  eq: string[];
  targets: string[];
  config: string[];
};

export type ValidationResult = {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  detected_columns: ColumnDetectionResult;
  missing_required_columns: string[];
  duplicate_wafer_id_count: number;
  total_missing_count: number;
  overall_missing_rate: number;
  schema_version: string | null;
  validation_mode: "training" | "inference" | "analysis" | null;
  config_completeness_rate: number;
  r_measurement_coverage: number;
  d_measurement_coverage: number;
  required_field_error_count: number;
  config_parse_error_count: number;
  target_consistency_rate: number | null;
  lot_structure_consistency_rate: number | null;
  duplicate_wafer_count: number;
  invalid_numeric_count: number;
  structural_unmeasured_count: number;
};

export type ValidationResponse = {
  success: boolean;
  filename: string;
  row_count: number;
  column_count: number;
  validation: ValidationResult;
};

export type DataSummary = {
  row_count: number;
  column_count: number;
  missing_count: number;
};

export type DataPreviewRow = Record<string, unknown>;

export type ProcessingSummary = {
  missing_strategy?: "native" | "median" | "model_specific" | string;
  missing_handling?: string;
  outlier_strategy?: "flag_only" | "iqr" | "none" | string;
  outlier_policy?: string;
  missing_indicator?: boolean;
  outlier_indicator?: boolean;
  missing_indicator_count?: number;
  outlier_indicator_count?: number;
  fallback_used?: boolean;
  step_feature_count?: number;
  r_column_count?: number;
  d_column_count?: number;
  categorical_column_count?: number;
  config_parsed?: boolean;
  config_column_count?: number;
  config_parser_version?: string;
  measurement_coverage?: { r?: number; d?: number };
  numeric_feature_count?: number;
  categorical_config_count?: number;
  removed_all_missing_columns?: string[];
  removed_constant_columns?: string[];
  removed_near_constant_columns?: string[];
  missing_imputed_columns?: number;
  winsorized_columns?: number;
  training_row_count?: number;
  lot_count?: number;
  split_method?: string;
  pipeline_version?: string;
  model_strategies?: Record<string, unknown>;
  model_outlier_strategies?: Record<string, unknown>;
  schema_version?: string;
  config_encoding?: string;
  config_strings_decomposed?: boolean;
  r_clipping?: string;
  d_clipping?: string;
  by_target?: Record<string, unknown>;
};

export type PreprocessResponse = {
  success: boolean;
  filename: string;
  before: DataSummary;
  after: DataSummary;
  changes: {
    filled_missing_values: number;
    clipped_outliers: number;
    added_indicator_columns: string[];
  };
  warnings: string[];
  preview: DataPreviewRow[];
  schema_version: string | null;
  measurement_coverage: { r: number; d: number };
  preprocessing_policy: Record<string, unknown>;
  config_summary: Record<string, unknown>;
  processing_summary: ProcessingSummary;
};

export type ModelMetrics = {
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  mse?: number | null;
};

export type DatasetSplit = {
  train_rows: number;
  validation_rows: number;
  test_rows: number;
  group_split_used: boolean;
  split_method: string;
};

export type ModelComparisonItem = {
  model_name: string;
  status: "success" | "failed";
  validation: ModelMetrics | null;
  selected: boolean;
  error_message: string | null;
};

export type MetricAggregate = {
  mean: number;
  std: number;
};

export type MetricSummary = Partial<Record<"r2" | "rmse" | "mae" | "mse", MetricAggregate>>;

export type EvaluationSummary = {
  metric_summary?: MetricSummary;
  aggregate_metrics?: MetricSummary;
  [key: string]: unknown;
};

export type CVSummary = {
  name?: string;
  group_column?: string;
  outer_folds?: number;
  inner_folds?: number;
  seed?: number;
  metric_summary?: MetricSummary;
  aggregate_metrics?: MetricSummary;
  fold_metrics?: unknown[];
  [key: string]: unknown;
};

export type TrainingResult = {
  success: boolean;
  target: string;
  best_model: string;
  split: DatasetSplit;
  metrics: {
    train: ModelMetrics;
    validation: ModelMetrics;
    test: ModelMetrics;
  };
  model_comparison: ModelComparisonItem[];
  feature_count: number;
  warnings: string[];
  artifacts: {
    model_file: string;
    metadata_file: string;
  };
  schema_version: string;
  missingness_sensitivity: Record<string, unknown> | null;
  evaluation_summary?: EvaluationSummary | null;
  model_id?: string | null;
  model_type?: string | null;
  selected_final_output?: string | null;
  cv?: CVSummary | null;
  cv_summary?: CVSummary | null;
  final_y_metrics?: Record<string, unknown> | null;
  target_metrics?: Record<string, unknown> | null;
  risk_metrics?: Record<string, unknown> | null;
  preprocessing?: ProcessingSummary | null;
  ensemble?: {
    enabled: boolean;
    selected: boolean;
    selected_type: string;
    size: number;
    base_models: string[];
    weights: Record<string, number>;
    best_single_metrics: ModelMetrics;
    ensemble_metrics: ModelMetrics;
    improvement_over_single: {
      rmse_relative: number | null;
      r2_absolute: number | null;
    };
    agreement: {
      mean_prediction_spread: number | null;
      max_prediction_spread: number | null;
      mean_pairwise_correlation: number | null;
    };
    target_configs: Record<string, {
      selected_type: string;
      method: string;
      base_models: string[];
      weights: Record<string, number>;
      best_single_model: string;
      improvement_over_single: { rmse_relative: number | null };
      fold_rmse_std: number;
    }>;
  } | null;
};

export type TrainResponse = TrainingResult;

export type TrainingJobState =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "interrupted";

export type TrainingJobResult = {
  model_id: string;
  target: string;
  best_model: string;
  test_metrics: ModelMetrics | null;
  feature_count: number;
  warning_count: number;
};

export type TrainingJobCreateResponse = {
  job_id: string;
  status: "queued";
};

export type TrainingJobStatusResponse = {
  job_id: string;
  status: TrainingJobState;
  stage: string;
  progress: number;
  elapsed_seconds: number;
  result: TrainingJobResult | null;
  error: string | null;
};

export type ModelSummary = {
  model_id: string;
  target: string;
  model_name: string;
  created_at: string;
  test_metrics: ModelMetrics;
  feature_count: number;
  compatibility: "compatible" | "legacy" | "incompatible" | "unknown_schema";
  schema_version: string | null;
  model_type?: string | null;
  bundle_type?: string | null;
  selected_final_output?: string | null;
  available_targets?: string[] | null;
  cv_summary?: CVSummary | null;
  available: boolean;
  loadable: boolean;
  compatibility_status: string;
  incompatibility_reason: string | null;
};

export type ModelListResponse = {
  success: boolean;
  models: ModelSummary[];
  total: number;
  warnings: string[];
};

export type DeleteModelResponse = {
  deleted: boolean;
  model_id: string;
  deleted_files: string[];
  missing_files: string[];
  metadata_deleted: boolean;
  bundle_deleted: boolean;
  removed_files?: string[];
  registry_removed: boolean;
  prediction_history_kept: boolean;
  analysis_history_kept: boolean;
  prediction_history_count?: number;
  analysis_history_count?: number;
};

export type ModelDetailMetrics = {
  r2: number | null;
  rmse: number | null;
  mse: number | null;
  mae: number | null;
};

export type TargetEnsembleConfig = {
  selected_type?: string | null;
  method?: string | null;
  base_models?: string[] | null;
  weights?: Record<string, number> | null;
  improvement_over_single?: { rmse_relative?: number | null } | null;
};

export type ModelDetail = {
  success: boolean;
  model_id: string;
  model_name: string | null;
  model_type: string | null;
  model_version: string | null;
  created_at: string | null;
  target: string | null;
  feature_count: number | null;
  feature_names?: string[] | null;
  dataset_split?: Record<string, number> | null;
  dataset_rows?: Record<string, number> | null;
  metrics?: Record<string, ModelDetailMetrics> | null;
  random_seed: number | null;
  split_method: string | null;
  preprocessing_version: string | null;
  preprocessing_config?: Record<string, unknown> | null;
  training_time_seconds: number | null;
  source_filename: string | null;
  model_file: string | null;
  metadata_file: string | null;
  storage_status: string;
  champion: boolean | null;
  sklearn_version: string | null;
  compatibility: "compatible" | "legacy" | "incompatible" | "unknown_schema";
  schema_version: string | null;
  schema_fingerprint: string | null;
  config_parser_version: string | null;
  missing_indicator_used: boolean | null;
  outlier_policy: string | null;
  group_column: string | null;
  target_leakage_check?: Record<string, unknown> | null;
  ensemble_enabled: boolean | null;
  ensemble_mode: string | null;
  ensemble_method: string | null;
  target_ensemble_configs?: Record<string, TargetEnsembleConfig> | null;
  target_metrics?: Record<string, unknown> | null;
  outer_fold_metrics?: unknown[] | null;
  inner_fold_metrics?: unknown[] | null;
  available_targets?: string[] | null;
  cv_summary?: Record<string, unknown> | null;
  ensemble_weights?: Record<string, unknown> | null;
  hybrid_summary?: Record<string, unknown> | null;
  risk_metrics?: Record<string, unknown> | null;
  preprocessing_summary?: Record<string, unknown> | null;
  training_config?: Record<string, unknown> | null;
  model_agreement_summary?: Record<string, unknown> | null;
  production_ensemble_retrained: boolean | null;
  available: boolean;
  loadable: boolean;
  compatibility_status: string;
  incompatibility_reason: string | null;
};

export type DatasetSummary = {
  dataset_id: string;
  kind: "bundled" | "uploaded";
  original_filename: string;
  uploaded_at: string | null;
  row_count: number;
  column_count: number;
  lot_min: string | null;
  lot_max: string | null;
  lot_count: number | null;
  warnings: string[];
  unmapped_columns: string[];
  deletable: boolean;
};

export type DatasetListResponse = { items: DatasetSummary[] };

export type DatasetUploadResponse = {
  success: boolean;
  dataset_id: string | null;
  blocking_errors: string[];
  warnings: string[];
  unmapped_columns: string[];
  row_count?: number | null;
  column_count?: number | null;
  lot_min?: string | null;
  lot_max?: string | null;
  lot_count?: number | null;
};

export type DatasetSchemaResponse = {
  dataset_id: string;
  steps_present: number[];
  max_step: number;
  r_columns: string[];
  d_columns: string[];
  config_columns: string[];
  target_columns: string[];
  unmapped_columns: string[];
  missing_rates: Record<string, number>;
  r_measurement_rate: number | null;
  d_measurement_rate: number | null;
};

export type RelationShape = "monotonic_increasing" | "monotonic_decreasing" | "u_shape" | "unclear";

export type ParetoFactor = {
  target: string;
  feature: string;
  kind: "R" | "D" | "Config";
  step: number;
  eps2: number;
  p_value: number;
  q_value: number;
  pearson_r: number | null;
  spearman_r: number | null;
  n_observed: number;
  contribution_pct: number;
  cumulative_pct: number;
  significant: boolean;
  relation_shape: RelationShape;
  optimal_center: number | null;
};

export type ScatterPoint = {
  x: number;
  y: number;
  lot_wafer_id: string | null;
  lot_id: string | null;
  in_range: boolean;
  config: string | null;
};

export type NormalRange = {
  lo: number | null;
  hi: number | null;
  one_sided: boolean;
  fallback_applied: boolean;
};

export type ReferenceLineKey =
  | "mean"
  | "q1"
  | "q3"
  | "iqr_lo"
  | "iqr_hi"
  | "s3_lo"
  | "s3_hi"
  | "s6_lo"
  | "s6_hi"
  | "warning_lo"
  | "warning_hi";

export type ReferenceLine = {
  key: ReferenceLineKey;
  value: number;
  drawable: boolean;
  alarm_relevant: boolean;
  formula: string;
  outside_count: number;
  /** key가 "warning_lo"/"warning_hi"일 때만 채워진다 (알람 판정 GBDT 전환
   * §C-4-1) -- 경고선 밖 실측 수율 차이(%p, 예측값 아님). 표본 30장 미만이면
   * null. */
  observed_yield_gap_pp: number | null;
};

export type BinProfile = {
  x_mean: number;
  y_mean: number;
  y_lo: number;
  y_hi: number;
  n: number;
  x_lo: number;
  x_hi: number;
  bin_span_ratio: number;
  /** Outlier-widened bin (its own [x_lo, x_hi] spans an outsized share of
   * the factor's overall range) -- rendered as a dashed curve segment. */
  sparse: boolean;
};

/** One method's (SPC or ML) 권장구간 fit + the F2 x 안정성 score it was
 * judged on -- spec "SPC / ML 방식 전환". `window`/`optimal_center` drive
 * the chart band when this method is selected; the rest backs the
 * comparison card's numbers only. */
export type MethodWindow = {
  window: [number, number];
  optimal_center: number;
  recall: number;
  precision: number;
  f2: number;
  width_sd: number;
  stability: number;
  score: number;
  /** Whether the control range actually cut into this method's raw
   * (unclamped) window -- drives the "(관리한계에 맞춰 조정됨)" tooltip. */
  clamped: boolean;
};

export type WindowMethod = "spc" | "ml";

export type MethodComparison = {
  spc: MethodWindow | null;
  ml: MethodWindow | null;
  /** Which method's window backs the alarm log / 개선 권장 목록 everywhere
   * else in the app -- the SPC/ML toggle only changes what's *displayed*
   * on this chart, never this (spec §2-2/§5). */
  adopted: WindowMethod;
  adopted_reason: string;
};

export type ScreeningScatterResponse = {
  points: ScatterPoint[];
  reference_lines: ReferenceLine[];
  normal_range: NormalRange;
  bins: BinProfile[];
  optimal_center: number | null;
  /** Set only when a classified optimal center existed but was dropped
   * (fell outside its own recommended window, or was picked from a
   * sparse bin) -- shown as the disabled-toggle tooltip reason instead
   * of the generic "단조 관계라..." message. */
  optimal_center_dropped_reason: string | null;
  eps2: number;
  spearman_r: number | null;
  p_value: number;
  q_value: number;
  significant: boolean;
  confidence_tier: ConfidenceTier;
  relation_shape: RelationShape;
  n: number;
  axis: { x_label: string; y_label: string };
  /** null for Config factors (no numeric x to fit either method on). */
  methods: MethodComparison | null;
};

export type CategoricalGroup = {
  category: string;
  n: number;
  mean: number;
  median: number;
  q1: number;
  q3: number;
  values: number[];
};

export type CategoricalScatterResponse = {
  groups: CategoricalGroup[];
  eps2: number;
  p_value: number;
  q_value: number;
  significant: boolean;
  confidence_tier: ConfidenceTier;
  n: number;
  axis: { x_label: string; y_label: string };
};

export type ControlRangeItem = {
  feature: string;
  target: string;
  kind: string;
  relation_shape: RelationShape;
  mean: number;
  std: number;
  q1: number;
  q3: number;
  lower: number | null;
  upper: number | null;
  one_sided: boolean;
  fallback_applied: boolean;
  band_width: number;
  n_observed: number;
  reference_lines: ReferenceLine[];
};

export type ControlRangeListResponse = {
  train_dataset_id: string;
  items: ControlRangeItem[];
  no_significant_factor_targets: string[];
};

/** 알람 판정 GBDT 전환 (spec §A-2) -- 관리한계 이탈량이 아니라 부트스트랩
 * 앙상블 예측 수율의 신뢰구간 상한 기준 등급. "개선 권고" 등급은 삭제됐다
 * (spec 알람 신뢰도 게이트 §B-1: 정밀도가 무작위 수준과 다르지 않았다). */
export type AlarmGrade = "심각" | "위험" | "주의";

export type AlarmItem = {
  lot_wafer_id: string;
  lot_id: string | null;
  grade: AlarmGrade;
  /** 0-100, 낮을수록 위험. 예측 수율 절대값·신뢰구간은 화면에 노출하지
   * 않는다 (spec §A-3: 오차가 Y 표준편차의 72~80%). */
  risk_percentile: number;
  reason: string;
};

export type AlarmListResponse = {
  train_dataset_id: string;
  eval_dataset_id: string;
  items: AlarmItem[];
  total: number;
  alarm_total: number;
  evaluated_total: number;
  alarm_share_warning: boolean;
  // 알람 신뢰도 게이트 (spec 알람 신뢰도 게이트 §A-2) -- auc_gate_passed가
  // false면 items/total/alarm_total 모두 0이다.
  auc_lower_bound: number | null;
  auc_gate_passed: boolean;
  auc_gate_threshold: number;
};

export type ReliabilityGrade = "높음" | "보통" | "낮음";

export type ReliabilityResponse = {
  dataset_id: string;
  eval_dataset_id: string;
  grade: ReliabilityGrade;
  total_score: number;
  auc_lower_bound: number | null;
  auc_score: number;
  // 알람 신뢰도 게이트 §D-3: AUC 항목에 게이트 통과 여부를 덧붙인다.
  auc_gate_passed: boolean;
  auc_gate_message: string | null;
  n_significant_factors: number;
  n_significant_score: number;
  max_eps2: number | null;
  max_eps2_score: number;
  n_train: number;
  n_train_score: number;
  coverage_pct: number | null;
  coverage_score: number;
  deduction_reasons: string[];
  low_holdout_sample: boolean;
  thresholds_disclaimer: string;
  target_fallback_tier: "per_target" | "final_yield_only" | "unanalyzable";
  target_fallback_message: string | null;
};

// -- 알림 연동 (설정 패널 신설 §C/§D) -----------------------------------

export type NotificationTiming = "on_analysis" | "daily_9am";
export type NotificationGrade = "심각" | "위험" | "주의";

export type SlackChannelSummary = {
  connected: boolean;
  target: string | null;
  webhook_masked: string | null;
  verified_at: string | null;
};

export type TelegramChannelSummary = {
  connected: boolean;
  target: string | null;
  chat_id_masked: string | null;
  verified_at: string | null;
};

export type GmailChannelSummary = {
  connected: boolean;
  pending: boolean;
  email: string | null;
  verified_at: string | null;
};

export type NotificationConditions = {
  grades: NotificationGrade[];
  timing: NotificationTiming;
};

export type NotificationSettingsSummary = {
  slack: SlackChannelSummary;
  telegram: TelegramChannelSummary;
  gmail: GmailChannelSummary;
  conditions: NotificationConditions;
};

export type SendTestResponse = { ok: boolean; error: string | null };

export type DispatchResponse = {
  skipped: boolean;
  reason: string | null;
  sent_count: number | null;
  results: Record<string, { ok: boolean; error: string | null }> | null;
};

// -- 전처리 방식 A/B/C 실시간 비교 (설정 패널 신설 §E) ----------------------

export type PreprocessingMode = "A" | "B" | "C";

export type PreprocessingModeResult = {
  mode: PreprocessingMode;
  label: string;
  r2: number;
  adopted: boolean;
};

export type PreprocessingComparisonResponse = {
  dataset_id: string;
  dataset_label: string;
  results: PreprocessingModeResult[];
  winner: PreprocessingMode;
  b_equals_c: boolean;
  holdout_note: string;
  winner_note: string | null;
};

// 사전 알람 로그 전면 개편 (spec §A-3) -- 등급 없는 wafer별 원시 예측치.
// 목표 수율/민감도를 조절할 때마다 이 배열을 다시 받아올 필요가 없다 --
// lib/alertsClassify.ts의 classifyWafer가 여기서 즉시 5분류를 계산한다.
export type WaferPrediction = {
  lot_wafer_id: string;
  lot_id: string | null;
  measured: boolean;
  pred_mean: number;
  pred_lo: number;
  pred_hi: number;
  reason: string | null;
};

export type AlertsDataResponse = {
  train_dataset_id: string;
  eval_dataset_id: string;
  total_wafers: number;
  sigma: number;
  train_y_min: number;
  train_y_max: number;
  train_y_median: number;
  train_y_p1: number;
  train_y_p99: number;
  predictions: WaferPrediction[];
  // 알람 신뢰도 게이트 -- AlarmListResponse와 같은 (train,eval) 쌍이면
  // 항상 일치한다.
  auc_lower_bound: number | null;
  auc_gate_passed: boolean;
  auc_gate_threshold: number;
  // B-1: holdout/factor_bands/measurement_bias는 삭제했다 -- 렌더하는
  // 화면이 없었다(estimatePrecisionRecall/representativeWafer도 죽은
  // 코드였다).
};

export type ConfidenceTier = "strong" | "moderate" | "weak" | "reference";

export type ParetoRankingItem = {
  feature: string;
  kind: string;
  step: number;
  eps2: number;
  p_value: number;
  q_value: number;
  significant: boolean;
  confidence_tier: ConfidenceTier;
  n_observed: number;
  contribution_pct: number;
  cumulative_pct: number;
};

export type ParetoRankingResponse = {
  dataset_id: string;
  target: string;
  total_factor_count: number;
  n80: number | null;
  fdr_pass_count: number;
  effect_size_pass_count: number;
  max_eps2: number | null;
  items: ParetoRankingItem[];
};

export type HeatmapMetric = "spearman" | "eps2";
export type HeatmapKind = "numeric" | "categorical";
export type ConfigHeatmapLevel = "model" | "eq" | "chamber";

export type HeatmapResponse = {
  dataset_id: string;
  metric: HeatmapMetric;
  kind: HeatmapKind;
  features: string[];
  targets: string[];
  values: Array<Array<number | null>>;
  n: number[][];
  q: Array<Array<number | null>>;
  significant: boolean[][];
  tier: Array<Array<ConfidenceTier | null>>;
  scale: { min: number; max: number };
  excluded_configs: number;
};

export type TargetPerformance = {
  target: string;
  no_factor_available: boolean;
  feature: string | null;
  kind: string | null;
  eps2: number | null;
  contribution_pct: number | null;
  relation_shape: RelationShape | null;
  optimal_center: number | null;
  p_value: number | null;
  confidence_tier: ConfidenceTier | null;
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  n: number | null;
};

export type ModelPerformanceResponse = {
  model_id: string | null;
  trained_at: string | null;
  source_filename: string | null;
  targets: TargetPerformance[];
  final_yield: TargetPerformance | null;
  row_count: number | null;
  feature_count: number | null;
};

// 자동 수집 파이프라인 §2-2 -- 승격 여부와 무관한 학습 시도 이력.
export type PromotionEvent = {
  created_at: string;
  candidate_model_id: string;
  promoted: number; // SQLite에서 0/1로 내려온다.
  reason: string;
  candidate_metric: number | null;
  active_metric: number | null;
  previous_model_id: string | null;
};

export type PromotionHistoryResponse = {
  items: PromotionEvent[];
};

// -- 학습·분석 결과 상태 유지 (탭 이동·재접속) --------------------------
// Server-persisted "latest result" for each of the 3 long-running tabs.
// `points` is never present on a restored ScreeningScatterResponse --
// see GET /api/state/latest's docstring (src/runtime/app_state.py) --
// so a restored entry is always `points: []` until the page independently
// refetches it.

// 지시서 I-3: 모델 학습 팝업의 SQL 연결 정보(비밀번호 제외)·Refresh 주기를
// 서버에 저장한다 -- 최근 학습 결과(performance)와 같은 레코드에 함께
// 실어 GET /api/state/latest 한 번으로 둘 다 복원한다.
export type LatestTrainingPayload = {
  performance: ModelPerformanceResponse;
  sqlHost?: string;
  sqlPort?: string;
  sqlDb?: string;
  sqlUser?: string;
  refreshIntervalMinutes?: number | null;
};

export type LatestTrainingRecord = {
  schema_version: number;
  created_at: string;
  dataset: string;
  payload: LatestTrainingPayload;
};

// Deliberately narrow (spec §6: 세 종류 합쳐 100KB 미만) -- per-factor
// scatter/categorical detail (관리한계·권장구간·최적중심 등) is NOT part of
// what gets persisted, even with points stripped: 25 factors' worth of
// reference lines/bins/methods comparison alone measured ~105KB against
// train.CSV, well over budget on its own. `paretoByTarget` is what the
// screening table/Pareto bars render from and is cheap (a few KB); the
// scatter detail is refetched via the same per-factor calls a live run
// already makes (frontend's fetchAllScatterData), same as points always were.
export type LatestAnalysisPayload = {
  activeTarget: string;
  paretoByTarget: Record<string, ParetoRankingResponse>;
  measurementExpansion?: MeasurementExpansionResponse | null;
  // 지시서 AJ: 저장된 스냅샷의 응답 형태·내용 규칙(예: PARETO_TOP_N
  // 5->10)이 여전히 유효한지 프론트가 직접 검사하는 버전 --
  // frontend/lib/snapshotVersion.ts의 ANALYSIS_SNAPSHOT_VERSION과
  // 다르면 복원하지 않는다. 백엔드 봉투의 schema_version(app_state.py)과는
  // 별개다 -- 그쪽이 필터링해 버리면 프론트는 "저장된 적 없음"과 "폐기됨"을
  // 구분할 수 없어 조용히 빈 화면이 된다.
  snapshotVersion: number;
};

export type LatestAnalysisRecord = {
  schema_version: number;
  created_at: string;
  dataset: string;
  payload: LatestAnalysisPayload;
};

// wafer 수만큼 커지는 predictions/holdout는 서버에 저장하지 않는다 (spec
// 학습·분석 결과 상태 유지와 같은 원칙 -- AnalysisState의
// alarmGradeByWaferId/scatterByKey와 동일하게, 재접속 시 가벼운 설정값만
// 복원하고 무거운 데이터는 배경에서 다시 불러온다). 목표 수율·민감도만
// 저장해 두면 재접속 시 사용자가 마지막으로 보던 설정 그대로 다시
// 조회할 수 있다.
export type LatestAlarmsPayload = {
  targetYield: number;
  sensitivity: number;
};

export type LatestAlarmsRecord = {
  schema_version: number;
  created_at: string;
  train_dataset: string;
  eval_dataset: string;
  payload: LatestAlarmsPayload;
};

export type LatestStateResponse = {
  training: LatestTrainingRecord | null;
  analysis: LatestAnalysisRecord | null;
  alarms: LatestAlarmsRecord | null;
  notifications: NotificationSettingsSummary;
  // 지시서 CB: 저장된 레코드 중 하나 이상이 더 이상 존재하지 않는
  // 데이터셋을 가리켜 서버가 통째로 버렸으면 true (dataset을 "train"으로
  // 바꿔치기하지 않는다 -- 옛 payload가 잘못된 라벨을 달고 뜨는 게 더
  // 나쁘다). 프론트는 이 신호로만 재실행 안내를 띄운다.
  dataset_fallback_applied: boolean;
};

// -- 즐겨찾기 (지시서 J) -------------------------------------------------
// 점 데이터는 절대 담지 않는다 (J-2) -- 열 때 이 파라미터로 API를 다시
// 불러 렌더한다.
export type FavoriteViewType = "scatter" | "box" | "pareto";

export type FavoriteSnapshot = {
  dataset: string;
  target: string;
  feature: string;
  viewType: FavoriteViewType;
  colorBy?: string;
  method?: string | null;
  isConfig: boolean;
};

export type FavoriteRecord = {
  favorite_id: string;
  created_at: string;
  snapshot: FavoriteSnapshot;
};

export type FavoriteListResponse = {
  items: FavoriteRecord[];
};

// 모니터링 홈 트리맵 -- Config 문자열은 서버에서 Model/EQ/Chamber로
// 분해하지 않는다(원문 그대로). n<30 회색 처리, ±3%p 고정 색 스케일은
// 프론트에서 처리한다(ConfigTreemap.tsx).
export type ConfigTreemapGroup = {
  config: string;
  n: number;
  mean: number;
  median: number;
  p5: number;
  p95: number;
};

export type ConfigTreemapResponse = {
  dataset_id: string;
  step: number;
  overall_mean: number;
  groups: ConfigTreemapGroup[];
};

export type ReportFactorEntry = {
  feature: string;
  kind: string;
  step: number;
  rank: number;
  eps2: number;
  contribution_pct: number;
  cumulative_pct: number;
  spearman_rho: number | null;
  p_value: number;
  q_value: number;
  grade: string;
  report_confidence: string;
  n_observed: number;
  n_missing_pct: number;
  relation: { shape: string; optimal_center: number | null; interpretation: string };
  binned_profile: { x_center: number; y_mean: number; n: number }[];
  control_limits: {
    lcl: number | null;
    ucl: number | null;
    one_sided: boolean;
    mean: number;
    std: number;
    q1: number;
    q3: number;
    sigma3: (number | null)[];
    sigma6: (number | null)[];
    sigma6_drawn: boolean;
  };
  band_stability: number;
  band_width: number | null;
  window: {
    lo: number;
    hi: number;
    mean_in_window: number | null;
    mean_overall: number;
    ratio: number | null;
    n_in_window: number;
  } | null;
  chamber_interaction: boolean;
  chamber_interaction_p: number | null;
  chamber_interaction_q: number | null;
  per_chamber_window: Record<string, { lo: number; hi: number; ratio: number | null; n: number }> | null;
  eval_result: { alarms: number; observed: number; mean_y_alarm: number | null; mean_y_normal: number | null };
};

export type ReportTargetEntry = {
  target: string;
  target_stats: { mean: number; std: number; q1: number; q3: number };
  factors: ReportFactorEntry[];
};

export type ReportAlarmRecord = {
  lot_wafer_id: string;
  lot_id: string | null;
  wafer_slot: number | null;
  step: number;
  feature: string;
  kind: string;
  target: string;
  value: number;
  normal_range: (number | null)[];
  deviation: number;
  direction: string;
  severity: string;
  actual_y_target: number | null;
  actual_y_final: number | null;
};

export type AnalysisReportResponse = {
  meta: {
    generated_at: string;
    app_version: string;
    dataset: Record<string, { name: string | null; rows: number | null; lots: number | null; lot_range: string | null }>;
  };
  method: {
    screening: string;
    contribution_denominator: string;
    control_limit: string;
    inclusion_rule: string;
    missing_policy: string;
  };
  summary: {
    targets_analyzed: number;
    factors_included: number;
    excluded_low_significance: number;
    alarm_wafers: number;
    normal_wafers: number;
    undecidable_wafers: number;
    mean_yield_alarm: number | null;
    mean_yield_normal: number | null;
    yield_gap_pp: number | null;
  };
  targets: ReportTargetEntry[];
  alarms: ReportAlarmRecord[];
  config_screening: {
    n_tested: number;
    n_significant_fdr: number;
    max_observed_eps2: number | null;
    max_observed_feature: string | null;
    max_observed_target: string | null;
    mde_eps2: number | null;
    median_n_per_group: number | null;
  };
  limitations: string[];
};

export type FactorPriority = {
  feature: string;
  target: string;
  measurement_rate: number;
  recommendation: string; // "+10%p" | "+15%p" | "유지"
  reason: string;
  additional_judged: number;
  yield_contribution_pp: number | null;
};

export type NewFactorDiscovery = {
  feature: string;
  target: string;
  kind: string;
};

export type MeasurementExpansionResponse = {
  train_dataset_id: string;
  eval_dataset_id: string;
  action_blocked_wafers: number;
  total_wafers: number;
  additional_judged: number;
  action_target: number;
  expected_yield_gain_pp: number | null;
  show_full_card: boolean;
  priorities: FactorPriority[];
  new_factor_discoveries: NewFactorDiscovery[];
};

