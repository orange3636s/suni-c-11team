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

export type ReferenceLineKey = "mean" | "q1" | "q3" | "iqr_lo" | "iqr_hi" | "s3_lo" | "s3_hi" | "s6_lo" | "s6_hi";

export type ReferenceLine = {
  key: ReferenceLineKey;
  value: number;
  drawable: boolean;
  alarm_relevant: boolean;
  formula: string;
  outside_count: number;
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

export type AlarmSeverity = "low" | "medium" | "high";

export type AlarmItem = {
  lot_wafer_id: string;
  lot_id: string | null;
  wafer_slot: number | null;
  step: number;
  feature: string;
  kind: string;
  target: string;
  value: number;
  normal_range: [number | null, number | null];
  deviation: number;
  direction: "above" | "below";
  severity: AlarmSeverity;
  actual_y: number | null;
};

export type AlarmListResponse = {
  train_dataset_id: string;
  eval_dataset_id: string;
  items: AlarmItem[];
  total: number;
};

export type FactorBandPoint = {
  count: number;
  mean_defect_rate: number | null;
};

export type FactorBand = {
  feature: string;
  target: string;
  kind: string;
  x_min: number;
  x_max: number;
  lcl: number | null;
  ucl: number | null;
  recommended_lo: number | null;
  recommended_hi: number | null;
  out_of_control: FactorBandPoint;
  out_of_recommended: FactorBandPoint;
  in_recommended: FactorBandPoint;
};

export type AlarmSummaryResponse = {
  train_dataset_id: string;
  eval_dataset_id: string;
  total_wafers: number;
  measured_wafers: number;
  counts: { alarm: number; out_of_recommended: number; in_recommended: number; unmeasured: number };
  band_yield: {
    alarm: number | null;
    out_of_recommended: number | null;
    in_recommended: number | null;
    unmeasured: number | null;
  };
  top_lots: Array<{ lot_id: string; alarm_count: number }>;
  measurement_bias_p: number | null;
  factor_bands: FactorBand[];
};

export type RecommendationTag = "priority" | "recommended" | "reference";

export type RecommendationItem = {
  lot_wafer_id: string;
  lot_id: string | null;
  step: number;
  feature: string;
  kind: string;
  target: string;
  value: number;
  recommended_range: [number, number];
  direction: "up" | "down";
  expected_improvement_pct: number | null;
  tag: RecommendationTag;
};

export type RecommendationListResponse = {
  train_dataset_id: string;
  eval_dataset_id: string;
  items: RecommendationItem[];
  total: number;
  excluded_alarm_count: number;
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
  items: ParetoRankingItem[];
};

export type HeatmapMetric = "spearman" | "eps2";

export type HeatmapResponse = {
  dataset_id: string;
  metric: HeatmapMetric;
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

