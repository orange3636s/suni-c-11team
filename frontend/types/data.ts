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

export type PredictionThresholds = {
  warning_threshold: number;
  danger_threshold: number;
};

export type PredictionRow = Record<string, unknown> & {
  Lot_Wafer_ID?: string | null;
  Lot_ID?: string | null;
  Wafer_ID?: string | null;
  Wafer_Slot?: number | null;
  risk_level?: "normal" | "warning" | "danger" | null;
  residual?: number | null;
  absolute_error?: number | null;
};

export type PredictionSummary = {
  total_rows: number;
  average_prediction: number;
  normal_count: number;
  warning_count: number;
  danger_count: number;
  evaluation: ModelMetrics | null;
};

export type PredictionResponse = {
  success: boolean;
  filename: string;
  model: {
    model_id: string;
    target: string;
    model_name: string;
  };
  summary: PredictionSummary;
  identifier_column: string;
  predictions: PredictionRow[];
  warnings: string[];
  truncated: boolean;
  preprocessing?: ProcessingSummary | null;
  prediction_id?: string | null;
  history_saved?: boolean;
  history_warning?: string | null;
};

export type HistoryStatus =
  | "running"
  | "completed"
  | "failed"
  | "partial"
  | "artifact_missing"
  | "artifact_corrupted"
  | "model_deleted"
  | "prediction_deleted";

export type PredictionHistorySummary = {
  prediction_id: string; created_at: string; completed_at?: string | null;
  status: HistoryStatus; source_filename: string | null; model_id: string | null;
  model_name_snapshot: string | null; row_count: number | null; lot_count: number | null;
  summary?: Record<string, unknown> | null; warning_count: number;
};

export type AnalysisHistorySummary = {
  analysis_id: string; prediction_id?: string | null; created_at: string;
  completed_at?: string | null;
  status: HistoryStatus; source_filename: string | null; model_id: string | null;
  model_name?: string | null; model_name_snapshot: string | null;
  row_count: number | null; lot_count: number | null;
  average_predicted_yield?: number | null; critical_count?: number | null;
  warning_wafer_count?: number | null; top_failure_target?: string | null;
  artifact_available?: boolean; summary?: Record<string, unknown> | null;
  default_target?: string | null; warning_count: number | null;
};

export type HistoryList<T> = { items: T[]; total: number; limit: number; offset: number };

export type HistoryResetSummary = {
  model_count: number;
  prediction_history_count: number;
  analysis_history_count: number;
  model_artifact_count?: number;
  prediction_artifact_count?: number;
  analysis_artifact_count?: number;
  report_snapshot_count?: number;
};

export type HistoryResetResponse = {
  success: boolean;
  deleted: {
    models: number;
    model_files: number;
    prediction_histories: number;
    prediction_artifacts: number;
    analysis_histories: number;
    analysis_artifacts: number;
    report_snapshots: number;
  };
  preserved: {
    alert_logs: boolean;
    automation_runs: boolean;
    source_csv: boolean;
  };
};

export type AnalysisHistoryListResponse = HistoryList<AnalysisHistorySummary>;
export type PredictionHistoryDetail = {
  metadata: PredictionHistorySummary & Record<string, unknown>;
  artifact: { response?: PredictionResponse; rows?: PredictionRow[]; warnings?: string[] } | null;
  linked_analysis_count: number;
};

export type ExplainOptions = {
  max_rows: number;
  top_n: number;
  per_wafer_top_n: number;
};

export type GlobalImportanceItem = {
  rank: number;
  feature: string;
  step: string;
  parameter_type: string;
  parameter_name: string;
  mean_abs_shap: number;
  mean_harmful_contribution: number;
  direction: string;
};

export type AggregateImportanceItem = {
  rank: number;
  mean_abs_shap: number;
  harmful_contribution: number;
  feature_count: number;
};

export type ExplainAnalysisSummary = {
  total_rows: number;
  analyzed_rows: number;
  sampling_used: boolean;
  sampling_strategy: string;
  explanation_method: string;
  is_fallback: boolean;
};

export type StepSummaryItem = AggregateImportanceItem & {
  step: string;
};

export type ParameterTypeSummaryItem = AggregateImportanceItem & {
  parameter_type: string;
};

export type LocalContributionItem = {
  feature: string;
  value: unknown;
  shap_value: number;
  harmful_contribution: number;
  beneficial_contribution: number;
  step: string;
  parameter_type: string;
};

export type WaferExplanation = {
  identifier: unknown;
  lot_id?: string | null;
  wafer_id?: string | null;
  wafer_slot?: number | null;
  slot?: string | number | null;
  prediction: number;
  risk_level: "normal" | "warning" | "danger" | null;
  base_value: number;
  top_negative_contributors: LocalContributionItem[];
  top_positive_contributors: LocalContributionItem[];
};

export type ExplainResponse = {
  success: boolean;
  filename: string;
  model: {
    model_id: string;
    target: string;
    model_name: string;
  };
  analysis_summary: ExplainAnalysisSummary;
  explanation_method: string;
  is_fallback: boolean;
  identifier_column: string;
  global_importance: GlobalImportanceItem[];
  step_summary: StepSummaryItem[];
  parameter_type_summary: ParameterTypeSummaryItem[];
  equipment_summary: {
    rank: number;
    equipment: string;
    mean_abs_shap: number;
    harmful_contribution: number;
  }[];
  wafer_explanations: WaferExplanation[];
  model_quality_warnings: string[];
  warnings: string[];
};

export type AssociationSummary = {
  pearson?: number | null;
  spearman?: number | null;
  eta_squared?: number | null;
  valid_count: number;
  excluded_count: number;
  category_count?: number;
  direction?: string;
  strength?: string;
};

export type RelationshipFeature = {
  rank?: number;
  feature: string;
  display_name: string;
  step: number | null;
  group: "R" | "D" | "EQ" | string;
  ranking_basis: string;
  score: number | null;
  signed_association: number | null;
  direction: string;
  valid_count: number | null;
  missing_count: number | null;
  missing_rate: number | null;
  category_count: number | null;
  is_categorical: boolean;
  p_value?: number | null;
  fdr_p_value?: number | null;
  effect_size?: number | null;
  mean_abs_shap?: number | null;
  pearson?: number | null;
  spearman?: number | null;
  coverage?: number | null;
  reason?: string | null;
  source_features?: string[];
  source_feature_count?: number | null;
};

export type EquipmentDistribution = {
  equipment: string;
  count: number;
  mean: number;
  median: number;
  q1: number;
  q3: number;
  minimum: number;
  maximum: number;
  whisker_min?: number;
  whisker_max?: number;
  outliers?: number[];
  outlier_count?: number;
  sample_warning: boolean | null;
};

export type RelationshipPath = {
  rank: number;
  step: number;
  response: string | null;
  defect: string;
  equipment: string | null;
  model?: string | null;
  chamber?: string | null;
  r_d: AssociationSummary | null;
  eq_d: AssociationSummary | null;
  d_y: AssociationSummary | null;
  r_y: AssociationSummary | null;
  eq_y: AssociationSummary | null;
  shap_importance: number;
  valid_count: number;
  missing_rate: number;
  confidence: "sufficient" | "caution" | "insufficient";
  path_score: number;
  path_status: string;
  interpretation: string;
  r_vs_d: { x: number; y: number }[];
  r_vs_y?: { x: number; y: number }[];
  eq_vs_d: EquipmentDistribution[];
  eq_vs_y?: EquipmentDistribution[];
  d_vs_y: { x: number; y: number }[];
};

export type NumericStatistic = {
  relation: string;
  feature: string;
  target: string;
  pearson: number | null;
  spearman: number | null;
  pearson_p_value: number | null;
  spearman_p_value: number | null;
  pearson_fdr_p_value: number | null;
  spearman_fdr_p_value: number | null;
  effect_size: number | null;
  valid_count: number;
  excluded_count: number;
  coverage: number;
  direction: string;
  strength: string;
  group?: "R" | "D" | "R_D" | string;
  scatter_data?: { x: number; y: number }[];
  scatter_sampled?: boolean;
  reason?: string | null;
};

export type CategoryTargetSummary = {
  category: string;
  count: number;
  coverage: number | null;
  mean: number | null;
  median: number | null;
  q1: number | null;
  q3: number | null;
  minimum: number | null;
  maximum: number | null;
  whisker_min: number | null;
  whisker_max: number | null;
  outliers: number[];
  outlier_count: number;
  sample_warning: boolean | null;
};

export type StatisticalTestResult = {
  statistic: number | null;
  p_value: number | null;
  fdr_p_value: number | null;
};

export type CategoricalStatistic = {
  relation: string;
  feature: string;
  target: string;
  valid_count: number;
  excluded_count: number;
  coverage: number;
  category_count: number;
  effect_size: number | null;
  anova: StatisticalTestResult;
  welch_anova: StatisticalTestResult;
  kruskal: StatisticalTestResult;
  group?: "Config" | string;
  source_type?: "raw_config" | "legacy_eq" | string;
  category_summary?: CategoryTargetSummary[];
  boxplot_data?: CategoryTargetSummary[];
  reason?: string | null;
};

export type RelationshipStatistics = {
  methods: string[];
  numeric: NumericStatistic[];
  categorical: CategoricalStatistic[];
  scatter_data: { x: number; y: number }[];
  boxplot_data: CategoryTargetSummary[];
  categorical_relationships: CategoricalStatistic[];
};

export type LotFeatureImportanceItem = {
  rank?: number;
  feature: string;
  display_name: string;
  step: string;
  group: string;
  mean_signed_shap: number;
  mean_abs_shap: number;
  adverse_contribution: number;
  improvement_contribution: number;
  sample_count: number;
  coverage: number;
  source_features: string[];
};

export type LotParetoItem = {
  rank?: number;
  feature: string;
  display_name: string;
  group: string;
  adverse_contribution: number;
  impact?: number;
  share: number;
  cumulative_share: number;
  within_threshold?: boolean;
  sample_count: number;
  coverage: number;
};

export type LotWaferItem = {
  identifier: unknown;
  lot_id: string;
  wafer_id: string | number | null;
  wafer_slot: number | null;
  prediction: number | null;
  predicted_value: number | null;
  predicted_yield: number | null;
  risk_level: "normal" | "warning" | "danger" | null;
  confidence: number | null;
  top_feature: string | null;
  top_step: string | null;
  top_config: string | null;
  shap_available: boolean | null;
};

export type LotCauseItem = {
  lot_id: string;
  wafer_count: number | null;
  analyzed_wafer_count: number | null;
  shap_coverage: number | null;
  average_predicted_value: number | null;
  average_predicted_yield: number | null;
  minimum_predicted_value: number | null;
  maximum_predicted_value: number | null;
  risk_extreme_predicted_value: number | null;
  risk_extreme_direction: "minimum" | "maximum" | null;
  critical_wafer_count: number | null;
  warning_wafer_count: number | null;
  normal_wafer_count: number | null;
  average_confidence: number | null;
  top_failure_target: string | null;
  top_failure_rate_target: string | null;
  top_failure_rate_average: number | null;
  top_fail_bit_count_target: string | null;
  top_fail_bit_count_average: number | null;
  feature_importance: Record<"all" | "r" | "d" | "config", LotFeatureImportanceItem[]>;
  pareto: Record<"all" | "r" | "d" | "config", LotParetoItem[]>;
  wafer_list: LotWaferItem[];
  top_causes: {
    feature: string | null;
    step: string | null;
    config: string | null;
    failure_target: string | null;
  };
};

export type LotCauseAnalysis = {
  target: string | null;
  aggregation: string | null;
  sampling_used: boolean | null;
  total_lot_count: number | null;
  excluded_row_count: number | null;
  lots: LotCauseItem[];
};

export type RelationshipAnalysisResponse = {
  success: boolean;
  filename: string;
  explanation: ExplainResponse | null;
  target: string;
  correlation_method: "pearson" | "spearman" | null;
  rankings: Record<
    "shap" | "correlation",
    Record<string, RelationshipFeature[]>
  >;
  pareto: {
    threshold: number;
    required_feature_count: number;
    cumulative_contribution: number;
    total_feature_count: number;
    total_impact: number;
    group_counts: Record<"R" | "D" | "EQ", number>;
    ranking_basis: string;
    caveat: string;
    features: (RelationshipFeature & {
      impact: number;
      share: number;
      cumulative_share: number;
      within_threshold: boolean;
    })[];
  };
  relationship_paths: RelationshipPath[];
  statistics: RelationshipStatistics;
  available_steps: number[];
  confidence_criteria: Record<string, string>;
  caveats: string[];
  analysis_unit: "wafer_observed_only" | "lot_aggregated" | null;
  config_summary: Record<string, unknown>;
  selection_bias_warnings: string[];
  analysis_result: AnalysisResult | null;
  report_snapshot: ReportResponse | null;
  lot_analysis: LotCauseAnalysis | null;
  analysis_id?: string | null;
  prediction_id?: string | null;
  history_saved?: boolean;
  history_warning?: string | null;
};

export type AnalysisHistoryDetail = {
  metadata: AnalysisHistorySummary & Record<string, unknown>;
  artifact: {
    response?: RelationshipAnalysisResponse;
    analysis_result?: AnalysisResult;
    report_snapshot?: ReportResponse;
    lot_analysis?: LotCauseAnalysis;
    [key: string]: unknown;
  } | null;
  source_prediction_deleted: boolean | null;
  linked_analysis_count?: number | null;
};

export type DashboardState =
  | "loading-history"
  | "history-empty"
  | "loading-analysis"
  | "ready"
  | "partial"
  | "analysis-not-found"
  | "artifact-missing"
  | "artifact-corrupted"
  | "api-error";

export type DashboardSectionState =
  | "loading"
  | "empty"
  | "unavailable"
  | "error"
  | "ready";

export type AnalysisSourceMetadata = {
  type: "analysis" | "empty";
  analysis_id: string | null;
  created_at: string | null;
  completed_at: string | null;
  status: string;
  source_filename: string | null;
  model_id: string | null;
  model_name: string | null;
  artifact_available: boolean;
  artifact_status: "available" | "missing" | "corrupted" | "not_applicable";
};

export type OverviewSummary = {
  wafer_count: number | null;
  lot_count: number | null;
  average_predicted_yield: number | null;
  minimum_predicted_yield: number | null;
  critical_count: number | null;
  warning_count: number | null;
  normal_count: number | null;
  low_confidence_count: number | null;
  risk_lot_count: number | null;
};

export type OverviewKpi = {
  label: string;
  value: number | null;
  unit?: string;
  detail: string;
  tone?: "default" | "warning" | "danger" | "normal";
};

export type OverviewModelMetrics = { r2: number | null; rmse: number | null; mae: number | null };

export type OverviewMultiY = {
  direct_y_mean: number | null;
  derived_y_mean: number | null;
  hybrid_y_mean: number | null;
  failure_rates: Record<string, number | null>;
  fail_bit_counts: Record<string, number | null>;
};

export type OverviewCauseItem = {
  rank?: number | null;
  feature?: string | null;
  display_name?: string | null;
  step?: string | number | null;
  equipment?: string | null;
  chamber?: string | null;
  mean_abs_shap?: number | null;
  impact?: number | null;
  score?: number | null;
  cumulative_share?: number | null;
  path_score?: number | null;
};

export type OverviewCauseSummary = {
  top_failure_target: string | null;
  top_features: OverviewCauseItem[];
  top_steps: OverviewCauseItem[];
  top_equipment: OverviewCauseItem[];
  top_chambers: OverviewCauseItem[];
};

export type OverviewRiskLot = {
  lot_id: string | null;
  wafer_count: number | null;
  average_predicted_yield: number | null;
  danger_count: number | null;
  warning_count: number | null;
  normal_count: number | null;
  danger_ratio: number | null;
  top_harmful_feature: string | null;
  top_harmful_step: string | null;
};

export type OverviewRiskWafer = {
  identifier: string | number | null;
  predicted_value: number | null;
  prediction: number | null;
  risk_level: string | null;
  top_harmful_features: string[];
  top_step: string | null;
};

export type OverviewRelationship = {
  relation: string | null;
  feature: string | null;
  target: string | null;
  response: string | null;
  defect: string | null;
  equipment: string | null;
  pearson: number | null;
  spearman: number | null;
  pearson_p_value: number | null;
  spearman_p_value: number | null;
  pearson_fdr_p_value: number | null;
  spearman_fdr_p_value: number | null;
  effect_size: number | null;
  path_score: number | null;
  valid_count: number | null;
  direction: string | null;
  interpretation: string | null;
};

export type OverviewAvailability = {
  summary: boolean;
  model_metrics: boolean;
  multi_y: boolean;
  causes: boolean;
  risk_lots: boolean;
  risk_wafers: boolean;
  pareto: boolean;
  relationships: boolean;
};

export type AnalysisOverviewResponse = {
  source: AnalysisSourceMetadata;
  summary: OverviewSummary;
  model_metrics: OverviewModelMetrics;
  multi_y: OverviewMultiY;
  causes: OverviewCauseSummary;
  risk_lots: OverviewRiskLot[];
  risk_wafers: OverviewRiskWafer[];
  pareto: OverviewCauseItem[];
  relationships: OverviewRelationship[];
  warnings: string[];
  availability: OverviewAvailability;
  source_type: "analysis" | "empty";
  source_id: string | null;
  created_at: string | null;
  source_label: string;
  filename: string | null;
  model: Record<string, unknown> | null;
  data_quality: Record<string, unknown>;
};

export type OverviewDashboardResponse = AnalysisOverviewResponse;

export type AnalysisResult = {
  analysis_id: string;
  analysis_version: string;
  created_at: string;
  model: {
    model_id: string;
    model_name: string | null;
    model_version: string | null;
    schema_version: string | null;
    compatibility: string | null;
    structure: string | null;
  };
  dataset: {
    filename: string;
    fingerprint: string | null;
    row_count: number;
    identifier_column: string | null;
  };
  target: { name: string; label: string; type: string | null; unit: string | null };
  metrics: Record<string, unknown>;
  multi_y: {
    average_direct_y: number | null;
    average_derived_y: number | null;
    average_ensemble_y: number | null;
    ensemble_weight: number | null;
    failure_rate_averages: Record<string, number | null>;
    fail_bit_count_averages: Record<string, number | null>;
    wafer_results: Array<{
      identifier: unknown;
      direct_y: number | null;
      derived_y: number | null;
      ensemble_y: number | null;
      direct_derived_gap: number | null;
      failure_rates: Record<string, number>;
      fail_bit_counts: Record<string, number>;
    }>;
  };
  risk: {
    warning_threshold: number;
    critical_threshold: number;
    normal_count: number;
    warning_count: number;
    critical_count: number;
    risk_probability: number | null;
  };
  confidence: Record<string, unknown>;
  feature_importance: Record<string, unknown>;
  shap: Record<string, unknown>;
  wafer_explanations: WaferExplanation[];
  relationships: RelationshipPath[];
  statistics: RelationshipStatistics;
  lot_analysis?: LotCauseAnalysis | null;
  risk_wafers: ReportRiskWafer[];
  lot_summary: ReportLotSummary[];
  data_quality: {
    r_measurement_coverage: number;
    d_measurement_coverage: number;
    config_completeness_rate: number;
    target_consistency_rate: number | null;
    config_parse_error_count: number;
    missing_indicator_used: boolean | null;
    outlier_policy: string | null;
    selection_bias_warnings: string[];
  };
  methodology: Record<string, unknown> & { notes?: string[] };
  report: { report_id: string | null; report_version: string | null };
  warnings: string[];
};

export type ReportOptions = {
  warning_threshold: number;
  danger_threshold: number;
  max_rows: number;
  top_n: number;
};

export type ReportExecutiveSummary = {
  total_wafers: number | null;
  average_predicted_yield: number;
  normal_count: number;
  warning_count: number;
  danger_count: number;
  risk_ratio: number;
  analyzed_rows: number | null;
  shap_sampling_used: boolean | null;
  sampling_strategy: string | null;
};

export type ReportModelInfo = {
  model_id: string;
  target: string;
  model_name: string;
  test_metrics: ModelMetrics;
};

export type ReportFinding = {
  severity: "info" | "warning" | "danger";
  title: string;
  description: string;
  evidence: string;
};

export type ReportRiskWafer = {
  identifier: unknown;
  predicted_value: number;
  risk_level: "normal" | "warning" | "danger" | null;
  actual_value: number | null;
  absolute_error: number | null;
  top_harmful_features: string[];
  top_step: string | null;
  top_parameter_type: string | null;
};

export type ReportLotSummary = {
  lot_id: string;
  wafer_count: number;
  average_predicted_yield: number;
  danger_count: number;
  warning_count: number;
  normal_count: number;
  danger_ratio: number;
  top_harmful_feature: string | null;
  top_harmful_step: string | null;
};

export type ReportRecommendation = {
  priority: "high" | "medium" | "low";
  title: string;
  description: string;
};

export type ReportResponse = {
  success: boolean;
  report_id: string;
  created_at: string;
  filename: string;
  model: ReportModelInfo;
  executive_summary: ReportExecutiveSummary;
  key_findings: ReportFinding[];
  top_risk_wafers: ReportRiskWafer[];
  lot_summary: ReportLotSummary[];
  top_features: GlobalImportanceItem[];
  top_steps: StepSummaryItem[];
  parameter_type_summary: (ParameterTypeSummaryItem & {
    ratio: number | null;
  })[];
  recommendations: ReportRecommendation[];
  model_quality_warnings: string[];
  methodology_notes: string[];
  explanation_method: string;
  is_fallback: boolean;
  warnings: string[];
  analysis_id: string | null;
  snapshot_metadata: Record<string, unknown> | null;
  lot_analysis?: LotCauseAnalysis | null;
  target_analysis?: {
    target: string | null;
    rankings: RelationshipAnalysisResponse["rankings"];
    pareto: RelationshipAnalysisResponse["pareto"];
    statistics: RelationshipStatistics;
  } | null;
  relationship_analysis?: {
    relationship_paths: RelationshipPath[];
    statistics: RelationshipStatistics;
  } | null;
};

export type AlertStatus = "New" | "Acknowledged" | "Resolved";
export type ExternalDeliveryStatus = "Not Configured" | "Pending" | "Sent" | "Failed";

export type AlertLogItem = {
  alert_id: string;
  created_at: string;
  analysis_id: string;
  model_id: string;
  model_version: string | null;
  lot_wafer_id: string;
  lot_id: string | null;
  predicted_y: number | null;
  direct_y: number | null;
  derived_y: number | null;
  critical_probability: number | null;
  warning_probability: number | null;
  risk_level: "danger" | "warning";
  confidence: "high" | "medium" | "low" | null;
  top_failure_target: string | null;
  top_feature: string | null;
  top_step: string | null;
  top_equipment: string | null;
  status: AlertStatus;
  external_delivery_status: ExternalDeliveryStatus;
  acknowledged_at: string | null;
  resolved_at: string | null;
};

export type AlertSummary = {
  total: number;
  new_count: number;
  acknowledged_count: number;
  resolved_count: number;
  critical_count: number;
  warning_count: number;
  external_not_configured_count: number;
};

export type AlertListResponse = { items: AlertLogItem[]; total: number; limit: number; offset: number };
