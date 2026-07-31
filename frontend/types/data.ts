export type ColumnDetectionResult = {
  id: string[];
  r: string[];
  d: string[];
  eq: string[];
  targets: string[];
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
};

export type ModelMetrics = {
  r2: number | null;
  rmse: number | null;
  mae: number | null;
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

export type TrainResponse = {
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
};

export type ModelSummary = {
  model_id: string;
  target: string;
  model_name: string;
  created_at: string;
  test_metrics: ModelMetrics;
  feature_count: number;
};

export type ModelListResponse = {
  success: boolean;
  models: ModelSummary[];
  warnings: string[];
};

export type ModelDetailMetrics = {
  r2: number | null;
  rmse: number | null;
  mse: number | null;
  mae: number | null;
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
  feature_names: string[];
  dataset_split: Record<string, number> | null;
  dataset_rows: Record<string, number> | null;
  metrics: Record<string, ModelDetailMetrics>;
  random_seed: number | null;
  split_method: string | null;
  preprocessing_version: string | null;
  preprocessing_config: Record<string, unknown> | null;
  training_time_seconds: number | null;
  source_filename: string | null;
  model_file: string | null;
  metadata_file: string | null;
  storage_status: string;
  champion: boolean | null;
  sklearn_version: string | null;
};

export type PredictionThresholds = {
  warning_threshold: number;
  danger_threshold: number;
};

export type PredictionRow = Record<string, unknown> & {
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
  sample_warning: boolean;
};

export type RelationshipPath = {
  rank: number;
  step: number;
  response: string | null;
  defect: string;
  equipment: string | null;
  r_d: AssociationSummary | null;
  eq_d: AssociationSummary | null;
  d_y: AssociationSummary;
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

export type RelationshipAnalysisResponse = {
  success: boolean;
  filename: string;
  explanation: ExplainResponse;
  target: string;
  correlation_method: "pearson" | "spearman";
  rankings: Record<
    "shap" | "correlation",
    Record<"overall" | "R" | "D" | "EQ", RelationshipFeature[]>
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
  available_steps: number[];
  confidence_criteria: Record<string, string>;
  caveats: string[];
};

export type ReportOptions = {
  warning_threshold: number;
  danger_threshold: number;
  max_rows: number;
  top_n: number;
};

export type ReportExecutiveSummary = {
  total_wafers: number;
  average_predicted_yield: number;
  normal_count: number;
  warning_count: number;
  danger_count: number;
  risk_ratio: number;
  analyzed_rows: number;
  shap_sampling_used: boolean;
  sampling_strategy: string;
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
};
