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
