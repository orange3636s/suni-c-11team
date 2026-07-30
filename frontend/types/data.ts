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
