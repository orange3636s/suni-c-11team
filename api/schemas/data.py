from typing import Any

from pydantic import BaseModel, Field


class ColumnDetectionResult(BaseModel):
    id: list[str] = Field(default_factory=list)
    r: list[str] = Field(default_factory=list)
    d: list[str] = Field(default_factory=list)
    eq: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    detected_columns: ColumnDetectionResult
    missing_required_columns: list[str] = Field(default_factory=list)
    duplicate_wafer_id_count: int = 0
    total_missing_count: int = 0
    overall_missing_rate: float = 0.0


class ValidationResponse(BaseModel):
    success: bool = True
    filename: str
    row_count: int
    column_count: int
    validation: ValidationResult


class DataSummary(BaseModel):
    row_count: int
    column_count: int
    missing_count: int


class PreprocessChanges(BaseModel):
    filled_missing_values: int
    clipped_outliers: int
    added_indicator_columns: list[str] = Field(default_factory=list)


class PreprocessResponse(BaseModel):
    success: bool = True
    filename: str
    before: DataSummary
    after: DataSummary
    changes: PreprocessChanges
    warnings: list[str] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)


class ModelMetrics(BaseModel):
    r2: float | None
    rmse: float | None
    mae: float | None


class DatasetSplit(BaseModel):
    train_rows: int
    validation_rows: int
    test_rows: int
    group_split_used: bool
    split_method: str


class ModelComparisonItem(BaseModel):
    model_name: str
    status: str
    validation: ModelMetrics | None = None
    selected: bool
    error_message: str | None = None


class ModelArtifacts(BaseModel):
    model_file: str
    metadata_file: str


class TrainResponse(BaseModel):
    success: bool = True
    target: str
    best_model: str
    split: DatasetSplit
    metrics: dict[str, ModelMetrics]
    model_comparison: list[ModelComparisonItem]
    feature_count: int
    warnings: list[str] = Field(default_factory=list)
    artifacts: ModelArtifacts


class ModelSummary(BaseModel):
    model_id: str
    target: str
    model_name: str
    created_at: str
    test_metrics: ModelMetrics
    feature_count: int


class ModelListResponse(BaseModel):
    success: bool = True
    models: list[ModelSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PredictionModelInfo(BaseModel):
    model_id: str
    target: str
    model_name: str


class PredictionSummary(BaseModel):
    total_rows: int
    average_prediction: float
    normal_count: int
    warning_count: int
    danger_count: int
    evaluation: ModelMetrics | None = None


class PredictionResponse(BaseModel):
    success: bool = True
    filename: str
    model: PredictionModelInfo
    summary: PredictionSummary
    identifier_column: str
    predictions: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False


class ExplainAnalysisSummary(BaseModel):
    total_rows: int
    analyzed_rows: int
    sampling_used: bool
    sampling_strategy: str
    explanation_method: str
    is_fallback: bool


class GlobalImportanceItem(BaseModel):
    rank: int
    feature: str
    step: str
    parameter_type: str
    parameter_name: str
    mean_abs_shap: float
    mean_harmful_contribution: float
    direction: str


class AggregateImportanceItem(BaseModel):
    rank: int
    mean_abs_shap: float
    harmful_contribution: float
    feature_count: int


class StepSummaryItem(AggregateImportanceItem):
    step: str


class ParameterTypeSummaryItem(AggregateImportanceItem):
    parameter_type: str


class EquipmentSummaryItem(BaseModel):
    rank: int
    equipment: str
    mean_abs_shap: float
    harmful_contribution: float


class LocalContributionItem(BaseModel):
    feature: str
    value: Any
    shap_value: float
    harmful_contribution: float
    beneficial_contribution: float
    step: str
    parameter_type: str


class WaferExplanation(BaseModel):
    identifier: Any
    prediction: float
    risk_level: str | None
    base_value: float
    top_negative_contributors: list[LocalContributionItem]
    top_positive_contributors: list[LocalContributionItem]


class ExplainModelInfo(BaseModel):
    model_id: str
    target: str
    model_name: str


class ExplainResponse(BaseModel):
    success: bool = True
    filename: str
    model: ExplainModelInfo
    analysis_summary: ExplainAnalysisSummary
    explanation_method: str
    is_fallback: bool
    identifier_column: str
    global_importance: list[GlobalImportanceItem]
    step_summary: list[StepSummaryItem]
    parameter_type_summary: list[ParameterTypeSummaryItem]
    equipment_summary: list[EquipmentSummaryItem]
    wafer_explanations: list[WaferExplanation]
    model_quality_warnings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
