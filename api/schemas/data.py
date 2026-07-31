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


class RelationshipAnalysisResponse(BaseModel):
    success: bool = True
    filename: str
    explanation: ExplainResponse
    target: str
    correlation_method: str
    rankings: dict[str, dict[str, list[dict[str, Any]]]]
    pareto: dict[str, Any]
    relationship_paths: list[dict[str, Any]]
    available_steps: list[int]
    confidence_criteria: dict[str, str]
    caveats: list[str]


class ReportExecutiveSummary(BaseModel):
    total_wafers: int
    average_predicted_yield: float
    normal_count: int
    warning_count: int
    danger_count: int
    risk_ratio: float
    analyzed_rows: int
    shap_sampling_used: bool
    sampling_strategy: str


class ReportModelInfo(BaseModel):
    model_id: str
    target: str
    model_name: str
    test_metrics: ModelMetrics


class ReportFinding(BaseModel):
    severity: str
    title: str
    description: str
    evidence: str


class ReportRiskWafer(BaseModel):
    identifier: Any
    predicted_value: float
    risk_level: str | None
    actual_value: float | None = None
    absolute_error: float | None = None
    top_harmful_features: list[str]
    top_step: str | None = None
    top_parameter_type: str | None = None


class ReportLotSummary(BaseModel):
    lot_id: str
    wafer_count: int
    average_predicted_yield: float
    danger_count: int
    warning_count: int
    normal_count: int
    danger_ratio: float
    top_harmful_feature: str | None = None
    top_harmful_step: str | None = None


class ReportRecommendation(BaseModel):
    priority: str
    title: str
    description: str


class ReportParameterTypeSummary(ParameterTypeSummaryItem):
    ratio: float | None


class ReportResponse(BaseModel):
    success: bool = True
    report_id: str
    created_at: str
    filename: str
    model: ReportModelInfo
    executive_summary: ReportExecutiveSummary
    key_findings: list[ReportFinding]
    top_risk_wafers: list[ReportRiskWafer]
    lot_summary: list[ReportLotSummary]
    top_features: list[GlobalImportanceItem]
    top_steps: list[StepSummaryItem]
    parameter_type_summary: list[ReportParameterTypeSummary]
    recommendations: list[ReportRecommendation]
    model_quality_warnings: list[str] = Field(default_factory=list)
    methodology_notes: list[str] = Field(default_factory=list)
    explanation_method: str
    is_fallback: bool
    warnings: list[str] = Field(default_factory=list)


class AnalyzeModelInfo(BaseModel):
    model_id: str
    target: str
    model_name: str
    test_r2: float | None
    test_rmse: float | None


class AnalyzeSummary(BaseModel):
    total_wafers: int
    average_predicted_yield: float
    normal_count: int
    warning_count: int
    danger_count: int
    risk_count: int
    risk_ratio: float
    minimum_predicted_yield: float | None


class AnalyzeAlert(BaseModel):
    required: bool
    severity: str
    reason: str
    danger_count: int
    warning_count: int


class AutomationMessage(BaseModel):
    title: str
    summary: str
    detail: str
    top_cause: str


class AnalyzeReportReference(BaseModel):
    included: bool
    report_id: str | None
    download_endpoint: str | None


class AnalyzeResponse(BaseModel):
    success: bool = True
    analysis_id: str
    created_at: str
    filename: str
    model: AnalyzeModelInfo
    summary: AnalyzeSummary
    alert: AnalyzeAlert
    automation_message: AutomationMessage
    top_findings: list[ReportFinding]
    top_risk_wafers: list[ReportRiskWafer]
    top_features: list[GlobalImportanceItem]
    top_steps: list[StepSummaryItem]
    parameter_type_summary: list[ReportParameterTypeSummary]
    model_quality_warnings: list[str] = Field(default_factory=list)
    report: AnalyzeReportReference
    warnings: list[str] = Field(default_factory=list)
