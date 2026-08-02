from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ColumnDetectionResult(BaseModel):
    id: list[str] = Field(default_factory=list)
    r: list[str] = Field(default_factory=list)
    d: list[str] = Field(default_factory=list)
    eq: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    config: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    detected_columns: ColumnDetectionResult
    missing_required_columns: list[str] = Field(default_factory=list)
    duplicate_wafer_id_count: int = 0
    total_missing_count: int = 0
    overall_missing_rate: float = 0.0
    schema_version: str | None = None
    validation_mode: str | None = None
    config_completeness_rate: float = 0.0
    r_measurement_coverage: float = 0.0
    d_measurement_coverage: float = 0.0
    required_field_error_count: int = 0
    config_parse_error_count: int = 0
    target_consistency_rate: float | None = None
    lot_structure_consistency_rate: float | None = None
    duplicate_wafer_count: int = 0
    invalid_numeric_count: int = 0
    structural_unmeasured_count: int = 0


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
    schema_version: str | None = None
    measurement_coverage: dict[str, float] = Field(default_factory=dict)
    preprocessing_policy: dict[str, Any] = Field(default_factory=dict)
    config_summary: dict[str, Any] = Field(default_factory=dict)
    processing_summary: dict[str, Any] = Field(default_factory=dict)


class ModelMetrics(BaseModel):
    r2: float | None
    rmse: float | None
    mae: float | None
    mse: float | None = None


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


class MetricAggregate(BaseModel):
    mean: float
    std: float


class MetricSummary(BaseModel):
    r2: MetricAggregate | None = None
    rmse: MetricAggregate | None = None
    mae: MetricAggregate | None = None
    mse: MetricAggregate | None = None


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    metric_summary: MetricSummary | None = None
    aggregate_metrics: MetricSummary | None = None


class CVSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    group_column: str | None = None
    outer_folds: int | None = None
    inner_folds: int | None = None
    seed: int | None = None
    metric_summary: MetricSummary | None = None
    aggregate_metrics: MetricSummary | None = None


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
    schema_version: str = "semicon_yield_v2"
    missingness_sensitivity: dict[str, Any] | None = None
    evaluation_summary: EvaluationSummary | None = Field(default_factory=EvaluationSummary)
    model_id: str | None = None
    model_type: str | None = None
    selected_final_output: str | None = None
    cv: CVSummary | None = None
    cv_summary: CVSummary | None = None
    final_y_metrics: dict[str, Any] | None = None
    target_metrics: dict[str, Any] | None = None
    risk_metrics: dict[str, Any] | None = None
    preprocessing: dict[str, Any] | None = None
    ensemble: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def align_cv_compatibility_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        evaluation = data.get("evaluation_summary")
        cv = data.get("cv") or data.get("cv_summary")
        evaluation_dict = dict(evaluation) if isinstance(evaluation, dict) else {}
        cv_dict = dict(cv) if isinstance(cv, dict) else {}
        if not cv_dict and evaluation_dict:
            cv_dict = dict(evaluation_dict)
        if not evaluation_dict and cv_dict:
            evaluation_dict = dict(cv_dict)
        metric_summary = (
            evaluation_dict.get("metric_summary")
            or evaluation_dict.get("aggregate_metrics")
            or cv_dict.get("metric_summary")
            or cv_dict.get("aggregate_metrics")
        )
        if metric_summary is None:
            return data
        evaluation_dict["metric_summary"] = metric_summary
        cv_dict["metric_summary"] = metric_summary
        cv_dict["aggregate_metrics"] = metric_summary
        data["evaluation_summary"] = evaluation_dict
        data["cv"] = cv_dict
        data["cv_summary"] = cv_dict
        return data


class ModelSummary(BaseModel):
    model_id: str
    target: str
    model_name: str
    created_at: str
    test_metrics: ModelMetrics
    feature_count: int
    compatibility: str = "unknown_schema"
    schema_version: str | None = None
    model_type: str | None = None
    bundle_type: str | None = None
    selected_final_output: str | None = None
    cv_summary: dict[str, Any] | None = None
    available_targets: list[str] = Field(default_factory=list)
    available: bool = True
    loadable: bool = True
    compatibility_status: str = "unknown_schema"
    incompatibility_reason: str | None = None


class ModelListResponse(BaseModel):
    success: bool = True
    models: list[ModelSummary] = Field(default_factory=list)
    total: int = 0
    warnings: list[str] = Field(default_factory=list)


class ModelDeleteResponse(BaseModel):
    deleted: bool = True
    model_id: str
    deleted_files: list[str] = Field(default_factory=list)
    missing_files: list[str] = Field(default_factory=list)
    metadata_deleted: bool = False
    bundle_deleted: bool = False
    # 이전 Frontend 계약과의 호환을 위해 동일 목록을 alias로 유지한다.
    removed_files: list[str] = Field(default_factory=list)
    registry_removed: bool = True
    prediction_history_kept: bool = True
    analysis_history_kept: bool = True
    prediction_history_count: int = 0
    analysis_history_count: int = 0


class ModelDetailMetrics(BaseModel):
    r2: float | None = None
    rmse: float | None = None
    mse: float | None = None
    mae: float | None = None


class ModelDetailResponse(BaseModel):
    success: bool = True
    model_id: str
    model_name: str | None = None
    model_type: str | None = None
    model_version: str | None = None
    created_at: str | None = None
    target: str | None = None
    feature_count: int | None = None
    feature_names: list[str] = Field(default_factory=list)
    dataset_split: dict[str, float] = Field(default_factory=dict)
    dataset_rows: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, ModelDetailMetrics] = Field(default_factory=dict)
    random_seed: int | None = None
    split_method: str | None = None
    preprocessing_version: str | None = None
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    training_time_seconds: float | None = None
    source_filename: str | None = None
    model_file: str | None = None
    metadata_file: str | None = None
    storage_status: str
    champion: bool | None = None
    sklearn_version: str | None = None
    compatibility: str = "unknown_schema"
    schema_version: str | None = None
    schema_fingerprint: str | None = None
    config_parser_version: str | None = None
    missing_indicator_used: bool | None = None
    outlier_policy: str | None = None
    group_column: str | None = None
    target_leakage_check: dict[str, Any] = Field(default_factory=dict)
    ensemble_enabled: bool | None = None
    ensemble_mode: str | None = None
    ensemble_method: str | None = None
    target_ensemble_configs: dict[str, Any] = Field(default_factory=dict)
    target_metrics: dict[str, Any] = Field(default_factory=dict)
    outer_fold_metrics: list[Any] = Field(default_factory=list)
    inner_fold_metrics: list[Any] = Field(default_factory=list)
    available_targets: list[str] = Field(default_factory=list)
    cv_summary: dict[str, Any] = Field(default_factory=dict)
    ensemble_weights: dict[str, Any] = Field(default_factory=dict)
    hybrid_summary: dict[str, Any] = Field(default_factory=dict)
    risk_metrics: dict[str, Any] = Field(default_factory=dict)
    preprocessing_summary: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    model_agreement_summary: dict[str, Any] = Field(default_factory=dict)
    production_ensemble_retrained: bool | None = None
    available: bool = True
    loadable: bool = True
    compatibility_status: str = "unknown_schema"
    incompatibility_reason: str | None = None


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
    preprocessing: dict[str, Any] | None = None
    prediction_id: str | None = None
    history_saved: bool = False
    history_warning: str | None = None
    artifact_available: bool | None = None
    preview_row_count: int | None = None


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
    lot_id: str | None = None
    wafer_id: str | None = None
    wafer_slot: int | None = None
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
    explanation: ExplainResponse | None = None
    target: str
    correlation_method: str
    rankings: dict[str, dict[str, list[dict[str, Any]]]]
    pareto: dict[str, Any]
    relationship_paths: list[dict[str, Any]]
    statistics: dict[str, Any] = Field(default_factory=dict)
    available_steps: list[int]
    confidence_criteria: dict[str, str]
    caveats: list[str]
    analysis_unit: str = "wafer_observed_only"
    config_summary: dict[str, Any] = Field(default_factory=dict)
    selection_bias_warnings: list[str] = Field(default_factory=list)
    lot_analysis: dict[str, Any] = Field(default_factory=dict)
    analysis_result: dict[str, Any] | None = None
    analysis_id: str | None = None
    prediction_id: str | None = None
    history_saved: bool = False
    history_warning: str | None = None
    artifact_available: bool | None = None


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


class AnalyzeResponse(BaseModel):
    success: bool = True
    analysis_id: str
    created_at: str
    filename: str
    model: AnalyzeModelInfo
    summary: AnalyzeSummary
    alert: AnalyzeAlert
    automation_message: AutomationMessage
    top_findings: list[dict[str, Any]]
    top_risk_wafers: list[dict[str, Any]]
    top_features: list[GlobalImportanceItem]
    top_steps: list[StepSummaryItem]
    parameter_type_summary: list[dict[str, Any]]
    model_quality_warnings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
