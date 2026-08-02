from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalysisHistorySummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis_id: str
    prediction_id: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    status: str
    source_filename: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    model_name_snapshot: str | None = None
    row_count: int | None = None
    lot_count: int | None = None
    average_predicted_yield: float | None = None
    critical_count: int | None = None
    warning_wafer_count: int | None = None
    top_failure_target: str | None = None
    artifact_available: bool = False
    default_target: str | None = None
    warning_count: int = 0
    summary: dict[str, Any] | None = None


class AnalysisHistoryListResponse(BaseModel):
    items: list[AnalysisHistorySummary]
    total: int
    limit: int
    offset: int


class AnalysisSourceMetadata(BaseModel):
    type: Literal["analysis", "empty"]
    analysis_id: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    status: str
    source_filename: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    artifact_available: bool = False
    artifact_status: Literal["available", "missing", "corrupted", "not_applicable"]


class OverviewSummary(BaseModel):
    wafer_count: int | None = None
    lot_count: int | None = None
    average_predicted_yield: float | None = None
    minimum_predicted_yield: float | None = None
    critical_count: int | None = None
    warning_count: int | None = None
    normal_count: int | None = None
    low_confidence_count: int | None = None
    risk_lot_count: int | None = None


class OverviewModelMetrics(BaseModel):
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None


class OverviewMultiY(BaseModel):
    predicted_y_mean: float | None = None
    failure_rates: dict[str, float | None] = Field(default_factory=dict)
    fail_bit_counts: dict[str, float | None] = Field(default_factory=dict)


class OverviewCauseSummary(BaseModel):
    top_failure_target: str | None = None
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    top_steps: list[dict[str, Any]] = Field(default_factory=list)
    top_equipment: list[dict[str, Any]] = Field(default_factory=list)
    top_chambers: list[dict[str, Any]] = Field(default_factory=list)


class OverviewAvailability(BaseModel):
    summary: bool = False
    model_metrics: bool = False
    multi_y: bool = False
    causes: bool = False
    risk_lots: bool = False
    risk_wafers: bool = False
    pareto: bool = False
    relationships: bool = False


class AnalysisOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: AnalysisSourceMetadata
    summary: OverviewSummary
    model_metrics: OverviewModelMetrics
    multi_y: OverviewMultiY
    causes: OverviewCauseSummary
    risk_lots: list[dict[str, Any]] = Field(default_factory=list)
    risk_wafers: list[dict[str, Any]] = Field(default_factory=list)
    pareto: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    availability: OverviewAvailability

    # Backward-compatible aliases for existing consumers.
    source_type: Literal["analysis", "empty"]
    source_id: str | None = None
    created_at: str | None = None
    source_label: str
    filename: str | None = None
    model: dict[str, Any] | None = None
    data_quality: dict[str, Any] = Field(default_factory=dict)
