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
