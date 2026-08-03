from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    dataset_id: str
    kind: str  # "bundled" | "uploaded"
    original_filename: str
    uploaded_at: str | None
    row_count: int
    column_count: int
    lot_min: str | None
    lot_max: str | None
    lot_count: int | None
    warnings: list[str] = Field(default_factory=list)
    unmapped_columns: list[str] = Field(default_factory=list)
    deletable: bool


class DatasetListResponse(BaseModel):
    items: list[DatasetSummary]


class DatasetUploadResponse(BaseModel):
    success: bool
    dataset_id: str | None
    blocking_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unmapped_columns: list[str] = Field(default_factory=list)
    row_count: int | None = None
    column_count: int | None = None
    lot_min: str | None = None
    lot_max: str | None = None
    lot_count: int | None = None


class DatasetDeleteResponse(BaseModel):
    success: bool
    dataset_id: str


class DatasetSchemaResponse(BaseModel):
    dataset_id: str
    steps_present: list[int]
    max_step: int
    r_columns: list[str]
    d_columns: list[str]
    config_columns: list[str]
    target_columns: list[str]
    unmapped_columns: list[str]
    missing_rates: dict[str, float]
