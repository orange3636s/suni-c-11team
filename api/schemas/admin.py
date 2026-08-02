from __future__ import annotations

from pydantic import BaseModel


class HistoryResetRequest(BaseModel):
    confirmation: str | None = None


class HistoryResetSummary(BaseModel):
    model_count: int
    prediction_history_count: int
    analysis_history_count: int
    model_artifact_count: int
    prediction_artifact_count: int
    analysis_artifact_count: int


class HistoryResetDeleted(BaseModel):
    models: int
    model_files: int
    prediction_histories: int
    prediction_artifacts: int
    analysis_histories: int
    analysis_artifacts: int


class HistoryResetPreserved(BaseModel):
    alert_logs: bool
    automation_runs: bool
    source_csv: bool


class HistoryResetResponse(BaseModel):
    success: bool
    deleted: HistoryResetDeleted
    preserved: HistoryResetPreserved
