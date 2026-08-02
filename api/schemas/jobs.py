from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TrainJobStatusValue = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "interrupted",
]


class TrainJobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class TrainJobMetrics(BaseModel):
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    mse: float | None = None


class TrainJobResult(BaseModel):
    model_id: str
    target: str
    best_model: str
    test_metrics: TrainJobMetrics | None = None
    feature_count: int
    warning_count: int


class TrainJobStatus(BaseModel):
    job_id: str
    status: TrainJobStatusValue
    stage: str
    progress: int = Field(ge=0, le=100)
    elapsed_seconds: float = Field(ge=0)
    result: TrainJobResult | None = None
    error: str | None = None

