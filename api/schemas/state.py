from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TrainingStateSaveRequest(BaseModel):
    dataset: str
    payload: dict[str, Any]


class AnalysisStateSaveRequest(BaseModel):
    dataset: str
    payload: dict[str, Any]


class AlarmsStateSaveRequest(BaseModel):
    train_dataset: str
    eval_dataset: str
    payload: dict[str, Any]


class StateSaveResponse(BaseModel):
    saved: bool


class LatestStateResponse(BaseModel):
    # Each is null when nothing has been saved yet (or the stored record's
    # schema_version is stale) -- never a 404 (spec §3-3).
    training: dict[str, Any] | None
    analysis: dict[str, Any] | None
    alarms: dict[str, Any] | None
    # 알림 연동 §D-3: 앱 마운트 시 1번의 요청으로 알림 설정도 함께 복원한다
    # -- 설정 패널을 위한 별도 요청을 만들지 않는다.
    notifications: dict[str, Any]
