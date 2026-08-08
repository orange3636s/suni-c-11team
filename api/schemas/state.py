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


class TrainingStateSaveResponse(StateSaveResponse):
    # H-3⑤: 자동 수집 주기 반영(스케줄러 reschedule/pause)은 상태 저장과
    # 별개 작업이다 -- 저장은 성공했지만 반영이 실패해도 `saved: true`만
    # 보이면 프런트가 "다음 실행부터 새 주기가 적용됐다"고 착각한다.
    schedule_applied: bool


class LatestStateResponse(BaseModel):
    # Each is null when nothing has been saved yet (or the stored record's
    # schema_version is stale) -- never a 404 (spec §3-3).
    training: dict[str, Any] | None
    analysis: dict[str, Any] | None
    alarms: dict[str, Any] | None
    # 알림 연동 §D-3: 앱 마운트 시 1번의 요청으로 알림 설정도 함께 복원한다
    # -- 설정 패널을 위한 별도 요청을 만들지 않는다.
    notifications: dict[str, Any]
    # 지시서 CB: 저장된 레코드 중 하나 이상이 더 이상 존재하지 않는
    # 데이터셋(삭제된 내장 데이터셋 등)을 가리켜 통째로 버려졌으면 true.
    # 프론트가 "이전에 선택한 데이터셋이 더 이상 없어 train으로
    # 전환했습니다" 안내를 띄우는 신호로만 쓰고, 화면을 조용히 비우지
    # 않는다.
    dataset_fallback_applied: bool = False
    # D-2: 복원 자체가 실패했으면(DB 손상, 조회 예외 등) true -- "저장된
    # 결과가 없음"과 구분해야 사용자가 결과가 사라진 줄 알고 (비싼)
    # 재분석을 다시 돌리지 않는다.
    degraded: bool = False
