from __future__ import annotations

import logging
from typing import Any

from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, Request

from api.schemas.state import (
    AlarmsStateSaveRequest,
    AnalysisStateSaveRequest,
    LatestStateResponse,
    StateSaveResponse,
    TrainingStateSaveRequest,
)
from api.settings import settings
from src.automation.ingest import AUTO_INGEST_JOB_ID
from src.notifications.settings_store import get_settings_summary
from src.runtime.app_state import get_latest_state, save_state
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/state", tags=["state"])


def _store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


@router.get("/latest", response_model=LatestStateResponse)
def get_latest() -> dict[str, Any]:
    """Called once per app mount (spec §4-2/§6) -- all three tabs' latest
    results in a single round trip so the frontend never needs to decide
    which ones to ask for.
    """
    store = _store()
    try:
        state = get_latest_state(store)
    except Exception:
        logger.warning("최근 결과 조회 실패", exc_info=True)
        state = {"training": None, "analysis": None, "alarms": None}
    try:
        notifications = get_settings_summary(store)
    except Exception:
        logger.warning("알림 설정 조회 실패", exc_info=True)
        notifications = {
            "slack": {"connected": False, "target": None, "webhook_masked": None, "verified_at": None},
            "telegram": {"connected": False, "target": None, "chat_id_masked": None, "verified_at": None},
            "gmail": {"connected": False, "pending": False, "email": None, "verified_at": None},
            "conditions": {"grades": ["심각"], "timing": "on_analysis"},
        }
    return {**state, "notifications": notifications}


def _apply_ingest_schedule(request: Request, refresh_interval_minutes: Any) -> None:
    """자동 수집 파이프라인 §1-2: 팝업에서 주기를 바꾸면 서버 재시작
    없이 다음 실행 간격에 반영한다. `null`이면 잡을 일시정지한다.
    스케줄러가 아직 뜨지 않은 상태(테스트 등)에서도 조용히 넘어간다.
    """
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return
    try:
        if isinstance(refresh_interval_minutes, (int, float)) and refresh_interval_minutes > 0:
            scheduler.reschedule_job(AUTO_INGEST_JOB_ID, trigger=IntervalTrigger(minutes=refresh_interval_minutes))
            scheduler.resume_job(AUTO_INGEST_JOB_ID)
        else:
            scheduler.pause_job(AUTO_INGEST_JOB_ID)
    except Exception:
        logger.exception("자동 수집 주기 반영 실패")


@router.post("/training", response_model=StateSaveResponse)
def save_training_state(body: TrainingStateSaveRequest, request: Request) -> dict[str, bool]:
    saved = save_state(_store(), "training", dataset={"dataset": body.dataset}, payload=body.payload)
    _apply_ingest_schedule(request, body.payload.get("refreshIntervalMinutes"))
    return {"saved": saved}


@router.post("/analysis", response_model=StateSaveResponse)
def save_analysis_state(body: AnalysisStateSaveRequest) -> dict[str, bool]:
    saved = save_state(_store(), "analysis", dataset={"dataset": body.dataset}, payload=body.payload)
    return {"saved": saved}


@router.post("/alarms", response_model=StateSaveResponse)
def save_alarms_state(body: AlarmsStateSaveRequest) -> dict[str, bool]:
    saved = save_state(
        _store(),
        "alarms",
        dataset={"train_dataset": body.train_dataset, "eval_dataset": body.eval_dataset},
        payload=body.payload,
    )
    return {"saved": saved}
