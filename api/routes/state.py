from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from api.schemas.state import (
    AlarmsStateSaveRequest,
    AnalysisStateSaveRequest,
    LatestStateResponse,
    StateSaveResponse,
    TrainingStateSaveRequest,
)
from api.settings import settings
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


@router.post("/training", response_model=StateSaveResponse)
def save_training_state(body: TrainingStateSaveRequest) -> dict[str, bool]:
    saved = save_state(_store(), "training", dataset={"dataset": body.dataset}, payload=body.payload)
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
