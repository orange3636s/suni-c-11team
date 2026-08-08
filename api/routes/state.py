from __future__ import annotations

import logging
from typing import Any

from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, Request

from api.routes.datasets import get_dataset_registry
from api.schemas.state import (
    AlarmsStateSaveRequest,
    AnalysisStateSaveRequest,
    LatestStateResponse,
    StateSaveResponse,
    TrainingStateSaveRequest,
    TrainingStateSaveResponse,
)
from api.settings import settings
from src.automation.ingest import AUTO_INGEST_JOB_ID
from src.automation.refresh import REFRESH_JOB_ID
from src.notifications.settings_store import get_settings_summary
from src.runtime.app_state import get_latest_state, is_state_degraded, save_state
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/state", tags=["state"])

# 각 저장 종류가 dataset_id를 싣는 필드 (지시서 CB) -- 하나라도 더 이상
# 존재하지 않는 데이터셋을 가리키면 그 레코드를 통째로 버린다.
_DATASET_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "training": ("dataset",),
    "analysis": ("dataset",),
    "alarms": ("train_dataset", "eval_dataset"),
}


def _store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


def _drop_records_for_missing_datasets(state: dict[str, Any]) -> bool:
    """지시서 CB: 저장된 결과가 이미 삭제된 데이터셋(구버전 내장
    데이터셋 등)을 가리키면 그 결과를 통째로 버린다 -- dataset 필드만
    "train"으로 바꿔치기하면, 사라진 데이터셋 스키마로 계산된 옛
    payload(예: 다른 Pareto 인자 목록)가 "train" 라벨을 달고 화면에
    뜨는 더 나쁜 상황이 된다(부분 복원 금지). `_dataframe_or_404`가
    그대로 404를 던지게 두면 앱이 빈 화면으로 굳으므로(예외 무시가
    아니라 폴백), 여기서 미리 걸러 반환값을 "저장된 적 없음"과 같은
    모양(null)으로 만들되, 프론트가 "폐기됐다"를 안내할 수 있도록
    별도 플래그를 반환한다."""
    registry = get_dataset_registry()
    fallback_applied = False
    for kind, fields in _DATASET_FIELDS_BY_KIND.items():
        record = state.get(kind)
        if not record:
            continue
        if any(registry.get_summary(record.get(field)) is None for field in fields):
            state[kind] = None
            fallback_applied = True
    return fallback_applied


@router.get("/latest", response_model=LatestStateResponse)
def get_latest() -> dict[str, Any]:
    """Called once per app mount (spec §4-2/§6) -- all three tabs' latest
    results in a single round trip so the frontend never needs to decide
    which ones to ask for.
    """
    store = _store()
    # D-2: 복원 실패(DB 손상 등)와 "저장된 결과 없음"을 구분해야 한다 --
    # 안 그러면 사용자는 결과가 사라진 줄 알고 (비싼) 재분석을 다시
    # 돌린다. get_latest_state가 예외를 던지는 경우와, 값은 읽혔지만
    # JSON이 깨져 조용히 None으로 매핑된 경우 둘 다 degraded로 잡는다.
    degraded = False
    try:
        state = get_latest_state(store)
        degraded = is_state_degraded(store)
    except Exception:
        logger.warning("최근 결과 조회 실패", exc_info=True)
        state = {"training": None, "analysis": None, "alarms": None}
        degraded = True
    try:
        dataset_fallback_applied = _drop_records_for_missing_datasets(state)
    except Exception:
        # 데이터셋 레지스트리 조회 자체가 실패해도 저장된 결과를 그대로
        # 돌려준다 -- 이 방어 로직 때문에 정상 복원까지 막히면 안 된다.
        logger.warning("데이터셋 존재 여부 확인 실패", exc_info=True)
        dataset_fallback_applied = False
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
    return {**state, "notifications": notifications, "dataset_fallback_applied": dataset_fallback_applied, "degraded": degraded}


def _apply_ingest_schedule(request: Request, refresh_interval_minutes: Any) -> bool:
    """자동 수집 파이프라인 §1-2: 팝업에서 주기를 바꾸면 서버 재시작
    없이 다음 실행 간격에 반영한다. `null`이면 잡을 일시정지한다.
    스케줄러가 아직 뜨지 않은 상태(테스트 등)에서도 조용히 넘어간다
    (그 경우는 반영 "실패"가 아니라 애초에 반영 대상이 없는 것이므로
    True를 반환한다). H-3⑤: 실제 reschedule/pause 호출이 예외를 던지면
    False를 반환해 호출부가 응답에 반영 실패를 실어 보낼 수 있게 한다.
    """
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return True
    try:
        if isinstance(refresh_interval_minutes, (int, float)) and refresh_interval_minutes > 0:
            trigger = IntervalTrigger(minutes=refresh_interval_minutes)
            scheduler.reschedule_job(AUTO_INGEST_JOB_ID, trigger=trigger)
            scheduler.resume_job(AUTO_INGEST_JOB_ID)
            # J-2: 리프레시 파이프라인도 같은 주기를 따른다 -- 별도 잡
            # id(REFRESH_JOB_ID)라 auto_ingest와 독립적으로 겹쳐 돌 수
            # 있지만, 사용자가 설정하는 주기 값은 하나뿐이다.
            scheduler.reschedule_job(REFRESH_JOB_ID, trigger=trigger)
            scheduler.resume_job(REFRESH_JOB_ID)
        else:
            scheduler.pause_job(AUTO_INGEST_JOB_ID)
            scheduler.pause_job(REFRESH_JOB_ID)
        return True
    except Exception:
        logger.exception("자동 수집 주기 반영 실패")
        return False


@router.post("/training", response_model=TrainingStateSaveResponse)
def save_training_state(body: TrainingStateSaveRequest, request: Request) -> dict[str, bool]:
    saved = save_state(_store(), "training", dataset={"dataset": body.dataset}, payload=body.payload)
    schedule_applied = _apply_ingest_schedule(request, body.payload.get("refreshIntervalMinutes"))
    return {"saved": saved, "schedule_applied": schedule_applied}


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


# -- J-3/J-4: 자동 갱신 파이프라인 스냅샷 -----------------------------


@router.get("/snapshot/meta")
def get_snapshot_meta() -> dict[str, Any]:
    """J-4: 프런트가 윈도우 포커스 복귀·60초 주기마다 부르는 가벼운
    엔드포인트 -- `created_at`만 돌려주고 본문 전체는 싣지 않는다.
    프런트는 이 값이 캐시된 값보다 최신일 때만 `GET /api/state/snapshot`
    전체를 다시 받는다.

    W-4: `bootstrap`은 첫 기동 부트스트랩(스냅샷이 아직 없을 때 1회
    학습+분석)의 진행 상태다 -- 스냅샷이 이미 있었던 적이 없으면(부트
    스트랩이 아직 시작 전이거나, 이미 끝나 상태를 지운 뒤 그대로 두는
    구버전 배포 등) null이다. 프런트는 이 값이 없거나 status가
    "done"이면 배너를 감춘다."""
    store = _store()
    meta = store.get_refresh_snapshot_meta()
    return {
        "created_at": meta.get("created_at") if meta else None,
        "bootstrap": store.get_bootstrap_status(),
    }


@router.get("/snapshot")
def get_snapshot() -> dict[str, Any]:
    """J-3: schema_version이 다르면(백엔드가 바뀐 뒤 남은 옛 스냅샷)
    복원하지 않고 `stale_version: true`만 알린다 -- 조용히 빈 화면을
    보여주지 않고, 다음 갱신 주기에 새 스키마로 다시 채워진다는 것을
    프런트가 안내할 수 있게 한다."""
    status = _store().get_refresh_snapshot_status()
    return {"snapshot": status["snapshot"], "stale_version": status["stale_version"]}
