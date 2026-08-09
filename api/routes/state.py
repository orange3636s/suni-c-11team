from __future__ import annotations

import logging
from typing import Any

from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from api.routes.datasets import get_dataset_registry
from api.schemas.state import (
    ActivateDatasetRequest,
    AlarmsStateSaveRequest,
    AnalysisStateSaveRequest,
    LatestStateResponse,
    StateSaveResponse,
    TrainingStateSaveRequest,
    TrainingStateSaveResponse,
)
from api.settings import settings
from src.automation.ingest import AUTO_INGEST_JOB_ID
from src.automation.refresh import REFRESH_JOB_ID, is_refresh_running, run_refresh_pipeline
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
            "conditions": {"grades": ["심각"], "timing": ["on_analysis"]},
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
        # AF: 모니터링의 "최신화" 버튼이 이 값으로 disabled 여부·툴팁을
        # 정한다 -- 주기 잡이 돌고 있어도 true가 된다(같은 락을 공유).
        "refresh_running": is_refresh_running(),
        # AG-3/AG-4: 활성화 직후(파이프라인이 아직 도는 중이라 스냅샷의
        # source.mode가 "manual"로 바뀌기 전)에도 배너가 "수동 모드"를
        # 바로 보여줄 수 있게, 스냅샷과 별개로 override 자체를 싣는다.
        "manual_eval_override": store.get_manual_eval_override(),
    }


@router.post("/refresh")
def trigger_refresh(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """AF/AG: 모니터링의 "최신화" 버튼과 업로드 연동이 부르는 단일
    진입점 -- 주기 잡과 같은 `run_refresh_pipeline`을 그대로 재사용한다
    (별도 파이프라인을 만들지 않는다). 실행 자체는 백그라운드로 넘겨
    응답을 블록하지 않는다 -- 완료는 기존 `created_at` 폴링이 감지한다.
    """
    if is_refresh_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="자동 갱신이 이미 진행 중입니다.",
        )
    background_tasks.add_task(run_refresh_pipeline)
    return {"triggered": True}


@router.post("/activate-dataset")
def activate_dataset(body: ActivateDatasetRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """AG: 원인 분석·알림 기록에서 새 파일을 업로드하면 이 엔드포인트가
    불린다 -- "업로드는 활성 평가 데이터셋 전환이다"(지시서 AG-1). 화면별로
    개별 재분석을 걸지 않고, 스냅샷 파이프라인을 한 번만 실행해 세 화면이
    같은 결과를 공유하게 한다. 학습은 절대 걸지 않는다(AG-2) -- 여기서는
    평가 데이터셋 포인터만 바꾸고, `_maybe_retrain`은 train_dataset를
    건드리지 않으므로 자동 학습 트리거가 없다."""
    registry = get_dataset_registry()
    summary = registry.get_summary(body.dataset_id)
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.")
    if is_refresh_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="자동 갱신이 이미 진행 중입니다. 잠시 후 다시 시도하세요.",
        )
    store = _store()
    store.set_manual_eval_override(body.dataset_id, summary["original_filename"])
    background_tasks.add_task(run_refresh_pipeline)
    return {"activated": True, "dataset_id": body.dataset_id}


@router.post("/deactivate-dataset")
def deactivate_dataset(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """AG-3: "자동 갱신으로 복귀" -- 수동 override를 지우고, 다음 파이프라인
    실행부터 SQL/폴백으로 되돌아간다. 되돌아간 결과를 바로 보여주기 위해
    한 번 즉시 실행한다."""
    store = _store()
    cleared = store.clear_manual_eval_override()
    if is_refresh_running():
        return {"deactivated": cleared, "triggered": False}
    background_tasks.add_task(run_refresh_pipeline)
    return {"deactivated": cleared, "triggered": True}


@router.get("/snapshot")
def get_snapshot() -> dict[str, Any]:
    """J-3: schema_version이 다르면(백엔드가 바뀐 뒤 남은 옛 스냅샷)
    복원하지 않고 `stale_version: true`만 알린다 -- 조용히 빈 화면을
    보여주지 않고, 다음 갱신 주기에 새 스키마로 다시 채워진다는 것을
    프런트가 안내할 수 있게 한다."""
    status = _store().get_refresh_snapshot_status()
    return {"snapshot": status["snapshot"], "stale_version": status["stale_version"]}
