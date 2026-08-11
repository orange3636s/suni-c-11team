from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.routes.analysis import _action_priority_payload, _fmea_payload
from api.routes.datasets import get_dataset_registry
from api.schemas.datasets import DatasetUploadResponse
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
from src.analysis import alarm_gbdt
from src.analysis.screening.schema import ALL_TARGET_COLUMNS
from src.automation import sql_source
from src.automation.refresh import is_refresh_running, run_refresh_pipeline
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
            "automation": {
                "enabled": False,
                "sql_host": "",
                "sql_port": "",
                "sql_db": "",
                "sql_user": "",
                "refresh_interval_minutes": 60,
                "last_run_at": None,
                "last_run_status": None,
                "last_run_sent_count": None,
            },
            "telegram_bot_username": None,
        }
    return {**state, "notifications": notifications, "dataset_fallback_applied": dataset_fallback_applied, "degraded": degraded}


@router.post("/training", response_model=TrainingStateSaveResponse)
def save_training_state(body: TrainingStateSaveRequest) -> dict[str, bool]:
    """SD그룹: SQL 연결·refresh time·자동화 on/off는 더 이상 이 슬롯에
    저장하지 않는다("알림·자동화 설정" 팝업이 `POST /api/notify/automation`
    으로 저장한다, 전용 슬롯 `automation:settings`) -- 이 라우트는 모델
    학습 팝업의 학습 성능 요약(performance)만 남긴다. `schedule_applied`는
    더 이상 이 저장이 스케줄러를 건드리지 않으므로 항상 true다(자동화
    잡의 재스케줄은 `POST /api/notify/automation`이 전담한다)."""
    saved = save_state(_store(), "training", dataset={"dataset": body.dataset}, payload=body.payload)
    return {"saved": saved, "schedule_applied": True}


def _with_fmea(dataset: str, payload: dict[str, Any]) -> dict[str, Any]:
    """지시서 JA-1: FMEA 분석표는 자동 갱신(`src/automation/refresh.py`)과
    수동 "다시 분석"(`POST /api/state/analysis`) 두 경로가 같은 계산을
    공유해야 한다 -- 프런트가 이미 `fmea`를 실어 보냈으면(현재는 안
    그런다, 앞으로도 프런트에서 계산하지 않는다) 그대로 두고, 없으면
    저장 시점에 여기서 채운다.

    JA-2: FMEA 계산 실패가 분석 저장 전체를 막으면 안 된다 -- Pareto·
    계측 확대는 이미 계산이 끝나 `payload`에 담겨 있으므로, 여기서
    예외가 나도 그 값들은 그대로 저장한다. 실패 사유는 `fmeaError`에
    남겨 화면이 "계산 안 됨"과 "계산 실패"를 구분할 수 있게 한다.
    """
    if payload.get("fmea") is not None:
        return payload
    try:
        fmea = _fmea_payload(dataset, ALL_TARGET_COLUMNS)
        payload = {
            **payload,
            "fmea": fmea,
            "fmeaError": None,
            "targetProvenance": fmea.get("target_provenance"),
        }
    except Exception:
        logger.exception("분석 저장: FMEA 분석표 계산 실패 dataset=%s", dataset)
        payload = {**payload, "fmea": None, "fmeaError": "FMEA 분석표 계산 중 오류가 발생했습니다."}
    return payload


def _with_action_priority(payload: dict[str, Any]) -> dict[str, Any]:
    """MB/MC: 모니터링 홈 블록①·②는 항상 train.CSV 기준(작업 지시서
    MB-6)이라 저장하려는 분석의 eval 데이터셋과 무관하다 -- `_with_fmea`와
    같은 "이미 있으면 건너뛰고, 없으면 저장 시점에 채우고, 실패해도
    나머지 저장은 막지 않는다" 정책을 따른다."""
    if payload.get("actionPriority") is not None:
        return payload
    try:
        action_priority = _action_priority_payload("train")
        return {**payload, "actionPriority": action_priority, "actionPriorityError": None}
    except Exception:
        logger.exception("분석 저장: 조치 우선순위 계산 실패")
        return {**payload, "actionPriority": None, "actionPriorityError": "조치 우선순위 계산 중 오류가 발생했습니다."}


@router.post("/analysis", response_model=StateSaveResponse)
def save_analysis_state(body: AnalysisStateSaveRequest) -> dict[str, bool]:
    payload = _with_fmea(body.dataset, body.payload)
    payload = _with_action_priority(payload)
    saved = save_state(_store(), "analysis", dataset={"dataset": body.dataset}, payload=payload)
    return {"saved": saved}


@router.post("/alarms", response_model=StateSaveResponse)
def save_alarms_state(body: AlarmsStateSaveRequest) -> dict[str, bool]:
    # 지시서 JD-2③: 이 요청이 오는 시점 자체가 이미 "사용자가 실제로
    # 조작했다"는 신호다 -- 프런트(alerts/page.tsx)가 userModified 플래그로
    # 가드해, 복원·초기화로 인한 setState는 이 엔드포인트를 아예 부르지
    # 않는다(JD-2②). 여기서는 그 저장이 어느 기본값 세대에서 만들어졌는지만
    # 감사용으로 찍어 둔다 -- 이 값으로 자동 무효화하지 않는다(진짜 사용자
    # 선택을 지울 위험).
    payload = {**body.payload, "defaultsVersion": alarm_gbdt.ALARM_DEFAULTS_VERSION}
    saved = save_state(
        _store(),
        "alarms",
        dataset={"train_dataset": body.train_dataset, "eval_dataset": body.eval_dataset},
        payload=payload,
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
        # SC-3: "모델 분석" 팝업의 [분석 시작] 버튼이 이 값으로 disabled
        # 여부·진행 표시를 정한다 -- 서버 기동 부트스트랩·학습 후 자동
        # 복구 실행이 돌고 있어도 true가 된다(같은 락을 공유).
        "refresh_running": is_refresh_running(),
        # SF-3: 네 화면(모니터링/트리맵/원인분석/수율예측)이 공유하는
        # 진행 표시("분석 진행 중… (2/4) 원인 분석") -- 실행 중이 아니면
        # null.
        "analysis_progress": store.get_analysis_progress(),
        # AG-3/AG-4: 등록 직후(파이프라인이 아직 도는 중이라 스냅샷의
        # source.mode가 "manual"로 바뀌기 전)에도 배너가 "수동 모드"를
        # 바로 보여줄 수 있게, 스냅샷과 별개로 override 자체를 싣는다.
        "manual_eval_override": store.get_manual_eval_override(),
    }


@router.post("/refresh")
def trigger_refresh(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """SC-3: "모델 분석" 팝업의 [분석 시작] 버튼이 부르는 단일 진입점 --
    네 화면(모니터링/Config별 트리맵/원인 분석/수율 예측)을 한 번에
    갱신하는 유일한 실행 경로다("새로고침 역할" 겸함 -- 서버 지연으로
    화면이 비었을 때도 이 버튼으로 복구한다). 실행 자체는 백그라운드로
    넘겨 응답을 블록하지 않는다 -- 팝업을 닫아도 계속 진행되고, 완료는
    `GET /api/state/snapshot/meta` 폴링이 감지한다.
    """
    if is_refresh_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="분석이 이미 진행 중입니다.",
        )
    background_tasks.add_task(run_refresh_pipeline)
    return {"triggered": True}


@router.post("/activate-dataset")
def activate_dataset(body: ActivateDatasetRequest) -> dict[str, Any]:
    """SC-2: "모델 분석" 팝업에서 파일을 선택하거나 데이터베이스에서
    불러오면 이 엔드포인트가 불려 그 데이터셋을 활성 분석 데이터로
    등록한다("한 번 등록되면 다시 바꿀 때까지 유지된다"). SC-3와
    분리했다 -- 등록만으로 4화면 분석이 자동으로 돌지 않는다. 실제
    계산은 사용자가 [분석 시작]을 눌러 `POST /api/state/refresh`를
    호출해야 시작된다. 학습은 절대 걸지 않는다(RB-3)."""
    registry = get_dataset_registry()
    summary = registry.get_summary(body.dataset_id)
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.")
    store = _store()
    store.set_manual_eval_override(body.dataset_id, summary["original_filename"])
    return {"activated": True, "dataset_id": body.dataset_id}


@router.post("/fetch-from-db", response_model=DatasetUploadResponse)
def fetch_from_db() -> dict[str, Any]:
    """SC-2 "데이터베이스에서 불러오기" -- "알림·자동화 설정"에 등록된
    서버와 같은 소스(`src/automation/sql_source.py`)에서 최신 배치를
    가져와 데이터셋으로 등록한다("자동화가 쓰는 것과 같은 소스다"). 등록만
    하고 활성화하지 않는다 -- 프런트가 이 응답의 `dataset_id`로
    `POST /api/state/activate-dataset`를 이어서 호출해야 분석 데이터로
    등록된다(파일 업로드 경로와 동일한 2단계).

    커서 기반 증분 조회(`fetch_incremental`)를 그대로 쓴다 -- 주기
    자동화 잡과 같은 커서 상태를 공유하므로, 자동화가 이미 가장 최근
    배치를 가져간 직후라면 "새 데이터 없음"일 수 있다(그 경우 404)."""
    store = _store()
    if not sql_source.is_sql_configured(store):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="서버가 설정되지 않았습니다. 알림·자동화 설정에서 서버를 먼저 등록하세요.",
        )
    dataframe = sql_source.fetch_incremental(store)
    if dataframe is None:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="데이터베이스 접속에 실패했습니다.")
    if dataframe.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="가져올 새 데이터가 없습니다.")

    registry = get_dataset_registry()
    filename = f"db_fetch_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.csv"
    content = dataframe.to_csv(index=False).encode("utf-8")
    return registry.upload(filename, content)


@router.post("/deactivate-dataset")
def deactivate_dataset() -> dict[str, Any]:
    """수동 override를 지워 다음 [분석 시작] 실행부터 SQL/폴백(내장
    test.CSV)으로 되돌아가게 한다. SC-2/SC-3 분리 이후 등록만 수행하고
    분석을 자동으로 다시 돌리지 않는다 -- 되돌아간 결과를 보려면
    사용자가 [분석 시작]을 눌러야 한다."""
    store = _store()
    cleared = store.clear_manual_eval_override()
    return {"deactivated": cleared}


@router.get("/snapshot")
def get_snapshot() -> dict[str, Any]:
    """J-3: schema_version이 다르면(백엔드가 바뀐 뒤 남은 옛 스냅샷)
    복원하지 않고 `stale_version: true`만 알린다 -- 조용히 빈 화면을
    보여주지 않고, 다음 갱신 주기에 새 스키마로 다시 채워진다는 것을
    프런트가 안내할 수 있게 한다."""
    status = _store().get_refresh_snapshot_status()
    return {
        "snapshot": status["snapshot"],
        "stale_version": status["stale_version"],
        "stale_model": bool(status.get("stale_model")),
    }
