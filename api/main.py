import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routes.data import (
    recover_interrupted_training_jobs,
    router as data_router,
)
from api.routes.analysis import router as analysis_router
from api.routes.chat import router as chat_router
from api.routes.datasets import get_dataset_registry, router as datasets_router
from api.routes.favorites import router as favorites_router
from api.routes.monitoring import router as monitoring_router
from api.routes.notify import (
    router as notify_router,
    run_daily_13_dispatch_job,
    run_daily_dispatch_job,
    run_notify_log_cleanup_job,
)
from api.routes.state import router as state_router
from api.settings import APP_VERSION, ENV_FILE_LOADED, settings
from src.automation.ingest import AUTO_INGEST_JOB_ID, DEFAULT_INGEST_MINUTES, run_auto_ingest_job
from src.automation.refresh import REFRESH_JOB_ID, run_refresh_pipeline
from src.notifications.telegram_bot import run_polling_loop
from src.runtime.app_state import get_latest_state
from src.runtime.datasets import BUNDLED_DATASET_FILES
from src.runtime.operation_coordinator import (
    HEAVY_JOB_MESSAGE,
    ActiveOperationError,
    OperationKind,
    operation_coordinator,
)
from src.runtime.migrations import run_startup_migrations
from src.runtime.store import RuntimeStore


SERVICE_NAME = "manufacturing-ai-api"

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(".env %s", "loaded" if ENV_FILE_LOADED else "not found (using OS env)")


def _is_deployment_environment() -> bool:
    return (
        settings.app_env.lower() == "production"
        or bool(os.environ.get("RAILWAY_ENVIRONMENT_ID"))
        or os.environ.get("RENDER", "").lower() == "true"
    )

if (
    _is_deployment_environment()
    and "FRONTEND_ORIGINS" not in os.environ
):
    logger.warning(
        "FRONTEND_ORIGINS is not set in production; only development "
        "origins are allowed."
    )

# H-3①: `asyncio.create_task(...)`의 반환값을 아무 데도 보관하지 않으면
# 그 Task 객체를 참조하는 것이 이벤트 루프의 내부 약한 참조뿐이라, GC가
# 도중에 태스크를 수거해 조용히 중단시킬 수 있다(asyncio 공식 문서가
# 명시적으로 경고하는 함정). 모듈 레벨 집합에 강한 참조를 들고 있다가
# 완료되면 콜백에서 스스로 빼도록 한다.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _warm_bundled_dataset_cache() -> None:
    """Pre-populates the module-level bundled-CSV cache (see
    src/runtime/datasets.py) so the first real request doesn't pay the
    full read+normalize cost for all bundled datasets. Runs in a
    threadpool from a fire-and-forget background task -- it must never
    delay startup or the Railway health check would fail and roll back
    the deploy.
    """
    registry = get_dataset_registry()
    for dataset_id in BUNDLED_DATASET_FILES:
        t0 = time.perf_counter()
        registry.get_dataframe(dataset_id)
        logger.info(
            "dataset warmup: %s ready in %.1fms", dataset_id, (time.perf_counter() - t0) * 1000
        )


async def _warmup_datasets_background() -> None:
    t0 = time.perf_counter()
    try:
        await asyncio.to_thread(_warm_bundled_dataset_cache)
    except Exception:
        logger.exception("데이터셋 워밍업 실패")
        return
    logger.info("dataset warmup complete in %.1fms", (time.perf_counter() - t0) * 1000)


# W-2: 첫 기동 스냅샷 부트스트랩 -- `run_refresh_pipeline`(기존 자동화
# 파이프라인)을 그대로 재사용한다. 새 파이프라인을 만들지 않는다.


def _bootstrap_runtime_store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


def _has_usable_champion(store: RuntimeStore) -> bool:
    """NE-4: `store.active_model()`은 DB의 `model_slots` 행이 있는지만
    보고, 그 `active_model_id`가 가리키는 아티팩트가 `models/`에 실제로
    있는지는 확인하지 않는다. 레지스트리 행만 남고 파일이 없는 상태(예:
    로컬 테스트로 생성된 모델을 커밋하지 않은 채 배포한 경우)에서는
    `active_model()`이 여전히 값을 반환해 `_run_bootstrap`이 재학습을
    건너뛰고, 이후 `run_refresh_pipeline`의 모델 로드가 조용히 실패해
    스냅샷이 저장되지 않는다 -- 그 결과가 "첫 스냅샷 생성 실패" 배너다.
    여기서 실제로 로드를 시도해 그 간극을 메운다."""
    active = store.active_model()
    model_id = str((active or {}).get("active_model_id") or "").strip()
    if not model_id:
        return False
    from src.ml.inference import InferenceInputError, ModelLoadError, load_prediction_model

    try:
        load_prediction_model(model_id, settings.model_dir)
    except (InferenceInputError, ModelLoadError):
        logger.warning("bootstrap: 챔피언 모델 ID는 있으나 로드할 수 없습니다 (%s) -- 재학습합니다.", model_id)
        return False
    return True


async def _train_bootstrap_champion() -> None:
    """NE-4 근본 원인: `run_refresh_pipeline`은 더 이상 학습을 트리거하지
    않는다(RB-4: 승격 게이트 제거 이후 학습은 "모델 학습" 팝업의 수동
    업로드 경로 하나뿐이다). 콜드 스타트가 옛 주석(W-2)대로 `run_refresh_
    pipeline`이 학습 Job을 제출하길 기다리기만 하면 챔피언이 전혀 없는
    환경(신규 배포, 또는 챔피언 포인터는 있으나 아티팩트가 없는 경우)에서
    영원히 끝나지 않는다 -- 실제로 재현했다. 수동 업로드와 같은 코드
    경로(`train_model`)를 내장 train.CSV로 그대로 재사용해 새 학습
    파이프라인을 만들지 않는다."""
    from fastapi import UploadFile

    from api.routes.data import train_model

    train_path = settings.bundled_dataset_dir / BUNDLED_DATASET_FILES["train"]
    with train_path.open("rb") as source:
        upload = UploadFile(file=source, filename=BUNDLED_DATASET_FILES["train"])
        await train_model(upload)


async def _run_bootstrap(store: RuntimeStore) -> None:
    store.set_bootstrap_status("running", "데이터 확인 중")
    try:
        # W-3: 챔피언 모델이 이미 있으면(볼륨은 살아있는데 스냅샷만
        # 없는 경우 등) 재학습하지 않는다 -- `run_refresh_pipeline` 내부의
        # 데이터 해시 비교(조건부 재학습)와 같은 원칙이다.
        if not _has_usable_champion(store):
            store.set_bootstrap_status("running", "학습 중")
            # WK-5: 콜드 스타트 결과로는 알림을 보내지 않는다.
            await _train_bootstrap_champion()
        store.set_bootstrap_status("running", "평가 · 원인분석 중")
        await asyncio.to_thread(run_refresh_pipeline, dispatch=False)
        if not store.has_valid_snapshot():
            raise RuntimeError("첫 스냅샷 생성에 실패했습니다 (원인분석/알람 판정 단계를 확인하세요).")
        store.set_bootstrap_status("done", None)
        logger.info("bootstrap: 첫 스냅샷 생성 완료")
    except Exception as exc:
        logger.exception("bootstrap: 첫 스냅샷 생성 실패")
        store.set_bootstrap_status("failed", None, error=str(exc))


async def _bootstrap_snapshot_background() -> None:
    """유효 스냅샷이 없을 때만, 단 한 번 실행한다. 기동을 블록하지 않도록
    `_spawn_background_task`로 fire-and-forget한다(89행 주석과 동일한
    이유 -- 헬스체크가 실패하면 배포가 롤백된다). NE-6: 수동 override로
    활성화된 평가 데이터셋이 있으면 그 자체가 사용자의 최근 작업이므로
    내장 데이터로 부트스트랩을 돌려 덮어쓰지 않는다."""
    store = _bootstrap_runtime_store()
    try:
        if store.has_valid_snapshot() or store.get_manual_eval_override() is not None:
            return
        if not store.acquire_bootstrap_lock():
            logger.info("bootstrap: 다른 인스턴스가 이미 진행 중이라 건너뜁니다.")
            return
        try:
            await _run_bootstrap(store)
        finally:
            store.release_bootstrap_lock()
    except Exception:
        logger.exception("bootstrap: 예기치 않은 오류")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        recover_interrupted_training_jobs()
    except Exception:
        logger.exception("중단된 학습 Job 시작 복구 실패")
    _spawn_background_task(_warmup_datasets_background())
    if _is_deployment_environment():
        try:
            results = run_startup_migrations(
                model_dir=settings.model_dir,
                store=RuntimeStore(
                    settings.runtime_db_path,
                    settings.runtime_artifact_dir,
                ),
            )
            logger.info("Startup Migration 결과: %s", results)
        except Exception:
            # Migration failures are recorded and logged, but readiness and
            # health endpoints must remain available for investigation.
            logger.exception("Startup Migration 실행 실패")

    # W-2: 첫 기동 스냅샷 부트스트랩 -- 유효 스냅샷이 있으면 내부에서
    # 즉시 반환한다(재학습 없음). 마이그레이션 다음에 두는 이유는 모델
    # 디렉터리·DB 스키마가 이미 정리된 상태에서 판단하기 위해서다.
    _spawn_background_task(_bootstrap_snapshot_background())

    # 알람 알림 연동 §C-3 Telegram -- 봇 토큰이 설정된 경우에만 long-polling
    # 루프를 띄운다. 설정되지 않은 개발/테스트 환경에서는 조용히 건너뛴다.
    telegram_stop_event = asyncio.Event()
    telegram_task: asyncio.Task | None = None
    if settings.telegram_bot_token:
        telegram_task = asyncio.create_task(run_polling_loop(settings.telegram_bot_token, telegram_stop_event))
        logger.info("Telegram 봇 polling 시작")
        # EA-5: 토큰만 있고 username이 없으면 폴링은 도는데 화면에는 봇
        # 링크가 안 보이는 어중간한 상태가 된다 -- 조용히 넘어가지 않고
        # 기동 로그에 남긴다.
        if not settings.telegram_bot_username:
            logger.warning("Telegram 알림 비활성: TELEGRAM_BOT_USERNAME 미설정")
    else:
        logger.info("TELEGRAM_BOT_TOKEN이 설정되지 않아 Telegram 알림 연동을 건너뜁니다.")
        if settings.telegram_bot_username:
            logger.warning("Telegram 알림 비활성: TELEGRAM_BOT_TOKEN 미설정")

    # 알람 알림 연동 §C-4 "매일 오전 9시" (지시서 N-2: 8시 -> 9시) -- n8n
    # 대신 APScheduler로 처리한다 (spec: "서비스가 늘면 메모리와 요금이
    # 증가한다").
    scheduler = AsyncIOScheduler()
    # A-5: 배포 컨테이너(Railway/Render)는 TZ가 UTC라 timezone 없이 hour=9를
    # 주면 KST 18시에 발송된다. misfire_grace_time도 기본값(1초)이라 09:00
    # 정각에 재시작이 겹치면 그날 발송이 통째로 스킵된다.
    scheduler.add_job(
        run_daily_dispatch_job,
        CronTrigger(hour=9, minute=0, timezone="Asia/Seoul"),
        id="daily_alarm_notification",
        misfire_grace_time=3600,
    )
    # DF그룹: 발송 시점 다중 선택에 "매일 오후 1시"가 추가됐다 -- 09:00 잡과
    # 별개 id로 등록한다. 실제로 보낼지는 run_daily_13_dispatch_job ->
    # dispatch_alarm_notifications가 저장된 conditions.timing에
    # TIMING_DAILY_13이 포함돼 있는지로 판단하므로, 이 잡 자체는 항상
    # 등록해 두고 조건 필터는 그쪽에 맡긴다(09:00 잡과 같은 패턴).
    scheduler.add_job(
        run_daily_13_dispatch_job,
        CronTrigger(hour=13, minute=0, timezone="Asia/Seoul"),
        id="daily_alarm_notification_13",
        misfire_grace_time=3600,
    )
    # H-3②: notify_sent_log 정리 -- 발송 잡과 겹치지 않도록 별도 id·시각으로
    # 등록한다(같은 id를 쓰면 재등록 시 서로 덮어쓴다).
    scheduler.add_job(
        run_notify_log_cleanup_job,
        CronTrigger(hour=3, minute=0, timezone="Asia/Seoul"),
        id="notify_sent_log_cleanup",
        misfire_grace_time=3600,
    )

    # 자동 수집 파이프라인 1단계 -- 별도 잡이다(위 발송 잡을 건드리지
    # 않는다). 주기는 환경변수가 아니라 저장된 사용자 설정
    # (state/training의 refreshIntervalMinutes)을 따른다 -- 없으면
    # DEFAULT_INGEST_MINUTES로 일단 등록해 두고 일시정지한다(설정된 적
    # 없다는 뜻이므로 자동 실행은 하지 않는다). 값이 있으면 그 주기로
    # 등록한다. max_instances=1 + coalesce=True로 학습 중복 실행을 막는다.
    initial_minutes: int | None = None
    try:
        training_state = get_latest_state(
            RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
        ).get("training")
        if training_state:
            raw_minutes = (training_state.get("payload") or {}).get("refreshIntervalMinutes")
            if isinstance(raw_minutes, (int, float)) and raw_minutes > 0:
                initial_minutes = int(raw_minutes)
    except Exception:
        logger.exception("자동 수집 주기 설정 조회 실패 -- 일시정지 상태로 시작합니다.")

    scheduler.add_job(
        run_auto_ingest_job,
        IntervalTrigger(minutes=initial_minutes or DEFAULT_INGEST_MINUTES),
        id=AUTO_INGEST_JOB_ID,
        max_instances=1,
        coalesce=True,
    )
    # J-2: 리프레시 파이프라인 -- 위 auto_ingest(감시 디렉터리 폴링)와는
    # 별개 잡이다. 같은 refreshIntervalMinutes를 따르므로 초기 등록도
    # 같은 initial_minutes를 쓴다. max_instances=1로 이전 사이클이 아직
    # 도는 중이면(학습이 오래 걸리는 경우 등) 겹쳐 실행하지 않는다.
    scheduler.add_job(
        run_refresh_pipeline,
        IntervalTrigger(minutes=initial_minutes or DEFAULT_INGEST_MINUTES),
        id=REFRESH_JOB_ID,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.start()
    if initial_minutes is None:
        scheduler.pause_job(AUTO_INGEST_JOB_ID)
        scheduler.pause_job(REFRESH_JOB_ID)
    # 저장 API(POST /api/state/training) 핸들러가 주기 변경 시
    # reschedule/pause할 수 있도록 앱 상태에 둔다 -- 서버 재시작을
    # 요구하지 않는다.
    app.state.scheduler = scheduler

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        if telegram_task is not None:
            telegram_stop_event.set()
            telegram_task.cancel()
            try:
                await telegram_task
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(
    title="Manufacturing AI API",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# GET /api/state/latest bundles 3 tabs' worth of results in one response
# (spec §3-3: "응답 크기가 100 KB를 넘으면 gzip 압축을 확인한다") -- applies
# to every response over 1KB, not just that endpoint, since it's free
# insurance for every other JSON payload too.
app.add_middleware(GZipMiddleware, minimum_size=1024)
_PROTECTED_OPERATION_PATHS: dict[str, OperationKind] = {
    "/api/train": "training",
}


@app.middleware("http")
async def protect_active_operations(request: Request, call_next):
    kind = (
        _PROTECTED_OPERATION_PATHS.get(request.url.path)
        if request.method == "POST"
        else None
    )
    if kind is None:
        return await call_next(request)
    try:
        with operation_coordinator.job(kind):
            return await call_next(request)
    except ActiveOperationError as exc:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc) or HEAVY_JOB_MESSAGE},
        )


app.include_router(analysis_router)
app.include_router(chat_router)
app.include_router(datasets_router)
app.include_router(data_router)
app.include_router(favorites_router)
app.include_router(monitoring_router)
app.include_router(notify_router)
app.include_router(state_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "제조 공정 불량 예측 & 원인 분석 AI"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness_check(response: Response) -> dict[str, str | bool]:
    model_directory_ready = settings.model_directory_ready()
    if not model_directory_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if model_directory_ready else "not_ready",
        "service": SERVICE_NAME,
        "model_directory_ready": model_directory_ready,
    }
