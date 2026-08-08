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
from api.routes.notify import router as notify_router, run_daily_dispatch_job, run_notify_log_cleanup_job
from api.routes.state import router as state_router
from api.settings import APP_VERSION, ENV_FILE_LOADED, settings
from src.automation.ingest import AUTO_INGEST_JOB_ID, DEFAULT_INGEST_MINUTES, run_auto_ingest_job
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

    # 알람 알림 연동 §C-3 Telegram -- 봇 토큰이 설정된 경우에만 long-polling
    # 루프를 띄운다. 설정되지 않은 개발/테스트 환경에서는 조용히 건너뛴다.
    telegram_stop_event = asyncio.Event()
    telegram_task: asyncio.Task | None = None
    if settings.telegram_bot_token:
        telegram_task = asyncio.create_task(run_polling_loop(settings.telegram_bot_token, telegram_stop_event))
        logger.info("Telegram 봇 polling 시작")
    else:
        logger.info("TELEGRAM_BOT_TOKEN이 설정되지 않아 Telegram 알림 연동을 건너뜁니다.")

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
    scheduler.start()
    if initial_minutes is None:
        scheduler.pause_job(AUTO_INGEST_JOB_ID)
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
