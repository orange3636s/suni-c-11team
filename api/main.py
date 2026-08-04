import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.data import (
    recover_interrupted_training_jobs,
    router as data_router,
)
from api.routes.analysis import router as analysis_router
from api.routes.chat import router as chat_router
from api.routes.datasets import router as datasets_router
from api.settings import APP_VERSION, settings
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

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        recover_interrupted_training_jobs()
    except Exception:
        logger.exception("중단된 학습 Job 시작 복구 실패")
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
    yield


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
