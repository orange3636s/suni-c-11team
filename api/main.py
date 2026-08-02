import logging
import os

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from api.routes.data import router as data_router
from api.routes.runtime import router as runtime_router
from api.settings import settings


APP_VERSION = "1.0.0"
SERVICE_NAME = "manufacturing-ai-api"

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

if (
    settings.app_env.lower() == "production"
    and "FRONTEND_ORIGINS" not in os.environ
):
    logger.warning(
        "FRONTEND_ORIGINS is not set in production; only development "
        "origins are allowed."
    )

app = FastAPI(
    title="Manufacturing AI API",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(data_router)
app.include_router(runtime_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "제조 공정 불량 예측 및 원인 분석 AI"}


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "environment": settings.app_env,
        "version": APP_VERSION,
        "model_directory_ready": settings.model_directory_ready(),
    }


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
