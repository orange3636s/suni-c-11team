from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src import upload_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH, override=False)
ENV_FILE_LOADED = _ENV_PATH.exists()

APP_VERSION = "1.0.0"
DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
VALID_LOG_LEVELS = {
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
}


def _parse_origins(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return DEFAULT_FRONTEND_ORIGINS
    origins = tuple(
        dict.fromkeys(
            origin.strip()
            for origin in raw_value.split(",")
            if origin.strip()
        )
    )
    return origins or DEFAULT_FRONTEND_ORIGINS


def _parse_bool(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name, "").strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "on"}


def _parse_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _storage_default(local_path: str, volume_path: str) -> Path:
    """Prefer an attached Railway Volume when no explicit path is set."""
    raw_mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if raw_mount:
        mount_path = Path(raw_mount).expanduser()
        if mount_path.is_absolute():
            return mount_path / volume_path
    return Path(local_path)


def _resolve_model_dir(raw_value: str | None) -> Path:
    configured_path = Path(
        raw_value or _storage_default("models", "models")
    ).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (PROJECT_ROOT / configured_path).resolve()


def _resolve_runtime_db(raw_value: str | None) -> Path:
    configured_path = Path(
        raw_value
        or _storage_default("data/runtime/dashboard.db", "runtime.db")
    ).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (PROJECT_ROOT / configured_path).resolve()


def _resolve_runtime_artifacts(raw_value: str | None) -> Path:
    configured_path = Path(
        raw_value or _storage_default("data/runtime", "artifacts")
    ).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (PROJECT_ROOT / configured_path).resolve()


def _resolve_training_jobs(raw_value: str | None) -> Path:
    configured_path = Path(
        raw_value
        or _storage_default(
            "data/runtime/training_jobs", "artifacts/training_jobs"
        )
    ).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (PROJECT_ROOT / configured_path).resolve()


def _resolve_dataset_upload_dir(raw_value: str | None) -> Path:
    configured_path = Path(
        raw_value or _storage_default("data/uploaded-datasets", "datasets")
    ).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (PROJECT_ROOT / configured_path).resolve()


@dataclass(frozen=True)
class Settings:
    app_env: str = field(
        default_factory=lambda: os.environ.get(
            "APP_ENV", "development"
        ).strip()
        or "development"
    )
    frontend_origins: tuple[str, ...] = field(
        default_factory=lambda: _parse_origins(
            os.environ.get("FRONTEND_ORIGINS")
        )
    )
    model_dir: Path = field(
        default_factory=lambda: _resolve_model_dir(
            os.environ.get("MODEL_DIR")
        )
    )
    runtime_db_path: Path = field(
        default_factory=lambda: _resolve_runtime_db(os.environ.get("RUNTIME_DB_PATH"))
    )
    runtime_artifact_dir: Path = field(
        default_factory=lambda: _resolve_runtime_artifacts(
            os.environ.get("RUNTIME_ARTIFACT_DIR")
        )
    )
    training_job_artifact_dir: Path = field(
        default_factory=lambda: _resolve_training_jobs(
            os.environ.get("TRAINING_JOB_ARTIFACT_DIR")
        )
    )
    dataset_upload_dir: Path = field(
        default_factory=lambda: _resolve_dataset_upload_dir(
            os.environ.get("DATASET_UPLOAD_DIR")
        )
    )
    bundled_dataset_dir: Path = field(
        default_factory=lambda: (PROJECT_ROOT / "data" / "bundled").resolve()
    )
    max_prediction_history: int = field(
        default_factory=lambda: _parse_positive_int("MAX_PREDICTION_HISTORY", 100)
    )
    max_analysis_history: int = field(
        default_factory=lambda: _parse_positive_int("MAX_ANALYSIS_HISTORY", 50)
    )
    max_runtime_artifact_mb: int = field(
        default_factory=lambda: _parse_positive_int("MAX_RUNTIME_ARTIFACT_MB", 1000)
    )
    max_upload_size_mb: int = field(
        default_factory=lambda: _parse_positive_int(
            "MAX_UPLOAD_SIZE_MB", upload_limits.max_upload_size_mb()
        )
    )
    log_level: str = field(
        default_factory=lambda: (
            os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
        )
    )
    upstage_api_key: str | None = field(
        default_factory=lambda: os.environ.get("UPSTAGE_API_KEY", "").strip() or None
    )
    upstage_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "UPSTAGE_BASE_URL", "https://api.upstage.ai/v1"
        ).strip()
    )
    upstage_model: str = field(
        default_factory=lambda: os.environ.get("UPSTAGE_MODEL", "solar-pro3").strip()
    )
    # 알림 연동 (알람 알림 연동 §C) -- 전부 선택값이다. 설정되지 않으면 해당
    # 채널은 "연결하기"를 눌러도 안내 메시지만 뜨고 실제 발송은 되지 않는다
    # (봇 토큰/SMTP 자격 증명 없이 발송을 시도하면 사용자가 원인을 알 수 없는
    # 실패를 겪는다).
    telegram_bot_token: str | None = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or None
    )
    telegram_bot_username: str | None = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_USERNAME", "").strip() or None
    )
    smtp_host: str | None = field(
        default_factory=lambda: os.environ.get("SMTP_HOST", "").strip() or None
    )
    smtp_port: int = field(
        default_factory=lambda: _parse_positive_int("SMTP_PORT", 587)
    )
    smtp_user: str | None = field(
        default_factory=lambda: os.environ.get("SMTP_USER", "").strip() or None
    )
    smtp_password: str | None = field(
        default_factory=lambda: os.environ.get("SMTP_PASSWORD", "").strip() or None
    )
    smtp_from_email: str | None = field(
        default_factory=lambda: os.environ.get("SMTP_FROM_EMAIL", "").strip() or None
    )
    # 인증 메일의 확인 링크가 가리킬 주소 -- A-7: `/api/notify/gmail/verify`는
    # FastAPI 라우트일 뿐 Next.js에 대응하는 페이지/rewrite가 없으므로,
    # 프런트엔드 오리진을 기본값으로 쓰면 메일 링크가 Next 404로 간다.
    # 명시적으로 설정되지 않았으면 None으로 두고, 호출부(api/routes/notify.py)가
    # 그 요청을 받은 API 서버 자신의 오리진(request.base_url)으로 채운다.
    notify_verify_base_url: str | None = field(
        default_factory=lambda: os.environ.get("NOTIFY_VERIFY_BASE_URL", "").strip() or None
    )
    # 자동 수집 파이프라인 1단계 (지시서 §1-1) -- 감시 디렉터리에 새 CSV가
    # 떨어지면 Y 유무로 갈라 처리한다. 주기는 여기 두지 않는다 -- 사용자가
    # 모델 학습·자동화 팝업에서 바꾼 값(`state/training`의
    # `refreshIntervalMinutes`)을 스케줄러가 그대로 따른다.
    auto_ingest_dir: str | None = field(
        default_factory=lambda: os.environ.get("AUTO_INGEST_DIR", "").strip() or None
    )
    auto_ingest_enabled: bool = field(
        default_factory=lambda: _parse_bool("AUTO_INGEST_ENABLED", False)
    )

    def __post_init__(self) -> None:
        if self.log_level not in VALID_LOG_LEVELS:
            raise ValueError(
                "LOG_LEVEL must be one of: "
                + ", ".join(sorted(VALID_LOG_LEVELS))
            )
        if "*" in self.frontend_origins:
            raise ValueError(
                "FRONTEND_ORIGINS cannot contain '*' when credentials "
                "are enabled."
            )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_model_directory(self) -> bool:
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            return self.model_dir.is_dir()
        except OSError:
            return False

    def model_directory_ready(self) -> bool:
        if not self.ensure_model_directory():
            return False
        try:
            probe = self.model_dir / ".write-probe"
            probe.touch(exist_ok=False)
            probe.unlink()
            return True
        except OSError:
            return False


settings = Settings()

if not settings.ensure_model_directory():
    logging.getLogger(__name__).error(
        "Model directory could not be created or accessed."
    )
