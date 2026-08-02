from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    admin_reset_secret: str | None = field(
        default_factory=lambda: (
            os.environ.get("ADMIN_RESET_SECRET", "").strip() or None
        )
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
            "MAX_UPLOAD_SIZE_MB", 20
        )
    )
    log_level: str = field(
        default_factory=lambda: (
            os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
        )
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
