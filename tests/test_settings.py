import pytest

from api.settings import (
    DEFAULT_FRONTEND_ORIGINS,
    PROJECT_ROOT,
    Settings,
    _parse_origins,
    _resolve_model_dir,
    _resolve_runtime_artifacts,
    _resolve_runtime_db,
    _resolve_training_jobs,
)


def test_parse_origins_uses_development_defaults() -> None:
    assert _parse_origins(None) == DEFAULT_FRONTEND_ORIGINS


def test_parse_origins_trims_deduplicates_and_ignores_empty_values() -> None:
    assert _parse_origins(
        "https://one.example, https://two.example,https://one.example,"
    ) == ("https://one.example", "https://two.example")


def test_relative_model_directory_is_resolved_from_project_root() -> None:
    assert _resolve_model_dir("models") == (PROJECT_ROOT / "models").resolve()


def test_absolute_model_directory_is_preserved() -> None:
    absolute_path = (PROJECT_ROOT / "models").resolve()
    assert _resolve_model_dir(str(absolute_path)) == absolute_path


def test_railway_volume_is_used_for_default_persistent_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume_root = (PROJECT_ROOT / "tests" / ".railway-volume-fixture").resolve()
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(volume_root))

    assert _resolve_model_dir(None) == (volume_root / "models").resolve()
    assert _resolve_runtime_db(None) == (volume_root / "runtime.db").resolve()
    assert _resolve_runtime_artifacts(None) == (volume_root / "artifacts").resolve()
    assert _resolve_training_jobs(None) == (
        volume_root / "artifacts" / "training_jobs"
    ).resolve()


def test_settings_rejects_wildcard_cors_origin() -> None:
    with pytest.raises(ValueError, match="cannot contain"):
        Settings(
            frontend_origins=("*",),
            model_dir=PROJECT_ROOT / "models",
        )
