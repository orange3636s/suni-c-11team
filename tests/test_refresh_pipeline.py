"""J-1/J-2: 리프레시 파이프라인의 데이터 소스 해석을 단위 테스트한다.

RB-3: refresh 파이프라인은 더 이상 학습을 트리거하지 않는다(자동
재학습 제출 경로 자체가 없다) -- `_maybe_retrain`은 삭제됐고,
`_current_model_meta`가 현재 챔피언을 읽기만 한다."""

from __future__ import annotations

from pathlib import Path

from src.automation import refresh
from src.runtime.datasets import DatasetRegistry
from src.runtime.store import RuntimeStore


def _store_and_registry(tmp_path: Path) -> tuple[RuntimeStore, DatasetRegistry]:
    store = RuntimeStore(tmp_path / "dashboard.db")
    bundled_root = Path(__file__).resolve().parents[1] / "data" / "bundled"
    registry = DatasetRegistry(store=store, upload_root=tmp_path / "uploads", bundled_root=bundled_root)
    return store, registry


def test_current_model_meta_reads_champion_without_training(tmp_path: Path) -> None:
    store, _registry = _store_and_registry(tmp_path)
    store.promote_if_better(
        model_id="m-existing", pipeline_version="v1", dataset_version=0,
        metadata={"model_id": "m-existing", "metrics": {"test": {"r2": 0.5}}},
    )

    result = refresh._current_model_meta(store)

    assert result["champion_version"] == "m-existing"
    assert result["trained_at"] is None
    assert "training_job_submitted" not in result


def test_current_model_meta_no_champion_yet(tmp_path: Path) -> None:
    store, _registry = _store_and_registry(tmp_path)

    result = refresh._current_model_meta(store)

    assert result["champion_version"] is None


def test_resolve_source_falls_back_when_sql_not_configured(tmp_path: Path) -> None:
    store, registry = _store_and_registry(tmp_path)
    errors: list[str] = []
    mode, train_id, eval_id, row_count = refresh._resolve_source(store, registry, errors)
    assert mode == "fallback"
    assert train_id == refresh.FALLBACK_TRAIN_DATASET
    assert eval_id == refresh.FALLBACK_EVAL_DATASET
