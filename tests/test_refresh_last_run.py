"""작업지시(Config 하이드레이션 실패 수정) T4 회귀 테스트: `run_refresh_pipeline`이
성공/실패를 `RuntimeStore`의 last_run 레코드에 남기는지, 그리고 Config
대표 인자 모델로도 실제 스냅샷이 저장되는지 확인한다."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.analysis import target_hydration
from src.analysis.screening.schema import parse_schema
from src.automation import refresh
from src.ml import pipeline as ml_pipeline
from src.runtime.datasets import DatasetRegistry
from src.runtime.store import RuntimeStore


def _run_refresh_pipeline_after_lock_is_free(timeout: float = 10.0) -> None:
    """`_refresh_lock`은 `src.automation.refresh` 모듈 전역이라 이 프로세스
    안의 다른 테스트(예: 부트스트랩 관련 TestClient 기동이 백그라운드로
    걸어 둔 실행)가 아직 붙잡고 있을 수 있다 -- 그 경우 `run_refresh_pipeline`은
    조용히 건너뛰고 반환하므로(기존의 올바른 동작, `test_refresh_manual_trigger.py`가
    이미 검증한다) 이 테스트의 관심사인 "이번 실행이 실제로 last_run/스냅샷을
    남기는지"를 확인할 수 없게 된다. 락이 빌 때까지 짧게 기다렸다가 돌린다."""
    deadline = time.monotonic() + timeout
    while refresh.is_refresh_running():
        if time.monotonic() > deadline:
            pytest.fail("다른 실행이 _refresh_lock을 오래 쥐고 있어 테스트를 진행할 수 없습니다.")
        time.sleep(0.05)
    refresh.run_refresh_pipeline()

CONFIG_FEATURE = "Step1_Config"
CATEGORIES = ("EQA", "EQB", "EQC")
_MEAN_BY_CATEGORY = {"EQA": 2.0, "EQB": 5.0, "EQC": 8.0}


def _store_and_registry(tmp_path: Path) -> tuple[RuntimeStore, DatasetRegistry]:
    store = RuntimeStore(tmp_path / "dashboard.db")
    bundled_root = Path(__file__).resolve().parents[1] / "data" / "bundled"
    registry = DatasetRegistry(store=store, upload_root=tmp_path / "uploads", bundled_root=bundled_root)
    return store, registry


def _synthetic_config_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = rng.choice(np.asarray(CATEGORIES), size=n)
    y1 = np.asarray([_MEAN_BY_CATEGORY[c] for c in cats]) + rng.normal(0, 0.05, size=n)
    return pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"L{i // 10:03d}W{i % 10:02d}" for i in range(n)],
            "Lot_ID": [f"L{i // 10:03d}" for i in range(n)],
            CONFIG_FEATURE: pd.Series(cats, dtype="object"),
            "Y1": y1,
        }
    )


class _AlwaysZeroModel:
    feature_names_in_ = np.asarray([])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features))


def _patch_config_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """T2: 학습 경로(`fit_target_pipeline`)로 실제 LightGBM을 학습시켜
    `target_hydration._active_model`이 그 모델을 돌려주게 한다 -- stage
    2("모델 추론")가 이 테스트에서는 mock 없이 실제로 도는 유일한
    무거운 단계다."""
    train_df = _synthetic_config_frame()
    schema = parse_schema(train_df)
    result = ml_pipeline.fit_target_pipeline(train_df, schema, "Y1")
    assert result.factors[0].kind == "Config"
    from src.ml.feature_builder import trained_categories_from_model

    details = {
        "feature": result.factors[0].feature,
        "kind": "Config",
        "relation_shape": result.factors[0].relation_shape,
        "optimal_center": result.factors[0].optimal_center,
        "categories": trained_categories_from_model(result.model),
    }

    class _Bundle:
        def model_for_target(self, target: str):
            return result.model if target == "Y1" else _AlwaysZeroModel()

    @dataclass
    class _Loaded:
        model_id: str = "config-model-1"

        def __post_init__(self) -> None:
            self.model = _Bundle()
            self.metadata = {
                "bundle_type": "screening_pareto_pipeline",
                "pipeline_version": "pipeline-v1",
                "available_targets": list(target_hydration.FAIL_RATE_TARGETS),
                "target_metrics": {t: details for t in target_hydration.FAIL_RATE_TARGETS},
            }

    loaded = _Loaded()
    monkeypatch.setattr(
        target_hydration,
        "_active_model",
        lambda _store, _model_dir: (loaded, {"active_model_id": loaded.model_id, "pipeline_version": "pipeline-v1"}),
    )
    target_hydration.invalidate_target_hydration_cache()


class _FakeYieldTable:
    candidates: list = []
    unmeasured_wafer_ids: list = []
    total_wafers = 0
    fallback_summary = type("_FB", (), {"rank_counts": {}, "none_count": 0, "total_combinations": 0})()


def _mock_heavy_unrelated_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """이 테스트의 관심사가 아닌 단계(FMEA/조치우선순위/트리맵워밍업/
    원인분석/수율예측)는 기존 test_manual_eval_override.py와 같은
    방식으로 무력화한다 -- 살려두는 건 stage 2(모델 추론)뿐이다."""
    monkeypatch.setattr(refresh, "_fmea_stage", lambda eid, e: (None, None))
    monkeypatch.setattr(refresh, "_action_priority_stage", lambda t, e: (None, None))
    monkeypatch.setattr(refresh, "_warmup_common_prerequisites", lambda eid: None)
    monkeypatch.setattr(refresh, "_pareto_stage", lambda eid, e: ({"Y1": {"items": []}}, None, []))
    monkeypatch.setattr(refresh, "_yield_prediction_stage", lambda r, t, eid, e: _FakeYieldTable())


# -- 7. refresh pipeline이 Config 모델로 새 스냅샷을 실제로 저장 -----------


def test_refresh_pipeline_saves_snapshot_with_config_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store, registry = _store_and_registry(tmp_path)
    # 번들 "test" 데이터셋을 쓴다 -- 정적 파일이라 store/registry 인스턴스와
    # 무관하게 어디서든 같은 내용으로 읽힌다(`_hydrated_targets_or_409`이
    # 내부적으로 프로덕션 설정 기반의 별도 레지스트리를 새로 만들기 때문에,
    # 업로드 데이터셋을 쓰면 그 레지스트리가 이 테스트의 DB 레코드를 보지
    # 못해 404가 난다 -- 번들 데이터셋은 이 문제 자체가 없다).
    monkeypatch.setattr(refresh, "_runtime_store", lambda: store)
    monkeypatch.setattr(refresh, "_resolve_source", lambda s, r, e: ("manual", "train", "test", 5))
    monkeypatch.setattr(refresh, "_current_model_meta", lambda s: {
        "champion_version": "config-model-1", "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": None,
    })
    monkeypatch.setattr(refresh, "get_latest_state", lambda s: {})
    _mock_heavy_unrelated_stages(monkeypatch)
    _patch_config_active_model(monkeypatch)

    _run_refresh_pipeline_after_lock_is_free()

    last_run = store.get_last_run()
    assert last_run is not None, "last_run이 기록되지 않았습니다"
    assert last_run["status"] == "succeeded", last_run
    status = store.get_refresh_snapshot_status()
    assert status["snapshot"] is not None


# -- 8. 강제 예외 주입 -> last_run.status == "failed" + 실패 단계·메시지 --


def test_forced_exception_records_failed_last_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store, registry = _store_and_registry(tmp_path)
    monkeypatch.setattr(refresh, "_runtime_store", lambda: store)
    monkeypatch.setattr(refresh, "_resolve_source", lambda s, r, e: ("manual", "train", "test", 5))
    monkeypatch.setattr(refresh, "_current_model_meta", lambda s: {
        "champion_version": None, "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": None,
    })
    monkeypatch.setattr(refresh, "get_latest_state", lambda s: {})
    monkeypatch.setattr(
        "api.routes.analysis._hydrated_targets_or_409",
        lambda dataset_id: (_ for _ in ()).throw(RuntimeError("강제 주입된 오류")),
    )

    _run_refresh_pipeline_after_lock_is_free()

    last_run = store.get_last_run()
    assert last_run is not None
    assert last_run["status"] == "failed"
    assert last_run["failed_stage"] == "hydrate_eval"
    assert last_run["error_message"] and "강제 주입된 오류" in last_run["error_message"]
    # SC-3: 실패했으니 스냅샷은 저장되지 않는다(원자적 저장 -- 기존 스냅샷 보존).
    assert store.get_refresh_snapshot_status()["snapshot"] is None


# -- 9. 성공 실행 후 실패 상태가 지워지고 새 스냅샷으로 교체 ----------------


def test_success_after_failure_replaces_failed_status_and_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store, registry = _store_and_registry(tmp_path)
    monkeypatch.setattr(refresh, "_runtime_store", lambda: store)
    monkeypatch.setattr(refresh, "_resolve_source", lambda s, r, e: ("manual", "train", "test", 5))
    monkeypatch.setattr(refresh, "_current_model_meta", lambda s: {
        "champion_version": None, "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": None,
    })
    monkeypatch.setattr(refresh, "get_latest_state", lambda s: {})

    # 1) 먼저 실패시킨다.
    monkeypatch.setattr(
        "api.routes.analysis._hydrated_targets_or_409",
        lambda dataset_id: (_ for _ in ()).throw(RuntimeError("첫 실행 강제 실패")),
    )
    _run_refresh_pipeline_after_lock_is_free()
    assert store.get_last_run()["status"] == "failed"
    assert store.get_refresh_snapshot_status()["snapshot"] is None

    # 2) 이번엔 성공시킨다 -- 실패 상태가 새 성공 상태로 교체돼야 한다.
    monkeypatch.undo()
    monkeypatch.setattr(refresh, "_runtime_store", lambda: store)
    monkeypatch.setattr(refresh, "_resolve_source", lambda s, r, e: ("manual", "train", "test", 5))
    monkeypatch.setattr(refresh, "_current_model_meta", lambda s: {
        "champion_version": None, "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": None,
    })
    monkeypatch.setattr(refresh, "get_latest_state", lambda s: {})
    _mock_heavy_unrelated_stages(monkeypatch)
    _patch_config_active_model(monkeypatch)

    _run_refresh_pipeline_after_lock_is_free()

    last_run = store.get_last_run()
    assert last_run["status"] == "succeeded"
    assert last_run["failed_stage"] is None
    assert last_run["error_message"] is None
    assert store.get_refresh_snapshot_status()["snapshot"] is not None
