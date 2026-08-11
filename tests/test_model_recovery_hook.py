"""RA-B4: `api/routes/analysis.py::_hydrated_targets_or_409` is the single
choke point every analysis/monitoring/notify route goes through when it
needs the champion model (`hydrate_targets` -> `_active_model`). Before
this fix it just turned any `TargetHydrationError` into a bare-string 409
and stopped there -- nothing ever triggered a retrain. Now it must also
call the shared runtime recovery hook (`api.main.ensure_usable_champion`,
the same lock/`bootstrap_status` machinery the boot-time bootstrap uses)
and shape the 409 detail as `{"message": ..., "recovering": bool}` so the
frontend can tell "genuinely stuck" apart from "fixing itself now" without
breaking existing plain-string detail parsing (`frontend/lib/api.ts`
`getErrorMessage` already reads `detail.message` when detail is a dict).

These tests stub out `hydrate_targets` and `ensure_usable_champion`
entirely -- the real end-to-end retrain path (real `_train_bootstrap_
champion` against the real bundled train.CSV) is already covered by
`tests/test_bootstrap_snapshot.py::test_run_bootstrap_skips_retrain_when_
champion_already_exists`, which completes in a few seconds. What's unique
to this file is the NEW wiring in `_hydrated_targets_or_409` itself: does
it call the recovery hook, and does it shape the response correctly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

import api.main as main_module
import api.routes.analysis as analysis_routes
from src.analysis.target_hydration import TargetHydrationError


class _FakeRegistry:
    """Just enough of `DatasetRegistry`'s surface for `_hydrated_targets_or_409`
    to get past dataset lookup and reach the `hydrate_targets` call."""

    def get_dataframe(self, dataset_id: str):
        return object()

    def content_version(self, dataset_id: str) -> str:
        return "v1"


@pytest.fixture()
def isolated_analysis_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Points `_hydrated_targets_or_409`'s `RuntimeStore(...)` construction
    at an isolated temp DB -- the real repo's data/runtime/dashboard.db
    must never be touched by a test run (same isolation principle as
    `test_state_endpoints.py::isolated_settings`)."""
    runtime_db = tmp_path / "runtime" / f"dashboard_{uuid4().hex}.db"
    artifact_root = tmp_path / "runtime"
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_db,
        runtime_artifact_dir=artifact_root,
        model_dir=tmp_path / "models",
    )
    monkeypatch.setattr(analysis_routes, "settings", test_settings)
    monkeypatch.setattr(analysis_routes, "get_dataset_registry", lambda: _FakeRegistry())
    return test_settings


def test_hydrated_targets_or_409_triggers_recovery_and_marks_response_recovering(
    isolated_analysis_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RA repro: no usable champion. `ensure_usable_champion` reports
    it just spawned (or is already running) a recovery -- the 409 must say
    so via `detail.recovering = True`, and the human-readable message from
    the original TargetHydrationError must still be present (frontend
    error extraction relies on `detail.message`)."""

    def raise_error(*args, **kwargs):
        raise TargetHydrationError("승인 모델을 사용할 수 없습니다: 존재하지 않거나 사용할 수 없는 모델 ID입니다.")

    monkeypatch.setattr(analysis_routes, "hydrate_targets", raise_error)

    calls: list[object] = []

    def fake_ensure_usable_champion(store) -> bool:
        calls.append(store)
        return False  # recovery just spawned / already in progress

    monkeypatch.setattr(main_module, "ensure_usable_champion", fake_ensure_usable_champion)

    with pytest.raises(HTTPException) as excinfo:
        analysis_routes._hydrated_targets_or_409("train")

    exc = excinfo.value
    assert exc.status_code == 409
    assert isinstance(exc.detail, dict)
    assert exc.detail["recovering"] is True
    assert "존재하지 않거나 사용할 수 없는 모델 ID입니다" in exc.detail["message"]
    # 복구 훅이 정확히 한 번, RuntimeStore 인스턴스를 들고 호출됐는지 --
    # 부트스트랩과 같은 락/상태를 공유하려면 진짜 store가 전달돼야 한다.
    assert len(calls) == 1


def test_hydrated_targets_or_409_marks_not_recovering_when_champion_already_usable(
    isolated_analysis_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """다른 종류의 TargetHydrationError(예: 파일은 멀쩡히 로드되는데
    Y1~Y5 서브모델 구성이 잘못된 경우)는 `ensure_usable_champion`의 빠른
    경로가 True를 반환한다 -- 이 경우 재학습을 스폰하지 않았으므로
    `recovering`은 False여야 한다(불필요한 재시도를 유도하는 오해를
    막는다)."""

    def raise_error(*args, **kwargs):
        raise TargetHydrationError("승인 모델에 필요한 Y1~Y5 서브모델이 없습니다: Y1")

    monkeypatch.setattr(analysis_routes, "hydrate_targets", raise_error)
    monkeypatch.setattr(main_module, "ensure_usable_champion", lambda store: True)

    with pytest.raises(HTTPException) as excinfo:
        analysis_routes._hydrated_targets_or_409("train")

    exc = excinfo.value
    assert exc.status_code == 409
    assert exc.detail["recovering"] is False
    assert "Y1~Y5 서브모델이 없습니다" in exc.detail["message"]


def test_hydrated_targets_or_409_still_raises_404_for_missing_dataset(
    isolated_analysis_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """회귀 방지: 데이터셋 자체가 없는 경로(DatasetNotFoundError)는 이
    변경과 무관하게 그대로 404를 유지해야 한다 -- 복구 훅은 모델 부재
    전용이지 데이터셋 부재까지 손대지 않는다."""
    from src.runtime.datasets import DatasetNotFoundError

    class _MissingRegistry:
        def get_dataframe(self, dataset_id: str):
            raise DatasetNotFoundError(dataset_id)

    monkeypatch.setattr(analysis_routes, "get_dataset_registry", lambda: _MissingRegistry())

    with pytest.raises(HTTPException) as excinfo:
        analysis_routes._hydrated_targets_or_409("does-not-exist")

    assert excinfo.value.status_code == 404
