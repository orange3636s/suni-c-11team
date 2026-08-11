"""Route-level test for POST /api/notify/dispatch (api/routes/notify.py) --
exercises `dispatch_now` end-to-end against real bundled train/test.CSV
(same convention as test_state_endpoints.py: route functions called
directly with `settings` monkeypatched, not full TestClient/HTTP) rather
than unit-testing an internal helper in isolation, since a call through
the actual route function is what catches wiring regressions (e.g. a
previous version of this test caught a `_cached_reliability` call with
the wrong arity -- that whole old alarm-grade pipeline, including
`_cached_reliability`, has since been retired in favor of the yield
prediction update pipeline this test now exercises).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.routes import analysis as analysis_routes
from api.routes import datasets as datasets_routes
from api.routes import notify as notify_routes
from api.schemas.notify import DispatchRequest
from src.notifications import settings_store

BUNDLED_ROOT = Path(__file__).resolve().parents[1] / "data" / "bundled"

pytestmark = pytest.mark.skipif(
    not (BUNDLED_ROOT / "train.CSV").exists() and not (BUNDLED_ROOT / "test.CSV").exists(),
    reason="data/bundled의 train/test.CSV가 없어 건너뜁니다.",
)


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime_db = tmp_path / "runtime" / f"dashboard_{uuid4().hex}.db"
    artifact_root = tmp_path / "runtime"
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_db,
        runtime_artifact_dir=artifact_root,
        dataset_upload_dir=tmp_path / "uploads",
        bundled_dataset_dir=BUNDLED_ROOT,
        model_dir=tmp_path / "models",
    )
    # notify.py/datasets.py/analysis.py 셋 다 `from api.settings import
    # settings`로 각자 모듈 스코프에 이름을 바인딩하므로 셋 다 갈아끼워야
    # 같은 격리된 RuntimeStore/bundled 경로를 본다 -- dispatch_now가 거치는
    # _dataframe_or_404/_hydrated_targets_or_409는 api/routes/analysis.py에
    # 정의돼 있어 그 모듈의 settings도 패치하지 않으면 실제(개발자 로컬)
    # runtime DB의 active_model을 보고 존재하지 않는 모델을 로드하려다
    # 죽는다(격리된 store에는 활성 모델이 없어 measured-only로 폴백해야
    # 한다).
    monkeypatch.setattr(notify_routes, "settings", test_settings)
    monkeypatch.setattr(datasets_routes, "settings", test_settings)
    monkeypatch.setattr(analysis_routes, "settings", test_settings)
    return test_settings


def test_dispatch_now_route_does_not_crash(
    isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """분석 실행 직후 발송 경로(TRIGGER_MANUAL, 수율 예측 갱신 파이프라인)가
    실제 train/test.CSV로 끝까지 죽지 않고 돌아 실제로 발송하는지 확인한다."""
    store = notify_routes._store()
    settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#eng-yield")
    settings_store.save_conditions(store, grades=["심각"])

    sent = {"called": False}

    def _fake_send(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        sent["called"] = True
        return True, None

    # yield_update_dispatch.dispatch_yield_update도 같은 `senders` 모듈
    # 객체를 참조하므로(둘 다 `from src.notifications import senders`),
    # 여기서 patch하면 그쪽 호출도 그대로 잡힌다.
    monkeypatch.setattr(notify_routes.senders, "send_slack_webhook", _fake_send)

    # eval_dataset도 "train"을 쓴다(test.CSV가 아니라) -- train.CSV는
    # Y1~Y5가 전량 실측이라 모델 기반 타깃 보강(hydrate_targets의
    # "measured-only" 경로)이 필요 없다. test.CSV는 일부러 결측을 남겨
    # 모델 예측 보강이 필요한데, 이 테스트는 격리된(빈) store를 쓰므로
    # 등록된 모델이 없어 그 경로를 거치면 409로 실패한다 -- 이 테스트의
    # 목적(발송 배관 자체가 죽지 않는지)과 무관한 전제조건이다.
    result = notify_routes.dispatch_now(DispatchRequest(train_dataset="train", eval_dataset="train", dashboard_url=None))

    assert isinstance(result, dict)
    assert "skipped" in result
    assert result["skipped"] is False
    assert sent["called"] is True
