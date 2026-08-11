import asyncio
import logging
import os
import threading
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
from api.routes.notify import (
    router as notify_router,
    run_daily_13_dispatch_job,
    run_daily_dispatch_job,
    run_notify_history_cleanup_job,
    run_notify_log_cleanup_job,
)
from api.routes.state import router as state_router
from api.settings import APP_VERSION, ENV_FILE_LOADED, settings
from src.automation.refresh import run_refresh_pipeline
from src.automation.yield_dispatch import (
    AUTOMATION_YIELD_DISPATCH_JOB_ID,
    run_automation_yield_dispatch_job,
)
from src.notifications.telegram_bot import run_polling_loop
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
# SD그룹: 자동화(주기 SQL 수율 예측 발송) 초기 등록 주기 -- "알림·자동화
# 설정"에 저장된 값이 없을 때만 쓰는 폴백이다(잡 자체는 자동화가 꺼져
# 있으면 곧바로 pause된다).
DEFAULT_AUTOMATION_MINUTES = 60

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


# W-2: 첫 기동 스냅샷 부트스트랩 -- `run_refresh_pipeline`(기존 자동화
# 파이프라인)을 그대로 재사용한다. 새 파이프라인을 만들지 않는다.


def _bootstrap_runtime_store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


class BundledTrainingDataMissingError(RuntimeError):
    """RA-B5: 부트스트랩/복구 재학습이 내장 학습 데이터(data/bundled/train.CSV)
    자체를 찾지 못해 실패한 경우 -- 다음 요청이 재시도해도 파일이
    저절로 생기지 않으므로, 다른 예외(일시적 실패)와 구분해 배너가
    재시도 없는 구체적 안내를 보여줄 수 있게 한다."""


# RA-B5: `bootstrap_status.reason`에 싣는 코드 -- 프런트 BootstrapStatusBanner가
# 이 문자열로 "정말 복구 불가능한" 경우만 골라 재시도 버튼 없는 구체적
# 안내를 보여준다. 다른 문자열이 새로 필요해지면 여기 추가한다.
BOOTSTRAP_FAILURE_REASON_DATA_MISSING = "bundled_train_data_missing"


def _has_usable_champion(store: RuntimeStore) -> bool:
    """NE-4: `store.active_model()`은 DB의 `model_slots` 행이 있는지만
    보고, 그 `active_model_id`가 가리키는 아티팩트가 `models/`에 실제로
    있는지는 확인하지 않는다. 레지스트리 행만 남고 파일이 없는 상태(예:
    로컬 테스트로 생성된 모델을 커밋하지 않은 채 배포한 경우)에서는
    `active_model()`이 여전히 값을 반환해 `_run_bootstrap`이 재학습을
    건너뛰고, 이후 `run_refresh_pipeline`의 모델 로드가 조용히 실패해
    스냅샷이 저장되지 않는다 -- 그 결과가 "첫 스냅샷 생성 실패" 배너다.
    여기서 실제로 로드를 시도해 그 간극을 메운다.

    RA-B3: 로드에 실패하면(파일이 실제로 없음) 레지스트리에 남은 포인터를
    그대로 두지 않는다 -- 이 함수가 매 요청(RA-B4의 런타임 복구 훅)마다
    불릴 수 있으므로, 지우지 않으면 같은 죽은 model_id에 대해 매번 같은
    로드 실패를 반복하게 된다. `previous_model_id`(롤백 대상)는 그
    아티팩트가 실제로 살아있으면 보존하고, 그것도 없을 때만 함께 지운다."""
    active = store.active_model()
    model_id = str((active or {}).get("active_model_id") or "").strip()
    if not model_id:
        return False
    from src.ml.inference import InferenceInputError, ModelLoadError, load_prediction_model

    try:
        load_prediction_model(model_id, settings.model_dir)
    except (InferenceInputError, ModelLoadError):
        logger.warning("bootstrap: 챔피언 모델 ID는 있으나 로드할 수 없습니다 (%s) -- 재학습합니다.", model_id)
        previous_id = str((active or {}).get("previous_model_id") or "").strip()
        clear_previous = False
        if previous_id:
            try:
                load_prediction_model(previous_id, settings.model_dir)
            except (InferenceInputError, ModelLoadError):
                clear_previous = True
        store.clear_active_model(also_clear_previous=clear_previous)
        logger.warning(
            "models: 레지스트리에서 존재하지 않는 모델 ID %s 제거%s",
            model_id,
            " (previous_model_id도 함께 제거)" if clear_previous else "",
        )
        return False
    return True


async def _train_bootstrap_champion() -> None:
    """NE-4 근본 원인: `run_refresh_pipeline`은 더 이상 학습을 트리거하지
    않는다(RB-4: 승격 게이트 제거 이후 학습은 "모델 학습" 팝업의 수동
    업로드 경로 하나뿐이다). 콜드 스타트가 옛 주석(W-2)대로 `run_refresh_
    pipeline`이 학습 Job을 제출하길 기다리기만 하면 챔피언이 전혀 없는
    환경(신규 배포, 또는 챔피언 포인터는 있으나 아티팩트가 없는 경우)에서
    영원히 끝나지 않는다 -- 실제로 재현했다. 수동 업로드와 같은 코드
    경로(`train_model`)를 내장 train.CSV로 그대로 재사용해 새 학습
    파이프라인을 만들지 않는다."""
    from fastapi import UploadFile

    from api.routes.data import train_model

    train_path = settings.bundled_dataset_dir / BUNDLED_DATASET_FILES["train"]
    if not train_path.is_file():
        # RA-B5: 이 파일 자체가 배포에 빠져 있으면 재시도해도 저절로
        # 생기지 않는다 -- 진짜 복구 불가능 케이스로 구분한다.
        raise BundledTrainingDataMissingError(str(train_path))
    with train_path.open("rb") as source:
        upload = UploadFile(file=source, filename=BUNDLED_DATASET_FILES["train"])
        await train_model(upload)


async def _run_bootstrap(store: RuntimeStore) -> None:
    store.set_bootstrap_status("running", "데이터 확인 중")
    try:
        # W-3: 챔피언 모델이 이미 있으면(볼륨은 살아있는데 스냅샷만
        # 없는 경우 등) 재학습하지 않는다 -- `run_refresh_pipeline` 내부의
        # 데이터 해시 비교(조건부 재학습)와 같은 원칙이다.
        if not _has_usable_champion(store):
            store.set_bootstrap_status("running", "학습 중")
            # WK-5: 콜드 스타트 결과로는 알림을 보내지 않는다.
            await _train_bootstrap_champion()
        store.set_bootstrap_status("running", "평가 · 원인분석 중")
        await asyncio.to_thread(run_refresh_pipeline, dispatch=False)
        if not store.has_valid_snapshot():
            raise RuntimeError("첫 스냅샷 생성에 실패했습니다 (원인분석/알람 판정 단계를 확인하세요).")
        store.set_bootstrap_status("done", None)
        logger.info("bootstrap: 첫 스냅샷 생성 완료")
    except BundledTrainingDataMissingError as exc:
        # RA-B5: 다른 예외와 달리 reason을 남긴다 -- BootstrapStatusBanner가
        # 이 코드일 때만 재시도 버튼 없는 구체적 안내("내장 학습 데이터를
        # 찾을 수 없습니다 (data/bundled/train.CSV)")로 갈아탄다.
        logger.error("bootstrap: 내장 학습 데이터를 찾을 수 없습니다 (%s)", exc)
        store.set_bootstrap_status(
            "failed",
            None,
            error=f"내장 학습 데이터를 찾을 수 없습니다 ({exc})",
            reason=BOOTSTRAP_FAILURE_REASON_DATA_MISSING,
        )
    except Exception as exc:
        # RA-B5: reason 없는 "failed"는 일시적 실패로 취급한다 -- 이
        # 상태 자체가 다음 재시도를 막지 않는다(B2/B4의 복구 훅은 오직
        # `_has_usable_champion`/`has_valid_snapshot`만 보고 재시도
        # 여부를 결정하지, 이 bootstrap_status 값을 게이트로 쓰지 않는다).
        logger.exception("bootstrap: 첫 스냅샷 생성 실패")
        store.set_bootstrap_status("failed", None, error=str(exc))


async def _bootstrap_snapshot_background() -> None:
    """유효 스냅샷이 없을 때만, 단 한 번 실행한다. 기동을 블록하지 않도록
    `_spawn_background_task`로 fire-and-forget한다(89행 주석과 동일한
    이유 -- 헬스체크가 실패하면 배포가 롤백된다). NE-6: 수동 override로
    활성화된 평가 데이터셋이 있으면 그 자체가 사용자의 최근 작업이므로
    내장 데이터로 부트스트랩을 돌려 덮어쓰지 않는다.

    RA-B1 (근본 원인): `has_valid_snapshot()`은 스냅샷 행의 스키마
    버전과, 그 안에 기록된 model_id가 현재 `model_slots.active_model_id`와
    같은지만 본다 -- 그 모델 파일이 `models/`에 실제로 있는지는 확인하지
    않는다. 스냅샷이 이미 저장된 뒤 `models/`가 비워지고(예: 배포 볼륨
    유실) 서버가 재시작되면, 이 게이트가 "유효한 스냅샷 있음"으로 잘못
    판단해 `_run_bootstrap`을 건너뛰었다 -- 그 결과 런타임 요청은 영원히
    409로 막히고 아무도 재학습을 트리거하지 않았다(실제 재현 시나리오).
    `_has_usable_champion`으로 파일 존재까지 확인해야만 건너뛴다."""
    store = _bootstrap_runtime_store()
    try:
        if store.get_manual_eval_override() is not None:
            return
        if store.has_valid_snapshot() and _has_usable_champion(store):
            return
        if not store.acquire_bootstrap_lock():
            logger.info("bootstrap: 다른 인스턴스가 이미 진행 중이라 건너뜁니다.")
            return
        try:
            await _run_bootstrap(store)
        finally:
            store.release_bootstrap_lock()
    except Exception:
        logger.exception("bootstrap: 예기치 않은 오류")


def ensure_usable_champion(store: RuntimeStore) -> bool:
    """RA-B2/B4: 런타임 요청(`api/routes/analysis.py::_hydrated_targets_or_409`
    등, 409를 올리기 직전)이 챔피언 모델을 로드하지 못했을 때 호출하는
    복구 훅. `_bootstrap_snapshot_background`/`_run_bootstrap`과 같은
    락(`acquire_bootstrap_lock`)과 같은 `bootstrap_status` 상태 머신을
    그대로 재사용한다 -- 그래야 `BootstrapStatusBanner`가 새 폴링
    인프라 없이 이 런타임발 재학습도 "첫 분석 진행 중" 배너로 보여준다.

    이 함수는 이제 (부트스트랩 1회가 아니라) 409가 발생하는 요청마다
    호출될 수 있으므로 빠른 경로(`_has_usable_champion`)가 저렴해야
    한다 -- 실제로 파일 존재를 확인하는 로드 시도 정도라 무겁지 않다.

    Returns:
        True  -- 챔피언을 이미 즉시 쓸 수 있다(호출부는 그대로 진행).
        False -- 복구가 필요하다. 이 호출로 막 새 복구를 스폰했거나,
                 이미 다른 복구/부트스트랩이 진행 중이라 스폰하지 않았다
                 -- 어느 쪽이든 호출부는 지금 이 요청에 대해서는 여전히
                 409를 올리되(재학습은 초 단위가 아니라 분 단위로 걸려
                 요청을 막고 기다릴 수 없다), 응답에 "복구 진행 중"
                 신호를 함께 실어 보낸다.
    """
    if _has_usable_champion(store):
        return True
    if not store.acquire_bootstrap_lock():
        logger.info("model recovery: 이미 다른 복구/부트스트랩이 진행 중이라 새로 시작하지 않습니다.")
        return False
    logger.warning("model recovery: 런타임 요청이 챔피언 모델 로드 실패를 감지해 재학습을 시작합니다.")

    def _runner() -> None:
        try:
            asyncio.run(_run_bootstrap(store))
        finally:
            store.release_bootstrap_lock()

    # RA-B4: 호출부(현재 요청을 처리 중인 FastAPI 스레드풀 워커)에는
    # 실행 중인 asyncio 이벤트 루프가 없다(동기 `def` 라우트 핸들러에서
    # 온 호출이므로) -- `asyncio.create_task`를 쓸 수 없다. 별도 데몬
    # 스레드에서 자신만의 이벤트 루프로 `_run_bootstrap`을 돌려, 현재
    # 요청을 절대 블록하지 않으면서도 기동 시 부트스트랩과 정확히 같은
    # 코루틴을 재사용한다.
    threading.Thread(target=_runner, name="model-recovery", daemon=True).start()
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # T5/T8-3: 방금 새로 뜬 프로세스에는 정의상 진행 중인 "분석 시작"
    # 파이프라인이 있을 수 없다 -- 있다면 그건 죽은 이전 프로세스(OOM
    # 강제 재시작 등)가 남긴 고아 진행 상태다. 그대로 두면 화면에 "분석
    # 진행 중… (N/8)"이 영구히 남는다.
    try:
        _boot_store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
        _boot_store.clear_orphaned_analysis_progress_on_boot()
        # 작업지시(Config 하이드레이션 실패 수정) T4: last_run도 같은 이유로
        # 부팅 시 정리한다 -- "running"으로 고아가 됐으면 failed로 바꾼다.
        _boot_store.reconcile_last_run_on_boot()
    except Exception:
        logger.exception("부팅 시 고아 analysis_progress/last_run 정리 실패")
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

    # W-2: 첫 기동 스냅샷 부트스트랩 -- 유효 스냅샷이 있으면 내부에서
    # 즉시 반환한다(재학습 없음). 마이그레이션 다음에 두는 이유는 모델
    # 디렉터리·DB 스키마가 이미 정리된 상태에서 판단하기 위해서다.
    _spawn_background_task(_bootstrap_snapshot_background())

    # 알람 알림 연동 §C-3 Telegram -- 봇 토큰이 설정된 경우에만 long-polling
    # 루프를 띄운다. 설정되지 않은 개발/테스트 환경에서는 조용히 건너뛴다.
    telegram_stop_event = asyncio.Event()
    telegram_task: asyncio.Task | None = None
    if settings.telegram_bot_token:
        telegram_task = asyncio.create_task(run_polling_loop(settings.telegram_bot_token, telegram_stop_event))
        logger.info("Telegram 봇 polling 시작")
        # EA-5: 토큰만 있고 username이 없으면 폴링은 도는데 화면에는 봇
        # 링크가 안 보이는 어중간한 상태가 된다 -- 조용히 넘어가지 않고
        # 기동 로그에 남긴다.
        if not settings.telegram_bot_username:
            logger.warning("Telegram 알림 비활성: TELEGRAM_BOT_USERNAME 미설정")
    else:
        logger.info("TELEGRAM_BOT_TOKEN이 설정되지 않아 Telegram 알림 연동을 건너뜁니다.")
        if settings.telegram_bot_username:
            logger.warning("Telegram 알림 비활성: TELEGRAM_BOT_TOKEN 미설정")

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
    # DF그룹: 발송 시점 다중 선택에 "매일 오후 1시"가 추가됐다 -- 09:00 잡과
    # 별개 id로 등록한다. 실제로 보낼지는 run_daily_13_dispatch_job ->
    # dispatch_alarm_notifications가 저장된 conditions.timing에
    # TIMING_DAILY_13이 포함돼 있는지로 판단하므로, 이 잡 자체는 항상
    # 등록해 두고 조건 필터는 그쪽에 맡긴다(09:00 잡과 같은 패턴).
    scheduler.add_job(
        run_daily_13_dispatch_job,
        CronTrigger(hour=13, minute=0, timezone="Asia/Seoul"),
        id="daily_alarm_notification_13",
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
    # SE-4: 알림 기록 보관 정책(최근 100건/30일) 보조 정리 -- 발송 잡과
    # 겹치지 않도록 다른 시각으로 등록한다.
    scheduler.add_job(
        run_notify_history_cleanup_job,
        CronTrigger(hour=3, minute=30, timezone="Asia/Seoul"),
        id="notify_history_cleanup",
        misfire_grace_time=3600,
    )

    # SD그룹: 자동화(주기 SQL 수율 예측 발송) -- "알림·자동화 설정" 팝업의
    # 전용 슬롯(automation:settings)에서 켜짐 여부·주기를 읽는다. 감시
    # 디렉터리 폴링 자동화(auto_ingest, 옛 src/automation/ingest.py)는
    # "자동화는 수율 예측만 계산해야 한다" 원칙에 위배돼 이번에 함께
    # 제거했다 -- 화면 재계산까지 하던 별개의 자동화 경로가 더는 없다.
    automation_enabled = False
    initial_minutes: int | None = None
    try:
        automation_settings = RuntimeStore(
            settings.runtime_db_path, settings.runtime_artifact_dir
        ).get_automation_settings()
        automation_enabled = bool(automation_settings.get("enabled"))
        raw_minutes = automation_settings.get("refreshIntervalMinutes")
        if isinstance(raw_minutes, (int, float)) and raw_minutes > 0:
            initial_minutes = int(raw_minutes)
    except Exception:
        logger.exception("자동화 설정 조회 실패 -- 일시정지 상태로 시작합니다.")

    scheduler.add_job(
        run_automation_yield_dispatch_job,
        IntervalTrigger(minutes=initial_minutes or DEFAULT_AUTOMATION_MINUTES),
        id=AUTOMATION_YIELD_DISPATCH_JOB_ID,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.start()
    if not automation_enabled:
        scheduler.pause_job(AUTOMATION_YIELD_DISPATCH_JOB_ID)
    # 저장 API(POST /api/notify/automation) 핸들러가 켜짐 여부·주기 변경
    # 시 reschedule/pause할 수 있도록 앱 상태에 둔다 -- 서버 재시작을
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
