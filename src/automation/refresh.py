"""J-2/J-3: 리프레시 파이프라인 -- 갱신 주기마다 데이터 취득(SQL 또는
폴백) -> 현재 챔피언으로 예측 -> 원인분석 -> 알람 판정 -> 모니터링
요약을 계산해 하나의 스냅샷으로 저장한다. 사용자가 버튼을 누르지
않아도 어느 탭이든 최신 결과가 즉시 보이게 하는 것이 목적이다(J-4가 이
스냅샷을 읽는다).

RB-3: 이 파이프라인은 학습을 트리거하지 않는다 -- SQL/폴백에서 받은
데이터는 항상 분석셋(eval)이고, 학습은 모델 학습 팝업의 수동 업로드
(또는 내장 train.csv)로만 일어난다.

`run_refresh_pipeline`은 APScheduler가 `refreshIntervalMinutes`마다
호출한다(`api/main.py`, job id `auto_refresh`) -- 기존 `auto_ingest`
(감시 디렉터리 폴링, `src/automation/ingest.py`)와는 별개의 잡이다.
이 잡은 예외를 절대 밖으로 던지지 않는다: 각 단계를 독립적으로
try/except하고, 실패는 로그 + 스냅샷의 `errors` 배열에 남긴다(단,
`except`로 삼키고 성공한 척하지 않는다 -- 화면에 실패가 보여야 한다).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from api.settings import settings
from src.automation import sql_source
from src.runtime.datasets import DatasetRegistry, parse_uploaded_csv
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

REFRESH_JOB_ID = "auto_refresh"
FALLBACK_TRAIN_DATASET = "train"
FALLBACK_EVAL_DATASET = "test"
TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")
_KST = ZoneInfo("Asia/Seoul")

# AF/AG: 주기 잡(APScheduler)과 수동 최신화 버튼·업로드 연동이 전부 이
# 함수 하나를 호출한다 -- 두 경로가 동시에 들어와도 파이프라인이 두 번
# 돌지 않도록 진입점 자체를 non-blocking 락으로 감싼다.
# operation_coordinator(training 전용, 이 함수 안에서 재학습 시 다시
# 잡는다)를 여기서 같이 쓰면 "이미 analysis를 잡은 채로 training을
# 잡으려는" 자기 자신과의 교착을 만들기 때문에 별도의 단순 락을 쓴다.
_refresh_lock = threading.Lock()


def is_refresh_running() -> bool:
    """AF: 최신화 버튼이 disabled 여부·툴팁을 결정하는 데 쓴다."""
    acquired = _refresh_lock.acquire(blocking=False)
    if acquired:
        _refresh_lock.release()
    return not acquired


def _runtime_store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


def _dataset_registry(store: RuntimeStore) -> DatasetRegistry:
    return DatasetRegistry(store, settings.dataset_upload_dir, settings.bundled_dataset_dir)


def run_refresh_pipeline(*, dispatch: bool = True) -> None:
    """`dispatch=False`는 WK그룹(콜드 스타트) 전용 -- 서버 최초 기동 시
    내장 test.csv로 돌리는 1회성 부트스트랩 분석 결과로는 알림을 보내지
    않는다. 주기 잡·수동 "최신화" 버튼은 항상 기본값(True)을 쓴다 --
    폴백/수동 모드도 발송 대상이라는 EB그룹 결정은 바뀌지 않는다."""
    if not _refresh_lock.acquire(blocking=False):
        logger.info("auto_refresh: 이미 다른 실행이 진행 중이라 이번 호출은 건너뜁니다.")
        return
    try:
        store = _runtime_store()
        try:
            _run_refresh_pipeline_inner(store, dispatch=dispatch)
        except Exception:
            # 개별 단계는 각자 try/except하지만, 예상하지 못한 예외가 여기까지
            # 올라오면 스케줄러 자체가 죽지 않도록 마지막 방어선에서 삼킨다.
            logger.exception("auto_refresh: 파이프라인 실행 중 예기치 않은 오류")
    finally:
        _refresh_lock.release()


def _warmup_common_prerequisites(eval_dataset_id: str) -> None:
    """UC (perf): pre-populate the prerequisites every scatter/heatmap card
    on the root-cause tab shares, right after a refresh snapshot is saved --
    so the user's next click doesn't pay a cold-start it doesn't need to.

    Deliberately narrow: only the SHARED prerequisites (target hydration,
    parsed schema). Individual scatter cards themselves are intentionally
    NOT precomputed here -- "개별 산점도는 워밍업 대상에서 제외".

    `hydrate_targets` for `eval_dataset_id` is very likely already warm by
    the time this runs -- `_analyze_and_score` (called just above, in
    `_run_refresh_pipeline_inner`) already calls into `_pareto_payload` (per
    target) and `_fmea_payload`, both of which call
    `_hydrated_targets_or_409(eval_dataset_id)` internally. Calling it again
    here is cheap insurance for the case where every per-target Pareto call
    failed but FMEA still populated it (or vice versa) -- after the UB fix,
    a cache hit is a dict lookup + a frozen-dataclass rebuild, not a
    dataframe copy, so re-touching it costs effectively nothing.

    `_cached_schema` is NOT touched by any of those functions --
    `_pareto_payload` and `_fmea_payload` call `parse_schema` directly (not
    through the per-request cache added for the scatter endpoint). That's
    the actual cold gap this warmup closes.
    """
    from api.routes.analysis import _cached_schema, _hydrated_targets_or_409

    hydrated = _hydrated_targets_or_409(eval_dataset_id)
    dataset_version = hydrated.provenance.dataset_version
    _cached_schema(eval_dataset_id, dataset_version)


def _warmup_common_prerequisites_background(eval_dataset_id: str) -> None:
    """Fire-and-forget: must never raise into the pipeline and must never
    delay `store.save_refresh_snapshot` (called just before this) -- a
    daemon thread start is effectively instant, so placement right after
    the snapshot save doesn't matter for that guarantee. A plain
    `threading.Thread` (this module's own existing concurrency primitive --
    see `_refresh_lock` above) rather than an asyncio task: this whole
    module is synchronous and may run inside APScheduler's executor thread
    pool rather than the event loop thread, so there is no guaranteed
    running loop here to schedule an asyncio task onto.
    """

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            _warmup_common_prerequisites(eval_dataset_id)
        except Exception:
            logger.exception(
                "auto_refresh: 워밍업 실패 -- 다음 조회가 콜드 스타트로 처리됩니다 (eval=%s)", eval_dataset_id
            )
            return
        logger.info(
            "auto_refresh: 워밍업 완료 %.1fms eval=%s", (time.perf_counter() - t0) * 1000, eval_dataset_id
        )

    threading.Thread(target=_run, daemon=True, name="refresh-warmup").start()


def _run_refresh_pipeline_inner(store: RuntimeStore, *, dispatch: bool = True) -> None:
    errors: list[str] = []
    registry = _dataset_registry(store)
    now_iso = datetime.now(timezone.utc).isoformat()

    # -- 1. 데이터 소스 해석 (J-1) -----------------------------------
    mode, train_dataset_id, eval_dataset_id, source_row_count = _resolve_source(store, registry, errors)

    # -- 2. 현재 챔피언 정보만 읽는다 (RB-3) -- refresh 파이프라인은 더
    # 이상 학습을 트리거하지 않는다. 학습은 모델 학습 팝업의 수동
    # 업로드로만 일어난다("자동화 없음" -- RA-3 데이터 흐름).
    model_meta = _current_model_meta(store)

    # -- 3. 챔피언으로 예측 + 4. 원인분석 ------------------------------
    analysis_block, train_dataset_for_analysis = _analyze_and_score(store, eval_dataset_id, errors)

    if analysis_block is None:
        # J-2: 원인분석 전 타깃 실패 -- 스냅샷 저장을 생략하고 기존
        # 스냅샷을 보존한다(빈 payload로 덮어쓰지 않는다).
        logger.warning("auto_refresh: 원인분석 전 타깃 실패 -- 스냅샷 저장 생략")
        return

    # AG: 헤더·모니터링이 "수동 · uploaded_0809.csv"처럼 파일명을 보여줄
    # 수 있게, eval_dataset_id뿐 아니라 원본 파일명도 스냅샷에 싣는다.
    eval_dataset_filename = None
    try:
        eval_summary = registry.get_summary(eval_dataset_id)
        eval_dataset_filename = eval_summary["original_filename"] if eval_summary else None
    except Exception:
        logger.exception("auto_refresh: 평가 데이터셋 파일명 조회 실패")

    snapshot = {
        "created_at": now_iso,
        "analysis_id": now_iso,
        "source": {
            "mode": mode,
            "train_dataset": train_dataset_id,
            "eval_dataset": eval_dataset_id,
            "eval_dataset_filename": eval_dataset_filename,
            "row_count": source_row_count,
        },
        "model": model_meta,
        "analysis": analysis_block,
        # 알람 등급(심각/위험/주의)·게이트 판정 파이프라인은 폐기됐다 --
        # 어떤 화면도 이 두 블록의 내용(counts/items_top/gate_passed/
        # predicted_yield/gap)을 더 이상 렌더링하지 않는 것으로 확인했다
        # (알림 발송은 수율 예측 갱신 파이프라인, 아래 7번으로 대체).
        # 스냅샷 스키마의 키 자체는 다른 소비처가 존재를 가정할 수 있어
        # 유지하되, 내용은 빈 상태로 남긴다.
        "alarms": None,
        "monitoring": {"predicted_yield": None, "gap": None, "gap_pareto": [], "treemap": None},
        "errors": errors,
    }
    store.save_refresh_snapshot(snapshot)
    _warmup_common_prerequisites_background(eval_dataset_id)
    # ZB-2: 콜드 스타트 부트스트랩(`_run_bootstrap`)이 한 번 실패하면
    # `bootstrap_status`가 store에 "failed"로 영구히 남는다 -- 그 이후의
    # 주기 잡·수동 "최신화"가 스냅샷 저장에 성공해도 아무도 그 상태를
    # 지우지 않아 "첫 분석에 실패했습니다" 배너가 계속 뜬다(실제로 재현
    # 확인함). dispatch=True는 콜드 스타트 자신의 내부 호출(dispatch=False)과
    # 구분되는, 진짜 이후의 정상 갱신이라는 뜻이므로 여기서 안전하게
    # 지운다 -- "failed"일 때만 건드리고, 지금 다른 콜드 스타트가
    # "running" 중이면(이론상 dispatch=True로는 오지 않지만 방어적으로)
    # 건드리지 않는다.
    if dispatch:
        current_bootstrap = store.get_bootstrap_status()
        if current_bootstrap is not None and current_bootstrap.get("status") == "failed":
            store.set_bootstrap_status("done", None)
    logger.info("auto_refresh: 스냅샷 저장 완료 mode=%s eval=%s", mode, eval_dataset_id)

    if not dispatch:
        # WK-5: 콜드 스타트 부트스트랩 -- 내장 test.csv 결과로는 알림을
        # 보내지 않는다. 스냅샷 저장(위)까지는 정상 진행한다.
        logger.info("auto_refresh: dispatch=False -- 발송 단계를 건너뜁니다 (콜드 스타트).")
        return

    # -- 6. (폐기) 옛 알람 등급 자동 발송 -- 수율 예측 갱신 발송(7번)으로
    # 완전히 대체됐다. 두 파이프라인을 동시에 돌리면 중복 발송이 되므로
    # 옛 호출은 남기지 않는다.
    #
    # EB그룹: 수동 모드(업로드로 활성화된 평가 데이터셋)도 발송 대상이다
    # -- source_note로 "[수동] 파일명" 출처를 붙이고, 수동 트리거는
    # 최소 간격을 둔다(연속 업로드가 연속 발송이 되지 않도록). 이 규칙은
    # `_dispatch_yield_update_for_refresh` -> `dispatch_yield_update`가
    # 그대로 적용한다.
    #
    # -- 7. 수율 예측 갱신 발송 (VE-1) -- 자동 갱신마다 발송 후보다(수동
    # 분석 실행과 달리 timing 설정 게이트가 없다). 분석이 일어날 때마다
    # 보낸다(24시간 dedupe 없음) -- 억제 규칙은 시간당 예산/최소 간격
    # 뿐이며 yield_update_dispatch 안에서 판단한다.
    try:
        _dispatch_yield_update_for_refresh(
            store,
            mode=mode,
            train_dataset_id=train_dataset_for_analysis,
            eval_dataset_id=eval_dataset_id,
            eval_dataset_filename=eval_dataset_filename,
            model_label=model_meta.get("champion_version"),
            now_iso=now_iso,
        )
    except Exception:
        logger.exception("auto_refresh: 수율 예측 갱신 발송 처리 실패")


def _resolve_source(
    store: RuntimeStore, registry: DatasetRegistry, errors: list[str]
) -> tuple[str, str, str, int]:
    """SQL 모드를 먼저 시도하고, 실패하거나 설정이 없으면 폴백(test)으로
    넘어간다. RB-3: SQL/폴백에서 받은 배치는 이제 항상 분석셋(eval)이다
    -- Y 유무로 학습/평가를 가르지 않는다(자동 학습 경로 자체가 없다).
    학습 대상(train_dataset)은 이 함수가 절대 바꾸지 않는다 -- 직전
    스냅샷의 것을 그대로 이어받거나(자동 학습을 걸지 않는다, AG-2),
    없으면 내장 train.CSV로 폴백한다. 학습 대상을 실제로 바꾸는 것은
    모델 학습 팝업의 수동 업로드/내장 train.csv뿐이다.

    AG-3: 업로드로 활성화된 "수동 평가 데이터셋"이 있으면 SQL/폴백보다
    먼저 그것을 쓴다 -- 주기 잡이 사용자가 올린 파일을 원래 소스로
    되돌리지 않는다("자동 갱신으로 복귀"를 눌러 override를 지우기
    전까지)."""
    manual = store.get_manual_eval_override()
    if manual is not None:
        previous = store.get_refresh_snapshot_status()["snapshot"]
        previous_source = (previous or {}).get("source") if previous else None
        previous_train = (previous_source or {}).get("train_dataset") if previous_source else None
        try:
            row_count = len(registry.get_dataframe(manual["dataset_id"]))
        except Exception:
            logger.exception("auto_refresh: 수동 평가 데이터셋을 읽지 못했습니다 -- %s", manual.get("dataset_id"))
            errors.append("업로드한 평가 데이터셋을 읽지 못했습니다.")
            row_count = 0
        return "manual", previous_train or FALLBACK_TRAIN_DATASET, manual["dataset_id"], row_count

    if sql_source.is_sql_configured(store):
        dataframe = sql_source.fetch_incremental(store)
        if dataframe is not None:
            previous = store.get_refresh_snapshot_status()["snapshot"]
            previous_source = (previous or {}).get("source") if previous else None
            previous_train = (previous_source or {}).get("train_dataset") if previous_source else None
            if dataframe.empty:
                # 신규 행이 없다 -- 평가 데이터셋은 바뀌지 않았으므로
                # 직전 스냅샷의 소스를 그대로 쓴다(첫 실행이면 폴백).
                if previous_source and previous_source.get("mode") == "sql":
                    return "sql", previous_source["train_dataset"], previous_source["eval_dataset"], previous_source.get("row_count", 0)
                # 첫 SQL 조회가 하필 0행이면 폴백으로 시작한다.
            else:
                filename = f"sql_refresh_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.csv"
                content = dataframe.to_csv(index=False).encode("utf-8")
                dataset_id = _register_dataset(registry, filename, content, errors)
                if dataset_id is not None:
                    return "sql", previous_train or FALLBACK_TRAIN_DATASET, dataset_id, len(dataframe)

    # 폴백: 내장 test.CSV로 평가 (J-1).
    try:
        eval_df = registry.get_dataframe(FALLBACK_EVAL_DATASET)
        row_count = len(eval_df)
    except Exception:
        errors.append("폴백 데이터셋(test.CSV)을 읽지 못했습니다.")
        row_count = 0
    return "fallback", FALLBACK_TRAIN_DATASET, FALLBACK_EVAL_DATASET, row_count


def _register_dataset(registry: DatasetRegistry, filename: str, content: bytes, errors: list[str]) -> str | None:
    try:
        result = registry.upload(filename, content)
    except Exception:
        logger.exception("auto_refresh: SQL 배치 등록 실패")
        errors.append("SQL로 받은 데이터를 등록하지 못했습니다.")
        return None
    if not result.get("success"):
        errors.append(f"SQL로 받은 데이터 등록 실패: {result.get('blocking_errors')}")
        return None
    return result["dataset_id"]


def _current_model_meta(store: RuntimeStore) -> dict[str, Any]:
    """RB-3: refresh 파이프라인은 더 이상 학습을 제출하지 않는다 --
    현재 활성 챔피언을 읽기만 한다. `trained_at`/`promoted`/`gate_reason`은
    이 파이프라인이 학습을 트리거하던 시절에도 항상 None이었다(학습이
    비동기라 완료를 여기서 기다리지 않았다) -- 그 필드 의미는 그대로
    유지한다."""
    return {
        "champion_version": _current_champion_id(store),
        "trained_at": None,
        "promoted": None,
        "gate_reason": None,
        "skipped_reason": None,
    }


def _current_champion_id(store: RuntimeStore) -> str | None:
    active = store.active_model()
    return active.get("active_model_id") if active else None


def _analyze_and_score(
    store: RuntimeStore, eval_dataset_id: str, errors: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    """원인분석(Pareto+계측 확대)을 계산한다. 전 타깃 Pareto 실패는
    analysis_block=None으로 알린다 -- 호출부가 스냅샷 저장 여부를 그
    신호로 결정한다(J-2 "부분 실패 정책"). 두 번째 반환값은 이번
    분석 사이클에 쓰인 train 데이터셋 id다(모니터링 홈 블록①·②가
    같은 기준을 써야 한다).

    이전에는 여기서 GBDT 알람 등급(심각/위험/주의)도 함께 계산했으나,
    그 등급 판정 결과(alarms_block.counts/items_top, monitoring_block.
    predicted_yield/gap)는 어떤 화면도 렌더링하지 않는 것으로 확인되어
    (알람/신뢰도 게이트 파이프라인 폐기, 수율 예측 갱신 발송으로 대체)
    제거했다 -- `_dispatch_yield_update_for_refresh`가 알림 발송을
    전담한다.
    """
    from api.routes.analysis import _action_priority_payload, _fmea_payload, _pareto_payload
    from src.runtime.app_state import get_latest_state

    registry = _dataset_registry(store)
    try:
        registry.get_dataframe(eval_dataset_id)
    except Exception:
        logger.exception("auto_refresh: 평가 데이터셋 로드 실패")
        errors.append(f"평가 데이터셋({eval_dataset_id})을 불러오지 못했습니다.")
        return None, None

    pareto_by_target: dict[str, Any] = {}
    failed_targets: list[str] = []
    for target in TARGETS:
        try:
            pareto_by_target[target] = _pareto_payload(eval_dataset_id, target, 10)
        except Exception:
            logger.exception("auto_refresh: Pareto 계산 실패 target=%s", target)
            failed_targets.append(target)

    if not pareto_by_target:
        errors.append("모든 타깃의 인자 스크리닝이 실패했습니다.")
        return None, None
    if failed_targets:
        errors.append(f"일부 타깃 스크리닝 실패: {', '.join(failed_targets)}")

    # 이번 분석 사이클의 train 데이터셋 id -- 블록①(조치 우선순위)도 같은
    # train.CSV 기준(작업 지시서 MB-6)이라 아래에서 재사용하고, 수율 예측
    # 갱신 발송(_dispatch_yield_update_for_refresh)도 같은 값을 쓴다.
    latest = get_latest_state(store)
    train_dataset_for_analysis = (
        (latest.get("alarms") or {}).get("train_dataset") if latest.get("alarms") else None
    ) or FALLBACK_TRAIN_DATASET

    # 모니터링 홈 FMEA 분석표(블록③ 데이터 한계의 원천) -- 다른 원인분석
    # 결과와 같은 스냅샷 저장 시점에 함께 계산한다. 별도 온디맨드 조회
    # 경로는 없다(IA-5). 실패해도 나머지 스냅샷 저장을 막지 않는다(J-2
    # 부분 실패 정책). `fmeaError`는 수동 분석 저장 경로(api/routes/state.py
    # `_with_fmea`)와 같은 필드명을 써서, 화면이 두 경로 어느 쪽에서
    # 왔든 같은 로직으로 "계산 안 됨"과 "계산 실패"를 구분할 수 있게
    # 한다(지시서 JA-3).
    fmea = None
    fmea_error = None
    try:
        fmea = _fmea_payload(eval_dataset_id, TARGETS)
    except Exception:
        logger.exception("auto_refresh: FMEA 분석표 계산 실패")
        errors.append("FMEA 분석표 계산에 실패했습니다.")
        fmea_error = "FMEA 분석표 계산 중 오류가 발생했습니다."

    # MB/MC: 모니터링 홈 블록①·② -- train.CSV 기준이라 eval 데이터셋과
    # 무관하다. 같은 "부분 실패 정책"(J-2)을 따른다.
    action_priority = None
    action_priority_error = None
    try:
        action_priority = _action_priority_payload(train_dataset_for_analysis)
    except Exception:
        logger.exception("auto_refresh: 조치 우선순위 계산 실패")
        errors.append("조치 우선순위 계산에 실패했습니다.")
        action_priority_error = "조치 우선순위 계산 중 오류가 발생했습니다."

    analysis_block = {
        "paretoByTarget": pareto_by_target,
        "fmea": fmea,
        "fmeaError": fmea_error,
        "actionPriority": action_priority,
        "actionPriorityError": action_priority_error,
        "target_provenance": next(
            (payload.get("target_provenance") for payload in pareto_by_target.values() if payload.get("target_provenance")),
            None,
        ),
    }

    return analysis_block, train_dataset_for_analysis


def _dispatch_yield_update_for_refresh(
    store: RuntimeStore,
    *,
    mode: str,
    train_dataset_id: str,
    eval_dataset_id: str,
    eval_dataset_filename: str | None,
    model_label: str | None,
    now_iso: str,
) -> None:
    """VE-1: 자동 갱신 스냅샷 저장 직후 수율 예측 갱신을 발송한다. 알람
    발송(`dispatch_new_alarms`)과 달리 AUC 게이트나 발송 시점(timing)
    조건이 없다 -- "자동 갱신마다 발송"이다. 억제는
    `yield_update_dispatch`의 신규분만/시간당 예산 두 규칙만 적용된다.
    지연 임포트는 순환 임포트 회피 목적이다(다른 신규 알람 발송 블록과
    같은 이유).
    """
    from api.routes.analysis import _hydrated_targets_or_409
    from src.analysis.yield_prediction import build_yield_prediction_table
    from src.automation.refresh_dispatch import _source_note_for
    from src.notifications.yield_update_dispatch import TRIGGER_REFRESH, dispatch_yield_update
    from src.notifications.yield_update_senders import build_yield_update_payload

    registry = _dataset_registry(store)
    train_df = registry.get_dataframe(train_dataset_id)
    eval_df = registry.get_dataframe(eval_dataset_id)
    hydrated = _hydrated_targets_or_409(eval_dataset_id)

    table = build_yield_prediction_table(
        train_df,
        eval_df,
        hydrated.dataframe,
        dataset_id=eval_dataset_id,
        train_dataset_id=train_dataset_id,
        train_dataset_version=registry.content_version(train_dataset_id),
    )

    timestamp_label = datetime.fromisoformat(now_iso).astimezone(_KST).strftime("%H:%M")
    payload = build_yield_update_payload(
        table,
        dataset_label=eval_dataset_filename or eval_dataset_id,
        timestamp_label=timestamp_label,
        source_note=_source_note_for(mode, eval_dataset_id),
        model_label=model_label,
    )
    dispatch_yield_update(store, payload, trigger=TRIGGER_REFRESH)
