"""사이드바 "모델 분석" 팝업의 [분석 시작] 버튼이 부르는 4화면 원자적
분석 파이프라인 -- 등록된 분석 데이터로 모니터링 홈·Config별 트리맵·원인
분석·수율 예측 네 화면을 한 번에 계산해 하나의 `analysis_id`를 공유하는
스냅샷으로 저장한다.

이 파이프라인은 학습을 트리거하지 않는다 -- 분석 데이터는 항상
분석셋(eval)이고, 학습은 모델 학습 팝업의 수동 업로드(또는 내장
train.csv)로만 일어난다.

주기 잡이 아니다. 이름(`refresh`)과 달리 `run_refresh_pipeline`은
① [분석 시작] 버튼(`POST /api/state/refresh`), ② 서버 기동 시 유효
스냅샷이 없을 때(부트스트랩), ③ 학습 완료 후 스냅샷이 활성 모델
기준으로 무효화됐을 때(1회, 조용히) 만 실행된다. 주기적으로 도는 것은
`src/automation/yield_dispatch.py`(수율 예측만 계산해 알림만 보낸다)다.

각 단계는 독립적으로 try/except하고, 실패는 로그 + 스냅샷의 `errors`
배열에 남긴다(단, `except`로 삼키고 성공한 척하지 않는다 -- 화면에
실패가 보여야 한다). 4화면(모니터링/Config별 트리맵/원인분석/수율예측)의
데이터 중 하나라도 완전히 실패하면 스냅샷을 저장하지 않는다("원자적
저장 -- 넷 다 성공해야 교체").

진행 표시(`analysis_progress`)는 화면 4개가 아니라 8단계로 쪼갠다 --
"모니터링 홈" 한 라벨 아래에 서로 다른 데이터셋에 대한 두 개의 무거운
계산(FMEA + 조치 우선순위)이 숨으면 어디서 오래 걸리는지 알 수 없다.
각 단계 결과는 `RuntimeStore.save_refresh_checkpoint`에도 남는다 --
화면이 보는 최종 스냅샷과는 별개로, 파이프라인이 죽었다가 같은 입력으로
재시작됐을 때 이미 끝낸 단계를 다시 계산하지 않기 위해서다(`analysis_id`가
실행 시각이 아니라 데이터셋+모델 버전의 해시라 "같은 입력 = 같은
analysis_id"가 성립한다).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from api.settings import settings
from src.runtime.app_state import get_latest_state
from src.runtime.datasets import DatasetRegistry, parse_uploaded_csv
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

REFRESH_JOB_ID = "auto_refresh"
FALLBACK_TRAIN_DATASET = "train"
FALLBACK_EVAL_DATASET = "test"
TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")

# 8단계 진행 표시. 순서가 곧 화면에 뜨는 (index/total)이다.
STAGE_ORDER = (
    "resolve",
    "hydrate_eval",
    "fmea",
    "action_priority",
    "treemap_warmup",
    "pareto",
    "yield_prediction",
    "save",
)
STAGE_LABELS = {
    "resolve": "데이터 확인",
    "hydrate_eval": "모델 추론 (분석 데이터)",
    "fmea": "데이터 한계 진단",
    "action_priority": "조치 우선순위 (학습 데이터)",
    "treemap_warmup": "Config별 트리맵",
    "pareto": "원인 분석",
    "yield_prediction": "수율 예측",
    "save": "저장",
}
TOTAL_STAGES = len(STAGE_ORDER)
# 이 주기로 하트비트를 갱신한다 -- 프런트의 60초 정지 판정보다
# 충분히 촘촘해야 정상 실행 중에 "중단됨"으로 오판하지 않는다.
HEARTBEAT_INTERVAL_SECONDS = 10.0


def _set_stage(
    store: RuntimeStore,
    stage_key: str,
    analysis_id: str,
    *,
    row_count: int | None = None,
    estimated_seconds: int | None = None,
) -> None:
    index = STAGE_ORDER.index(stage_key) + 1
    store.set_analysis_progress(
        stage=STAGE_LABELS[stage_key],
        index=index,
        total=TOTAL_STAGES,
        analysis_id=analysis_id,
        row_count=row_count,
        estimated_seconds=estimated_seconds,
    )


def _estimate_total_seconds(row_count: int) -> int:
    """화면에 "약 N분 예상"을 보여주기 위한 대략적인 추정치다 -- 정밀한
    SLA가 아니라 "끊긴 게 아니라 원래 이 정도 걸린다"를 체감시키는
    용도다. 표본 상한 덕에 대부분 단계는 행 수와 거의 무관하지만, 수율
    예측·모델 추론은 전량 O(n)이라 행 수에 비례해 늘어난다."""
    return 20 + max(0, row_count) // 1500


@contextmanager
def _heartbeat_during_pipeline(store: RuntimeStore):
    """파이프라인이 도는 내내(단계 하나가 몇 초가 걸리든) 백그라운드
    스레드가 주기적으로 하트비트를 갱신한다 -- 프로세스가 죽으면 이 스레드도
    함께 죽으므로, 하트비트가 멈춘다는 것 자체가 "죽었다"는 신호가 된다."""
    stop = threading.Event()

    def _tick() -> None:
        while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            try:
                store.touch_analysis_progress_heartbeat()
            except Exception:
                logger.exception("auto_refresh: 하트비트 갱신 실패")

    thread = threading.Thread(target=_tick, name="auto_refresh-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def _compute_analysis_id(registry: DatasetRegistry, eval_dataset_id: str, train_dataset_id: str, store: RuntimeStore) -> str:
    """실행 시각이 아니라 입력의 해시를 쓴다 -- 같은
    데이터셋·같은 모델로 다시 돌리면 항상 같은 analysis_id가 나오므로,
    죽었다가 재시작된 실행이 체크포인트를 "내 것"으로 알아볼 수 있다."""
    try:
        eval_version = registry.content_version(eval_dataset_id)
    except Exception:
        eval_version = "unknown"
    try:
        train_version = registry.content_version(train_dataset_id)
    except Exception:
        train_version = "unknown"
    model_id = _current_champion_id(store) or "none"
    raw = f"{eval_dataset_id}:{eval_version}:{train_dataset_id}:{train_version}:{model_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class _PipelineOutcome:
    """`run_refresh_pipeline`이 이 결과를
    `RuntimeStore.finish_last_run_*`에 그대로 기록한다.
    `success=True`는 정확히 "스냅샷을 저장했다"와 같다(원자적 저장 지점)
    -- 부분 실패(`errors`에만 쌓이고 스냅샷은 저장되는 경우, 예: FMEA
    실패)는 `success=True`로 남는다. 그건 errors 배너가 이미 따로
    알린다; `last_run.status`는 "이번 실행이 화면을 갱신했는가"
    만 답한다."""

    success: bool
    failed_stage: str | None = None
    error_message: str | None = None


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
    """이 파이프라인 자체는 알림을 전혀 보내지 않는다
    (알림은 전적으로 "알림·자동화 설정"의 책임 -- 아래 `dispatch` 인자와는
    무관하다). `dispatch=False`는 콜드 스타트 전용으로, 서버 최초 기동
    시 내장 test.csv로 돌리는 1회성 부트스트랩 실행이 "이후의 진짜 정상
    실행"과 구분되게 표시하는 데만 쓰인다(부트스트랩 실패 상태를 지우는
    로직이 이 값을 본다). [분석 시작] 버튼·학습
    후 자동 복구 실행은 항상 기본값(True)을 쓴다."""
    if not _refresh_lock.acquire(blocking=False):
        logger.info("auto_refresh: 이미 다른 실행이 진행 중이라 이번 호출은 건너뜁니다.")
        return
    try:
        store = _runtime_store()
        outcome = _PipelineOutcome(False, None, "파이프라인이 시작되지 못했습니다.")
        try:
            outcome = _run_refresh_pipeline_inner(store, dispatch=dispatch)
        except Exception as exc:
            # 개별 단계는 각자 try/except하지만, 예상하지 못한 예외가 여기까지
            # 올라오면 스케줄러 자체가 죽지 않도록 마지막 방어선에서 삼킨다.
            # MemoryError를 포함한 어떤 예외든(개별 단계가 던지지 않은
            # 예상 밖 실패까지) 진행 상태를 지우지 않고는 빠져나가지 않는다 --
            # 안 지우면 화면에 "분석 진행 중… (N/8)"이 영원히 남는다.
            logger.exception("auto_refresh: 파이프라인 실행 중 예기치 않은 오류")
            outcome = _PipelineOutcome(False, None, f"예기치 않은 오류: {exc}")
        finally:
            # 어떤 경로로 빠져나가든(성공/부분실패/예기치 않은 예외)
            # last_run이 항상 최종 상태로 남아야 한다 -- 안 남기면 실패가
            # 로그에만 남고 프런트는 영원히 "triggered: true" 이후를
            # 기다린다.
            if outcome is None:
                # 방어적 처리 -- `_run_refresh_pipeline_inner`는 항상
                # `_PipelineOutcome`을 반환하는 계약이지만, 테스트 더블처럼
                # 그 계약을 지키지 않는 호출부가 있어도 여기서 죽지 않는다.
                outcome = _PipelineOutcome(False, None, "파이프라인이 결과를 반환하지 않았습니다.")
            if outcome.success:
                store.finish_last_run_success()
            else:
                store.finish_last_run_failure(stage=outcome.failed_stage, message=outcome.error_message)
            store.clear_analysis_progress()
    finally:
        _refresh_lock.release()


def _warmup_common_prerequisites(eval_dataset_id: str) -> None:
    """2단계("Config별 트리맵"): 타깃 하이드레이션 + 스키마 파싱 +
    상관관계 히트맵을 먼저 데워둔다 -- Config별 트리맵을 포함해 원인 분석
    탭의 모든 scatter/heatmap 카드가 공유하는 선행 조건이다. 이제는
    파이프라인의 2단계로서 *동기적으로* 호출되어 원자성 게이트 역할도
    겸한다(이 단계가 실패하면 스냅샷을 저장하지 않는다 -- "넷 다 성공해야
    교체").

    Deliberately narrow: only the SHARED prerequisites (target hydration,
    parsed schema, heatmap). Individual scatter cards themselves are
    intentionally NOT precomputed here -- "개별 산점도는 워밍업 대상에서 제외".

    히트맵(`_cached_heatmap`)은 train 10k 기준 ~5.7초가 걸린다 -- 여기서
    데우지 않으면 원인 분석 탭 첫 진입이 그 시간을 그대로 문다.
    `_cached_schema`를 데우는 김에 같은 자리에서 데운다.
    """
    from api.routes.analysis import TARGET_HYDRATION_VERSION, _cached_heatmap, _cached_schema, _hydrated_targets_or_409

    hydrated = _hydrated_targets_or_409(eval_dataset_id)
    provenance = hydrated.provenance
    dataset_version = provenance.dataset_version
    _cached_schema(eval_dataset_id, dataset_version)
    _cached_heatmap(
        eval_dataset_id,
        dataset_version,
        provenance.model_id or "measured-only",
        provenance.model_version or "none",
        TARGET_HYDRATION_VERSION,
    )


def _fmea_stage(eval_dataset_id: str, errors: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    """모니터링 홈 블록③(데이터 한계/FMEA). 실패해도 나머지 단계의 저장을
    막지 않는다(부분 실패 정책) -- `fmeaError`로 "계산 안 됨"과
    "계산 실패"를 구분한다."""
    from api.routes.analysis import _fmea_payload

    try:
        return _fmea_payload(eval_dataset_id, TARGETS), None
    except Exception:
        logger.exception("auto_refresh: FMEA 분석표 계산 실패")
        errors.append("FMEA 분석표 계산에 실패했습니다.")
        return None, "FMEA 분석표 계산 중 오류가 발생했습니다."


def _action_priority_stage(train_dataset_for_analysis: str, errors: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    """모니터링 홈 블록①·② -- train.CSV 기준이라 eval 데이터셋과
    무관하다. FMEA와 같은 부분 실패 정책을 따른다."""
    from api.routes.analysis import _action_priority_payload

    try:
        return _action_priority_payload(train_dataset_for_analysis), None
    except Exception:
        logger.exception("auto_refresh: 조치 우선순위 계산 실패")
        errors.append("조치 우선순위 계산에 실패했습니다.")
        return None, "조치 우선순위 계산 중 오류가 발생했습니다."


def _pareto_stage(eval_dataset_id: str, errors: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """3단계("원인 분석"): 타깃별 Pareto 랭킹. 전 타깃 실패는 호출부가
    빈 dict로 판정해 스냅샷 저장을 생략한다."""
    from api.routes.analysis import _pareto_payload

    pareto_by_target: dict[str, Any] = {}
    failed_targets: list[str] = []
    for target in TARGETS:
        try:
            pareto_by_target[target] = _pareto_payload(eval_dataset_id, target, 10)
        except Exception:
            logger.exception("auto_refresh: Pareto 계산 실패 target=%s", target)
            failed_targets.append(target)

    target_provenance = next(
        (payload.get("target_provenance") for payload in pareto_by_target.values() if payload.get("target_provenance")),
        None,
    )
    return pareto_by_target, target_provenance, failed_targets


def _yield_prediction_stage(
    registry: DatasetRegistry, train_dataset_id: str, eval_dataset_id: str, errors: list[str]
) -> Any | None:
    """4단계("수율 예측"): 수율 예측 표를 계산한다. 반환값은
    `YieldPredictionTable`(frozen dataclass) 그대로다 -- 호출부가 이
    회차의 알림 발송(`_dispatch_yield_update_for_refresh`)에 그대로
    넘기고, 스냅샷에는 `serialize_yield_prediction_table`로 JSON-safe
    변환한 사본을 캐시한다. `GET /api/alerts/ranking`이 그 캐시를
    재사용해 "화면에 보이는 수치 = 알림에 적힌 수치"를 보장한다(같은
    analysis_id 아래 다시 계산하지 않는다)."""
    from api.routes.analysis import _hydrated_targets_or_409
    from src.analysis.yield_prediction import build_yield_prediction_table

    try:
        train_df = registry.get_dataframe(train_dataset_id)
        eval_df = registry.get_dataframe(eval_dataset_id)
        hydrated = _hydrated_targets_or_409(eval_dataset_id)
        return build_yield_prediction_table(
            train_df,
            eval_df,
            hydrated.dataframe,
            dataset_id=eval_dataset_id,
            train_dataset_id=train_dataset_id,
            train_dataset_version=registry.content_version(train_dataset_id),
        )
    except Exception:
        logger.exception("auto_refresh: 수율 예측 계산 실패")
        errors.append("수율 예측 계산에 실패했습니다.")
        return None


def _run_refresh_pipeline_inner(store: RuntimeStore, *, dispatch: bool = True) -> _PipelineOutcome:
    errors: list[str] = []
    registry = _dataset_registry(store)
    now_iso = datetime.now(timezone.utc).isoformat()

    with _heartbeat_during_pipeline(store):
        # -- 1/8 데이터 확인 --------------------------------------------
        mode, train_dataset_id, eval_dataset_id, source_row_count = _resolve_source(store, registry, errors)
        analysis_id = _compute_analysis_id(registry, eval_dataset_id, train_dataset_id, store)
        # 이 실행의 진행상태 기록을
        # 시작한다 -- 아래 어떤 단계에서 멈추더라도(중간에 `return`하든,
        # 이 함수를 감싼 바깥 try/except가 예상 밖 예외를 잡든)
        # `run_refresh_pipeline`이 이 레코드를 최종 상태로 갱신한다.
        store.start_last_run(analysis_id=analysis_id)
        checkpoint = store.get_refresh_checkpoint(analysis_id)
        completed = set((checkpoint or {}).get("completed_stages") or [])
        if checkpoint:
            logger.info("auto_refresh: 체크포인트 발견(analysis_id=%s) -- 완료 단계 %s", analysis_id, sorted(completed))
        # 이후 모든 단계 배너에 "N행 · 약 M분 예상"을 함께 싣는다.
        estimated_seconds = _estimate_total_seconds(source_row_count)

        def _stage(stage_key: str) -> None:
            _set_stage(store, stage_key, analysis_id, row_count=source_row_count, estimated_seconds=estimated_seconds)

        _stage("resolve")
        t_stage = time.perf_counter()
        # -- 현재 챔피언 정보만 읽는다 -- 이 파이프라인은 학습을
        # 트리거하지 않는다. 학습은 모델 학습 팝업의 수동 업로드로만
        # 일어난다.
        model_meta = _current_model_meta(store)
        try:
            registry.get_dataframe(eval_dataset_id)
        except Exception:
            logger.exception("auto_refresh: 평가 데이터셋 로드 실패")
            message = f"평가 데이터셋({eval_dataset_id})을 불러오지 못했습니다."
            errors.append(message)
            return _PipelineOutcome(False, "resolve", message)
        logger.info("auto_refresh[1/8 데이터 확인] %.1fs", time.perf_counter() - t_stage)

        latest = get_latest_state(store)
        train_dataset_for_analysis = (
            (latest.get("alarms") or {}).get("train_dataset") if latest.get("alarms") else None
        ) or FALLBACK_TRAIN_DATASET

        # -- 2/8 모델 추론 (분석 데이터) -----------------------------------
        # 하이드레이션은 FMEA 단계 안이 아니라 여기서 별도 단계로 돈다 --
        # 다른 단계 라벨 뒤에 숨으면 진행 표시가 그 시간 동안 멈춰 보인다.
        _stage("hydrate_eval")
        t_stage = time.perf_counter()
        try:
            from api.routes.analysis import _hydrated_targets_or_409

            _hydrated_targets_or_409(eval_dataset_id)
        except Exception as exc:
            logger.exception("auto_refresh: 분석 데이터 모델 추론 실패 -- 스냅샷 저장 생략")
            message = f"분석 데이터 모델 추론에 실패했습니다: {exc}"
            errors.append("분석 데이터 모델 추론에 실패했습니다.")
            return _PipelineOutcome(False, "hydrate_eval", message)
        logger.info("auto_refresh[2/8 모델 추론] %.1fs", time.perf_counter() - t_stage)

        # -- 3/8 데이터 한계 진단 -------------------------------------------
        _stage("fmea")
        if "fmea" in completed:
            fmea, fmea_error = checkpoint.get("fmea"), checkpoint.get("fmea_error")
            logger.info("auto_refresh[3/8 데이터 한계 진단] 체크포인트 재사용")
        else:
            t_stage = time.perf_counter()
            fmea, fmea_error = _fmea_stage(eval_dataset_id, errors)
            logger.info("auto_refresh[3/8 데이터 한계 진단] %.1fs", time.perf_counter() - t_stage)
            store.save_refresh_checkpoint(analysis_id, "fmea", {"fmea": fmea, "fmea_error": fmea_error})

        # -- 4/8 조치 우선순위 (학습 데이터) ---------------------------------
        # train.CSV가 바뀌지 않는 한 순위/권장구간은 이미 프로세스
        # 전역 캐시(`_ranked_rows_for_provenance`/`compare_methods`)로
        # 사실상 즉시 반환되지만(analysis.py의 _action_priority_payload
        # 독스트링 참고), 재시작으로 그 캐시까지 날아간 경우를 위해
        # 체크포인트로 한 번 더 방어한다.
        _stage("action_priority")
        if "action_priority" in completed:
            action_priority = checkpoint.get("action_priority")
            action_priority_error = checkpoint.get("action_priority_error")
            logger.info("auto_refresh[4/8 조치 우선순위] 체크포인트 재사용")
        else:
            t_stage = time.perf_counter()
            action_priority, action_priority_error = _action_priority_stage(train_dataset_for_analysis, errors)
            logger.info("auto_refresh[4/8 조치 우선순위] %.1fs", time.perf_counter() - t_stage)
            store.save_refresh_checkpoint(
                analysis_id,
                "action_priority",
                {"action_priority": action_priority, "action_priority_error": action_priority_error},
            )

        # -- 5/8 Config별 트리맵 (선행 조건 워밍업) --------------------------
        _stage("treemap_warmup")
        t_stage = time.perf_counter()
        try:
            _warmup_common_prerequisites(eval_dataset_id)
        except Exception as exc:
            logger.exception("auto_refresh: Config별 트리맵 선행 조건 계산 실패 -- 스냅샷 저장 생략")
            message = f"Config별 트리맵 계산에 실패했습니다: {exc}"
            errors.append("Config별 트리맵 계산에 실패했습니다.")
            return _PipelineOutcome(False, "treemap_warmup", message)
        logger.info("auto_refresh[5/8 Config별 트리맵] %.1fs", time.perf_counter() - t_stage)

        # -- 6/8 원인 분석 -----------------------------------------------
        _stage("pareto")
        if "pareto" in completed:
            pareto_by_target = checkpoint.get("pareto_by_target") or {}
            target_provenance = checkpoint.get("target_provenance")
            failed_targets = checkpoint.get("failed_targets") or []
            logger.info("auto_refresh[6/8 원인 분석] 체크포인트 재사용")
        else:
            t_stage = time.perf_counter()
            pareto_by_target, target_provenance, failed_targets = _pareto_stage(eval_dataset_id, errors)
            logger.info("auto_refresh[6/8 원인 분석] %.1fs", time.perf_counter() - t_stage)
            if pareto_by_target:
                store.save_refresh_checkpoint(
                    analysis_id,
                    "pareto",
                    {
                        "pareto_by_target": pareto_by_target,
                        "target_provenance": target_provenance,
                        "failed_targets": failed_targets,
                    },
                )
        if not pareto_by_target:
            message = "모든 타깃의 인자 스크리닝이 실패했습니다."
            errors.append(message)
            logger.warning("auto_refresh: 원인분석 전 타깃 실패 -- 스냅샷 저장 생략")
            return _PipelineOutcome(False, "pareto", message)
        if failed_targets:
            errors.append(f"일부 타깃 스크리닝 실패: {', '.join(failed_targets)}")

        # -- 7/8 수율 예측 ------------------------------------------------
        # 절대 표본을 쓰지 않는다 -- 웨이퍼별 순위가 산출물이라
        # 체크포인트로도 건너뛰지 않는다(매번 전량 재계산).
        _stage("yield_prediction")
        t_stage = time.perf_counter()
        yield_table = _yield_prediction_stage(registry, train_dataset_for_analysis, eval_dataset_id, errors)
        logger.info("auto_refresh[7/8 수율 예측] %.1fs", time.perf_counter() - t_stage)
        if yield_table is None:
            logger.warning("auto_refresh: 수율 예측 계산 실패 -- 스냅샷 저장 생략")
            message = errors[-1] if errors else "수율 예측 계산에 실패했습니다."
            return _PipelineOutcome(False, "yield_prediction", message)

        # -- 8/8 저장 -----------------------------------------------------
        _stage("save")
        from src.analysis.yield_prediction import serialize_yield_prediction_table

        analysis_block = {
            "paretoByTarget": pareto_by_target,
            "fmea": fmea,
            "fmeaError": fmea_error,
            "actionPriority": action_priority,
            "actionPriorityError": action_priority_error,
            "target_provenance": target_provenance,
            "yieldPrediction": serialize_yield_prediction_table(
                yield_table, train_dataset_id=train_dataset_for_analysis, eval_dataset_id=eval_dataset_id
            ),
        }

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
            "analysis_id": analysis_id,
            "source": {
                "mode": mode,
                "train_dataset": train_dataset_id,
                "eval_dataset": eval_dataset_id,
                "eval_dataset_filename": eval_dataset_filename,
                "row_count": source_row_count,
            },
            "model": model_meta,
            "analysis": analysis_block,
            # 알림 발송은 수율 예측 갱신 파이프라인이 담당한다 -- 이
            # 스냅샷에는 알람 등급/게이트 판정 블록이 없다.
            "monitoring": {"predicted_yield": None, "gap": None, "gap_pareto": [], "treemap": None},
            "errors": errors,
        }
        store.save_refresh_snapshot(snapshot)
        # 화면이 보는 스냅샷이 이 회차를 온전히 반영하므로 재시작-이어하기용
        # 체크포인트는 더 이상 필요 없다.
        store.clear_refresh_checkpoint()
    # `run_refresh_pipeline`의 바깥 try/finally가
    # `clear_analysis_progress()`를 반드시 호출하므로 여기서 따로
    # 지우지 않는다(이 함수의 모든 return 경로 -- 성공/부분 실패 모두
    # -- 가 그 finally를 통과한다).
    # 콜드 스타트 부트스트랩(`_run_bootstrap`)이 한 번 실패하면
    # `bootstrap_status`가 store에 "failed"로 영구히 남는다 -- 그 이후의
    # [분석 시작]이 스냅샷 저장에 성공해도 아무도 그 상태를 지우지 않아
    # "첫 분석에 실패했습니다" 배너가 계속 뜬다(실제로 재현 확인함).
    # dispatch=True는 콜드 스타트 자신의 내부 호출(dispatch=False)과
    # 구분되는, 진짜 이후의 정상 실행이라는 뜻이므로 여기서 안전하게
    # 지운다.
    if dispatch:
        current_bootstrap = store.get_bootstrap_status()
        if current_bootstrap is not None and current_bootstrap.get("status") == "failed":
            store.set_bootstrap_status("done", None)
    logger.info("auto_refresh: 스냅샷 저장 완료 mode=%s eval=%s", mode, eval_dataset_id)

    # "모델 분석"([분석 시작])은 네 화면을 갱신할 뿐, 알림을 보내지
    # 않는다 -- 알림 발송은 전적으로 "알림·자동화 설정"의 주기 SQL 잡
    # (`src/automation/yield_dispatch.py`) 책임이다. 역할 분리: "모델
    # 분석 = 네 화면 갱신", "알림·자동화 = 화면 없이 수율 예측만 돌려
    # 알림 발송". 이 함수는 알림 파이프라인을 부르지 않는다.
    if not dispatch:
        logger.info("auto_refresh: dispatch=False (콜드 스타트) -- 알림은 애초에 이 파이프라인 책임이 아니다.")

    return _PipelineOutcome(True)


def _resolve_source(
    store: RuntimeStore, registry: DatasetRegistry, errors: list[str]
) -> tuple[str, str, str, int]:
    """"모델 분석"이 쓰는 분석 데이터는 등록된 것(수동 override -- 파일
    업로드 또는 "데이터베이스에서 불러오기") 하나뿐이고, 없으면 내장
    test.CSV로 폴백한다("기본값"). 이 파이프라인은 SQL을 직접 조회하지
    않는다 -- SQL 조회는 ①"데이터베이스에서 불러오기" 버튼
    (`POST /api/state/fetch-from-db`, 결과를 수동 override로 등록)과
    ②"알림·자동화 설정"의 주기 자동화(`src/automation/yield_dispatch.py`,
    화면을 건드리지 않고 수율 예측만 계산)만 한다. 여기서 SQL을 조용히
    자동 선점하면 "한 번 등록되면 다시 바꿀 때까지 유지된다"는 원칙이
    깨진다.

    학습 대상(train_dataset)은 이 함수가 절대 바꾸지 않는다 -- 직전
    스냅샷의 것을 그대로 이어받거나(자동 학습을 걸지 않는다), 없으면
    내장 train.CSV로 폴백한다. 학습 대상을 실제로 바꾸는 것은 모델 학습
    팝업의 수동 업로드/내장 train.csv뿐이다."""
    manual = store.get_manual_eval_override()
    if manual is not None:
        previous = store.get_refresh_snapshot_status()["snapshot"]
        previous_source = (previous or {}).get("source") if previous else None
        previous_train = (previous_source or {}).get("train_dataset") if previous_source else None
        try:
            row_count = len(registry.get_dataframe(manual["dataset_id"]))
        except Exception:
            logger.exception("auto_refresh: 등록된 분석 데이터를 읽지 못했습니다 -- %s", manual.get("dataset_id"))
            errors.append("등록된 분석 데이터를 읽지 못했습니다.")
            row_count = 0
        return "manual", previous_train or FALLBACK_TRAIN_DATASET, manual["dataset_id"], row_count

    # 폴백: 내장 test.CSV로 평가.
    try:
        eval_df = registry.get_dataframe(FALLBACK_EVAL_DATASET)
        row_count = len(eval_df)
    except Exception:
        errors.append("폴백 데이터셋(test.CSV)을 읽지 못했습니다.")
        row_count = 0
    return "fallback", FALLBACK_TRAIN_DATASET, FALLBACK_EVAL_DATASET, row_count


def _current_model_meta(store: RuntimeStore) -> dict[str, Any]:
    """이 파이프라인은 학습을 제출하지 않고 현재 활성 챔피언을 읽기만
    한다 -- 그래서 `trained_at`/`promoted`는 항상 None이다(이 회차가
    학습한 것이 아니므로 채울 값이 없다)."""
    return {
        "champion_version": _current_champion_id(store),
        "trained_at": None,
        "promoted": None,
        "skipped_reason": None,
    }


def _current_champion_id(store: RuntimeStore) -> str | None:
    active = store.active_model()
    return active.get("active_model_id") if active else None


