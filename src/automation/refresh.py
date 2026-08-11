"""SC그룹("모델 분석"): 사이드바 "모델 분석" 팝업의 [분석 시작] 버튼이
부르는 4화면 원자적 분석 파이프라인 -- 등록된 분석 데이터로 모니터링
홈·Config별 트리맵·원인 분석·수율 예측 네 화면을 한 번에 계산해 하나의
`analysis_id`를 공유하는 스냅샷으로 저장한다.

RB-3: 이 파이프라인은 학습을 트리거하지 않는다 -- 분석 데이터는 항상
분석셋(eval)이고, 학습은 모델 학습 팝업의 수동 업로드(또는 내장
train.csv)로만 일어난다.

SD그룹 이후 이 파이프라인은 더 이상 APScheduler 주기 잡이 아니다 --
주기적으로 도는 것은 `src/automation/yield_dispatch.py`(수율 예측만
계산해 알림만 보낸다)이고, 이 파이프라인(`run_refresh_pipeline`)은
① [분석 시작] 버튼(`POST /api/state/refresh`), ② 서버 기동 시 유효
스냅샷이 없을 때(부트스트랩), ③ 학습 완료 후 스냅샷이 활성 모델
기준으로 무효화됐을 때(1회, 조용히) 만 실행된다 -- 이름(`refresh`)은
과거 주기 잡이던 시절의 흔적으로 남아 있다.

각 단계는 독립적으로 try/except하고, 실패는 로그 + 스냅샷의 `errors`
배열에 남긴다(단, `except`로 삼키고 성공한 척하지 않는다 -- 화면에
실패가 보여야 한다). SC-3: 4단계(모니터링/Config별 트리맵/원인분석/
수율예측) 중 하나라도 완전히 실패하면 스냅샷을 저장하지 않는다("원자적
저장 -- 넷 다 성공해야 교체").
"""

from __future__ import annotations

import logging
import threading
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
    """이 파이프라인 자체는 SC/SD그룹 이후 알림을 전혀 보내지 않는다
    (알림은 전적으로 "알림·자동화 설정"의 책임 -- 아래 `dispatch` 인자와는
    무관하다). `dispatch=False`는 WK그룹(콜드 스타트) 전용으로, 서버
    최초 기동 시 내장 test.csv로 돌리는 1회성 부트스트랩 실행이 "이후의
    진짜 정상 실행"과 구분되게 표시하는 데만 쓰인다(부트스트랩 실패
    상태를 지우는 로직이 이 값을 본다, ZB-2 참고). [분석 시작] 버튼·학습
    후 자동 복구 실행은 항상 기본값(True)을 쓴다."""
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
    """SC-2단계("Config별 트리맵"): 타깃 하이드레이션 + 스키마 파싱 +
    상관관계 히트맵을 먼저 데워둔다 -- Config별 트리맵을 포함해 원인 분석
    탭의 모든 scatter/heatmap 카드가 공유하는 선행 조건이다. 이제는
    파이프라인의 2단계로서 *동기적으로* 호출되어 원자성 게이트 역할도
    겸한다(이 단계가 실패하면 스냅샷을 저장하지 않는다 -- SC-3 "넷 다
    성공해야 교체").

    Deliberately narrow: only the SHARED prerequisites (target hydration,
    parsed schema, heatmap). Individual scatter cards themselves are
    intentionally NOT precomputed here -- "개별 산점도는 워밍업 대상에서 제외".

    E-4(perf): 히트맵(`_cached_heatmap`, ~5.7초 @ train 10k)은 예전에
    이 워밍업 대상이 아니었다 -- 원인 분석 탭 첫 진입에서 그 5.7초를
    그대로 물었다. `_cached_schema`를 데우는 김에 같은 자리에서 데운다.
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
    막지 않는다(J-2 부분 실패 정책) -- `fmeaError`로 "계산 안 됨"과
    "계산 실패"를 구분한다."""
    from api.routes.analysis import _fmea_payload

    try:
        return _fmea_payload(eval_dataset_id, TARGETS), None
    except Exception:
        logger.exception("auto_refresh: FMEA 분석표 계산 실패")
        errors.append("FMEA 분석표 계산에 실패했습니다.")
        return None, "FMEA 분석표 계산 중 오류가 발생했습니다."


def _action_priority_stage(train_dataset_for_analysis: str, errors: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    """모니터링 홈 블록①·② -- train.CSV 기준(작업 지시서 MB-6)이라 eval
    데이터셋과 무관하다. FMEA와 같은 부분 실패 정책을 따른다."""
    from api.routes.analysis import _action_priority_payload

    try:
        return _action_priority_payload(train_dataset_for_analysis), None
    except Exception:
        logger.exception("auto_refresh: 조치 우선순위 계산 실패")
        errors.append("조치 우선순위 계산에 실패했습니다.")
        return None, "조치 우선순위 계산 중 오류가 발생했습니다."


def _pareto_stage(eval_dataset_id: str, errors: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """SC-3단계("원인 분석"): 타깃별 Pareto 랭킹. 전 타깃 실패는 호출부가
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
    """SC-4단계("수율 예측"): 수율 예측 표를 계산한다. 반환값은
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


def _run_refresh_pipeline_inner(store: RuntimeStore, *, dispatch: bool = True) -> None:
    errors: list[str] = []
    registry = _dataset_registry(store)
    now_iso = datetime.now(timezone.utc).isoformat()
    analysis_id = now_iso

    # -- 1. 데이터 소스 해석 (J-1) -----------------------------------
    mode, train_dataset_id, eval_dataset_id, source_row_count = _resolve_source(store, registry, errors)

    # -- 2. 현재 챔피언 정보만 읽는다 (RB-3) -- 이 파이프라인은 학습을
    # 트리거하지 않는다. 학습은 모델 학습 팝업의 수동 업로드로만
    # 일어난다.
    model_meta = _current_model_meta(store)

    try:
        registry.get_dataframe(eval_dataset_id)
    except Exception:
        logger.exception("auto_refresh: 평가 데이터셋 로드 실패")
        errors.append(f"평가 데이터셋({eval_dataset_id})을 불러오지 못했습니다.")
        return

    latest = get_latest_state(store)
    train_dataset_for_analysis = (
        (latest.get("alarms") or {}).get("train_dataset") if latest.get("alarms") else None
    ) or FALLBACK_TRAIN_DATASET

    # -- SC-3: 4단계 -- 하나라도 완전히 실패하면 스냅샷을 저장하지 않고
    # (기존 스냅샷 보존) 진행 상태를 지운다("원자적 저장").
    store.set_analysis_progress(stage="모니터링 홈", index=1, total=4, analysis_id=analysis_id)
    fmea, fmea_error = _fmea_stage(eval_dataset_id, errors)
    action_priority, action_priority_error = _action_priority_stage(train_dataset_for_analysis, errors)

    store.set_analysis_progress(stage="Config별 트리맵", index=2, total=4, analysis_id=analysis_id)
    try:
        _warmup_common_prerequisites(eval_dataset_id)
    except Exception:
        logger.exception("auto_refresh: Config별 트리맵 선행 조건 계산 실패 -- 스냅샷 저장 생략")
        errors.append("Config별 트리맵 계산에 실패했습니다.")
        store.clear_analysis_progress()
        return

    store.set_analysis_progress(stage="원인 분석", index=3, total=4, analysis_id=analysis_id)
    pareto_by_target, target_provenance, failed_targets = _pareto_stage(eval_dataset_id, errors)
    if not pareto_by_target:
        errors.append("모든 타깃의 인자 스크리닝이 실패했습니다.")
        logger.warning("auto_refresh: 원인분석 전 타깃 실패 -- 스냅샷 저장 생략")
        store.clear_analysis_progress()
        return
    if failed_targets:
        errors.append(f"일부 타깃 스크리닝 실패: {', '.join(failed_targets)}")

    store.set_analysis_progress(stage="수율 예측", index=4, total=4, analysis_id=analysis_id)
    yield_table = _yield_prediction_stage(registry, train_dataset_for_analysis, eval_dataset_id, errors)
    if yield_table is None:
        logger.warning("auto_refresh: 수율 예측 계산 실패 -- 스냅샷 저장 생략")
        store.clear_analysis_progress()
        return

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
        # 알람 등급(심각/위험/주의)·게이트 판정 파이프라인은 폐기됐다 --
        # 어떤 화면도 이 두 블록의 내용(counts/items_top/gate_passed/
        # predicted_yield/gap)을 더 이상 렌더링하지 않는 것으로 확인했다
        # (알림 발송은 수율 예측 갱신 파이프라인, 아래로 대체).
        # 스냅샷 스키마의 키 자체는 다른 소비처가 존재를 가정할 수 있어
        # 유지하되, 내용은 빈 상태로 남긴다.
        "alarms": None,
        "monitoring": {"predicted_yield": None, "gap": None, "gap_pareto": [], "treemap": None},
        "errors": errors,
    }
    store.save_refresh_snapshot(snapshot)
    store.clear_analysis_progress()
    # ZB-2: 콜드 스타트 부트스트랩(`_run_bootstrap`)이 한 번 실패하면
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

    # SC/SD그룹: "모델 분석"([분석 시작])은 네 화면을 갱신할 뿐, 알림을
    # 보내지 않는다 -- 알림 발송은 전적으로 "알림·자동화 설정"의 자동화
    # (주기 SQL 잡, `src/automation/yield_dispatch.py`)와 매일 09:00/13:00
    # 예약 잡의 책임이다(작업 지시서 최상단 역할표: "모델 분석 = 네 화면
    # 갱신", "알림·자동화 = 화면 없이 수율 예측만 돌려 알림 발송"). 이
    # 함수는 알림 파이프라인을 부르지 않는다.
    if not dispatch:
        logger.info("auto_refresh: dispatch=False (콜드 스타트) -- 알림은 애초에 이 파이프라인 책임이 아니다.")


def _resolve_source(
    store: RuntimeStore, registry: DatasetRegistry, errors: list[str]
) -> tuple[str, str, str, int]:
    """SC-2: "모델 분석"이 쓰는 분석 데이터는 등록된 것(수동 override --
    파일 업로드 또는 "데이터베이스에서 불러오기") 하나뿐이고, 없으면
    내장 test.CSV로 폴백한다("기본값"). 이 파이프라인은 더 이상 SQL을
    직접 조회하지 않는다 -- SQL 조회는 ①"데이터베이스에서 불러오기"
    버튼(`POST /api/state/fetch-from-db`, 결과를 수동 override로 등록)과
    ②"알림·자동화 설정"의 주기 자동화(`src/automation/yield_dispatch.py`,
    화면을 건드리지 않고 수율 예측만 계산)만 한다 -- "모델 분석"이 SQL을
    조용히 자동 선점하면 "한 번 등록되면 다시 바꿀 때까지 유지된다"(SC-2)
    는 원칙이 깨진다.

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


