"""J-2/J-3: 리프레시 파이프라인 -- 갱신 주기마다 데이터 취득(SQL 또는
폴백) -> (신규 데이터일 때만) 학습 + 승격 게이트 -> 챔피언 예측 ->
원인분석 -> 알람 판정 -> 모니터링 요약을 계산해 하나의 스냅샷으로
저장한다. 사용자가 버튼을 누르지 않아도 어느 탭이든 최신 결과가 즉시
보이게 하는 것이 목적이다(J-4가 이 스냅샷을 읽는다).

`run_refresh_pipeline`은 APScheduler가 `refreshIntervalMinutes`마다
호출한다(`api/main.py`, job id `auto_refresh`) -- 기존 `auto_ingest`
(감시 디렉터리 폴링, `src/automation/ingest.py`)와는 별개의 잡이다.
이 잡은 예외를 절대 밖으로 던지지 않는다: 각 단계를 독립적으로
try/except하고, 실패는 로그 + 스냅샷의 `errors` 배열에 남긴다(단,
`except`로 삼키고 성공한 척하지 않는다 -- 화면에 실패가 보여야 한다).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from api.settings import settings
from src.automation import sql_source
from src.ml.dataset import has_target_column
from src.runtime.datasets import DatasetRegistry, parse_uploaded_csv
from src.runtime.operation_coordinator import ActiveOperationError
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

REFRESH_JOB_ID = "auto_refresh"
FALLBACK_TRAIN_DATASET = "train"
FALLBACK_EVAL_DATASET = "test"
# app_state 키 -- 마지막으로 학습에 실제로 제출한 데이터의 내용 해시.
# 같은 해시면(신규 행이 없으면) 재학습을 건너뛴다(J-1 "재학습 조건").
DATA_HASH_STATE_KEY = "automation:last_train_hash"
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


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class _RetryLater(Exception):
    """학습 Job 슬롯이 이미 사용 중일 때만 쓴다 -- src/automation/ingest.py의
    동명 예외와 같은 용도(일시적 혼잡, 다음 주기에 재시도)."""


def run_refresh_pipeline() -> None:
    if not _refresh_lock.acquire(blocking=False):
        logger.info("auto_refresh: 이미 다른 실행이 진행 중이라 이번 호출은 건너뜁니다.")
        return
    try:
        store = _runtime_store()
        try:
            _run_refresh_pipeline_inner(store)
        except Exception:
            # 개별 단계는 각자 try/except하지만, 예상하지 못한 예외가 여기까지
            # 올라오면 스케줄러 자체가 죽지 않도록 마지막 방어선에서 삼킨다.
            logger.exception("auto_refresh: 파이프라인 실행 중 예기치 않은 오류")
    finally:
        _refresh_lock.release()


def _run_refresh_pipeline_inner(store: RuntimeStore) -> None:
    errors: list[str] = []
    registry = _dataset_registry(store)
    now_iso = datetime.now(timezone.utc).isoformat()

    # -- 1. 데이터 소스 해석 (J-1) -----------------------------------
    mode, train_dataset_id, eval_dataset_id, source_row_count = _resolve_source(store, registry, errors)

    # -- 2. (신규 데이터일 때만) 학습 + 승격 게이트 (J-1) --------------
    model_meta = _maybe_retrain(store, registry, mode, errors)

    # -- 3. 챔피언으로 예측 + 4. 원인분석 + 5. 알람 판정 ---------------
    analysis_block, alarms_block, monitoring_block, alarm_items_for_dispatch, train_dataset_for_alarms = (
        _analyze_and_score(store, eval_dataset_id, errors)
    )

    if analysis_block is None:
        # J-2: 원인분석 전 타깃 실패 -- 스냅샷 저장을 생략하고 기존
        # 스냅샷을 보존한다(빈 payload로 덮어쓰지 않는다).
        logger.warning("auto_refresh: 원인분석 전 타깃 실패 -- 스냅샷 저장 생략")
        return
    if alarms_block is None:
        logger.warning("auto_refresh: 알람 판정 실패 -- 스냅샷 저장 생략")
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
        "source": {
            "mode": mode,
            "train_dataset": train_dataset_id,
            "eval_dataset": eval_dataset_id,
            "eval_dataset_filename": eval_dataset_filename,
            "row_count": source_row_count,
        },
        "model": model_meta,
        "analysis": analysis_block,
        "alarms": alarms_block,
        "monitoring": monitoring_block,
        "errors": errors,
    }
    store.save_refresh_snapshot(snapshot)
    logger.info("auto_refresh: 스냅샷 저장 완료 mode=%s eval=%s", mode, eval_dataset_id)

    # -- 6. 신규 알람 자동 발송 (J-5) -- 별도 모듈, 게이트/폴백/발송
    # 조건은 그쪽에서 판단한다. 임포트를 함수 안에 둔 것은 순환 임포트
    # 회피(dispatch.py가 이 모듈을 다시 참조하지 않지만, notify 쪽
    # 모듈들이 얽혀 있어 안전하게 지연 임포트한다) 목적이다.
    from src.automation import refresh_dispatch

    # AG-3: 수동 모드(업로드로 활성화된 평가 데이터셋)에서는 자동 발송을
    # 중단한다 -- 사용자가 임의로 올린 파일의 판정 결과를 폰으로 보내면
    # 안 된다.
    if mode == "manual":
        logger.info("auto_refresh: 수동 모드라 신규 알람 발송을 건너뜁니다.")
        return

    try:
        refresh_dispatch.dispatch_new_alarms(
            store,
            mode=mode,
            train_dataset_id=train_dataset_for_alarms,
            eval_dataset_id=eval_dataset_id,
            alarm_items=alarm_items_for_dispatch,
            gate_passed=alarms_block["gate_passed"],
            snapshot_created_at=now_iso,
        )
    except Exception:
        logger.exception("auto_refresh: 신규 알람 발송 처리 실패")


def _resolve_source(
    store: RuntimeStore, registry: DatasetRegistry, errors: list[str]
) -> tuple[str, str, str, int]:
    """SQL 모드를 먼저 시도하고, 실패하거나 설정이 없으면 폴백(train/test)
    으로 넘어간다. SQL 모드에서 얻은 배치가 학습용(Y 있음)인지 평가용
    (Y 없음)인지는 기존 파일 기반 자동 수집과 동일한 기준
    (`has_target_column`)으로 가른다.

    AG-3: 업로드로 활성화된 "수동 평가 데이터셋"이 있으면 SQL/폴백보다
    먼저 그것을 쓴다 -- 주기 잡이 사용자가 올린 파일을 원래 소스로
    되돌리지 않는다("자동 갱신으로 복귀"를 눌러 override를 지우기
    전까지). 학습 대상(train_dataset)은 건드리지 않는다 -- 직전
    스냅샷의 것을 그대로 이어받는다(자동 학습을 걸지 않는다, AG-2)."""
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
            if dataframe.empty:
                # 신규 행이 없다 -- 학습/평가 데이터셋은 바뀌지 않았으므로
                # 직전 스냅샷의 소스를 그대로 쓴다(첫 실행이면 폴백).
                previous = store.get_refresh_snapshot_status()["snapshot"]
                previous_source = (previous or {}).get("source") if previous else None
                if previous_source and previous_source.get("mode") == "sql":
                    return "sql", previous_source["train_dataset"], previous_source["eval_dataset"], previous_source.get("row_count", 0)
                # 첫 SQL 조회가 하필 0행이면 폴백으로 시작한다.
            else:
                filename = f"sql_refresh_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.csv"
                content = dataframe.to_csv(index=False).encode("utf-8")
                if has_target_column(dataframe):
                    dataset_id = _register_dataset(registry, filename, content, errors)
                    if dataset_id is not None:
                        # 학습용 배치다 -- eval은 이전 스냅샷의 eval을
                        # 유지한다(SQL 모드에서 평가 대상은 별도 조회로
                        # 갱신되며, 이 사이클은 학습 데이터만 새로 받았다).
                        previous = store.get_refresh_snapshot_status()["snapshot"]
                        previous_eval = ((previous or {}).get("source") or {}).get("eval_dataset")
                        return "sql", dataset_id, previous_eval or dataset_id, len(dataframe)
                else:
                    dataset_id = _register_dataset(registry, filename, content, errors)
                    if dataset_id is not None:
                        previous = store.get_refresh_snapshot_status()["snapshot"]
                        previous_train = ((previous or {}).get("source") or {}).get("train_dataset")
                        return "sql", previous_train or FALLBACK_TRAIN_DATASET, dataset_id, len(dataframe)

    # 폴백: 내장 train.CSV로 학습, test.CSV로 평가 (J-1).
    try:
        train_df = registry.get_dataframe(FALLBACK_TRAIN_DATASET)
        row_count = len(train_df)
    except Exception:
        errors.append("폴백 데이터셋(train.CSV)을 읽지 못했습니다.")
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


def _maybe_retrain(
    store: RuntimeStore, registry: DatasetRegistry, mode: str, errors: list[str]
) -> dict[str, Any]:
    """J-1 "재학습 조건" -- 데이터 내용 해시가 이전과 같고 챔피언이 이미
    있으면 학습을 건너뛴다. 승격 게이트(`RuntimeStore.promote_if_better`)
    자체는 이미 구현돼 있어(수동 학습과 공유) 여기서 다시 만들지 않는다
    -- 그 게이트를 통과시키는 학습 Job 제출 경로(`api.routes.data`)를
    그대로 재사용한다.
    """
    train_dataset_id = FALLBACK_TRAIN_DATASET if mode == "fallback" else None
    try:
        if mode == "fallback":
            content = (registry.bundled_root / "train.CSV").read_bytes()
        else:
            snapshot = store.get_refresh_snapshot_status()["snapshot"]
            train_dataset_id = ((snapshot or {}).get("source") or {}).get("train_dataset")
            if not train_dataset_id:
                return {"champion_version": _current_champion_id(store), "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": "학습 대상 데이터셋 없음"}
            content = registry.get_dataframe(train_dataset_id).to_csv(index=False).encode("utf-8")
    except Exception:
        logger.exception("auto_refresh: 학습 대상 데이터 읽기 실패")
        errors.append("학습 대상 데이터를 읽지 못했습니다.")
        return {"champion_version": _current_champion_id(store), "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": "학습 대상 데이터 읽기 실패"}

    content_hash = _content_hash(content)
    stored_hash = (store.get_app_state(DATA_HASH_STATE_KEY) or {}).get("value")
    champion_exists = store.active_model() is not None

    if champion_exists and stored_hash == content_hash:
        return {
            "champion_version": _current_champion_id(store),
            "trained_at": None,
            "promoted": None,
            "gate_reason": None,
            "skipped_reason": "데이터 내용 변경 없음 -- 재학습 생략",
        }

    filename = f"{train_dataset_id or 'train'}_auto_refresh.csv"
    try:
        from api.routes.data import _run_persisted_training_job, get_training_job_manager
        from src.runtime.training_jobs import new_training_job_id

        upload_result = registry.upload(filename, content)
        if not upload_result.get("success"):
            errors.append(f"자동 재학습용 데이터셋 등록 실패: {upload_result.get('blocking_errors')}")
            return {"champion_version": _current_champion_id(store), "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": "데이터셋 등록 실패"}
        dataset_id = upload_result["dataset_id"]

        manager = get_training_job_manager()
        job_id = new_training_job_id()
        from functools import partial

        input_path = manager.allocate_input_path(job_id)
        input_path.write_bytes(content)
        manager.submit(
            job_id=job_id,
            source_filename=filename,
            input_path=input_path,
            runner=partial(_run_persisted_training_job, input_path, filename, {}),
        )
        # 학습 Job은 비동기로 돈다 -- 이번 사이클은 제출까지만 하고,
        # 예측/분석/알람은 (아직 승격 전인) 기존 챔피언으로 계속한다.
        # 해시는 제출 시점에 기록한다: 같은 내용으로 매 사이클 중복
        # 제출하지 않기 위함이지, 승격 성공 여부와는 무관하다.
        store.set_app_state(DATA_HASH_STATE_KEY, {"value": content_hash})
        return {
            "champion_version": _current_champion_id(store),
            "trained_at": None,
            "promoted": None,
            "gate_reason": None,
            "skipped_reason": None,
            "training_job_submitted": job_id,
        }
    except ActiveOperationError:
        logger.info("auto_refresh: 다른 무거운 작업이 실행 중이라 학습을 다음 주기로 미룹니다.")
        return {
            "champion_version": _current_champion_id(store),
            "trained_at": None,
            "promoted": None,
            "gate_reason": None,
            "skipped_reason": "다른 작업이 실행 중이라 이번 주기는 건너뜀",
        }
    except Exception:
        logger.exception("auto_refresh: 자동 재학습 제출 실패")
        errors.append("자동 재학습을 제출하지 못했습니다 -- 기존 챔피언으로 계속합니다.")
        return {
            "champion_version": _current_champion_id(store),
            "trained_at": None,
            "promoted": None,
            "gate_reason": None,
            "skipped_reason": "학습 제출 실패",
        }


def _current_champion_id(store: RuntimeStore) -> str | None:
    active = store.active_model()
    return active.get("active_model_id") if active else None


def _analyze_and_score(
    store: RuntimeStore, eval_dataset_id: str, errors: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """원인분석(Pareto+계측 확대) + 알람 판정 + 모니터링 요약을 계산한다.
    전 타깃 Pareto 실패는 analysis_block=None으로, 알람 계산 실패는
    alarms_block=None으로 알린다 -- 호출부가 스냅샷 저장 여부를 그
    신호로 결정한다(J-2 "부분 실패 정책"). 다섯 번째 반환값은 알람
    판정에 실제로 쓰인 train 데이터셋 id다(J-5의 신뢰도 재계산이 같은
    기준을 써야 한다)."""
    from api.routes.analysis import _pareto_payload, _scored_wafers, get_measurement_expansion
    from src.runtime.app_state import get_latest_state

    registry = _dataset_registry(store)
    try:
        eval_df = registry.get_dataframe(eval_dataset_id)
    except Exception:
        logger.exception("auto_refresh: 평가 데이터셋 로드 실패")
        errors.append(f"평가 데이터셋({eval_dataset_id})을 불러오지 못했습니다.")
        return None, None, None, [], None

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
        return None, None, None, [], None
    if failed_targets:
        errors.append(f"일부 타깃 스크리닝 실패: {', '.join(failed_targets)}")

    measurement_expansion = None
    try:
        measurement_expansion = get_measurement_expansion(eval_dataset_id)
    except Exception:
        logger.exception("auto_refresh: 계측 확대 계산 실패")
        errors.append("계측 확대 계산에 실패했습니다.")

    analysis_block = {
        "paretoByTarget": pareto_by_target,
        "measurementExpansion": measurement_expansion,
    }

    # 알람 판정 -- 저장된 목표 수율·민감도를 그대로 따른다(A-3 원칙과
    # 동일: 여러 화면의 판정 기준이 어긋나면 안 된다).
    latest = get_latest_state(store)
    alarms_payload = ((latest.get("alarms") or {}).get("payload")) or {}
    from src.analysis import alarm_gbdt

    target_yield = alarms_payload.get("targetYield", alarm_gbdt.DEFAULT_TARGET_YIELD)
    sensitivity = alarms_payload.get("sensitivity", alarm_gbdt.DEFAULT_SENSITIVITY)
    train_dataset_for_alarms = (
        (latest.get("alarms") or {}).get("train_dataset") if latest.get("alarms") else None
    ) or FALLBACK_TRAIN_DATASET

    try:
        # 존재 검증만 필요하다 -- _scored_wafers는 더 이상 train_df를
        # 받지 않는다(판정이 점추정 기준으로 바뀌며 sigma 계산이 없어짐,
        # spec §CA-1).
        registry.get_dataframe(train_dataset_for_alarms)
        scored, auc_lo, gate_passed = _scored_wafers(
            train_dataset_for_alarms, eval_dataset_id, eval_df,
            target=target_yield, sensitivity=sensitivity,
        )
    except Exception:
        logger.exception("auto_refresh: 알람 판정 실패")
        errors.append("알람 판정에 실패했습니다.")
        return analysis_block, None, None, [], train_dataset_for_alarms

    counts: dict[str, int] = {"심각": 0, "위험": 0, "주의": 0, "정상": 0, "판별불가": 0}
    for item in scored:
        key = item.grade if item.grade in counts else "판별불가"
        counts[key] += 1
    alarm_items = sorted(
        (item for item in scored if item.grade in ("심각", "위험", "주의")),
        key=lambda item: item.risk_percentile,
    )
    items_top = [
        {
            "lot_wafer_id": item.lot_wafer_id,
            "lot_id": item.lot_id,
            "grade": item.grade,
            "risk_percentile": item.risk_percentile,
        }
        for item in alarm_items[:200]
    ]
    alarms_block = {
        "gate_passed": gate_passed,
        "target_yield": target_yield,
        "sensitivity": sensitivity,
        "counts": counts,
        "items_top": items_top,
        "total": len(alarm_items),
    }
    # spec §BC-2: 계측 없이 등급이 매겨진 wafer는 사유를 댈 수 없으므로
    # 자동 발송(dispatch_new_alarms) 대상에서 제외한다 -- alarms_block/
    # items_top(화면 표시용)은 걸러내지 않는다.
    alarm_items_for_dispatch = [
        {
            "lot_wafer_id": item.lot_wafer_id,
            "lot_id": item.lot_id,
            "grade": item.grade,
            "risk_percentile": item.risk_percentile,
            "reason": "",
        }
        for item in alarm_items
        if item.measured
    ]

    monitoring_block = _build_monitoring_block(scored, target_yield, measurement_expansion, errors)
    return analysis_block, alarms_block, monitoring_block, alarm_items_for_dispatch, train_dataset_for_alarms


def _build_monitoring_block(
    scored: list[Any], target_yield: float, measurement_expansion: dict[str, Any] | None, errors: list[str]
) -> dict[str, Any]:
    measured = [item for item in scored if item.measured]
    if measured:
        point = float(np.mean([item.pred_mean for item in measured]))
        lo = float(np.mean([item.pred_lo for item in measured]))
        hi = float(np.mean([item.pred_hi for item in measured]))
    else:
        point = lo = hi = None
    gap = None
    if point is not None:
        gap = {"lo": round(target_yield - hi, 2), "hi": round(target_yield - lo, 2)}
    return {
        "predicted_yield": {"point": point, "lo": lo, "hi": hi} if point is not None else None,
        "gap": gap,
        "gap_pareto": (measurement_expansion or {}).get("priorities", []),
        # 트리맵은 스텝별 상호작용 조회라 그 자체는 온디맨드로 유지한다
        # (K/J 공통 원칙: "모든 상호작용이 오프라인으로 되는 것이 목표가
        # 아니다") -- 스냅샷에는 담지 않는다.
        "treemap": None,
    }
