"""자동 수집 파이프라인 1단계 (작업 지시서 "자동 수집 파이프라인") --
감시 디렉터리(`settings.auto_ingest_dir`)를 폴링해 새 CSV를 처리한다.

RB-3: "자동화·모델학습·모델분석 3분리" 지시서 이후로는 Y 컬럼 유무와
무관하게 모든 파일이 평가(분석) 데이터셋으로 등록된다 -- 자동 수집이
학습을 트리거하는 경로 자체가 없다(refresh time은 분석만 갱신한다,
학습은 모델 학습 팝업의 수동 업로드로만 일어난다).

    평가 데이터셋 등록 -> 최신 챔피언 모델로 예측(있으면) -> 원인분석
    Pareto·계측 확대 재계산 -> state/analysis, state/alarms 스냅샷
    갱신(created_at 포함)

`run_auto_ingest_job`은 APScheduler가 주기적으로 호출한다
(`api/main.py`) -- 알림 발송 잡과 같은 best-effort 원칙으로, 파일 하나가
실패해도 스케줄러 루프 자체는 절대 죽지 않는다.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.settings import settings
from src.analysis import alarm_gbdt
from src.runtime.app_state import get_latest_state, save_state
from src.runtime.datasets import DatasetRegistry
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

AUTO_INGEST_JOB_ID = "auto_ingest"
PROCESSED_SUBDIR = "processed"
FAILED_SUBDIR = "failed"
# api/main.py가 시작 시 저장된 refreshIntervalMinutes로 덮어쓴다 -- 이
# 값은 그게 아직 없을 때(최초 기동)만 쓰는 폴백이다.
DEFAULT_INGEST_MINUTES = 60
TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")


def _runtime_store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


def _dataset_registry(store: RuntimeStore) -> DatasetRegistry:
    return DatasetRegistry(store, settings.dataset_upload_dir, settings.bundled_dataset_dir)


def new_csv_files(directory: str | Path) -> list[Path]:
    """감시 디렉터리 바로 아래의 `*.csv`만 대상으로 한다 -- `processed/`,
    `failed/` 하위 디렉터리는 glob("*.csv")가 애초에 내려가지 않는다."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*.csv") if path.is_file())


def move_to(path: Path, subdir: str) -> Path:
    """처리(성공/실패)한 파일을 물리적으로 옮긴다 -- 파일명 기록만으로
    "처리됨"을 판단하면 재시작 후 그 기록이 사라져 같은 파일을 다시
    학습하게 된다(지시서: "물리적으로 옮길 것"). 목적지에 동명 파일이
    있으면(같은 이름으로 여러 번 넣은 경우) 타임스탬프를 붙여 덮어쓰지
    않는다."""
    destination_dir = path.parent / subdir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / path.name
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        destination = destination_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.move(str(path), str(destination))
    return destination


def run_auto_ingest_job() -> None:
    if not settings.auto_ingest_enabled or not settings.auto_ingest_dir:
        return
    for path in new_csv_files(settings.auto_ingest_dir):
        try:
            _process_incoming_csv(path)
        except Exception:
            logger.exception("auto_ingest_failed file=%s", path.name)
            move_to(path, FAILED_SUBDIR)
        else:
            move_to(path, PROCESSED_SUBDIR)


def _process_incoming_csv(path: Path) -> None:
    content = path.read_bytes()
    filename = path.name
    store = _runtime_store()
    registry = _dataset_registry(store)
    # RB-3: Y 유무와 무관하게 항상 평가(분석) 데이터셋으로 등록한다 --
    # 자동 수집이 학습을 트리거하는 경로가 없다.
    _ingest_eval_csv(store, registry, filename, content)


def _ingest_eval_csv(store: RuntimeStore, registry: DatasetRegistry, filename: str, content: bytes) -> None:
    upload_result = registry.upload(filename, content)
    if not upload_result.get("success"):
        raise ValueError(f"데이터셋 등록 실패: {upload_result.get('blocking_errors')}")
    dataset_id = upload_result["dataset_id"]

    _predict_with_champion(store, registry, dataset_id)
    _refresh_analysis_snapshot(store, dataset_id)
    logger.info("auto_ingest: 평가 데이터셋 등록 및 분석 갱신 완료 dataset=%s file=%s", dataset_id, filename)


def _predict_with_champion(store: RuntimeStore, registry: DatasetRegistry, dataset_id: str) -> None:
    """챔피언 모델이 없으면(최초 실행, 학습 이력 없음) 조용히 건너뛴다 --
    에러로 처리하지 않는다."""
    from src.ml.inference import InferenceInputError, load_latest_model_bundle, predict_dataframe

    try:
        loaded = load_latest_model_bundle(store, settings.model_dir)
    except InferenceInputError:
        logger.info("auto_ingest: 챔피언 모델 없음 -- 예측 건너뜀 dataset=%s", dataset_id)
        return
    try:
        dataframe = registry.get_dataframe(dataset_id)
        prediction = predict_dataframe(dataframe, loaded)
        logger.info(
            "auto_ingest: 챔피언 모델 예측 완료 dataset=%s n=%d",
            dataset_id, len(prediction.lot_wafer_id),
        )
    except Exception:
        logger.exception("auto_ingest: 챔피언 모델 예측 실패 dataset=%s", dataset_id)


def _refresh_analysis_snapshot(store: RuntimeStore, dataset_id: str) -> None:
    """원인분석·알림 이력 상태 스냅샷을 새 데이터셋 기준으로 갱신한다.
    `save_state`가 매번 `created_at`을 새로 찍으므로, 모니터링 홈의
    캐시 무효화(=created_at 비교)가 자동으로 동작한다."""
    from api.routes.analysis import _action_priority_payload, _pareto_payload
    from src.analysis.screening.selector import PARETO_TOP_N

    pareto_by_target: dict[str, Any] = {}
    for target in TARGETS:
        try:
            pareto_by_target[target] = _pareto_payload(dataset_id, target, PARETO_TOP_N)
        except Exception:
            logger.exception("auto_ingest: Pareto 계산 실패 dataset=%s target=%s", dataset_id, target)

    if not pareto_by_target:
        # B-6: 전 타깃이 실패하면(예: 스키마가 맞지 않는 파일) 빈 payload로
        # 저장하지 않는다 -- 안 그러면 기존에 정상이던 스냅샷(모니터링
        # 홈·원인 분석 탭이 읽는)을 이 깨진 결과가 그대로 덮어써 버린다.
        # analysis와 alarms 스냅샷은 서로의 기준(dataset)을 가리키므로
        # 둘 다 저장을 생략해 일관성을 유지한다.
        logger.warning("auto_ingest: 전 타깃 Pareto 실패 -- 스냅샷 갱신 생략 dataset=%s", dataset_id)
        return

    latest = get_latest_state(store)
    previous_analysis = latest.get("analysis") or {}
    previous_analysis_payload = previous_analysis.get("payload") or {}
    previous_alarms = latest.get("alarms") or {}
    train_dataset_id = previous_alarms.get("train_dataset") or "train"

    # MB/MC: 모니터링 홈 블록①·② -- train.CSV 기준(작업 지시서 MB-6)이라
    # 이 ingest 경로(SQL 자동 수집)에서도 함께 갱신한다.
    action_priority = None
    try:
        action_priority = _action_priority_payload(train_dataset_id)
    except Exception:
        logger.exception("auto_ingest: 조치 우선순위 계산 실패 dataset=%s", dataset_id)

    save_state(
        store,
        "analysis",
        dataset={"dataset": dataset_id},
        payload={
            "activeTarget": previous_analysis_payload.get("activeTarget", "Y1"),
            "paretoByTarget": pareto_by_target,
            "actionPriority": action_priority,
        },
    )

    previous_alarms_payload = previous_alarms.get("payload") or {}
    save_state(
        store,
        "alarms",
        dataset={
            # train_dataset(정상범위 산출 기준)은 건드리지 않는다 -- 이
            # 파일은 eval(판정 대상)로만 등록됐다.
            "train_dataset": previous_alarms.get("train_dataset") or "train",
            "eval_dataset": dataset_id,
        },
        payload={
            "targetYield": previous_alarms_payload.get("targetYield", alarm_gbdt.DEFAULT_TARGET_YIELD),
            "sensitivity": previous_alarms_payload.get("sensitivity", alarm_gbdt.DEFAULT_SENSITIVITY),
        },
    )
