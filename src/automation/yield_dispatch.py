"""SD그룹("알림·자동화 설정"): "자동화 사용"이 켜져 있는 동안
refreshIntervalMinutes마다 실행되는 주기 잡 -- SQL에서 최신 배치를
받아 활성 모델(모델 학습에 저장된 것)로 **수율 예측만** 계산해 알림만
보낸다.

이 잡은 모니터링 홈·Config별 트리맵·원인 분석을 계산하지 않고,
`src/automation/refresh.py`의 스냅샷도 절대 건드리지 않는다 -- 화면
갱신은 오직 사용자가 "모델 분석" 팝업에서 [분석 시작]을 눌렀을 때만
일어난다("하지 말 것": 자동화가 화면 스냅샷을 갱신하게 하지 마라).

학습도 트리거하지 않는다 -- 모델이 없으면 건너뛰고 사유를 남긴다(SD-3).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from api.settings import settings
from src.automation import sql_source
from src.column_detection import detect_feature_columns
from src.notifications.yield_update_dispatch import TRIGGER_REFRESH, dispatch_yield_update
from src.notifications.yield_update_senders import build_yield_update_payload
from src.runtime.datasets import DatasetRegistry
from src.runtime.store import RuntimeStore
from src.schema_loader import load_data_schema

logger = logging.getLogger(__name__)

AUTOMATION_YIELD_DISPATCH_JOB_ID = "automation_yield_dispatch"
_KST = ZoneInfo("Asia/Seoul")


def _runtime_store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


def _dataset_registry(store: RuntimeStore) -> DatasetRegistry:
    return DatasetRegistry(store, settings.dataset_upload_dir, settings.bundled_dataset_dir)


def _looks_like_eval_batch(columns: list[str]) -> bool:
    """SD-2: "Step 인자 컬럼 존재 + y·y1~y5 일부 또는 전부 결측"인 배치인지
    확인한다. 컬럼 자체의 존재 여부만 본다(셀 값 결측은 호출부가
    데이터프레임으로 이미 갖고 있어 별도로 확인하지 않는다 -- 목표
    컬럼이 아예 없는 배치도 "결측"의 일종으로 취급한다). 목적은 "이게
    학습용 완전 라벨 배치가 아니라 분석용 배치처럼 보이는가"를 가리는
    최소 판단이다 -- 업로드 검증(SB-3)만큼 엄격하지 않다."""
    schema_config = load_data_schema()
    detected = detect_feature_columns(columns, schema_config)
    return bool(detected["r_columns"] or detected["d_columns"] or detected["config_columns"])


def run_automation_yield_dispatch_job() -> None:
    store = _runtime_store()
    automation = store.get_automation_settings()
    if not automation.get("enabled"):
        return
    if not sql_source.is_sql_configured(store):
        logger.info("automation_yield_dispatch: SQL 설정 없음 -- 건너뜁니다.")
        return

    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        dataframe = sql_source.fetch_incremental(store)
    except Exception:
        logger.exception("automation_yield_dispatch: SQL 조회 실패")
        store.save_automation_settings(lastRunAt=now_iso, lastRunStatus="error", lastRunSentCount=0)
        return
    if dataframe is None or dataframe.empty:
        logger.info("automation_yield_dispatch: 새 데이터 없음 -- 건너뜁니다.")
        return

    if not _looks_like_eval_batch(list(dataframe.columns)):
        store.record_notify_history(
            trigger=TRIGGER_REFRESH,
            channels=[],
            dataset_label=None,
            model_version=None,
            status="skipped",
            skip_reason="분석 데이터 모양이 아님 (Step 인자 컬럼 없음)",
        )
        store.save_automation_settings(lastRunAt=now_iso, lastRunStatus="skipped", lastRunSentCount=0)
        logger.info("automation_yield_dispatch: 평가 데이터 모양이 아님 -- 건너뜁니다.")
        return

    active_model = store.active_model()
    if not active_model:
        store.record_notify_history(
            trigger=TRIGGER_REFRESH,
            channels=[],
            dataset_label=None,
            model_version=None,
            status="skipped",
            skip_reason="학습된 모델 없음",
        )
        store.save_automation_settings(lastRunAt=now_iso, lastRunStatus="skipped", lastRunSentCount=0)
        logger.info("automation_yield_dispatch: 학습된 모델 없음 -- 건너뜁니다.")
        return

    registry = _dataset_registry(store)
    filename = f"automation_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.csv"
    content = dataframe.to_csv(index=False).encode("utf-8")
    try:
        upload_result = registry.upload(filename, content)
    except Exception:
        logger.exception("automation_yield_dispatch: 배치 등록 실패")
        store.save_automation_settings(lastRunAt=now_iso, lastRunStatus="error", lastRunSentCount=0)
        return
    if not upload_result.get("success"):
        logger.warning("automation_yield_dispatch: 배치 등록 거부 -- %s", upload_result.get("blocking_errors"))
        store.save_automation_settings(lastRunAt=now_iso, lastRunStatus="error", lastRunSentCount=0)
        return
    eval_dataset_id = upload_result["dataset_id"]

    # SD-3: 모델 학습에 저장된 모델을 그대로 쓴다 -- 자동화는 재학습하지
    # 않는다. train 데이터셋은 가장 최근 "모델 분석" 스냅샷이 쓴 것을
    # 계승한다(자동화가 학습 대상을 바꾸지 않는다).
    snapshot = store.get_refresh_snapshot_status()["snapshot"]
    train_dataset_id = ((snapshot or {}).get("source") or {}).get("train_dataset") or "train"

    try:
        from api.routes.analysis import _hydrated_targets_or_409
        from src.analysis.yield_prediction import build_yield_prediction_table

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
        timestamp_label = datetime.now(_KST).strftime("%H:%M")
        payload = build_yield_update_payload(
            table,
            dataset_label=filename,
            timestamp_label=timestamp_label,
            model_label=active_model.get("active_model_id"),
        )
        result = dispatch_yield_update(store, payload, trigger=TRIGGER_REFRESH)
    except Exception:
        logger.exception("automation_yield_dispatch: 수율 예측 계산/발송 실패")
        store.save_automation_settings(lastRunAt=now_iso, lastRunStatus="error", lastRunSentCount=0)
        return

    sent_count = 0
    if not result.get("skipped"):
        sent_count = sum(1 for item in (result.get("results") or {}).values() if item.get("ok"))
    store.save_automation_settings(
        lastRunAt=now_iso,
        lastRunStatus="skipped" if result.get("skipped") else "sent",
        lastRunSentCount=sent_count,
    )
    logger.info(
        "automation_yield_dispatch: 완료 eval=%s skipped=%s sent_count=%d",
        eval_dataset_id, result.get("skipped"), sent_count,
    )
