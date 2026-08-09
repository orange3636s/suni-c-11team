"""J-5: 갱신 시 신규 알람 자동 발송.

`src/notifications/dispatch.py`의 `dispatch_alarm_notifications`가 이미
등급 필터·발송 시점 필터·채널별 24시간 dedupe(등급 악화 시 예외)를
전부 처리하므로 여기서 다시 구현하지 않는다 -- 이 모듈은 그 위에
자동 갱신 파이프라인 고유의 세 가지만 더한다:

1. 차단 조건(게이트 미달/폴백 모드)을 `dispatch_alarm_notifications`
   호출 자체를 건너뛰는 방식으로 추가한다(그 함수는 이 두 신호를
   모른다).
2. 이전 스냅샷 대비 "신규" 알람만 후보로 추린다(§C-7 24시간 dedupe와는
   별개 -- 스냅샷은 주기가 24시간보다 훨씬 촘촘할 수 있어 그 dedupe만
   믿으면 매 사이클 재전송된다).
3. 시간당 발송 예산과, 대량 발생 시 요약 처리를 기록한다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.notifications import dispatch, senders, settings_store
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

# 이 개수를 넘는 신규 알람은 "대량 발생"으로 기록한다 -- 실제 메시지는
# senders.py가 이미 상위 5건 + "외 N건"으로 요약하므로(모든 발송 경로가
# 공유하는 형식) 별도 포맷을 새로 만들지 않는다. 여기서는 이 사이클이
# 대량이었다는 사실 자체를 기록해 알림 기록 화면에서 구분할 수 있게 한다.
NEW_ALARM_SUMMARY_THRESHOLD = 10
HOURLY_SEND_BUDGET = 6
_BUDGET_META_ALERT_STATE_KEY = "automation:hourly_budget_meta_alert_hour"


def _previous_snapshot_alarm_ranks(store: RuntimeStore) -> dict[tuple[str, str], int]:
    """(lot_wafer_id, grade) -> 등급 순위. 직전 스냅샷의 `items_top`
    (상위 200건)만 본다 -- 전체를 담지 않는 J-3 저장 정책과 같은 이유다.
    200건 밖으로 밀려났던 항목이 이번에 다시 잡히면 "신규"로 오인될 수
    있다는 한계가 있다(문서화된 트레이드오프)."""
    status = store.get_refresh_snapshot_status()
    snapshot = status["snapshot"]
    if not snapshot:
        return {}
    items = ((snapshot.get("alarms") or {}).get("items_top")) or []
    rank = {"주의": 0, "위험": 1, "심각": 2}
    return {(item["lot_wafer_id"], item["grade"]): rank.get(item["grade"], -1) for item in items}


def _is_new_or_escalated(item: dict[str, Any], previous_ranks: dict[tuple[str, str], int]) -> bool:
    rank = {"주의": 0, "위험": 1, "심각": 2}
    key = (item["lot_wafer_id"], item["grade"])
    if key in previous_ranks:
        return False
    # 같은 wafer가 다른(더 나쁜) 등급으로도 없었는지 확인한다.
    current_rank = rank.get(item["grade"], -1)
    best_previous = max(
        (r for (wafer, _grade), r in previous_ranks.items() if wafer == item["lot_wafer_id"]),
        default=-1,
    )
    return current_rank > best_previous


def _record(store: RuntimeStore, *, new_alarm_count: int, blocked_reason: str | None, summarized: bool, channels: dict[str, Any]) -> None:
    try:
        store.record_refresh_dispatch(
            new_alarm_count=new_alarm_count, blocked_reason=blocked_reason, summarized=summarized, channels=channels
        )
    except Exception:
        logger.exception("auto_refresh: 발송 기록 저장 실패")


def _send_budget_meta_alert(store: RuntimeStore) -> None:
    """시간당 예산 초과는 조용히 넘어가지 않는다 -- "알람 시스템의 고장은
    침묵이 아니라 소음으로 온다"(J-5). 같은 시간(UTC 시간 버킷) 안에는
    한 번만 보낸다."""
    current_hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    marker = store.get_app_state(_BUDGET_META_ALERT_STATE_KEY) or {}
    if marker.get("hour") == current_hour:
        return
    store.set_app_state(_BUDGET_META_ALERT_STATE_KEY, {"hour": current_hour})

    message = f"[SUNI] 알람 자동 발송이 시간당 예산({HOURLY_SEND_BUDGET}건)을 초과해 이번 시간 동안 멈췄습니다. 알림 설정을 확인해 주세요."
    slack = settings_store.get_slack(store)
    if slack:
        try:
            senders.send_slack_webhook(slack["webhook_url"], {"text": message})
        except Exception:
            logger.exception("auto_refresh: 예산 초과 메타 알림 Slack 발송 실패")
    telegram = settings_store.get_telegram(store)
    from api.settings import settings as api_settings

    if telegram and api_settings.telegram_bot_token:
        try:
            senders.send_telegram_message(api_settings.telegram_bot_token, telegram["chat_id"], message)
        except Exception:
            logger.exception("auto_refresh: 예산 초과 메타 알림 Telegram 발송 실패")
    gmail = settings_store.get_gmail(store)
    if gmail and gmail.get("verified") and api_settings.smtp_host and api_settings.smtp_user and api_settings.smtp_password and api_settings.smtp_from_email:
        try:
            senders.send_gmail(
                smtp_host=api_settings.smtp_host,
                smtp_port=api_settings.smtp_port,
                smtp_user=api_settings.smtp_user,
                smtp_password=api_settings.smtp_password,
                from_email=api_settings.smtp_from_email,
                to_email=gmail["email"],
                subject="[SUNI] 알람 발송 시간당 예산 초과",
                html_body=f"<p>{message}</p>",
            )
        except Exception:
            logger.exception("auto_refresh: 예산 초과 메타 알림 Gmail 발송 실패")


def dispatch_new_alarms(
    store: RuntimeStore,
    *,
    mode: str,
    train_dataset_id: str,
    eval_dataset_id: str,
    alarm_items: list[dict[str, Any]],
    gate_passed: bool,
    snapshot_created_at: str,
) -> None:
    """호출부(`src/automation/refresh.py`)는 스냅샷 저장이 성공했을
    때만 이 함수를 부른다 -- "스냅샷 저장이 생략된 경우" 차단 조건은
    자연히 만족된다(애초에 호출되지 않으므로)."""
    if not gate_passed:
        _record(store, new_alarm_count=0, blocked_reason="게이트 미달", summarized=False, channels={})
        return
    if mode == "fallback":
        _record(store, new_alarm_count=0, blocked_reason="폴백 모드(SQL 미연결) -- 자동 발송 제외", summarized=False, channels={})
        return

    conditions = settings_store.get_conditions(store)
    if settings_store.TIMING_ON_ANALYSIS not in (conditions.get("timing") or []):
        _record(store, new_alarm_count=0, blocked_reason="발송 시점 설정에 on_analysis 없음", summarized=False, channels={})
        return

    grades = set(conditions.get("grades") or [])
    candidates = [item for item in alarm_items if item.get("grade") in grades]

    previous_ranks = _previous_snapshot_alarm_ranks(store)
    new_items = [item for item in candidates if _is_new_or_escalated(item, previous_ranks)]
    if not new_items:
        _record(store, new_alarm_count=0, blocked_reason=None, summarized=False, channels={})
        return

    since_hour = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    if store.notifications_sent_since(since_hour) >= HOURLY_SEND_BUDGET:
        _send_budget_meta_alert(store)
        _record(
            store,
            new_alarm_count=len(new_items),
            blocked_reason=f"시간당 발송 예산({HOURLY_SEND_BUDGET}건) 초과",
            summarized=False,
            channels={},
        )
        return

    summarized = len(new_items) > NEW_ALARM_SUMMARY_THRESHOLD

    from api.routes.analysis import _cached_reliability

    reliability = _cached_reliability(train_dataset_id, eval_dataset_id)

    result = dispatch.dispatch_alarm_notifications(
        store,
        trigger=settings_store.TIMING_ON_ANALYSIS,
        dataset_id=eval_dataset_id,
        dataset_label=eval_dataset_id,
        alarms=new_items,
        reliability_grade=reliability["grade"],
        reliability_score=reliability["total_score"],
    )
    blocked_reason = result.get("reason") if result.get("skipped") else None
    _record(
        store,
        new_alarm_count=len(new_items),
        blocked_reason=blocked_reason,
        summarized=summarized,
        channels=result.get("results", {}),
    )
