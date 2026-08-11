"""VE-1: 수율 예측이 갱신될 때마다(자동 갱신, 사용자 파일 업로드, 또는
"분석 실행 직후" 설정이 켜진 수동 분석 실행 -- 즉 분석이 실제로 새로
수행될 때마다) 연결된 Telegram·Gmail·Slack으로 발송한다.

억제 규칙은 시간당 예산과 (수동 트리거만) 최소 간격, 두 가지만 유지한다.
24시간/신규분 dedupe는 없다 -- 분석이 일어날 때마다 보낸다. 같은 wafer가
직전 발송에도 나왔든 상관없다(이 시스템은 "새 이벤트가 있었는가"가 아니라
"최신 분석 결과가 무엇인가"를 알리는 것이 목적이므로, 알람 발송의
(dataset, wafer, grade, channel) 24시간 dedupe와 성격이 다르다).

발송은 best-effort다 -- 실패해도 예외를 올리지 않는다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.notifications import senders, settings_store
from src.notifications.yield_update_senders import (
    YieldUpdatePayload,
    build_slack_yield_update,
    build_telegram_yield_update_chunks,
    build_yield_update_email_html,
)
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

HOURLY_SEND_BUDGET = 6
MANUAL_MIN_INTERVAL_MINUTES = 10

_HOURLY_STATE_KEY = "yield_update:hourly_sent_at"
_MANUAL_INTERVAL_STATE_KEY = "yield_update:manual_last_sent_at"

TRIGGER_REFRESH = "refresh"  # 자동 갱신 -- 항상 발송 후보(VE-1)
TRIGGER_MANUAL = "manual_analysis"  # 수동 분석 실행 -- timing에 on_analysis 포함 시만
# YD: 수율 예측 화면의 "알림 전송" 버튼 -- 사용자가 직접 눌러 지금
# 보내라는 명시적 의도이므로 TRIGGER_MANUAL과 달리 "분석 실행 직후"
# 타이밍 설정 여부와 무관하게 시도한다. 그래도 최소 간격(10분)·시간당
# 예산은 TRIGGER_MANUAL과 동일하게 적용한다("하지 말 것": 억제 규칙을
# 우회하는 옵션을 만들지 마라).
TRIGGER_MANUAL_BUTTON = "manual_button"
# DF그룹: 알림 설정 화면의 "매일 오전 9시"/"매일 오후 1시" 발송 시점 옵션
# -- 저장된 발송 시점 설정(`conditions["timing"]`)에 각각의 대응 값
# (settings_store.TIMING_DAILY_9AM/TIMING_DAILY_13)이 포함돼 있을 때만
# 실제로 보낸다. APScheduler 잡 등록은 api/main.py, 잡 본문은
# api/routes/notify.py의 run_daily_dispatch_job/run_daily_13_dispatch_job.
TRIGGER_DAILY_9AM = "daily_9am"
TRIGGER_DAILY_13 = "daily_13"

# trigger -> 발송을 허용하려면 conditions["timing"]에 있어야 하는 값.
# TRIGGER_REFRESH/TRIGGER_MANUAL_BUTTON은 timing 설정과 무관하게 항상
# 후보다(각각 "자동 갱신마다", "사용자가 지금 보내라고 명시적으로 누름").
_TRIGGER_REQUIRED_TIMING = {
    TRIGGER_MANUAL: settings_store.TIMING_ON_ANALYSIS,
    TRIGGER_DAILY_9AM: settings_store.TIMING_DAILY_9AM,
    TRIGGER_DAILY_13: settings_store.TIMING_DAILY_13,
}
_TIMING_LABEL = {
    settings_store.TIMING_ON_ANALYSIS: "분석 실행 직후",
    settings_store.TIMING_DAILY_9AM: "매일 오전 9시",
    settings_store.TIMING_DAILY_13: "매일 오후 1시",
}


def _within_hourly_budget(store: RuntimeStore) -> bool:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=1)
    record = store.get_app_state(_HOURLY_STATE_KEY) or {}
    recent = [t for t in record.get("sent_at", []) if _parse_iso(t) and _parse_iso(t) >= since]
    if len(recent) >= HOURLY_SEND_BUDGET:
        store.set_app_state(_HOURLY_STATE_KEY, {"sent_at": recent})
        return False
    recent.append(now.isoformat())
    store.set_app_state(_HOURLY_STATE_KEY, {"sent_at": recent})
    return True


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _manual_interval_blocked_until(store: RuntimeStore) -> str | None:
    record = store.get_app_state(_MANUAL_INTERVAL_STATE_KEY) or {}
    last_sent_at = _parse_iso(record.get("last_sent_at", ""))
    if last_sent_at is None:
        return None
    next_allowed_at = last_sent_at + timedelta(minutes=MANUAL_MIN_INTERVAL_MINUTES)
    if datetime.now(timezone.utc) < next_allowed_at:
        return next_allowed_at.strftime("%H:%M")
    return None


def _mark_manual_dispatch_sent(store: RuntimeStore) -> None:
    store.set_app_state(_MANUAL_INTERVAL_STATE_KEY, {"last_sent_at": datetime.now(timezone.utc).isoformat()})


def dispatch_yield_update(store: RuntimeStore, payload: YieldUpdatePayload, *, trigger: str) -> dict[str, Any]:
    try:
        required_timing = _TRIGGER_REQUIRED_TIMING.get(trigger)
        if required_timing is not None:
            conditions = settings_store.get_conditions(store)
            if required_timing not in (conditions.get("timing") or []):
                return {"skipped": True, "reason": f"발송 시점 설정에 '{_TIMING_LABEL[required_timing]}'가 없음"}
        if trigger in (TRIGGER_MANUAL, TRIGGER_MANUAL_BUTTON):
            blocked_until = _manual_interval_blocked_until(store)
            if blocked_until is not None:
                return {
                    "skipped": True,
                    "reason": f"직전 발송 후 {MANUAL_MIN_INTERVAL_MINUTES}분이 지나지 않아 발송하지 않았습니다 (다음 발송 가능 {blocked_until})",
                }

        from api.settings import settings as api_settings

        slack = settings_store.get_slack(store)
        telegram = settings_store.get_telegram(store)
        gmail_record = settings_store.get_gmail(store)
        gmail_ready = bool(
            gmail_record
            and gmail_record.get("verified")
            and api_settings.smtp_host
            and api_settings.smtp_user
            and api_settings.smtp_password
            and api_settings.smtp_from_email
        )
        telegram_ready = bool(telegram and api_settings.telegram_bot_token)
        if not (slack or telegram_ready or gmail_ready):
            return {"skipped": True, "reason": "연결된 채널 없음"}

        if not _within_hourly_budget(store):
            return {"skipped": True, "reason": f"시간당 발송 예산({HOURLY_SEND_BUDGET}건) 초과"}

        results: dict[str, Any] = {}
        if slack:
            ok, error = senders.send_slack_webhook(slack["webhook_url"], build_slack_yield_update(payload))
            results["slack"] = {"ok": ok, "error": error}

        if telegram_ready:
            # Telegram 메시지 하드 한계(4096자)를 넘을 수 있어 블록 단위로
            # 나눠 여러 번 보낸다(블록 중간을 자르거나 타깃 섹션을 누락하지
            # 않는다 -- build_telegram_yield_update_chunks 참고). 청크
            # 하나라도 실패하면 전체를 실패로 보고한다(부분 실패의 채널
            # 결과를 ok=True로 위장하지 않는다).
            chunks = build_telegram_yield_update_chunks(payload)
            chunk_ok = True
            chunk_error: str | None = None
            for chunk in chunks:
                ok, error = senders.send_telegram_message(api_settings.telegram_bot_token, telegram["chat_id"], chunk, parse_mode=None)
                if not ok:
                    chunk_ok = False
                    chunk_error = error
                    break
            results["telegram"] = {"ok": chunk_ok, "error": chunk_error}

        if gmail_ready:
            ok, error = senders.send_gmail(
                smtp_host=api_settings.smtp_host,
                smtp_port=api_settings.smtp_port,
                smtp_user=api_settings.smtp_user,
                smtp_password=api_settings.smtp_password,
                from_email=api_settings.smtp_from_email,
                to_email=gmail_record["email"],
                subject=f"[SUNI] 수율 예측 갱신 ({payload.timestamp_label})",
                html_body=build_yield_update_email_html(payload),
            )
            results["gmail"] = {"ok": ok, "error": error}

        if trigger in (TRIGGER_MANUAL, TRIGGER_MANUAL_BUTTON):
            _mark_manual_dispatch_sent(store)

        return {"skipped": False, "results": results}
    except Exception:
        logger.exception("수율 예측 갱신 발송 중 오류")
        return {"skipped": True, "reason": "발송 중 오류 발생"}
