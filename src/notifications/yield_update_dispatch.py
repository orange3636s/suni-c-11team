"""VE-1: 수율 예측이 갱신될 때마다(자동 갱신, 또는 "분석 실행 직후"
설정이 켜진 수동 분석 실행) 연결된 Telegram·Gmail·Slack으로 발송한다.

억제 규칙은 알람 발송(`src/notifications/dispatch.py`,
`src/automation/refresh_dispatch.py`)과 같은 세 가지를 유지한다 --
신규분만, 시간당 예산, (수동 트리거만) 최소 간격. 다만 알람은 (dataset,
wafer, grade, channel) 단위로 dedupe하는데(§C-7), 수율 예측 갱신은
"등급"이 없는 표+요약 메시지라 그 스키마에 맞지 않는다 -- 대신 이번에
보낼 내용 전체의 지문(fingerprint)을 직전 발송과 비교해 "신규분만"을
판정한다. 그래서 별도의 `app_state` 키를 쓰고 `notify_sent_log`는
건드리지 않는다(알람의 24시간 dedupe 로그에 다른 스키마의 이력을
섞지 않는다).

발송은 best-effort다 -- 실패해도 예외를 올리지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.notifications import senders, settings_store
from src.notifications.yield_update_senders import (
    YieldUpdatePayload,
    build_slack_yield_update,
    build_telegram_yield_update_text,
    build_yield_update_email_html,
)
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

HOURLY_SEND_BUDGET = 6
MANUAL_MIN_INTERVAL_MINUTES = 10

_FINGERPRINT_STATE_KEY = "yield_update:last_sent_fingerprint"
_HOURLY_STATE_KEY = "yield_update:hourly_sent_at"
_MANUAL_INTERVAL_STATE_KEY = "yield_update:manual_last_sent_at"

TRIGGER_REFRESH = "refresh"  # 자동 갱신 -- 항상 발송 후보(VE-1)
TRIGGER_MANUAL = "manual_analysis"  # 수동 분석 실행 -- timing에 on_analysis 포함 시만


def _fingerprint(payload: YieldUpdatePayload) -> str:
    """이번에 보낼 내용의 지문 -- 직전 발송과 같으면 "신규분 없음"으로
    스킵한다. 반올림된(표시) 값 기준으로 비교한다 -- 부동소수 노이즈로
    매번 "신규"로 오판하지 않도록."""
    source = json.dumps(
        {
            "top10": [(item.lot_wafer_id, round(item.y, 2), item.reliability_count) for item in payload.top10],
            "targets": [
                (
                    block.target,
                    block.unavailable_reason,
                    [(item.lot_wafer_id, round(item.value, 2)) for item in block.items],
                )
                for block in payload.target_blocks
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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
        if trigger == TRIGGER_MANUAL:
            conditions = settings_store.get_conditions(store)
            if settings_store.TIMING_ON_ANALYSIS not in (conditions.get("timing") or []):
                return {"skipped": True, "reason": "발송 시점 설정에 '분석 실행 직후'가 없음"}
            blocked_until = _manual_interval_blocked_until(store)
            if blocked_until is not None:
                return {
                    "skipped": True,
                    "reason": f"직전 발송 후 {MANUAL_MIN_INTERVAL_MINUTES}분이 지나지 않아 발송하지 않았습니다 (다음 발송 가능 {blocked_until})",
                }

        fingerprint = _fingerprint(payload)
        last = store.get_app_state(_FINGERPRINT_STATE_KEY) or {}
        if last.get("fingerprint") == fingerprint:
            return {"skipped": True, "reason": "신규분 없음 (직전 발송과 동일한 내용)"}

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
            text = build_telegram_yield_update_text(payload)
            ok, error = senders.send_telegram_message(api_settings.telegram_bot_token, telegram["chat_id"], text, parse_mode=None)
            results["telegram"] = {"ok": ok, "error": error}

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

        store.set_app_state(_FINGERPRINT_STATE_KEY, {"fingerprint": fingerprint, "sent_at": datetime.now(timezone.utc).isoformat()})
        if trigger == TRIGGER_MANUAL:
            _mark_manual_dispatch_sent(store)

        return {"skipped": False, "results": results}
    except Exception:
        logger.exception("수율 예측 갱신 발송 중 오류")
        return {"skipped": True, "reason": "발송 중 오류 발생"}
