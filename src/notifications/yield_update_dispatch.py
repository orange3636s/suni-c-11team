"""수율 예측이 갱신될 때마다(Refresh Time 주기 자동화 또는 사용자가
직접 누른 수동 발송 -- 즉 분석이 실제로 새로 수행될 때마다) 연결된
Telegram·Gmail·Slack으로 발송한다.

어떤 트리거도 저장된 시각/조건 설정으로 막히지 않는다. 자동 발송 경로는
Refresh Time 주기 잡(src/automation/yield_dispatch.py) 하나뿐이며 매
주기마다 무조건 시도한다.

억제 규칙은 시간당 예산과 (수동 트리거만) 최소 간격, 두 가지뿐이다.
24시간/신규분 dedupe는 하지 않는다 -- 분석이 일어날 때마다 보내며, 같은
wafer가 직전 발송에도 나왔든 상관없다. 이 발송의 목적은 "새 이벤트가
있었는가"가 아니라 "최신 분석 결과가 무엇인가"를 알리는 것이라, 알람
발송의 (dataset, wafer, grade, channel) 24시간 dedupe와 성격이 다르다.

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
    build_telegram_yield_update_text,
    build_yield_update_email_html,
)
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

HOURLY_SEND_BUDGET = 6
MANUAL_MIN_INTERVAL_MINUTES = 10

_HOURLY_STATE_KEY = "yield_update:hourly_sent_at"
_MANUAL_INTERVAL_STATE_KEY = "yield_update:manual_last_sent_at"

# "알림·자동화 설정"의 Refresh Time 주기 자동화(SQL) 실행 -- 유일한 자동
# 발송 경로이며, 별도의 조건 없이 매 주기마다 무조건 시도한다(억제
# 규칙은 시간당 예산만 적용).
TRIGGER_REFRESH = "refresh"
TRIGGER_MANUAL = "manual_analysis"  # 수동 "분석 실행 직후" 발송(POST /api/notify/dispatch)
# 수율 예측 화면의 "알림 전송" 버튼 -- 사용자가 직접 눌러 지금 보내라는
# 명시적 의도다. 최소 간격(10분)·시간당 예산은 TRIGGER_MANUAL과 동일하게
# 적용한다 -- 억제 규칙을 우회하는 옵션은 만들지 않는다.
TRIGGER_MANUAL_BUTTON = "manual_button"


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
    """이 함수의 모든 종료 경로(발송/건너뜀 전부)가 `notify_history`에
    기록을 남긴다 -- 알림 기록 화면이 "왜 안 보냈는지"까지 보여줄 수
    있어야 한다."""

    def _record(*, status: str, skip_reason: str | None, results: dict[str, Any] | None) -> None:
        channels = [name for name, result in (results or {}).items() if result.get("ok")]
        sent_count = len(channels)
        message_text = build_telegram_yield_update_text(payload) if results is not None else None
        try:
            store.record_notify_history(
                trigger=trigger,
                channels=channels,
                dataset_label=payload.dataset_label,
                model_version=payload.model_label,
                status=status,
                skip_reason=skip_reason,
                message_text=message_text,
                sent_count=sent_count,
            )
        except Exception:
            # 이력 기록 실패가 발송 자체의 성공/실패 판정을 가리면
            # 안 된다 -- best-effort로 로그만 남긴다.
            logger.exception("notify_history 기록 실패 trigger=%s", trigger)

    def _skip(reason: str) -> dict[str, Any]:
        _record(status="skipped", skip_reason=reason, results=None)
        return {"skipped": True, "reason": reason}

    try:
        if trigger in (TRIGGER_MANUAL, TRIGGER_MANUAL_BUTTON):
            blocked_until = _manual_interval_blocked_until(store)
            if blocked_until is not None:
                return _skip(
                    f"직전 발송 후 {MANUAL_MIN_INTERVAL_MINUTES}분이 지나지 않아 발송하지 않았습니다 (다음 발송 가능 {blocked_until})"
                )

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
            return _skip("연결된 채널 없음")

        if not _within_hourly_budget(store):
            return _skip(f"시간당 발송 예산({HOURLY_SEND_BUDGET}건) 초과")

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

        # 시도한 채널이 하나도 성공하지 못했으면 "발송됨"으로 남기지
        # 않는다 -- 알림 기록 화면은 status가 sent가 아닐 때만 사유를
        # 보여주므로, 그대로 두면 "TOP 0 · 발송됨"만 뜨고 왜 못 갔는지
        # 알 수 없다.
        if results and not any(item.get("ok") for item in results.values()):
            errors = "; ".join(f"{name}({item.get('error') or '실패'})" for name, item in results.items())
            _record(status="skipped", skip_reason=f"채널 발송 실패: {errors}", results=results)
        else:
            _record(status="sent", skip_reason=None, results=results)
        return {"skipped": False, "results": results}
    except Exception:
        logger.exception("수율 예측 갱신 발송 중 오류")
        return _skip("발송 중 오류 발생")
