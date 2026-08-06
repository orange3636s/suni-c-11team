"""알림 채널 연결 정보 영속 저장 (알람 알림 연동 §D) -- `localStorage`가 아니라
기존 `RuntimeStore`의 `app_state` 키-값 테이블에 저장한다. 브라우저를 닫거나
다른 기기에서 접속해도 유지되어야 하기 때문이다 (§D-1).

Webhook URL과 chat_id는 인증 정보에 준한다 (§D-2) -- API로 내보낼 때는 항상
마스킹된 값만 담고, 실제 발송(src.notifications.senders/dispatch)에서만
원본을 조회해 쓴다.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from src.runtime.store import RuntimeStore

STATE_KEYS = {
    "slack": "notify:slack",
    "telegram": "notify:telegram",
    "gmail": "notify:gmail",
    "conditions": "notify:conditions",
}

ChannelName = Literal["slack", "telegram", "gmail"]
AlarmGrade = Literal["심각", "위험", "주의"]

DEFAULT_GRADES: list[AlarmGrade] = ["심각", "위험"]  # spec §C-4: 기본값, 주의는 기본 해제
TIMING_ON_ANALYSIS = "on_analysis"
TIMING_DAILY_8AM = "daily_8am"
VALID_TIMINGS = {TIMING_ON_ANALYSIS, TIMING_DAILY_8AM}
DEFAULT_CONDITIONS = {"grades": list(DEFAULT_GRADES), "timing": TIMING_ON_ANALYSIS}

SLACK_WEBHOOK_DOMAIN = "hooks.slack.com"
GMAIL_TOKEN_TTL_MINUTES = 60  # 인증 메일 링크 유효 시간


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- 마스킹 (spec §D-2) -----------------------------------------------------


def mask_slack_webhook(url: str) -> str:
    """`https://hooks.slack.com/services/T00.../B00.../xxx` ->
    `hooks.slack.com/…/xxx****` -- 도메인과 마지막 세그먼트 앞 3자만 남긴다."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    visible = tail[:3]
    return f"{SLACK_WEBHOOK_DOMAIN}/…/{visible}****" if visible else f"{SLACK_WEBHOOK_DOMAIN}/…/****"


def mask_chat_id(chat_id: str) -> str:
    """`123456789` -> `1234****` -- 앞 4자리만 남긴다."""
    chat_id = str(chat_id)
    visible = chat_id[:4]
    return f"{visible}****" if visible else "****"


def is_valid_slack_webhook_url(url: str) -> bool:
    if not url or not url.startswith("https://"):
        return False
    host = url[len("https://") :].split("/", 1)[0]
    return host == SLACK_WEBHOOK_DOMAIN


# -- Slack --------------------------------------------------------------


def get_slack(store: RuntimeStore) -> dict[str, Any] | None:
    return store.get_app_state(STATE_KEYS["slack"])


def save_slack(store: RuntimeStore, *, webhook_url: str, channel: str | None) -> dict[str, Any]:
    record = {"webhook_url": webhook_url, "channel": channel, "verified_at": _now_iso()}
    store.set_app_state(STATE_KEYS["slack"], record)
    return record


# -- Telegram -------------------------------------------------------------


def get_telegram(store: RuntimeStore) -> dict[str, Any] | None:
    return store.get_app_state(STATE_KEYS["telegram"])


def save_telegram(store: RuntimeStore, *, chat_id: str, username: str | None) -> dict[str, Any]:
    record = {"chat_id": str(chat_id), "username": username, "verified_at": _now_iso()}
    store.set_app_state(STATE_KEYS["telegram"], record)
    return record


# -- Gmail ------------------------------------------------------------------

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email.strip()))


def get_gmail(store: RuntimeStore) -> dict[str, Any] | None:
    return store.get_app_state(STATE_KEYS["gmail"])


def start_gmail_verification(store: RuntimeStore, *, email: str) -> str:
    """인증 메일 발송 전 '대기 중' 레코드를 저장하고 토큰을 반환한다
    (spec §C-3 Gmail: "확인 전에는 대기 중 상태"). 인증 전 상태는 절대
    `verified: true`로 남지 않는다 -- 확인 없이 저장하면 타인의 주소로
    알림을 보낼 수 있다."""
    token = secrets.token_urlsafe(24)
    record = {
        "email": email,
        "verified": False,
        "token": token,
        "requested_at": _now_iso(),
        "verified_at": None,
    }
    store.set_app_state(STATE_KEYS["gmail"], record)
    return token


def verify_gmail(store: RuntimeStore, *, token: str) -> bool:
    record = get_gmail(store)
    if not record or record.get("verified") or record.get("token") != token:
        return False
    record["verified"] = True
    record["verified_at"] = _now_iso()
    record.pop("token", None)
    store.set_app_state(STATE_KEYS["gmail"], record)
    return True


# -- 발송 조건 (spec §C-4) --------------------------------------------------


def get_conditions(store: RuntimeStore) -> dict[str, Any]:
    return store.get_app_state(STATE_KEYS["conditions"]) or dict(DEFAULT_CONDITIONS)


def save_conditions(store: RuntimeStore, *, grades: list[str], timing: str) -> dict[str, Any]:
    if timing not in VALID_TIMINGS:
        raise ValueError(f"알 수 없는 발송 시점입니다: {timing}")
    valid_grades = [g for g in grades if g in ("심각", "위험", "주의")]
    record = {"grades": valid_grades, "timing": timing}
    store.set_app_state(STATE_KEYS["conditions"], record)
    return record


# -- 해제 (spec §D-4) ---------------------------------------------------


def disconnect(store: RuntimeStore, channel: ChannelName) -> bool:
    return store.delete_app_state(STATE_KEYS[channel])


# -- 요약 (마스킹된, API 응답/state-latest 공용) ----------------------------


def get_settings_summary(store: RuntimeStore) -> dict[str, Any]:
    slack = get_slack(store)
    telegram = get_telegram(store)
    gmail = get_gmail(store)
    conditions = get_conditions(store)

    return {
        "slack": (
            {
                "connected": True,
                "target": slack.get("channel"),
                "webhook_masked": mask_slack_webhook(slack["webhook_url"]),
                "verified_at": slack.get("verified_at"),
            }
            if slack
            else {"connected": False, "target": None, "webhook_masked": None, "verified_at": None}
        ),
        "telegram": (
            {
                "connected": True,
                "target": telegram.get("username"),
                "chat_id_masked": mask_chat_id(telegram["chat_id"]),
                "verified_at": telegram.get("verified_at"),
            }
            if telegram
            else {"connected": False, "target": None, "chat_id_masked": None, "verified_at": None}
        ),
        "gmail": (
            {
                "connected": bool(gmail.get("verified")),
                "pending": not gmail.get("verified", False),
                "email": gmail.get("email"),
                "verified_at": gmail.get("verified_at"),
            }
            if gmail
            else {"connected": False, "pending": False, "email": None, "verified_at": None}
        ),
        "conditions": conditions,
    }
