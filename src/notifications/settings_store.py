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

from api.settings import settings
from src.runtime.store import RuntimeStore

STATE_KEYS = {
    "slack": "notify:slack",
    "telegram": "notify:telegram",
    "gmail": "notify:gmail",
    "conditions": "notify:conditions",
}
# A-8: 인증 대기 중인 재연결 시도는 이 별도 키에 저장한다 -- 예전에는
# `STATE_KEYS["gmail"]`(연결 완료 레코드와 같은 키) 하나를 pending으로
# 덮어썼기 때문에, 5분 안에 인증을 끝내지 못하면 `get_gmail`이 만료된
# pending 레코드를 지우면서 "덮어써지기 전까지 멀쩡했던 기존 연결"까지
# 함께 사라졌다. pending 전용 키를 두면 5분 TTL은 이 키에만 적용되고,
# 연결 완료(verified) 레코드는 verify 성공 시에만 이 키의 내용으로
# 교체되므로 만료 대상이 될 수 없다.
GMAIL_PENDING_STATE_KEY = "notify:gmail:pending"

ChannelName = Literal["slack", "telegram", "gmail"]
AlarmGrade = Literal["심각", "위험", "주의"]

DEFAULT_GRADES: list[AlarmGrade] = ["심각"]  # 지시서 N-3: 기본값은 심각만, 위험·주의는 기본 해제
TIMING_ON_ANALYSIS = "on_analysis"
TIMING_DAILY_9AM = "daily_9am"
TIMING_DAILY_13 = "daily_13"
VALID_TIMINGS = {TIMING_ON_ANALYSIS, TIMING_DAILY_9AM, TIMING_DAILY_13}
DEFAULT_TIMINGS: list[str] = [TIMING_ON_ANALYSIS]
DEFAULT_CONDITIONS = {"grades": list(DEFAULT_GRADES), "timing": list(DEFAULT_TIMINGS)}

# 지시서 N-2: 발송 시각을 오전 8시 -> 9시로 옮기며 타이밍 값 이름도
# daily_8am -> daily_9am으로 바꿨다. 이미 "daily_8am"으로 저장된 기존
# 사용자 설정을 읽을 때 깨진 값으로 남기지 않도록 여기서 변환한다.
_LEGACY_TIMING_MIGRATIONS = {"daily_8am": TIMING_DAILY_9AM}


def _normalize_timings(raw: Any) -> list[str]:
    """DF그룹: 발송 시점이 단일 문자열(구버전 저장값)이든 배열(신규)이든
    항상 배열로 정규화한다. 값이 아예 없던 적이 없으면(레코드 자체가
    없음) 기본값으로 폴백하지만, 사용자가 명시적으로 빈 배열을
    저장했다면(전부 해제) 그 선택을 그대로 존중한다 -- 조용히 기본값으로
    되돌리면 "발송 안 함"을 선택할 방법이 없어진다."""
    if raw is None:
        return list(DEFAULT_TIMINGS)
    if isinstance(raw, str):
        migrated = _LEGACY_TIMING_MIGRATIONS.get(raw, raw)
        return [migrated] if migrated in VALID_TIMINGS else list(DEFAULT_TIMINGS)
    if isinstance(raw, list):
        result: list[str] = []
        for value in raw:
            migrated = _LEGACY_TIMING_MIGRATIONS.get(value, value)
            if migrated in VALID_TIMINGS and migrated not in result:
                result.append(migrated)
        return result
    return list(DEFAULT_TIMINGS)

SLACK_WEBHOOK_DOMAIN = "hooks.slack.com"

# 지시서 W: 인증 대기(pending) 레코드가 무기한 남아 있으면 사용자가 메일을
# 못 받거나 주소를 잘못 입력했을 때 영영 복구할 수 없다. 조회 시점에
# 만료를 판정한다 -- 별도 백그라운드 정리 잡은 두지 않는다.
PENDING_TTL_SECONDS = 300  # 5분


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_pending_expired(record: dict[str, Any]) -> bool:
    """연결이 완료된(verified) 레코드는 절대 만료시키지 않는다 -- 이 함수는
    "인증 대기 중" 레코드에만 적용된다."""
    if record.get("verified"):
        return False
    requested = record.get("requested_at")
    if not requested:
        return True  # 시각 없는 레코드는 만료 처리
    try:
        requested_at = datetime.fromisoformat(requested)
    except ValueError:
        return True
    age = (datetime.now(timezone.utc) - requested_at).total_seconds()
    return age > PENDING_TTL_SECONDS


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
    """연결 완료(verified) 레코드만 반환한다 -- A-8: 이 키는 verify_gmail이
    성공했을 때만 쓰여지므로 만료 판정 자체가 필요 없다(영구 보관)."""
    return store.get_app_state(STATE_KEYS["gmail"])


def get_gmail_pending(store: RuntimeStore) -> dict[str, Any] | None:
    """지시서 W: 만료된 인증 대기 레코드는 읽는 즉시 지우고 `None`을
    반환한다. A-8: 이 pending 레코드는 연결 완료 레코드(`STATE_KEYS["gmail"]`)와
    별도 키에 저장되므로, 5분 안에 인증을 끝내지 못해 여기서 지워져도
    기존에 연결 완료돼 있던 이메일에는 영향이 없다."""
    record = store.get_app_state(GMAIL_PENDING_STATE_KEY)
    if record and _is_pending_expired(record):
        store.delete_app_state(GMAIL_PENDING_STATE_KEY)
        return None
    return record


def start_gmail_verification(store: RuntimeStore, *, email: str) -> str:
    """인증 메일 발송 전 '대기 중' 레코드를 별도 키(`GMAIL_PENDING_STATE_KEY`)에
    저장하고 토큰을 반환한다 (spec §C-3 Gmail: "확인 전에는 대기 중 상태").
    A-8: 기존 연결 완료 레코드(`STATE_KEYS["gmail"]`)는 절대 건드리지
    않는다 -- 재인증 시도가 5분 내에 끝나지 않아도 이미 연결된 이메일은
    그대로 살아 있어야 한다."""
    token = secrets.token_urlsafe(24)
    record = {
        "email": email,
        "token": token,
        "requested_at": _now_iso(),
    }
    store.set_app_state(GMAIL_PENDING_STATE_KEY, record)
    return token


def verify_gmail(store: RuntimeStore, *, token: str) -> bool:
    """pending 레코드가 유효하면 연결 완료 레코드로 승격시켜
    `STATE_KEYS["gmail"]`에 쓰고, pending 키는 지운다."""
    pending = get_gmail_pending(store)
    if not pending or pending.get("token") != token:
        return False
    record = {
        "email": pending["email"],
        "verified": True,
        "verified_at": _now_iso(),
    }
    store.set_app_state(STATE_KEYS["gmail"], record)
    store.delete_app_state(GMAIL_PENDING_STATE_KEY)
    return True


# -- 발송 조건 (spec §C-4) --------------------------------------------------


def get_conditions(store: RuntimeStore) -> dict[str, Any]:
    record = store.get_app_state(STATE_KEYS["conditions"])
    if not record:
        return dict(DEFAULT_CONDITIONS)
    return {**record, "timing": _normalize_timings(record.get("timing"))}


def save_conditions(store: RuntimeStore, *, grades: list[str], timing: list[str]) -> dict[str, Any]:
    invalid = [t for t in timing if t not in VALID_TIMINGS]
    if invalid:
        raise ValueError(f"알 수 없는 발송 시점입니다: {invalid[0]}")
    valid_grades = [g for g in grades if g in ("심각", "위험", "주의")]
    # dict.fromkeys로 순서를 유지한 채 중복만 제거한다.
    record = {"grades": valid_grades, "timing": list(dict.fromkeys(timing))}
    store.set_app_state(STATE_KEYS["conditions"], record)
    return record


# -- 해제 (spec §D-4) ---------------------------------------------------


def disconnect(store: RuntimeStore, channel: ChannelName) -> bool:
    if channel == "gmail":
        # A-8: pending을 별도 키로 분리했으므로, 연결 해제 시 진행 중이던
        # 재인증 시도도 함께 지워야 한다 -- 안 지우면 이미 "연결 해제"한
        # 뒤에 예전 인증 링크를 클릭해 유령 연결이 되살아날 수 있다.
        store.delete_app_state(GMAIL_PENDING_STATE_KEY)
    return store.delete_app_state(STATE_KEYS[channel])


# -- 요약 (마스킹된, API 응답/state-latest 공용) ----------------------------


def get_settings_summary(store: RuntimeStore) -> dict[str, Any]:
    slack = get_slack(store)
    telegram = get_telegram(store)
    gmail = get_gmail(store)
    gmail_pending = None if gmail else get_gmail_pending(store)
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
                "connected": True,
                "pending": False,
                "email": gmail.get("email"),
                "verified_at": gmail.get("verified_at"),
            }
            if gmail
            else (
                {"connected": False, "pending": True, "email": gmail_pending.get("email"), "verified_at": None}
                if gmail_pending
                else {"connected": False, "pending": False, "email": None, "verified_at": None}
            )
        ),
        "conditions": conditions,
        # SD-1: "알림·자동화 설정" 팝업의 자동화 섹션 -- 비밀번호는 절대
        # 포함하지 않는다(환경변수 DB_PASSWORD로만 받는다).
        "automation": _automation_summary(store),
        # EA그룹: 봇 username을 백엔드 단일 소스로 -- 프런트가 자기
        # 환경변수를 따로 읽고 없으면 존재하지 않는 봇 이름으로 링크를
        # 만들던 문제(하드코딩 폴백)를 없앤다. 가공하지 않고 그대로
        # 내려보낸다 -- 미설정(None)이면 그대로 null. 토큰은 여기 절대
        # 포함하지 않는다.
        "telegram_bot_username": settings.telegram_bot_username,
    }


def _automation_summary(store: RuntimeStore) -> dict[str, Any]:
    record = store.get_automation_settings()
    return {
        "enabled": bool(record.get("enabled")),
        "sql_host": record.get("sqlHost") or "",
        "sql_port": record.get("sqlPort") or "",
        "sql_db": record.get("sqlDb") or "",
        "sql_user": record.get("sqlUser") or "",
        "refresh_interval_minutes": record.get("refreshIntervalMinutes") or 60,
        "last_run_at": record.get("lastRunAt"),
        "last_run_status": record.get("lastRunStatus"),
        "last_run_sent_count": record.get("lastRunSentCount"),
    }
