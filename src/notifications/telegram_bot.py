"""Telegram 봇 인증 코드 흐름 (알람 알림 연동 §C-3 Telegram) -- 텔레그램
정책상 봇은 사용자보다 먼저 대화를 시작할 수 없으므로, 사용자 이름을 직접
입력받는 방식은 애초에 동작하지 않는다. 대신:

  1. 사용자가 봇 링크를 열어 `/start`를 보낸다
  2. 이 모듈이 long-polling으로 그 메시지를 받아 6자리 코드를 생성하고,
     봇이 그 코드를 답장으로 보낸다 (chat_id는 이 시점에만 얻을 수 있다)
  3. 사용자가 그 코드를 대시보드에 입력하면 `resolve_code()`로 chat_id를
     account에 연결한다

코드는 메모리에만 10분간 보관한다 (재시작하면 사라지지만, 유효 시간이
짧아 재시작 자체가 드문 개발 환경에서는 재발급을 다시 받으면 그만이다 --
DB에 남겨 봐야 만료된 코드가 계속 쌓이기만 한다).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.notifications.senders import send_telegram_message

logger = logging.getLogger(__name__)

CODE_TTL_MINUTES = 10
POLL_TIMEOUT_SECONDS = 30


@dataclass
class PendingCode:
    chat_id: str
    username: str | None
    expires_at: datetime


_pending_codes: dict[str, PendingCode] = {}


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _prune_expired() -> None:
    now = datetime.now(timezone.utc)
    expired = [code for code, pending in _pending_codes.items() if pending.expires_at < now]
    for code in expired:
        _pending_codes.pop(code, None)


def register_start(chat_id: str, username: str | None) -> str:
    """`/start` 수신 시 코드를 발급하고 등록한다. 이미 이 chat_id로 발급된
    코드가 있으면 재사용하지 않고 새로 발급한다 -- 이전 코드가 이미 다른
    화면에 노출되어 있을 수 있으므로 무효화하지 않는다(그냥 둘 다 유효)."""
    _prune_expired()
    code = _generate_code()
    while code in _pending_codes:
        code = _generate_code()
    _pending_codes[code] = PendingCode(
        chat_id=str(chat_id),
        username=username,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=CODE_TTL_MINUTES),
    )
    return code


def resolve_code(code: str) -> dict[str, Any] | None:
    """대시보드에 입력한 코드를 chat_id로 바꾼다. 1회용 -- 성공하면 즉시
    제거해 같은 코드를 재사용할 수 없게 한다."""
    _prune_expired()
    pending = _pending_codes.pop(code.strip(), None)
    if pending is None:
        return None
    return {"chat_id": pending.chat_id, "username": pending.username}


async def _handle_update(bot_token: str, update: dict[str, Any]) -> None:
    message = update.get("message") or {}
    text = str(message.get("text") or "").strip()
    if not text.startswith("/start"):
        return
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return
    username = chat.get("username")
    display = f"@{username}" if username else None
    code = register_start(str(chat_id), display)
    ok, error = await asyncio.to_thread(
        send_telegram_message,
        bot_token,
        str(chat_id),
        f"SUNI 대시보드 인증 코드: `{code}`\n10분 이내에 대시보드에 입력해 주세요\\.",
    )
    if not ok:
        logger.warning("Telegram 인증 코드 응답 발송 실패: %s", error)


async def run_polling_loop(bot_token: str, stop_event: asyncio.Event) -> None:
    """앱 lifespan 동안 계속 도는 long-polling 루프. 앱 종료 시
    `stop_event`를 set하면 다음 반복에서 빠져나온다."""
    offset: int | None = None
    async with httpx.AsyncClient(timeout=POLL_TIMEOUT_SECONDS + 10) as client:
        while not stop_event.is_set():
            try:
                params: dict[str, Any] = {"timeout": POLL_TIMEOUT_SECONDS}
                if offset is not None:
                    params["offset"] = offset
                response = await client.get(
                    f"https://api.telegram.org/bot{bot_token}/getUpdates", params=params
                )
                body = response.json()
                if not body.get("ok"):
                    logger.warning("Telegram getUpdates 오류: %s", body)
                    await asyncio.sleep(5)
                    continue
                for update in body.get("result", []):
                    offset = update["update_id"] + 1
                    await _handle_update(bot_token, update)
            except httpx.HTTPError as exc:
                logger.warning("Telegram polling 오류: %s", exc)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Telegram polling 루프에서 알 수 없는 오류")
                await asyncio.sleep(5)
