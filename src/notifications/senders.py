"""채널 전송 -- Slack webhook/Telegram sendMessage/Gmail SMTP의 공용
저수준 전송 함수. 메시지 조립(어떤 내용을 어떤 포맷으로 담을지)은 이
모듈의 몫이 아니다 -- `src/notifications/yield_update_senders.py`가
전담한다(수율 예측 갱신 발송이 유일한 발신 파이프라인이다).
"""

from __future__ import annotations

import logging
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SEND_TIMEOUT_SECONDS = 10


def send_slack_webhook(webhook_url: str, body: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        response = httpx.post(webhook_url, json=body, timeout=SEND_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return True, None
        return False, f"Slack 응답 {response.status_code}: {response.text[:200]}"
    except httpx.HTTPError as exc:
        return False, str(exc)


def send_slack_test(webhook_url: str) -> tuple[bool, str | None]:
    body = {
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*SUNI 알람 연동 테스트 발송입니다.* 이 메시지가 보이면 연결이 정상입니다."},
            }
        ]
    }
    return send_slack_webhook(webhook_url, body)


# -- Telegram --------------------------------------------------------------

_MARKDOWN_V2_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!])")


def escape_markdown_v2(text: str) -> str:
    return _MARKDOWN_V2_SPECIAL.sub(r"\\\1", text)


def send_telegram_message(bot_token: str, chat_id: str, text: str, *, parse_mode: str | None = "MarkdownV2") -> tuple[bool, str | None]:
    """`parse_mode=None` sends plain text -- 수율 예측 갱신 발송은
    Telegram에서 MarkdownV2 이스케이프 없이 고정폭 정렬 텍스트로 보낸다.
    연결 테스트 메시지(`send_telegram_test`)만 기본값(MarkdownV2)을 쓴다."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        response = httpx.post(url, json=payload, timeout=SEND_TIMEOUT_SECONDS)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if response.status_code == 200 and body.get("ok"):
            return True, None
        return False, f"Telegram 응답 {response.status_code}: {body.get('description') or response.text[:200]}"
    except httpx.HTTPError as exc:
        return False, str(exc)


def send_telegram_test(bot_token: str, chat_id: str) -> tuple[bool, str | None]:
    return send_telegram_message(bot_token, chat_id, escape_markdown_v2("SUNI 알람 연동 테스트 발송입니다. 이 메시지가 보이면 연결이 정상입니다."))


# -- Gmail (SMTP) ------------------------------------------------------------


def send_gmail(
    *,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    subject: str,
    html_body: str,
) -> tuple[bool, str | None]:
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email
    message.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=SEND_TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, [to_email], message.as_string())
        return True, None
    except (smtplib.SMTPException, OSError) as exc:
        return False, str(exc)


def send_gmail_test(*, smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str, from_email: str, to_email: str) -> tuple[bool, str | None]:
    return send_gmail(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=from_email,
        to_email=to_email,
        subject="[SUNI] 알람 연동 테스트 발송",
        html_body="<p>SUNI 알람 연동 테스트 발송입니다. 이 메일이 보이면 연결이 정상입니다.</p>",
    )
