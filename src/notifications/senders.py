"""채널별 발송 (알람 알림 연동 §C-6) -- Slack은 Block Kit, Telegram은
MarkdownV2, 메일은 HTML로 각각 포맷한다. 세 플랫폼을 하나의 포맷으로
통일하지 않는다 (연결 방식이 서로 다른 것과 같은 이유: 플랫폼마다 요구하는
형식이 다르다).

메시지에는 예측 수율 절대값을 쓰지 않는다 (§C-6) -- 위험 순위(하위 N%)만
표기한다. 예측 오차가 Y 표준편차의 70~80%에 달해(알람 판정 GBDT 전환 §A-3
검증), 정밀해 보이는 절대값은 실제보다 정확하다는 인상을 준다.
"""

from __future__ import annotations

import logging
import re
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SEND_TIMEOUT_SECONDS = 10
MAX_LISTED_ITEMS = 5


@dataclass
class AlarmNotificationItem:
    lot_wafer_id: str
    risk_percentile: float  # 0-100, 하위 퍼센트 (낮을수록 위험)
    grade: str
    reason: str


@dataclass
class AlarmNotificationPayload:
    dataset_label: str
    timestamp_label: str  # "2026-08-07 09:12" 형식으로 이미 포맷된 문자열
    items: list[AlarmNotificationItem]
    grade_counts: dict[str, int]  # 심각/위험/주의 -> 건수 (표시 대상 등급만)
    reliability_grade: str
    reliability_score: int
    dashboard_url: str | None = None

    @property
    def total(self) -> int:
        return sum(self.grade_counts.values())


def _grade_counts_line(counts: dict[str, int]) -> str:
    order = ["심각", "위험", "주의"]
    parts = [f"{grade} {counts[grade]}건" for grade in order if counts.get(grade)]
    return " · ".join(parts)


def _item_lines(payload: AlarmNotificationPayload) -> list[tuple[str, str]]:
    """(제목 줄, 사유 줄) 튜플의 리스트를 최대 MAX_LISTED_ITEMS개까지 반환한다."""
    lines: list[tuple[str, str]] = []
    for item in payload.items[:MAX_LISTED_ITEMS]:
        title = f"{item.lot_wafer_id}   하위 {item.risk_percentile:.1f}%   {item.grade}"
        lines.append((title, item.reason))
    return lines


def _remainder_note(payload: AlarmNotificationPayload) -> str | None:
    remaining = len(payload.items) - MAX_LISTED_ITEMS
    return f"외 {remaining}건" if remaining > 0 else None


def format_plain_summary(payload: AlarmNotificationPayload) -> str:
    lines = [
        f"[SUNI] 알람 {payload.total}건 발생",
        f"데이터셋  {payload.dataset_label}        {payload.timestamp_label}",
        "",
        _grade_counts_line(payload.grade_counts),
        "",
    ]
    for title, reason in _item_lines(payload):
        lines.append(title)
        lines.append(f"  {reason}")
    remainder = _remainder_note(payload)
    if remainder:
        lines.append(remainder)
    lines.append("")
    lines.append(f"분석 신뢰도  {payload.reliability_grade} ({payload.reliability_score}점)")
    if payload.dashboard_url:
        lines.append(f"대시보드에서 확인 → {payload.dashboard_url}")
    return "\n".join(lines)


# -- Slack (Block Kit) -------------------------------------------------------


def build_slack_blocks(payload: AlarmNotificationPayload) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"[SUNI] 알람 {payload.total}건 발생", "emoji": True},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"데이터셋 *{payload.dataset_label}* · {payload.timestamp_label}"}],
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{_grade_counts_line(payload.grade_counts)}*"}},
    ]
    for title, reason in _item_lines(payload):
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*\n{reason}"}})
    remainder = _remainder_note(payload)
    if remainder:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": remainder}]})
    footer = f"분석 신뢰도 *{payload.reliability_grade}* ({payload.reliability_score}점)"
    if payload.dashboard_url:
        footer += f" · <{payload.dashboard_url}|대시보드에서 확인>"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]})
    return {"blocks": blocks}


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


def send_slack_alarm(webhook_url: str, payload: AlarmNotificationPayload) -> tuple[bool, str | None]:
    return send_slack_webhook(webhook_url, build_slack_blocks(payload))


# -- Telegram (MarkdownV2) --------------------------------------------------

_MARKDOWN_V2_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!])")


def escape_markdown_v2(text: str) -> str:
    return _MARKDOWN_V2_SPECIAL.sub(r"\\\1", text)


def build_telegram_text(payload: AlarmNotificationPayload) -> str:
    lines = [
        f"*\\[SUNI\\] 알람 {payload.total}건 발생*",
        escape_markdown_v2(f"데이터셋 {payload.dataset_label} · {payload.timestamp_label}"),
        "",
        escape_markdown_v2(_grade_counts_line(payload.grade_counts)),
        "",
    ]
    for title, reason in _item_lines(payload):
        lines.append(f"*{escape_markdown_v2(title)}*")
        lines.append(escape_markdown_v2(f"  {reason}"))
    remainder = _remainder_note(payload)
    if remainder:
        lines.append(escape_markdown_v2(remainder))
    lines.append("")
    lines.append(escape_markdown_v2(f"분석 신뢰도 {payload.reliability_grade} ({payload.reliability_score}점)"))
    if payload.dashboard_url:
        lines.append(f"[대시보드에서 확인]({payload.dashboard_url})")
    return "\n".join(lines)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str | None]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        response = httpx.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2", "disable_web_page_preview": True},
            timeout=SEND_TIMEOUT_SECONDS,
        )
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if response.status_code == 200 and body.get("ok"):
            return True, None
        return False, f"Telegram 응답 {response.status_code}: {body.get('description') or response.text[:200]}"
    except httpx.HTTPError as exc:
        return False, str(exc)


def send_telegram_test(bot_token: str, chat_id: str) -> tuple[bool, str | None]:
    return send_telegram_message(bot_token, chat_id, escape_markdown_v2("SUNI 알람 연동 테스트 발송입니다. 이 메시지가 보이면 연결이 정상입니다."))


def send_telegram_alarm(bot_token: str, chat_id: str, payload: AlarmNotificationPayload) -> tuple[bool, str | None]:
    return send_telegram_message(bot_token, chat_id, build_telegram_text(payload))


# -- Gmail (SMTP, HTML) ------------------------------------------------------


def build_email_html(payload: AlarmNotificationPayload) -> str:
    rows = "".join(
        f"<tr><td style='padding:6px 10px;font-weight:600'>{title}</td></tr>"
        f"<tr><td style='padding:0 10px 10px;color:#555;font-size:13px'>{reason}</td></tr>"
        for title, reason in _item_lines(payload)
    )
    remainder = _remainder_note(payload)
    remainder_html = f"<p style='color:#888;font-size:12px'>{remainder}</p>" if remainder else ""
    dashboard_html = (
        f"<p><a href='{payload.dashboard_url}'>대시보드에서 확인 →</a></p>" if payload.dashboard_url else ""
    )
    return f"""
    <div style="font-family:sans-serif;max-width:520px">
      <h2 style="margin:0 0 4px">[SUNI] 알람 {payload.total}건 발생</h2>
      <p style="color:#555;margin:0 0 12px">데이터셋 {payload.dataset_label} &middot; {payload.timestamp_label}</p>
      <p style="font-weight:600">{_grade_counts_line(payload.grade_counts)}</p>
      <table style="border-collapse:collapse;width:100%">{rows}</table>
      {remainder_html}
      <p style="margin-top:16px">분석 신뢰도 {payload.reliability_grade} ({payload.reliability_score}점)</p>
      {dashboard_html}
    </div>
    """


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


def send_gmail_alarm(
    *, smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str, from_email: str, to_email: str, payload: AlarmNotificationPayload
) -> tuple[bool, str | None]:
    return send_gmail(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=from_email,
        to_email=to_email,
        subject=f"[SUNI] 알람 {payload.total}건 발생",
        html_body=build_email_html(payload),
    )
