"""채널별 발송 (알람 알림 연동 §C-6) -- Slack은 Block Kit, Telegram은
MarkdownV2, 메일은 HTML로 각각 포맷한다. 세 플랫폼을 하나의 포맷으로
통일하지 않는다 (연결 방식이 서로 다른 것과 같은 이유: 플랫폼마다 요구하는
형식이 다르다).

메시지에는 예측 수율 절대값을 쓰지 않는다 (§C-6) -- 위험 순위(하위 N%)만
표기한다. 예측 오차가 Y 표준편차의 70~80%에 달해(알람 판정 GBDT 전환 §A-3
검증), 정밀해 보이는 절대값은 실제보다 정확하다는 인상을 준다.
"""

from __future__ import annotations

import html
import logging
import re
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

from src.analysis.alarm_gbdt import DEFAULT_SENSITIVITY, DEFAULT_TARGET_YIELD, classify_margin

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
    # EB그룹: 수동 업로드/폴백(데모) 모드에서 발송할 때 본문 맨 위에 붙는
    # 출처 한 줄 -- 예: "[데모] 내장 데이터 기준 -- 실제 공정 데이터가
    # 아닙니다", "[수동] uploaded_0809.csv 업로드 결과". 원본(미이스케이프)
    # 텍스트이며, 채널별 이스케이프는 각 build_* 함수가 담당한다. SQL
    # 연동 자동 갱신 경로에서는 None(표시 없음).
    source_note: str | None = None
    # GD-2: 판정 기준 -- 이게 없으면 받는 사람이 무엇에 대한 알람인지
    # 모른다("심각 8건"이 무슨 기준으로 심각인지 알 수 없다). 개별 wafer의
    # 예측 수율 절대값은 여전히 안 보여주지만(위 모듈 docstring, §C-6),
    # 목표 수율·민감도와 그 둘로 정해지는 판정 컷(문턱값 하나)은 특정
    # wafer의 예측이 아니라 이번 판정 자체의 설정값이라 같은 문제가 없다.
    target_yield: float = DEFAULT_TARGET_YIELD
    sensitivity: float = DEFAULT_SENSITIVITY

    @property
    def total(self) -> int:
        return sum(self.grade_counts.values())

    @property
    def judgment_cut(self) -> float:
        """가장 느슨한 등급("주의")의 컷 -- src.analysis.alarm_gbdt.classify_wafer
        와 같은 공식(목표 - margin)이다. 알람 여부 자체가 이 컷 하나로
        정해지므로, 개별 등급 컷 셋을 전부 나열하지 않고 이것만 보여준다."""
        return self.target_yield - classify_margin(self.sensitivity)


def _grade_counts_line(counts: dict[str, int]) -> str:
    order = ["심각", "위험", "주의"]
    parts = [f"{grade} {counts[grade]}건" for grade in order if counts.get(grade)]
    return " · ".join(parts)


def _criteria_line(payload: AlarmNotificationPayload) -> str:
    """GD-2: 목표·민감도·판정 컷 -- 이게 없으면 받는 사람이 무엇에 대한
    알람인지 모른다. "수율"이라는 단어를 쓰지 않는다 -- 개별 wafer의
    예측 수율 절대값 노출 금지(§C-6)와 혼동하지 않도록, 이 값들은
    이번 판정의 설정값이라는 점을 문구로도 구분한다."""
    return f"목표 {payload.target_yield:.1f}% · 민감도 {payload.sensitivity:.2f} · 판정 컷 {payload.judgment_cut:.1f}%"


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
    lines = []
    if payload.source_note:
        lines.append(payload.source_note)
        lines.append("")
    lines += [
        f"[SUNI] 알람 {payload.total}건 발생",
        f"데이터셋  {payload.dataset_label}        {payload.timestamp_label}",
        "",
        _grade_counts_line(payload.grade_counts),
        _criteria_line(payload),
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
    blocks: list[dict[str, Any]] = []
    if payload.source_note:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"*{payload.source_note}*"}]})
    blocks += [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"[SUNI] 알람 {payload.total}건 발생", "emoji": True},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"데이터셋 *{payload.dataset_label}* · {payload.timestamp_label}"}],
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{_grade_counts_line(payload.grade_counts)}*"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": _criteria_line(payload)}]},
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
    lines = []
    if payload.source_note:
        lines.append(f"*{escape_markdown_v2(payload.source_note)}*")
        lines.append("")
    lines += [
        f"*\\[SUNI\\] 알람 {payload.total}건 발생*",
        escape_markdown_v2(f"데이터셋 {payload.dataset_label} · {payload.timestamp_label}"),
        "",
        escape_markdown_v2(_grade_counts_line(payload.grade_counts)),
        escape_markdown_v2(_criteria_line(payload)),
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
    # H-3④: dataset_label은 사용자가 올린 CSV 원본 파일명이라 그대로
    # HTML에 꽂으면 <script>같은 태그를 주입할 수 있다 -- 여기서 만드는
    # 모든 동적 문자열에 이스케이프를 적용한다(다른 채널은 이미
    # Markdown/MarkdownV2 이스케이프가 있었는데 이 함수만 빠져 있었다).
    rows = "".join(
        f"<tr><td style='padding:6px 10px;font-weight:600'>{html.escape(title)}</td></tr>"
        f"<tr><td style='padding:0 10px 10px;color:#555;font-size:13px'>{html.escape(reason)}</td></tr>"
        for title, reason in _item_lines(payload)
    )
    remainder = _remainder_note(payload)
    remainder_html = f"<p style='color:#888;font-size:12px'>{html.escape(remainder)}</p>" if remainder else ""
    dashboard_html = (
        f"<p><a href='{html.escape(payload.dashboard_url)}'>대시보드에서 확인 →</a></p>"
        if payload.dashboard_url
        else ""
    )
    # EB그룹: 출처 한 줄도 사용자가 올린 파일명을 포함할 수 있으므로(예:
    # "[수동] <script>.csv") 다른 동적 문자열과 동일하게 반드시
    # html.escape를 거친다.
    source_html = (
        f"<p style='color:#b45309;font-weight:600;margin:0 0 10px'>{html.escape(payload.source_note)}</p>"
        if payload.source_note
        else ""
    )
    return f"""
    <div style="font-family:sans-serif;max-width:520px">
      {source_html}
      <h2 style="margin:0 0 4px">[SUNI] 알람 {payload.total}건 발생</h2>
      <p style="color:#555;margin:0 0 12px">데이터셋 {html.escape(payload.dataset_label)} &middot; {html.escape(payload.timestamp_label)}</p>
      <p style="font-weight:600">{html.escape(_grade_counts_line(payload.grade_counts))}</p>
      <p style="color:#555;font-size:13px;margin:0 0 12px">{html.escape(_criteria_line(payload))}</p>
      <table style="border-collapse:collapse;width:100%">{rows}</table>
      {remainder_html}
      <p style="margin-top:16px">분석 신뢰도 {html.escape(payload.reliability_grade)} ({payload.reliability_score}점)</p>
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
