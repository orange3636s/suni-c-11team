from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class SlackChannelSummary(BaseModel):
    connected: bool
    target: str | None
    webhook_masked: str | None
    verified_at: str | None


class TelegramChannelSummary(BaseModel):
    connected: bool
    target: str | None
    chat_id_masked: str | None
    verified_at: str | None


class GmailChannelSummary(BaseModel):
    connected: bool
    pending: bool
    email: str | None
    verified_at: str | None


class NotificationConditions(BaseModel):
    grades: list[str]
    timing: list[str]


class AutomationSettingsSummary(BaseModel):
    """SD-1: "알림·자동화 설정" 팝업의 자동화 섹션. 비밀번호는 절대
    포함하지 않는다 -- 환경변수(DB_PASSWORD)로만 받는다."""

    enabled: bool
    sql_host: str
    sql_port: str
    sql_db: str
    sql_user: str
    refresh_interval_minutes: int
    last_run_at: str | None = None
    last_run_status: str | None = None
    last_run_sent_count: int | None = None


class NotificationSettingsSummary(BaseModel):
    slack: SlackChannelSummary
    telegram: TelegramChannelSummary
    gmail: GmailChannelSummary
    conditions: NotificationConditions
    automation: AutomationSettingsSummary
    # EA그룹: 텔레그램 봇 username 단일 소스 -- 토큰은 절대 포함하지 않는다.
    # 미설정이면 null.
    telegram_bot_username: str | None = None


class SlackConnectRequest(BaseModel):
    webhook_url: str
    channel: str | None = None


class SlackTestRequest(BaseModel):
    # D-3: 이미 연결된 채널을 테스트할 때는 생략한다 -- 연결 요약에는
    # 마스킹된 webhook_masked만 있고 원본 URL이 없으므로, 프런트가 다시
    # 보낼 방법이 없다. 생략하면 서버가 저장된 값을 쓴다.
    webhook_url: str | None = None


class TelegramVerifyRequest(BaseModel):
    code: str


class GmailConnectRequest(BaseModel):
    email: str


class ConditionsSaveRequest(BaseModel):
    grades: list[str]
    timing: list[str]


class SendTestResponse(BaseModel):
    ok: bool
    error: str | None = None


class DispatchRequest(BaseModel):
    train_dataset: str
    eval_dataset: str
    dashboard_url: str | None = None


class DispatchResponse(BaseModel):
    skipped: bool
    reason: str | None = None
    sent_count: int | None = None
    results: dict[str, Any] | None = None


class AutomationSaveRequest(BaseModel):
    """SD-1: 비밀번호 필드가 없다 -- 서버 환경변수(DB_PASSWORD)로만 받는다."""

    enabled: bool
    sql_host: str = ""
    sql_port: str = ""
    sql_db: str = ""
    sql_user: str = ""
    refresh_interval_minutes: int = 60


class AutomationTestResponse(BaseModel):
    ok: bool
    error: str | None = None


class NotifyHistoryItem(BaseModel):
    id: int
    sent_at: str
    trigger: str
    channels: list[str]
    dataset_label: str | None
    model_version: str | None
    status: Literal["sent", "skipped"]
    skip_reason: str | None
    message_text: str | None
    sent_count: int


class NotifyHistoryListResponse(BaseModel):
    items: list[NotifyHistoryItem]
