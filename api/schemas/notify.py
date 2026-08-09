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


class NotificationSettingsSummary(BaseModel):
    slack: SlackChannelSummary
    telegram: TelegramChannelSummary
    gmail: GmailChannelSummary
    conditions: NotificationConditions


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
