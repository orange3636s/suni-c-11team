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
    timing: str


class NotificationSettingsSummary(BaseModel):
    slack: SlackChannelSummary
    telegram: TelegramChannelSummary
    gmail: GmailChannelSummary
    conditions: NotificationConditions


class SlackConnectRequest(BaseModel):
    webhook_url: str
    channel: str | None = None


class SlackTestRequest(BaseModel):
    webhook_url: str


class TelegramVerifyRequest(BaseModel):
    code: str


class GmailConnectRequest(BaseModel):
    email: str


class ConditionsSaveRequest(BaseModel):
    grades: list[str]
    timing: str


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
