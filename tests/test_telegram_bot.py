"""Tests for src/notifications/telegram_bot.py's in-memory code exchange
(spec §C-3 Telegram: bots can't message a user first, so a /start-issued
code is the only way to learn a chat_id)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.notifications import telegram_bot


def test_register_and_resolve_code_roundtrip():
    code = telegram_bot.register_start("555", "@tester")
    resolved = telegram_bot.resolve_code(code)
    assert resolved == {"chat_id": "555", "username": "@tester"}


def test_resolve_code_is_single_use():
    code = telegram_bot.register_start("555", "@tester")
    assert telegram_bot.resolve_code(code) is not None
    assert telegram_bot.resolve_code(code) is None


def test_resolve_unknown_code_returns_none():
    assert telegram_bot.resolve_code("000000") is None


def test_expired_code_is_pruned(monkeypatch):
    code = telegram_bot.register_start("777", None)
    # Force it into the past without waiting 10 real minutes.
    telegram_bot._pending_codes[code].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert telegram_bot.resolve_code(code) is None
