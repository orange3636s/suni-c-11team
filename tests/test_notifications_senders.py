"""Tests for src/notifications/senders.py -- this module is now pure
channel transport (Slack webhook / Telegram sendMessage / Gmail SMTP).
Message assembly moved entirely to
src/notifications/yield_update_senders.py (see tests/test_yield_update_senders.py
for that coverage) after the old alarm-grade notification pipeline was
retired. The only pure/testable-without-network logic left here is the
Telegram MarkdownV2 escaper used by `send_telegram_test`.
"""

from __future__ import annotations

from src.notifications.senders import escape_markdown_v2


def test_markdown_v2_escaping_covers_special_chars():
    escaped = escape_markdown_v2("Step1_D1 = 14.0 (테스트 12.0 초과)")
    for ch in "_()=.":
        assert f"\\{ch}" in escaped
