"""Tests for src/notifications/senders.py -- message formatting only (no
live network calls). Covers spec §C-6: no absolute predicted-yield value,
only risk percentile; max 5 items listed with "외 N건" for the rest.
"""

from __future__ import annotations

from src.notifications.senders import (
    AlarmNotificationItem,
    AlarmNotificationPayload,
    build_slack_blocks,
    build_telegram_text,
    escape_markdown_v2,
    format_plain_summary,
)


def _payload(n_items: int) -> AlarmNotificationPayload:
    items = [
        AlarmNotificationItem(lot_wafer_id=f"L{i:03d}W01", risk_percentile=i * 0.1, grade="심각", reason="Step1_D1 = 14.0 (경고선 12.0 초과)")
        for i in range(n_items)
    ]
    return AlarmNotificationPayload(
        dataset_label="train.CSV",
        timestamp_label="2026-08-07 09:12",
        items=items,
        grade_counts={"심각": n_items},
        reliability_grade="높음",
        reliability_score=83,
        dashboard_url="https://example.com/alerts",
    )


def test_plain_summary_has_no_absolute_yield_value():
    text = format_plain_summary(_payload(2))
    # The message must only ever show risk percentile, never a yield number
    # -- structurally guaranteed since AlarmNotificationItem has no yield
    # field at all, but assert the rendered text follows the same rule.
    assert "하위 0.0%" in text
    assert "하위 0.1%" in text
    assert "수율" not in text


def test_plain_summary_caps_listed_items_at_five():
    text = format_plain_summary(_payload(8))
    assert "외 3건" in text
    assert text.count("경고선 12.0 초과") == 5


def test_plain_summary_no_remainder_note_when_five_or_fewer():
    text = format_plain_summary(_payload(3))
    assert "외" not in text


def test_slack_blocks_has_header_and_item_sections():
    blocks = build_slack_blocks(_payload(2))["blocks"]
    assert blocks[0]["type"] == "header"
    assert "알람 2건 발생" in blocks[0]["text"]["text"]
    item_blocks = [b for b in blocks if b["type"] == "section" and "L000W01" in b.get("text", {}).get("text", "")]
    assert len(item_blocks) == 1


def test_markdown_v2_escaping_covers_special_chars():
    escaped = escape_markdown_v2("Step1_D1 = 14.0 (경고선 12.0 초과)")
    for ch in "_()=.":
        assert f"\\{ch}" in escaped


def test_telegram_text_is_escaped_and_contains_grade_counts():
    text = build_telegram_text(_payload(1))
    assert "심각 1건" in text.replace("\\", "")
