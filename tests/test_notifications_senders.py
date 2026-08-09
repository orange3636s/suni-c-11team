"""Tests for src/notifications/senders.py -- message formatting only (no
live network calls). Covers spec §C-6: no absolute predicted-yield value,
only risk percentile; max 5 items listed with "외 N건" for the rest.
"""

from __future__ import annotations

from src.notifications.senders import (
    AlarmNotificationItem,
    AlarmNotificationPayload,
    build_email_html,
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


def test_email_html_escapes_user_supplied_dataset_label():
    """H-3④: dataset_label은 사용자가 올린 CSV 원본 파일명이다 -- 악의적인
    파일명(예: <script> 태그)이 메일 HTML에 그대로 렌더되면 안 된다."""
    payload = _payload(1)
    payload.dataset_label = "<script>alert(1)</script>.csv"
    html_body = build_email_html(payload)
    assert "<script>alert(1)</script>" not in html_body
    assert "&lt;script&gt;" in html_body


# -- EB그룹: 발송 출처(source_note) 표시 -------------------------------


def test_source_note_absent_by_default_in_all_channels():
    """source_note를 안 주면(SQL 자동 갱신 경로) 기존 메시지 형태가
    그대로 유지된다 -- 새 표시가 추가되지 않는다."""
    payload = _payload(1)
    assert payload.source_note is None
    blocks = build_slack_blocks(payload)["blocks"]
    assert blocks[0]["type"] == "header"
    assert "[데모]" not in str(blocks) and "[수동]" not in str(blocks)
    assert "[데모]" not in build_telegram_text(payload)
    assert "[데모]" not in build_email_html(payload)


def test_slack_blocks_prepend_source_note_context_block():
    payload = _payload(1)
    payload.source_note = "[데모] 내장 데이터 기준 — 실제 공정 데이터가 아닙니다"
    blocks = build_slack_blocks(payload)["blocks"]
    assert blocks[0]["type"] == "context"
    assert payload.source_note in blocks[0]["elements"][0]["text"]
    assert blocks[1]["type"] == "header"


def test_telegram_text_escapes_and_prepends_source_note():
    payload = _payload(1)
    payload.source_note = "[수동] uploaded_0809.csv 업로드 결과"
    text = build_telegram_text(payload)
    unescaped = text.replace("\\", "")
    # 볼드(*...*)로 감싸므로 정확히 맨 앞은 아니지만, [SUNI] 알람 줄보다는 앞에 와야 한다.
    assert "[수동] uploaded_0809.csv 업로드 결과" in unescaped
    assert unescaped.index("[수동] uploaded_0809.csv 업로드 결과") < unescaped.index("[SUNI]")


def test_email_html_escapes_source_note_filename():
    """EB-3: 수동 업로드 파일명이 그대로 출처 문구에 들어가므로, 다른
    동적 문자열과 동일하게 반드시 HTML 이스케이프를 거쳐야 한다."""
    payload = _payload(1)
    payload.source_note = "[수동] <script>alert(1)</script>.csv 업로드 결과"
    html_body = build_email_html(payload)
    assert "<script>alert(1)</script>" not in html_body
    assert "&lt;script&gt;" in html_body
