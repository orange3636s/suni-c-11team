"""Tests for src/notifications/yield_update_senders.py (VE-2~VE-5) --
두 블록 조립, 20% 기여율 필터, 요약 문장 조건 분기, 채널별 형식(Slack
코드블록/Telegram 평문/Gmail 이스케이프)을 검증한다. LLM을 쓰지 않으므로
전부 결정적(deterministic)이다."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.analysis.yield_prediction import (
    CoreFactorCell,
    FallbackSummary,
    Recommendation,
    ReliabilityInfo,
    YieldCandidate,
    YieldPredictionTable,
    YieldSummary,
)
from src.notifications.yield_update_senders import (
    build_slack_yield_update,
    build_telegram_yield_update_text,
    build_yield_update_email_html,
    build_yield_update_payload,
)

FAIL_TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")


def _candidate(
    lot_wafer_id: str,
    y: float,
    reliability_count: int = 5,
    *,
    y1_feature: str | None = None,
    y1_contribution: float | None = None,
    y1_value: float = 1.0,
) -> YieldCandidate:
    core_factors = {t: CoreFactorCell(feature=None, contribution_pct=None, rank_used=None, factor_value=None) for t in FAIL_TARGETS}
    core_factors["Y1"] = CoreFactorCell(
        feature=y1_feature, contribution_pct=y1_contribution, rank_used=(1 if y1_feature else None), factor_value=42.0
    )
    y_components = {t: 1.0 for t in FAIL_TARGETS}
    y_components["Y1"] = y1_value
    return YieldCandidate(
        lot_wafer_id=lot_wafer_id,
        lot_id=None,
        y=y,
        y_components=y_components,
        cells={},
        core_factors=core_factors,
        reliability=ReliabilityInfo(count=reliability_count, measured=(), unmeasured=()),
        recommendation=Recommendation(text="", adjustable_targets=(), measurement_gap_targets=()),
    )


def _table(candidates: list[YieldCandidate]) -> YieldPredictionTable:
    primary_factors = {t: SimpleNamespace(feature=f"{t}PrimaryFactor") for t in FAIL_TARGETS}
    ys = [c.y for c in candidates]
    summary = YieldSummary(
        predicted_mean=(sum(ys) / len(ys)) if ys else 0.0,
        predicted_min=min(ys) if ys else 0.0,
        predicted_max=max(ys) if ys else 0.0,
        bottom_n=10,
        bottom_mean=(sum(sorted(ys)[:10]) / 10) if len(ys) >= 10 else None,
        judgeable_count=len(candidates),
        total_wafers=len(candidates),
        histogram=[],
        mode_loss=[],
    )
    return YieldPredictionTable(
        candidates=candidates,
        unmeasured_wafer_ids=[],
        total_wafers=len(candidates),
        fallback_summary=FallbackSummary(rank_counts={1: 0}, none_count=0, total_combinations=0),
        summary=summary,
        primary_factors=primary_factors,
    )


def test_top10_takes_first_ten_candidates_only():
    candidates = [_candidate(f"W{i:02d}", y=float(i)) for i in range(15)]
    table = _table(candidates)
    payload = build_yield_update_payload(table, dataset_label="test.csv", timestamp_label="14:20")
    assert len(payload.top10) == 10
    assert [item.lot_wafer_id for item in payload.top10] == [f"W{i:02d}" for i in range(10)]


def test_target_block_filters_by_contribution_threshold_and_sorts_desc():
    eligible_low = _candidate("W_LOW", y=50, y1_feature="Step18_R1", y1_contribution=25.0, y1_value=3.0)
    eligible_high = _candidate("W_HIGH", y=51, y1_feature="Step18_R1", y1_contribution=80.0, y1_value=9.0)
    ineligible = _candidate("W_INELIGIBLE", y=52, y1_feature="Step99_R1", y1_contribution=4.0, y1_value=99.0)
    table = _table([eligible_low, eligible_high, ineligible])
    payload = build_yield_update_payload(table, dataset_label="test.csv", timestamp_label="14:20")

    y1_block = next(b for b in payload.target_blocks if b.target == "Y1")
    assert y1_block.unavailable_reason is None
    assert [item.lot_wafer_id for item in y1_block.items] == ["W_HIGH", "W_LOW"]  # 값(불량률) 높은 순
    assert all(item.lot_wafer_id != "W_INELIGIBLE" for item in y1_block.items)


def test_target_block_reports_reason_when_no_wafer_qualifies():
    candidates = [_candidate("W1", y=50)]  # Y1 인자 없음(<20%)
    table = _table(candidates)
    payload = build_yield_update_payload(table, dataset_label="test.csv", timestamp_label="14:20")
    y1_block = next(b for b in payload.target_blocks if b.target == "Y1")
    assert y1_block.items == ()
    assert y1_block.unavailable_reason == "Y1PrimaryFactor이 계측된 wafer가 없습니다"


def test_summary_sentence_dominant_loss_mode():
    candidates = [_candidate(f"W{i}", y=50, y1_value=20.0) for i in range(10)]  # Y1이 나머지(각 1.0)보다 압도적
    payload = build_yield_update_payload(_table(candidates), dataset_label="d", timestamp_label="t")
    assert payload.summary_sentence is not None
    assert payload.summary_sentence.startswith("Y1 불량이 전체 손실의")


def test_summary_sentence_low_reliability_majority():
    candidates = [_candidate(f"W{i}", y=50, reliability_count=1, y1_value=1.0) for i in range(10)]
    payload = build_yield_update_payload(_table(candidates), dataset_label="d", timestamp_label="t")
    assert payload.summary_sentence == "상위 10건 중 10건이 핵심 인자 미계측 상태입니다."


def test_summary_sentence_all_reliable():
    candidates = [_candidate(f"W{i}", y=50, reliability_count=5, y1_value=1.0) for i in range(10)]
    payload = build_yield_update_payload(_table(candidates), dataset_label="d", timestamp_label="t")
    assert payload.summary_sentence == "상위 10건 모두 핵심 인자가 계측되어 있습니다."


def test_summary_sentence_omitted_when_no_condition_matches():
    """VE-5: 조건에 안 걸리면 억지로 문장을 만들지 않는다."""
    candidates = [
        _candidate("W0", y=50, reliability_count=2, y1_value=1.0),
        _candidate("W1", y=50, reliability_count=3, y1_value=1.0),
        _candidate("W2", y=50, reliability_count=4, y1_value=1.0),
    ]
    payload = build_yield_update_payload(_table(candidates), dataset_label="d", timestamp_label="t")
    assert payload.summary_sentence is None


def test_slack_message_wraps_tables_in_code_block():
    table = _table([_candidate("W1", y=50, y1_feature="Step18_R1", y1_contribution=80.0, y1_value=5.0)])
    payload = build_yield_update_payload(table, dataset_label="test.csv", timestamp_label="14:20", dashboard_url="https://dash.example/x")
    body = build_slack_yield_update(payload)
    texts = [el["text"] for block in body["blocks"] for el in ([block["text"]] if "text" in block else block.get("elements", []))]
    assert any("```" in t for t in texts)
    assert any("W1" in t for t in texts)


def test_telegram_message_is_plain_text_without_markdown_escaping():
    table = _table([_candidate("W1", y=50.5, y1_feature="Step18_R1", y1_contribution=80.0, y1_value=5.0)])
    payload = build_yield_update_payload(table, dataset_label="test.csv", timestamp_label="14:20")
    text = build_telegram_yield_update_text(payload)
    assert "\\." not in text  # MarkdownV2 이스케이프가 없어야 한다(VE-4: 일반 텍스트)
    assert "50.50%" in text
    assert "[SUNI] 수율 예측 갱신 (14:20)" in text


def test_gmail_html_escapes_untrusted_dataset_label():
    table = _table([_candidate("W1", y=50, y1_feature="Step18_R1", y1_contribution=80.0, y1_value=5.0)])
    payload = build_yield_update_payload(table, dataset_label="<script>alert(1)</script>.csv", timestamp_label="14:20")
    html_body = build_yield_update_email_html(payload)
    assert "<script>alert(1)</script>" not in html_body
    assert "&lt;script&gt;" in html_body


def test_gmail_html_includes_unavailable_reason():
    candidates = [_candidate("W1", y=50)]
    payload = build_yield_update_payload(_table(candidates), dataset_label="d", timestamp_label="t")
    html_body = build_yield_update_email_html(payload)
    assert "해당 없음" in html_body
