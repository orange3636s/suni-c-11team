"""SUNI 챗봇 질문 라우팅 골든 세트 (작업지시서 PART 3).

LLM 호출 없이 `_resolve_mode`/`_resolve_report_kind`/`_context_level`만
검사한다 -- 실제 답변 품질은 수동 점검(작업지시서 PART 3의 두 번째 표)
영역이라 여기서 단언하지 않는다.
"""

from __future__ import annotations

import pytest

from api.routes.chat import ChatRequest, _context_level, _resolve_mode, _resolve_report_kind

GOLDEN_CASES = [
    # (message, expected_mode, expected_report_kind_or_None, expected_level)
    ("Step28_R1이 왜 위험해?", "chat", None, "full"),
    ("L401W07 알람이 왜 떴어?", "chat", None, "full"),
    ("히트맵 회색 칸은 뭐야?", "chat", None, "none"),
    ("이 모델 믿을 만해?", "chat", None, "none"),
    ("어떤 모델로 학습해?", "chat", None, "none"),
    ("왜 상관계수를 안 써?", "chat", None, "none"),
    ("CMP가 뭐야?", "chat", None, "none"),
    ("Step18은 무슨 공정이야?", "chat", None, "none"),
    ("Y2 원인이 뭐야?", "chat", None, "digest"),
    ("모니터링 보고서 써줘", "report", "monitoring", "full"),
    ("트리맵 보고서 만들어줘", "report", "treemap", "full"),
    ("수율 예측 보고서", "report", "yield", "full"),
    ("보고서 써줘", "report", "rootcause", "full"),
    ("방금 답 요약해줘", "chat", None, "digest"),
    ("지금 뭐부터 조치해야 해?", "chat", None, "digest"),
]


@pytest.mark.parametrize("message, expected_mode, expected_kind, expected_level", GOLDEN_CASES)
def test_routing_golden_set(message, expected_mode, expected_kind, expected_level):
    request = ChatRequest(message=message, dataset="train")
    mode = _resolve_mode(request)
    assert mode == expected_mode, f"mode mismatch for {message!r}"

    level = _context_level(request, mode)
    assert level == expected_level, f"context level mismatch for {message!r}"

    if mode == "report":
        kind = _resolve_report_kind(request)
        assert kind == expected_kind, f"report_kind mismatch for {message!r}"


def test_report_kind_chip_overrides_keyword_regex():
    """프런트 칩이 report_kind를 명시하면, 메시지 텍스트가 다른 종류를
    암시하더라도 칩이 우선한다 -- 정규식은 타이핑 입력에 대한 폴백일
    뿐이다."""
    request = ChatRequest(message="트리맵 얘기는 없지만 보고서 줘", mode="report", report_kind="yield", dataset="train")
    assert _resolve_report_kind(request) == "yield"


def test_report_kind_defaults_to_rootcause_when_no_pattern_matches():
    request = ChatRequest(message="보고서 만들어줘", mode="report", dataset="train")
    assert _resolve_report_kind(request) == "rootcause"
