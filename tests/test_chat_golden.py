"""SUNI 챗봇 프롬프트 조립 골든 테스트 (지시서 A-5).

LLM 응답 자체는 비결정적이라 단언하지 않는다. 여기서는 `_build_messages`/
`_context_level`이 조립하는 메시지 구조와, `prompts/chat_system.md`의
정적인 내용만 검증한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from api.routes import chat as chat_module
from api.routes.chat import (
    HISTORY_TURNS,
    REPORT_KEYWORDS,
    ChatHistoryTurn,
    ChatRequest,
    _build_messages,
    _chat_system_prompt,
    _context_level,
    _report_system_prompt,
    _resolve_mode,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _fake_report_payload(dataset: str) -> dict:
    return {
        "summary": {
            "targets_analyzed": 5,
            "factors_included": 3,
            "excluded_low_significance": 2,
            "alarm_wafers": 4,
            "normal_wafers": 90,
            "undecidable_wafers": 6,
            "mean_yield_alarm": 80.0,
            "mean_yield_normal": 90.0,
            "yield_gap_pp": 10.0,
        },
        "targets": [
            {
                "target": "Y1",
                "target_stats": {"mean": 1.0, "std": 0.5, "q1": 0.5, "q3": 1.5},
                "factors": [
                    {
                        "feature": "Step28_R1",
                        "kind": "R",
                        "step": 28,
                        "rank": 1,
                        "adj_r2": 0.159,
                        "degree": 2,
                        "contribution_pct": 40.0,
                        "cumulative_pct": 40.0,
                        "p_value": 0.0001,
                        "q_value": 0.0002,
                        "grade": "강함",
                        "report_confidence": "강함",
                        "n_observed": 100,
                        "n_missing_pct": 10.0,
                        "relation": {
                            "shape": "monotonic_increasing",
                            "optimal_center": None,
                            "interpretation": "값이 클수록 불량률이 상승한다.",
                        },
                        "control_limits": {
                            "lcl": 1.0,
                            "ucl": 5.0,
                            "one_sided": False,
                            "mean": 3.0,
                            "std": 1.0,
                            "q1": 2.0,
                            "q3": 4.0,
                            "sigma3": [1.0, 5.0],
                            "sigma6": [0.0, 6.0],
                            "sigma6_drawn": True,
                        },
                        "band_stability": 0.1,
                        "band_width": 4.0,
                        "window": {
                            "lo": 1.0,
                            "hi": 3.0,
                            "mean_in_window": 1.0,
                            "mean_overall": 2.0,
                            "ratio": 0.5,
                            "n_in_window": 50,
                        },
                        "chamber_interaction": False,
                        "chamber_interaction_p": None,
                        "chamber_interaction_q": None,
                        "per_chamber_window": None,
                        "eval_result": {"alarms": 2, "observed": 50, "mean_y_alarm": 80.0, "mean_y_normal": 90.0},
                    }
                ],
            },
            {"target": "Y2", "target_stats": {"mean": 2.0, "std": 1.0, "q1": 1.0, "q3": 3.0}, "factors": []},
        ],
        "alarms": [],
        "config_screening": {
            "n_tested": 600,
            "n_significant_fdr": 0,
            "max_observed_adj_r2": 0.01,
            "max_observed_feature": None,
            "max_observed_target": None,
            "mde_adj_r2": 0.02,
            "median_n_per_group": 10,
        },
        "limitations": ["한계 문장 1", "한계 문장 2"],
        "meta": {
            "target_provenance": {
                "train": {"uses_predictions": False},
                "eval": {"uses_predictions": False},
            }
        },
    }


@pytest.fixture(autouse=True)
def _patch_report_payload(monkeypatch):
    # _grounding_block/_digest_context가 build_analysis_report 전체 파이프라인
    # (pandas 데이터셋 로드 등)을 타지 않도록, chat.py가 유일하게 의존하는
    # 진입점(_build_report_payload)만 가짜로 바꾼다.
    monkeypatch.setattr(chat_module, "_build_report_payload", _fake_report_payload)


def _history(n_turns: int) -> list[ChatHistoryTurn]:
    turns: list[ChatHistoryTurn] = []
    for i in range(n_turns):
        turns.append(ChatHistoryTurn(role="user", content=f"질문 {i}"))
        turns.append(ChatHistoryTurn(role="assistant", content=f"답변 {i}"))
    return turns


def test_grounding_survives_six_turns():
    """A-1: history가 12개(6턴)여도 system 메시지에 근거 JSON 블록이
    여전히 존재해야 한다 -- 예전에는 user 메시지 슬라이딩 윈도우 밖으로
    밀려나 4턴째부터 사라졌다."""
    request = ChatRequest(message="Step28_R1은 어떤가요?", dataset="train", history=_history(6))
    messages = _build_messages(request, "chat")

    system_messages = [m for m in messages if m["role"] == "system"]
    assert len(system_messages) == 2, "system 프롬프트 + 근거 블록, 두 개여야 한다"
    grounding = system_messages[1]["content"]
    assert "Step28_R1" in grounding
    assert '"summary"' in grounding or "summary" in grounding


def test_history_window_matches_frontend():
    """두 창이 어긋나면 한쪽만 고친 게 된다 -- HISTORY_TURNS*2(백엔드)와
    AiPanel.tsx의 HISTORY_MESSAGES(프런트)가 반드시 같은 값이어야 한다."""
    ai_panel_path = (
        Path(__file__).resolve().parents[1] / "frontend" / "components" / "ai-panel" / "AiPanel.tsx"
    )
    source = ai_panel_path.read_text(encoding="utf-8")
    match = re.search(r"HISTORY_MESSAGES\s*=\s*(\d+)", source)
    assert match is not None, "AiPanel.tsx에서 HISTORY_MESSAGES 상수를 찾지 못했습니다"
    frontend_window = int(match.group(1))
    assert HISTORY_TURNS * 2 == frontend_window


def test_report_mode_keeps_history():
    """A-2: report 모드도 history를 반영해야 한다 -- 예전에는 report
    분기가 history를 아예 읽지 않아 보고서 후속 질문이 새 대화 취급됐다."""
    request = ChatRequest(message="이 결과 더 설명해줘", dataset="train", history=_history(2))
    messages = _build_messages(request, "report")

    roles_and_content = [(m["role"], m["content"]) for m in messages]
    assert any(role == "user" and content == "질문 0" for role, content in roles_and_content)
    assert any(role == "assistant" and content == "답변 1" for role, content in roles_and_content)


def test_report_keywords_exclude_generic():
    """A-2: "요약해줘"·"정리해줘"는 후속 질문에도 흔히 쓰이는 일반 동사라
    report로 오분류되면 안 된다."""
    assert "요약해줘" not in REPORT_KEYWORDS
    assert "정리해줘" not in REPORT_KEYWORDS
    assert _resolve_mode(ChatRequest(message="방금 답 요약해줘", dataset="train")) == "chat"
    assert _resolve_mode(ChatRequest(message="정리해줘", dataset="train")) == "chat"
    assert _resolve_mode(ChatRequest(message="보고서 만들어줘", dataset="train")) == "report"


def test_context_level_none_for_ui_question():
    request = ChatRequest(message="이 화면은 무엇을 보여주나요?", dataset="train")
    assert _context_level(request, "chat") == "none"


def test_context_level_full_for_factor_question():
    request = ChatRequest(message="Step28_R1은 어떤가요?", dataset="train")
    assert _context_level(request, "chat") == "full"

    wafer_request = ChatRequest(message="L401W07 알람이 왜 떴어?", dataset="train")
    assert _context_level(wafer_request, "chat") == "full"

    report_request = ChatRequest(message="아무 말", dataset="train")
    assert _context_level(report_request, "report") == "full"


def test_context_level_defaults_to_digest():
    request = ChatRequest(message="가장 신뢰도 높은 인자는?", dataset="train")
    assert _context_level(request, "chat") == "digest"


# A-3: 폐기된 알람 등급(심각/위험/주의/정상/미분류) 판정 체계와 그 상수는
# src/analysis/alarm_gbdt.py 자체가 "전부 폐기됐다"고 명시하고, 어떤
# 컴포넌트도 그 판정을 렌더링하지 않는다(확인: frontend 전체에서 민감도/
# 목표 수율 슬라이더 UI가 존재하지 않는다) -- 이 문구들이 프롬프트에
# 남아 있으면 챗봇이 죽은 기능을 근거로 답한다.
_DEAD_TERMS = ("미분류", "게이트 미달", "4.0%p", "0.8%p", "판정 컷")


def test_prompt_has_no_dead_features():
    prompt = _chat_system_prompt()
    for term in _DEAD_TERMS:
        assert term not in prompt, f"폐기된 판정 체계 문구가 남아 있습니다: {term!r}"


def test_report_prompt_has_no_dead_features():
    """report_system.md는 chat_system.md와 별도 파일이라 위 검사가 커버하지
    않는다 -- 한때 여기만 목표 수율·민감도 절대 컷·미분류 두 사유 체계를
    시스템 상수처럼 서술하고 있었다(챗봇 프롬프트와 서로 다른 화면
    설명을 갖는 회귀였다). 같은 죽은 용어 목록을 여기도 강제한다."""
    prompt = _report_system_prompt()
    for term in _DEAD_TERMS:
        assert term not in prompt, f"폐기된 판정 체계 문구가 남아 있습니다: {term!r}"


_EXPECTED_SCREENS = ("모니터링 홈", "원인 분석", "수율 예측", "Config별 트리맵", "알림 기록", "즐겨찾기")


def test_prompt_covers_current_screens():
    prompt = _chat_system_prompt()
    for screen in _EXPECTED_SCREENS:
        assert screen in prompt, f"화면 안내에 {screen!r}이 빠져 있습니다"
