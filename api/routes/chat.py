"""SUNI chatbot endpoint: streams an Upstage (OpenAI-compatible) completion
over SSE, grounded in the same JSON build_analysis_report already produces
for /api/analysis/report and /api/analysis/context (see analysis.py's
_build_report_payload docstring). Two system prompts (prompts/*.md), picked
by a plain keyword check -- no LLM round-trip for intent classification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import APIError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.routes.analysis import (
    REPORT_EVAL_DATASET_ID,
    _action_priority_payload,
    _build_report_payload,
    _fmea_payload,
    _hydrated_targets_or_409,
    get_alerts_ranking,
)
from api.routes.datasets import get_dataset_registry
from api.settings import settings
from src.analysis.report import build_chat_context
from src.analysis.rounding import round_floats
from src.analysis.sampling import ANALYSIS_SAMPLE_MAX_ROWS, stratified_sample
from src.analysis.screening.schema import ALL_TARGET_COLUMNS, parse_schema
from src.analysis.screening.selector import DEFAULT_MIN_N_CATEGORICAL
from src.runtime.datasets import DatasetNotFoundError

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
# "요약해줘"·"정리해줘"는 뺐다 -- 후속 질문에서도 흔히 쓰이는 일반 동사라
# ("방금 답 요약해줘") report 모드로 잘못 분류돼 매번 보고서를 처음부터
# 다시 쓰는 원인이었다. 프런트의 REPORT_KEYWORD_PATTERN(AiPanel.tsx)과
# 반드시 같은 키워드 집합을 유지해야 한다.
REPORT_KEYWORDS = ("보고서", "리포트", "report")
CHAT_TIMEOUT_SECONDS = 90
# 근거 JSON을 system 메시지로 옮긴 뒤(_grounding_block)로는 history가 순수
# 대화용이라 늘려도 근거가 밀려날 위험이 없다. 프런트의
# AiPanel.tsx#HISTORY_MESSAGES(= 이 값 * 2)와 반드시 같은 턴수를 봐야 한다 --
# 한쪽만 고치면 둘 중 더 짧은 창이 실효 창이 되어 이 파일의 상수를 바꾼
# 의미가 없어진다.
HISTORY_TURNS = 6
MAX_RETRIES = 1
# 프런트의 idle 타임아웃(frontend/lib/api.ts의 CHAT_STREAM_IDLE_TIMEOUT_MS,
# 30초)보다 훨씬 짧게 잡는다 -- 컨텍스트 빌드가 이보다 오래 걸리는 동안
# 하트비트가 안 나가면 그 자체가 idle로 잡혀 끊긴다.
HEARTBEAT_INTERVAL_SECONDS = 5
NO_ANALYSIS_MESSAGE = (
    "원인 분석을 먼저 실행해 주세요. 원인 분석 탭에서 실행 버튼을 누르면 "
    "분석 결과를 바탕으로 보고서를 작성할 수 있습니다.\n\n[원인 분석 탭으로 이동](/root-cause)"
)

# 질문 종류와 무관하게 항상 전체 JSON을 붙이면(수 KB~수십 KB) 매
# 메시지가 그만큼의 토큰을 태운다. 구체적인 인자·wafer를 지목한 질문만
# 전체(`full`)를 받고, 나머지는 요약(`digest`)이나 근거 없음(`none`)으로
# 낮춘다. ContextLevel 판정은 키워드 매칭이 아니라 인자명/wafer ID
# 정규식과 report 모드 여부로만 한다 -- "요약해줘" 같은 일반 동사를
# 키워드로 잡으면 평범한 질문이 그대로 오분류된다.
ContextLevel = Literal["none", "digest", "full"]

FACTOR_ID_PATTERN = re.compile(r"Step\d+_(?:[RD]\d+|Config)", re.IGNORECASE)
WAFER_ID_PATTERN = re.compile(r"L\d+W\d+", re.IGNORECASE)

# 화면·기능·컨트롤의 동작 원리를 묻는 질문 -- 데이터 해석이 아니라
# prompts/chat_system.md의 "## 대시보드 기능 안내"/"## 예측 구간과 판정
# 체계" 절만으로 답이 나오므로 근거 JSON이 전혀 필요 없다. 오탐의 비용이
# 비대칭이다: 기능 질문을 여기서 놓쳐 digest로 흘려보내도 답의 품질은
# 그대로다(그냥 JSON을 조금 더 보낼 뿐)이지만, 데이터 질문을 여기 잘못
# 걸리면 근거 없이 답하게 된다 -- 그래서 애매하면 이 목록에 넣지 않는다.
UI_QUESTION_PATTERN = re.compile(
    r"화면|기능|탭에서|탭은|컨트롤|자동\s*갱신|자동화는|리프레시\s*주기|승격\s*게이트|"
    r"언제\s*발송|발송\s*이력|이미지\s*저장|저장\s*버튼|즐겨찾기(는|가|란)?|"
    r"Color\s*by|그룹\s*강조|SPC.{0,6}ML|민감도(를|는)?\s*(올리면|낮추면|조절)|"
    # 시각화를 "어떻게 읽는가"를 묻는 질문 -- chat_system.md의 "무엇을
    # 알고 싶은가 → 어떤 화면을 보는가"/"각 시각화를 읽을 때" 절만으로
    # 답이 나온다(이번 분석의 구체적인 값이 아니라 화면 자체의 해석
    # 규칙을 묻는 것이므로, "회색 셀은 뭐야" 같은 질문에 데이터를 함께
    # 실어 보낼 필요가 없다).
    r"히트맵|회색\s*(셀|칸)|파레토|박스플롯|Box\s*Plot|IQR"
)

# 계산 방식·검증 성적을 묻는 질문 -- chat_system.md의 "## 분석 방법"/
# "## 검증 성적" 절만으로 답이 나온다(분석 실행마다 바뀌지 않는 시스템
# 상수라 근거 JSON이 필요 없다). UI_QUESTION_PATTERN과 같은 이유로 오탐
# 비용이 비대칭이다 -- 애매하면 여기 넣지 않는다.
METHOD_QUESTION_PATTERN = re.compile(
    r"어떻게\s*(계산|학습|판단|고르|뽑)|무슨\s*모델|어떤\s*모델|LightGBM|"
    r"정확도|믿을\s*만|신뢰(할|도가|성)|검증|성능(이|은)|왜\s*상관계수|"
    r"표본\s*(수|기준|게이트)|FDR|부트스트랩"
)

# 일반 반도체 공정 지식 질문 -- 우리 데이터와 무관하다(chat_system.md의
# "## 반도체 도메인 지식" 절만으로 답이 나온다). "(이|가)\s*뭐(야|예요|인가)"
# 같은 범용 캐치올은 일부러 넣지 않는다 -- "Y2 원인이 뭐야?"처럼 실제로는
# JSON 근거가 필요한 데이터 질문까지 걸려버려서, 애매하면 이 목록에
# 넣지 않는 편이 안전하다(이 파일 상단 주석의 비대칭 위험 원칙).
DOMAIN_QUESTION_PATTERN = re.compile(
    r"CMP|포토(리소)?|식각|증착|이온\s*주입|오버레이|Cpk?\b|SPC(가|는|란)|"
    r"FDC|FMEA|웨이퍼(란|가\s*뭐)|로트(란|가\s*뭐)|챔버(란|가\s*뭐)|"
    r"무슨\s*공정|어떤\s*공정|공정(이|은)\s*(뭐|무엇)"
)


def _context_level(request: ChatRequest, mode: Literal["report", "chat"]) -> ContextLevel:
    if mode == "report":
        return "full"
    # 인자·wafer ID가 있으면 무조건 데이터 질문이다 -- 아래 다른 패턴보다
    # 먼저 본다. 데이터 질문을 "none"으로 잘못 보내면 근거 없이 답하게
    # 되는 반면, 기능/방법론 질문을 "digest"로 흘려보내는 것은 그저 JSON을
    # 조금 더 보내는 것뿐이라 위험이 비대칭이다.
    if FACTOR_ID_PATTERN.search(request.message) or WAFER_ID_PATTERN.search(request.message):
        return "full"
    if UI_QUESTION_PATTERN.search(request.message):
        return "none"
    if METHOD_QUESTION_PATTERN.search(request.message):
        return "none"
    if DOMAIN_QUESTION_PATTERN.search(request.message):
        return "none"
    return "digest"


ReportKind = Literal["rootcause", "monitoring", "treemap", "yield"]

# 명시적 report_kind가 없을 때(칩이 아니라 타이핑으로 들어온 요청)만
# 쓰는 폴백 -- 어디에도 안 걸리면 기존 동작 그대로 "rootcause"다.
REPORT_KIND_PATTERNS: tuple[tuple[ReportKind, "re.Pattern[str]"], ...] = (
    ("monitoring", re.compile(r"모니터링|홈\s*(화면|보고서)|우선순위|한계\s*진단")),
    ("treemap", re.compile(r"트리맵|Config\s*별|장비\s*별|레시피\s*별")),
    ("yield", re.compile(r"수율\s*예측|예측\s*순위|우선\s*검토\s*웨이퍼")),
)


def _resolve_report_kind(request: ChatRequest) -> ReportKind:
    if request.report_kind is not None:
        return request.report_kind
    for kind, pattern in REPORT_KIND_PATTERNS:
        if pattern.search(request.message):
            return kind
    return "rootcause"


# digest에 남기는 인자 필드 -- 타깃별 1위 인자를 "이름·등급·설명력·관계
# 형태" 정도로만 요약한다. `_text` 형제 필드가 있으면 함께 남겨 LLM이
# 반올림을 다시 하지 않게 한다.
_DIGEST_FACTOR_KEYS = (
    "feature",
    "kind",
    "step",
    "grade",
    "grade_text",
    "report_confidence",
    "report_confidence_text",
    "adj_r2",
    "adj_r2_text",
    "degree",
    "p_value",
    "p_value_text",
    "relation",
    # digest는 알람 레코드를 전혀 싣지 않으므로(요약만), "지금 뭐부터
    # 조치해야 해?" 같은 일반 우선순위 질문은 타깃별 1위 인자에 붙은
    # 이 필드가 유일한 근거다 -- 없으면(build_chat_context가 게이트를
    # 통과시키지 못했으면) 이 키 자체가 없다.
    "action",
)


def _digest_factor(factor: dict[str, Any]) -> dict[str, Any]:
    return {key: factor[key] for key in _DIGEST_FACTOR_KEYS if key in factor}


def _digest_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": context["summary"],
        "config_screening": context["config_screening"],
        "limitations": context["limitations"],
        "targets": [
            {
                "target": target_entry["target"],
                "top_factor": _digest_factor(target_entry["factors"][0]) if target_entry["factors"] else None,
            }
            for target_entry in context["targets"]
        ],
    }


_REPORT_PROMPT_FILES: dict[ReportKind, str] = {
    "rootcause": "report_rootcause.md",
    "monitoring": "report_monitoring.md",
    "treemap": "report_treemap.md",
    "yield": "report_yield.md",
}


@lru_cache(maxsize=len(_REPORT_PROMPT_FILES))
def _report_prompt(kind: ReportKind) -> str:
    return (PROMPTS_DIR / _REPORT_PROMPT_FILES[kind]).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _chat_system_prompt() -> str:
    return (PROMPTS_DIR / "chat_system.md").read_text(encoding="utf-8")


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    mode: Literal["report", "chat"] | None = None
    # 프런트가 보고서 칩으로 명시하면 이 값을 쓰고, 없으면(타이핑)
    # REPORT_KIND_PATTERNS로 폴백한다 -- _resolve_report_kind 참고.
    report_kind: ReportKind | None = None
    dataset: str = "train"
    history: list[ChatHistoryTurn] = Field(default_factory=list)


def _resolve_mode(request: ChatRequest) -> Literal["report", "chat"]:
    if request.mode in ("report", "chat"):
        return request.mode
    return "report" if any(keyword in request.message for keyword in REPORT_KEYWORDS) else "chat"


TREEMAP_TOP_STEPS_PER_TARGET = 3
TREEMAP_TOP_GROUPS_PER_STEP = 5
YIELD_CANDIDATE_TRUNCATE_KEEP = 50
_CONFIG_STEP_PATTERN = re.compile(r"Step(\d+)_Config$")


def _action_priority_rows(train_dataset_id: str) -> list[dict[str, Any]] | None:
    """Thin, separately-mockable wrapper around `_action_priority_payload`
    -- tests patch this directly instead of exercising the real
    pandas/FMEA pipeline (see test_chat_golden.py's `_patch_report_payload`
    fixture)."""
    return _action_priority_payload(train_dataset_id)["rows"]


def _rootcause_context(dataset: str) -> dict[str, Any]:
    """근거등급 [측정]의 원천 -- build_analysis_report/build_chat_context가
    이미 만드는 원인 분석 JSON에, 모니터링 홈의 조치 우선순위 표
    (`_action_priority_payload`)를 얹어 알람 레코드·타깃 1위 인자에
    `action`을 붙인다(report.py의 `_build_action` 참고). 조치 우선순위
    계산이 실패해도 원인 분석 자체는 계속 답할 수 있어야 하므로
    action_rows만 None으로 낮추고 계속 진행한다.
    """
    try:
        action_rows = _action_priority_rows("train")
    except Exception:  # noqa: BLE001 -- action 없이도 근거 JSON은 계속 구성해야 한다
        logger.exception("조치 우선순위 계산에 실패해 action 없이 근거를 구성합니다.")
        action_rows = None
    return build_chat_context(_build_report_payload(dataset), action_rows=action_rows)


def _monitoring_chat_payload(dataset: str) -> dict[str, Any]:
    """report_monitoring.md의 유일한 근거 -- 모니터링 홈 블록①(조치
    우선순위)·블록③(데이터 한계 진단)을 그대로 재사용한다(둘 다
    api/routes/analysis.py가 /api/state 저장 경로와 공유하는 순수 함수).
    새 계산을 하지 않는다.
    """
    return {
        "action_priority": _action_priority_payload("train"),
        "data_limitations": _fmea_payload(dataset, ALL_TARGET_COLUMNS),
    }


def _treemap_chat_payload(dataset: str) -> dict[str, Any]:
    """report_treemap.md의 유일한 근거 -- Config별 트리맵 화면
    (api/routes/monitoring.py#get_config_treemap)과 같은 그룹 통계를
    직접 집계한다. 그 라우트를 스텝마다 그대로 재호출하지 않는 이유는
    그 함수의 유의성 검정(`_is_config_significant`)이 매 호출마다 전체
    스텝을 다시 훑어(스텝수 × 스텝수) 30스텝 × 5타깃 조합에서는 비용이
    과하기 때문이다 -- 이 보고서는 유의성 배지 없이 순수 평균 격차만
    보여주고, 그 사실을 프롬프트가 한계로 명시한다.
    """
    hydrated = _hydrated_targets_or_409(dataset)
    schema = parse_schema(hydrated.dataframe)
    df, _sample_info = stratified_sample(
        hydrated.dataframe, max_rows=ANALYSIS_SAMPLE_MAX_ROWS, dataset_version=hydrated.provenance.dataset_version
    )

    targets_out = []
    for target in schema.target_cols:
        steps: list[dict[str, Any]] = []
        for config_col in schema.config_cols:
            valid = df[[config_col, target]].dropna()
            if valid.empty:
                continue
            grouped = valid.groupby(config_col)[target].agg(["mean", "count"]).reset_index()
            if grouped.empty:
                continue
            match = _CONFIG_STEP_PATTERN.match(config_col)
            groups_sorted = grouped.sort_values("mean", ascending=False)
            steps.append(
                {
                    "step": int(match.group(1)) if match else None,
                    "config_col": config_col,
                    "gap": float(grouped["mean"].max() - grouped["mean"].min()),
                    "n_groups": int(len(grouped)),
                    "n_small_sample_groups": int((grouped["count"] < DEFAULT_MIN_N_CATEGORICAL).sum()),
                    "top_groups": [
                        {
                            "config": str(row[config_col]),
                            "n": int(row["count"]),
                            "mean": float(row["mean"]),
                            "small_sample": bool(row["count"] < DEFAULT_MIN_N_CATEGORICAL),
                        }
                        for _, row in groups_sorted.head(TREEMAP_TOP_GROUPS_PER_STEP).iterrows()
                    ],
                }
            )
        steps.sort(key=lambda s: s["gap"], reverse=True)
        targets_out.append({"target": target, "steps": steps[:TREEMAP_TOP_STEPS_PER_TARGET]})

    return round_floats({"dataset": dataset, "min_n_categorical": DEFAULT_MIN_N_CATEGORICAL, "targets": targets_out})


def _yield_chat_payload(dataset: str) -> dict[str, Any]:
    """report_yield.md의 유일한 근거 -- /api/alerts/ranking과 동일한 함수를
    그대로 호출해 화면 수치와 절대 어긋나지 않게 한다. train은 채팅이
    보고 있는 `dataset`(보고서 모드에서는 항상 학습 데이터셋 선택을
    가리킨다), eval은 원인 분석 보고서와 같은 고정값을 쓴다.
    """
    ranking = get_alerts_ranking(train=dataset, eval=REPORT_EVAL_DATASET_ID)
    candidates = ranking.get("candidates", [])
    truncated = len(candidates) > YIELD_CANDIDATE_TRUNCATE_KEEP
    # `cells`는 화면의 색상 타일용 필드(direction/shade)라 텍스트 보고서엔
    # 쓸모가 없고, 값 대부분이 `core_factors`와 겹친다 -- 근거 JSON
    # 크기를 절반 가까이 줄이려고 뺀다(report_yield.md는 core_factors만
    # 문서화한다).
    trimmed_candidates = [{k: v for k, v in c.items() if k != "cells"} for c in candidates[:YIELD_CANDIDATE_TRUNCATE_KEEP]]
    return round_floats(
        {
            "total_wafers": ranking.get("total_wafers"),
            "candidates": trimmed_candidates,
            "candidates_truncated": truncated,
            "candidates_total": len(candidates),
            "unmeasured_count": ranking.get("unmeasured_count"),
            "fallback_summary": ranking.get("fallback_summary"),
        }
    )


def _grounding_payload(dataset: str, mode: Literal["report", "chat"], kind: ReportKind | None) -> dict[str, Any]:
    # report_kind는 report 모드에서만 의미가 있다(chat 모드는 항상
    # 원인 분석 JSON 하나로 답한다 -- 보고서 4종은 report 모드 전용).
    if mode == "report" and kind == "monitoring":
        return _monitoring_chat_payload(dataset)
    if mode == "report" and kind == "treemap":
        return _treemap_chat_payload(dataset)
    if mode == "report" and kind == "yield":
        return _yield_chat_payload(dataset)
    return _rootcause_context(dataset)


def _grounding_block(dataset: str, level: ContextLevel, mode: Literal["report", "chat"], kind: ReportKind | None) -> str:
    # Grouped {summary, records} alarms (see build_chat_context's
    # docstring) -- individual-wafer questions ("L401W07 알람이 왜 떴어?")
    # need the record-level data, not just the aggregate counts a raw
    # report payload would give.
    #
    # This block must be sent as a `system` message, not a `user` one: the
    # frontend/backend trim history to the last few turns, so a user-role
    # block would drop out of that sliding window after a few turns and
    # leave the model answering ungrounded. System messages are never
    # subject to that window, so it survives every turn.
    context = _grounding_payload(dataset, mode, kind)
    # 원인 분석(rootcause) 외 세 보고서는 항상 report 모드로만 진입하므로
    # level은 언제나 "full"이다 -- _digest_context는 rootcause 구조
    # (summary/config_screening/limitations/targets)를 가정하므로 다른
    # 종류의 payload에는 적용하지 않는다.
    payload = context if level == "full" else _digest_context(context)
    intro = (
        "아래는 이번 분석의 결과 요약 JSON이다(타깃별 1위 인자만 포함)."
        if level == "digest"
        else "아래는 이번 분석의 결과 전체 JSON이다."
    )
    return (
        f"{intro} 이후 사용자의 모든 질문에 이 JSON만을 근거로 답한다. "
        "JSON에 없는 수치나 사실을 만들어내지 않는다.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    )


def _build_messages(request: ChatRequest, mode: Literal["report", "chat"]) -> list[dict[str, str]]:
    # report 모드도 일반 chat과 같은 경로를 탄다 -- 질문 종류에 따라
    # 갈라지는 것은 system 프롬프트뿐이고 history는 두 모드 모두 읽는다.
    # 그래야 이미 만들어진 보고서에 대한 후속 질문("Y2는 왜 저래?")이
    # 같은 대화의 연장으로 처리된다.
    kind = _resolve_report_kind(request) if mode == "report" else None
    system_prompt = _report_prompt(kind) if mode == "report" and kind is not None else _chat_system_prompt()
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # 화면·기능 질문(`none`)은 근거 JSON 자체가 필요 없다 --
    # chat_system.md의 "대시보드 기능 안내"/"분석 방법"/"검증 성적"/
    # "반도체 도메인 지식" 절만으로 답이 나온다.
    level = _context_level(request, mode)
    if level != "none":
        messages.append({"role": "system", "content": _grounding_block(request.dataset, level, mode, kind)})

    for turn in request.history[-(HISTORY_TURNS * 2) :]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": request.message})
    return messages


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _static_stream(text: str) -> AsyncIterator[str]:
    """Delivers a fixed message through the same SSE protocol as a real
    completion (so the frontend's streaming/rendering pipeline treats it
    identically to an LLM answer) without any Upstage call. Used for the
    "원인 분석 미실행" guidance -- same entry point (`/api/chat`) regardless
    of whether the request came from a typed message or an example chip,
    so the two can never behave differently.
    """
    yield _sse({"delta": text})
    yield _sse({"done": True})


async def _stream_completion(request: ChatRequest, mode: Literal["report", "chat"]) -> AsyncIterator[str]:
    # 컨텍스트 빌드(_build_messages, CPU-bound pandas 파이프라인 --
    # train.CSV 10회 재랭킹 등, 수 초가 걸릴 수 있다)를 StreamingResponse
    # 반환 *전에* 끝내면 첫 바이트가 그만큼 늦게 나가, 프런트의 idle
    # 타이머(30초, 응답 시작 전부터 이미 돌고 있다)가 "총소요 제한"처럼
    # 동작해 버린다. 대신 스트림을 연 직후 SSE 주석 하트비트를 먼저 보내
    # 첫 바이트를 즉시 내보내고, 빌드는 백그라운드 스레드에서 돌리며 그동안
    # 주기적으로 하트비트를 더 보내 idle 타이머가 계속 리셋되게 한다.
    yield ": ping\n\n"
    build_task = asyncio.ensure_future(run_in_threadpool(_build_messages, request, mode))
    while not build_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(build_task), timeout=HEARTBEAT_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
    try:
        messages = build_task.result()
    except Exception as exc:  # noqa: BLE001 -- 스트림이 이미 열려 있어 여기서 SSE 에러 프레임으로만 알릴 수 있다
        logger.exception("분석 컨텍스트 구성에 실패했습니다: %s", exc)
        yield _sse({"error": "분석 결과를 불러오지 못했습니다. 다시 시도해 주세요."})
        return

    client = AsyncOpenAI(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
        timeout=CHAT_TIMEOUT_SECONDS,
    )
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            stream = await client.chat.completions.create(
                model=settings.upstage_model,
                messages=messages,
                stream=True,
                timeout=CHAT_TIMEOUT_SECONDS,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield _sse({"delta": delta})
            yield _sse({"done": True})
            return
        except (APIError, APITimeoutError, TimeoutError) as exc:
            last_error = exc
            logger.warning("Upstage 호출 실패 (시도 %d/%d): %s", attempt + 1, MAX_RETRIES + 1, exc)

    logger.error("Upstage 호출이 재시도 후에도 실패했습니다: %s", last_error)
    yield _sse({"error": "답변을 생성하지 못했습니다. 다시 시도해 주세요."})


@router.post("/chat")
async def post_chat(request: ChatRequest) -> StreamingResponse:
    # "원인 분석 미실행"은 백엔드가 즉시 판단해 안내하고 끝낸다 -- LLM 호출도,
    # API 키도 필요 없다. 프런트는 실행된 분석이 없으면 dataset을 빈 문자열로
    # 보낸다(analysisDataset이 세션에만 존재하는 프런트 상태이므로, "판단"은
    # 여기서 그 신호를 받아 내려야 한다). 타이핑이든 예시 칩이든 항상 이
    # 엔드포인트를 거치므로 두 경로가 서로 다르게 동작할 수 없다.
    if not request.dataset:
        return StreamingResponse(
            _static_stream(NO_ANALYSIS_MESSAGE),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not settings.upstage_api_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM이 연결되지 않았습니다.")

    registry = get_dataset_registry()
    try:
        registry.get_dataframe(request.dataset)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="원인 분석을 먼저 실행해 주세요.") from exc

    mode = _resolve_mode(request)
    # 컨텍스트 빌드(CPU-bound pandas 파이프라인)는 여기서 기다리지 않는다
    # -- _stream_completion 제너레이터 안에서 돌려 첫 바이트를 즉시
    # 내보낸다(위 함수 docstring 참고).
    return StreamingResponse(
        _stream_completion(request, mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
