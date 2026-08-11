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

from api.routes.analysis import _build_report_payload
from api.routes.datasets import get_dataset_registry
from api.settings import settings
from src.analysis.report import build_chat_context
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

# A-4: 질문 종류와 무관하게 항상 전체 JSON을 붙이면(수 KB~수십 KB) 매
# 메시지가 그만큼의 토큰을 태운다. 구체적인 인자·wafer를 지목한 질문만
# 전체(`full`)를 받고, 나머지는 요약(`digest`)이나 근거 없음(`none`)으로
# 낮춘다. ContextLevel 판정은 키워드 매칭이 아니라 인자명/wafer ID
# 정규식과 report 모드 여부로만 한다 -- A-2에서 "요약해줘" 같은 일반
# 동사가 report 모드를 오분류시킨 것과 같은 함정을 여기서도 피한다.
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
    r"Color\s*by|그룹\s*강조|SPC.{0,6}ML|민감도(를|는)?\s*(올리면|낮추면|조절)"
)


def _context_level(request: ChatRequest, mode: Literal["report", "chat"]) -> ContextLevel:
    if mode == "report":
        return "full"
    if FACTOR_ID_PATTERN.search(request.message) or WAFER_ID_PATTERN.search(request.message):
        return "full"
    if UI_QUESTION_PATTERN.search(request.message):
        return "none"
    return "digest"


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
    "eps2",
    "eps2_text",
    "p_value",
    "p_value_text",
    "relation",
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


@lru_cache(maxsize=1)
def _report_system_prompt() -> str:
    return (PROMPTS_DIR / "report_system.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _chat_system_prompt() -> str:
    return (PROMPTS_DIR / "chat_system.md").read_text(encoding="utf-8")


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    mode: Literal["report", "chat"] | None = None
    dataset: str = "train"
    history: list[ChatHistoryTurn] = Field(default_factory=list)


def _resolve_mode(request: ChatRequest) -> Literal["report", "chat"]:
    if request.mode in ("report", "chat"):
        return request.mode
    return "report" if any(keyword in request.message for keyword in REPORT_KEYWORDS) else "chat"


def _grounding_block(dataset: str, level: ContextLevel) -> str:
    # Grouped {summary, records} alarms (see build_chat_context's
    # docstring) -- individual-wafer questions ("L401W07 알람이 왜 떴어?")
    # need the record-level data, not just the aggregate counts a raw
    # report payload would give.
    #
    # A-1 fix: this used to be a `user` message, competing for space in the
    # same sliding history window the frontend/backend trim to the last few
    # turns -- by the 4th turn the window no longer contained it and the
    # model kept answering (or refusing) ungrounded. A `system` message is
    # never subject to that window, so it survives every turn.
    context = build_chat_context(_build_report_payload(dataset))
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
    # A-2: report 모드도 일반 chat과 같은 경로를 탄다 -- system 프롬프트만
    # 갈라진다. 이전에는 report 모드가 history를 아예 안 읽어서, 이미
    # 만들어진 보고서에 대한 후속 질문("Y2는 왜 저래?")이 완전히 새
    # 대화처럼 취급됐다.
    system_prompt = _report_system_prompt() if mode == "report" else _chat_system_prompt()
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # A-4: 화면·기능 질문(`none`)은 근거 JSON 자체가 필요 없다 --
    # chat_system.md의 "대시보드 기능 안내" 절만으로 답이 나온다.
    level = _context_level(request, mode)
    if level != "none":
        messages.append({"role": "system", "content": _grounding_block(request.dataset, level)})

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
    so the two never behave differently (spec 재지시: "같은 요청인데 경로에
    따라 동작이 다르면 안 된다").
    """
    yield _sse({"delta": text})
    yield _sse({"done": True})


async def _stream_completion(request: ChatRequest, mode: Literal["report", "chat"]) -> AsyncIterator[str]:
    # 지시서 D-1: 컨텍스트 빌드(_build_messages, CPU-bound pandas 파이프라인 --
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
    # 지시서 D-1: 컨텍스트 빌드(CPU-bound pandas 파이프라인)는 더 이상 여기서
    # 기다리지 않는다 -- _stream_completion 제너레이터 안으로 옮겨 첫 바이트를
    # 즉시 내보낸다(위 함수 docstring 참고).
    return StreamingResponse(
        _stream_completion(request, mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
