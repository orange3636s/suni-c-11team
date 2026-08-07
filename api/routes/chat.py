"""SUNI chatbot endpoint: streams an Upstage (OpenAI-compatible) completion
over SSE, grounded in the same JSON build_analysis_report already produces
for /api/analysis/report and /api/analysis/context (see analysis.py's
_build_report_payload docstring). Two system prompts (prompts/*.md), picked
by a plain keyword check -- no LLM round-trip for intent classification.
"""

from __future__ import annotations

import json
import logging
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
REPORT_KEYWORDS = ("보고서", "리포트", "report", "요약해줘", "정리해줘")
CHAT_TIMEOUT_SECONDS = 90
HISTORY_TURNS = 2
MAX_RETRIES = 1
NO_ANALYSIS_MESSAGE = (
    "원인 분석을 먼저 실행해 주세요. 원인 분석 탭에서 실행 버튼을 누르면 "
    "분석 결과를 바탕으로 보고서를 작성할 수 있습니다.\n\n[원인 분석 탭으로 이동](/root-cause)"
)


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


def _context_user_message(dataset: str, message: str) -> str:
    # Grouped {summary, records} alarms (see build_chat_context's
    # docstring) -- individual-wafer questions ("L401W07 알람이 왜 떴어?")
    # need the record-level data, not just the aggregate counts a raw
    # report payload would give.
    context = build_chat_context(_build_report_payload(dataset))
    return (
        f"다음은 분석 결과 JSON이다.\n\n```json\n{json.dumps(context, ensure_ascii=False)}\n```\n\n"
        f"요청: {message}"
    )


def _build_messages(request: ChatRequest, mode: Literal["report", "chat"]) -> list[dict[str, str]]:
    if mode == "report":
        return [
            {"role": "system", "content": _report_system_prompt()},
            {"role": "user", "content": _context_user_message(request.dataset, request.message)},
        ]

    messages: list[dict[str, str]] = [{"role": "system", "content": _chat_system_prompt()}]
    if not request.history:
        # First chat turn: attach the full analysis JSON once.
        messages.append({"role": "user", "content": _context_user_message(request.dataset, request.message)})
        return messages

    # Follow-up turns: only the last few turns, no JSON re-send (spec §3-5).
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


async def _stream_completion(messages: list[dict[str, str]]) -> AsyncIterator[str]:
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
    # Building the grounding context runs the full screening/report pipeline
    # (CPU-bound pandas work) -- off the event loop so a chat request
    # doesn't stall every other request on this single-worker process.
    messages = await run_in_threadpool(_build_messages, request, mode)

    return StreamingResponse(
        _stream_completion(messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
