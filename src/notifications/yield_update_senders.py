"""수율 예측 갱신 발송 -- 두 블록(예측 수율 낮은 순 TOP 10, 타깃별
불량률 높은 순 TOP 3)을 세 채널 형식으로 조립한다.

LLM을 쓰지 않는다 -- 표·수치·인자명은 전부 `yield_prediction`이 이미
계산한 분석 데이터에서 직접 채우고(여기서 다시 계산하지 않는다), 요약
문장만 정해진 조건 분기로 고른다. 반올림 규칙은 화면(수율 예측 표)과
`yield_formatting`을 공유해 어긋나지 않는다.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from src.analysis.yield_formatting import format_contribution_pct, format_reliability_fraction, format_yield_pct
from src.analysis.yield_prediction import CONTRIBUTION_THRESHOLD, FAIL_RATE_TARGETS, YieldPredictionTable

DASHBOARD_DISCLAIMER = "예측 수율의 절대값은 정확도가 낮습니다. 검토 우선순위로 활용하세요."

TOP10_LIMIT = 10
TOP3_LIMIT = 3
# 요약 문장 조건.
DOMINANT_LOSS_SHARE_THRESHOLD = 30.0
LOW_RELIABILITY_MAX_COUNT = 2  # "신뢰도 2/5 미만"
LOW_RELIABILITY_MAJORITY_FRACTION = 0.5
STRONG_RELIABILITY_MIN_COUNT = 3  # "신뢰도 3/5 이상"


@dataclass(frozen=True)
class YieldUpdateTop10Item:
    lot_wafer_id: str
    y: float
    reliability_count: int


@dataclass(frozen=True)
class YieldUpdateTargetItem:
    lot_wafer_id: str
    value: float
    feature: str
    contribution_pct: float


@dataclass(frozen=True)
class YieldUpdateTargetBlock:
    target: str
    items: tuple[YieldUpdateTargetItem, ...]
    # "해당 없음" 사유 -- 자격 있는 wafer가 없어도 섹션 자체는
    # 생략하지 않는다.
    unavailable_reason: str | None


@dataclass(frozen=True)
class YieldUpdatePayload:
    dataset_label: str
    timestamp_label: str
    top10: tuple[YieldUpdateTop10Item, ...]
    target_blocks: tuple[YieldUpdateTargetBlock, ...]
    summary_sentence: str | None = None
    source_note: str | None = None
    model_label: str | None = None
    dashboard_url: str | None = None


def _summary_sentence(top10_candidates: list) -> str | None:
    """조건에 안 걸리면 문장을 만들지 않는다 -- 억지로 채우지 않는다.
    아래 분기 순서가 곧 우선순위다."""
    if not top10_candidates:
        return None

    loss_by_target = {t: 0.0 for t in FAIL_RATE_TARGETS}
    for candidate in top10_candidates:
        for target in FAIL_RATE_TARGETS:
            loss_by_target[target] += candidate.y_components.get(target, 0.0)
    total_loss = sum(loss_by_target.values())
    if total_loss > 0:
        dominant = max(loss_by_target, key=lambda t: loss_by_target[t])
        share = loss_by_target[dominant] / total_loss * 100.0
        if share >= DOMINANT_LOSS_SHARE_THRESHOLD:
            return f"{dominant} 불량이 전체 손실의 {share:.0f}%를 차지합니다."

    low_count = sum(1 for c in top10_candidates if c.reliability.count < LOW_RELIABILITY_MAX_COUNT)
    if low_count >= len(top10_candidates) * LOW_RELIABILITY_MAJORITY_FRACTION:
        return f"상위 10건 중 {low_count}건이 핵심 인자 미계측 상태입니다."

    if all(c.reliability.count >= STRONG_RELIABILITY_MIN_COUNT for c in top10_candidates):
        return "상위 10건 모두 핵심 인자가 계측되어 있습니다."

    return None


def build_yield_update_payload(
    table: YieldPredictionTable,
    *,
    dataset_label: str,
    timestamp_label: str,
    source_note: str | None = None,
    model_label: str | None = None,
    dashboard_url: str | None = None,
) -> YieldUpdatePayload:
    top10_candidates = table.candidates[:TOP10_LIMIT]
    top10 = tuple(
        YieldUpdateTop10Item(lot_wafer_id=c.lot_wafer_id, y=c.y, reliability_count=c.reliability.count)
        for c in top10_candidates
    )

    target_blocks: list[YieldUpdateTargetBlock] = []
    for target in FAIL_RATE_TARGETS:
        # "기여율 임계 이상 인자가 계측된 wafer만" -- 구간 조정
        # 자격(core_factors[target])과 같은 임계를 재사용한다.
        eligible = [
            c
            for c in table.candidates
            if (cell := c.core_factors[target]).contribution_pct is not None and cell.contribution_pct >= CONTRIBUTION_THRESHOLD
        ]
        eligible.sort(key=lambda c: c.y_components.get(target, 0.0), reverse=True)
        if eligible:
            items = tuple(
                YieldUpdateTargetItem(
                    lot_wafer_id=c.lot_wafer_id,
                    value=c.y_components.get(target, 0.0),
                    feature=c.core_factors[target].feature,
                    contribution_pct=c.core_factors[target].contribution_pct,
                )
                for c in eligible[:TOP3_LIMIT]
            )
            target_blocks.append(YieldUpdateTargetBlock(target=target, items=items, unavailable_reason=None))
        else:
            primary = table.primary_factors.get(target)
            reason = f"{primary.feature}이 계측된 wafer가 없습니다" if primary is not None else "핵심 인자를 산출할 수 없습니다"
            target_blocks.append(YieldUpdateTargetBlock(target=target, items=(), unavailable_reason=reason))

    return YieldUpdatePayload(
        dataset_label=dataset_label,
        timestamp_label=timestamp_label,
        top10=top10,
        target_blocks=tuple(target_blocks),
        summary_sentence=_summary_sentence(top10_candidates),
        source_note=source_note,
        model_label=model_label,
        dashboard_url=dashboard_url,
    )


# -- 공용 텍스트 조립 (Slack 코드블록 / Telegram 고정폭 텍스트가 공유) ------


def _header_line(payload: YieldUpdatePayload) -> str:
    return f"[SUNI] 수율 예측 갱신 ({payload.timestamp_label})"


def _source_line(payload: YieldUpdatePayload) -> str:
    line = f"소스: {payload.dataset_label}"
    if payload.model_label:
        line += f" · 모델 {payload.model_label}"
    return line


def _render_table(title: str, header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i]) for i in range(len(header))]

    def _row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    lines = [title, _row(header)]
    lines.extend(_row(r) for r in rows)
    return "\n".join(lines)


def _top10_table_text(payload: YieldUpdatePayload) -> str:
    rows = [[item.lot_wafer_id, format_yield_pct(item.y), format_reliability_fraction(item.reliability_count)] for item in payload.top10]
    return _render_table("예측 수율이 낮은 WF TOP 10", ["LOT_WF_ID", "Y", "신뢰도"], rows)


def _target_block_text(block: YieldUpdateTargetBlock) -> str:
    title = f"{block.target} 불량률 높은 순"
    if not block.items:
        return f"{title}\n해당 없음 — {block.unavailable_reason}"
    rows = [
        [item.lot_wafer_id, format_yield_pct(item.value), f"{item.feature} ({format_contribution_pct(item.contribution_pct)})"]
        for item in block.items
    ]
    return _render_table(title, ["LOT_WF_ID", block.target, "핵심인자"], rows)


def _footer_lines(payload: YieldUpdatePayload) -> list[str]:
    lines = [DASHBOARD_DISCLAIMER]
    if payload.dashboard_url:
        lines.append(f"대시보드에서 확인 → {payload.dashboard_url}")
    return lines


def _compose_body_lines(payload: YieldUpdatePayload) -> list[str]:
    lines: list[str] = []
    if payload.source_note:
        lines.append(payload.source_note)
        lines.append("")
    lines.append(_header_line(payload))
    lines.append(_source_line(payload))
    lines.append("")
    lines.append(_top10_table_text(payload))
    lines.append("")
    for block in payload.target_blocks:
        lines.append(_target_block_text(block))
        lines.append("")
    if payload.summary_sentence:
        lines.append(payload.summary_sentence)
        lines.append("")
    lines.extend(_footer_lines(payload))
    return lines


# -- Slack (마크다운 코드블록으로 표 정렬) -----------------------------------


def build_slack_yield_update(payload: YieldUpdatePayload) -> dict:
    blocks: list[dict] = []
    if payload.source_note:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"*{payload.source_note}*"}]})
    blocks.append({"type": "header", "text": {"type": "plain_text", "text": _header_line(payload), "emoji": True}})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": _source_line(payload)}]})
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"```{_top10_table_text(payload)}```"}})
    for block in payload.target_blocks:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"```{_target_block_text(block)}```"}})
    if payload.summary_sentence:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": payload.summary_sentence}})
    footer = DASHBOARD_DISCLAIMER
    if payload.dashboard_url:
        footer += f" · <{payload.dashboard_url}|대시보드에서 확인>"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]})
    return {"blocks": blocks}


# -- Telegram (일반 텍스트, 고정폭 정렬) -------------------------------------

TELEGRAM_MAX_CHARS = 4096  # Telegram sendMessage 하드 한계


def build_telegram_yield_update_text(payload: YieldUpdatePayload) -> str:
    """단일 문자열 전체 -- 짧은 페이로드/테스트/하위 호환용. 실제 발송
    (`yield_update_dispatch.dispatch_yield_update`)은 4096자 한계를 넘을
    수 있으므로 `build_telegram_yield_update_chunks`를 쓴다."""
    return "\n".join(_compose_body_lines(payload))


def _telegram_blocks(payload: YieldUpdatePayload) -> list[str]:
    """청크 분할의 최소 단위 -- 헤더+소스 한 덩어리, TOP10 표 한 덩어리,
    Y1~Y5 타깃 섹션 각각 한 덩어리, 요약 문장, 꼬리말. 분할은 이 목록
    경계에서만 일어난다 -- 표 중간이나 타깃 섹션 중간에서 자르지 않는다."""
    blocks: list[str] = []
    header = _header_line(payload)
    if payload.source_note:
        header = f"{payload.source_note}\n{header}"
    blocks.append(f"{header}\n{_source_line(payload)}")
    blocks.append(_top10_table_text(payload))
    for block in payload.target_blocks:
        blocks.append(_target_block_text(block))
    if payload.summary_sentence:
        blocks.append(payload.summary_sentence)
    blocks.append("\n".join(_footer_lines(payload)))
    return blocks


def build_telegram_yield_update_chunks(payload: YieldUpdatePayload, *, max_chars: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Telegram 하드 한계(4096자)를 넘으면 여러 메시지로 나눈다. 항상
    `_telegram_blocks`의 온전한 블록 단위로만 자른다 -- 표를 반토막
    내거나 타깃 섹션을 통째로 누락하지 않는다. 대개(TOP10 10행 + 타깃당
    최대 3행 기준) 한 청크(길이 1개 리스트)로 끝난다 -- 실측치는
    프로젝트 README/작업 보고 참고.

    블록 하나가 그 자체로 `max_chars`를 넘는 극단적인 경우(예: 매우 긴
    피처명이 반복)에는 그 블록만 단독 청크로 내보낸다 -- 잘라서 깨진
    표를 보내는 것보다 낫다.
    """
    full_text = build_telegram_yield_update_text(payload)
    if len(full_text) <= max_chars:
        return [full_text]

    blocks = _telegram_blocks(payload)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        separator_len = 2 if current else 0  # "\n\n"으로 이어붙일 때
        if current and current_len + separator_len + len(block) > max_chars:
            chunks.append("\n\n".join(current))
            current = [block]
            current_len = len(block)
        else:
            current.append(block)
            current_len += separator_len + len(block)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# -- Gmail (HTML 표, 이스케이프 필수) ----------------------------------------


def build_yield_update_email_html(payload: YieldUpdatePayload) -> str:
    """dataset_label·source_note·인자명은 사용자가 올린 파일명 등 신뢰할
    수 없는 문자열을 담을 수 있으므로 빠짐없이 `html.escape`를 거친다."""

    def esc(value: object) -> str:
        return html.escape(str(value))

    def _table_html(header: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th style='text-align:left;padding:4px 8px;border-bottom:1px solid #ddd'>{esc(h)}</th>" for h in header)
        body = "".join(
            "<tr>" + "".join(f"<td style='padding:4px 8px'>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows
        )
        return f"<table style='border-collapse:collapse;width:100%;font-size:13px'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    top10_rows = [[item.lot_wafer_id, format_yield_pct(item.y), format_reliability_fraction(item.reliability_count)] for item in payload.top10]
    sections = [f"<h3 style='margin:16px 0 6px'>예측 수율이 낮은 WF TOP 10</h3>{_table_html(['LOT_WF_ID', 'Y', '신뢰도'], top10_rows)}"]
    for block in payload.target_blocks:
        title_html = f"<h3 style='margin:16px 0 6px'>{esc(block.target)} 불량률 높은 순</h3>"
        if not block.items:
            sections.append(f"{title_html}<p style='color:#888'>해당 없음 — {esc(block.unavailable_reason)}</p>")
            continue
        rows = [
            [item.lot_wafer_id, format_yield_pct(item.value), f"{item.feature} ({format_contribution_pct(item.contribution_pct)})"]
            for item in block.items
        ]
        sections.append(f"{title_html}{_table_html(['LOT_WF_ID', block.target, '핵심인자'], rows)}")

    summary_html = f"<p>{esc(payload.summary_sentence)}</p>" if payload.summary_sentence else ""
    source_html = (
        f"<p style='color:#b45309;font-weight:600;margin:0 0 10px'>{esc(payload.source_note)}</p>" if payload.source_note else ""
    )
    dashboard_html = f"<p><a href='{esc(payload.dashboard_url)}'>대시보드에서 확인 →</a></p>" if payload.dashboard_url else ""

    return f"""
    <div style="font-family:sans-serif;max-width:640px">
      {source_html}
      <h2 style="margin:0 0 4px">{esc(_header_line(payload))}</h2>
      <p style="color:#555;margin:0 0 12px">{esc(_source_line(payload))}</p>
      {''.join(sections)}
      {summary_html}
      <p style="color:#888;font-size:12px;margin-top:16px">{esc(DASHBOARD_DISCLAIMER)}</p>
      {dashboard_html}
    </div>
    """
