"""Human-readable `_text` companions for the LLM chat/report context JSON
(spec "LLM 답변·보고서 서술 다듬기" §7) -- pre-formats numbers, ranges,
booleans, and grade strings using the exact same rounding conventions the
screen already uses (frontend/lib/numberFormat.ts, and the per-field
`toFixed(n)` calls scattered across ScatterChart.tsx/alerts/page.tsx/
root-cause/page.tsx), so the model never has to invent its own digit count,
bracket notation, or scientific notation -- and reliably gets at least one
of those wrong when left to it.

Every function here is a pure formatter: given a raw numeric/boolean value,
return the exact string the screen would show (or `None` when there's
nothing worth saying, e.g. `chamber_interaction=False`). `add_display_text`
walks the report structure and adds `<field>_text` siblings without
removing or altering the original numeric fields -- both the download
report and any downstream computation still have the raw numbers.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def _quantize(value: float, decimals: int) -> str:
    # Anchored via str(value) rather than the Decimal(float) constructor,
    # so this rounds the same *decimal* digits a human reads off the
    # number -- not the binary float's exact (often longer, surprising)
    # value. ROUND_HALF_UP mirrors JS `toFixed`'s round-half-away-from-zero
    # behavior, unlike Python's own round()/format() which round half to
    # even.
    quantum = Decimal(1).scaleb(-decimals)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def format_1dp(value: float | None) -> str | None:
    """인자 값 · 관리한계 · 권장 구간 (화면의 `toFixed(1)`과 동일)."""
    if value is None:
        return None
    return _quantize(value, 1)


def format_2dp(value: float | None) -> str | None:
    """불량률 · 수율."""
    if value is None:
        return None
    return _quantize(value, 2)


def format_3dp(value: float | None) -> str | None:
    """설명력(eps2) -- 화면의 `eps2.toFixed(3)`과 동일."""
    if value is None:
        return None
    return _quantize(value, 3)


def format_4dp(value: float | None) -> str | None:
    """p값/q값 -- 화면의 formatPValue/formatQValue와 동일. 아주 작은 값도
    지수 표기 없이 고정 소수점("0.0000")으로 표시된다."""
    if value is None:
        return None
    return _quantize(value, 4)


def format_pct1(value: float | None) -> str | None:
    """기여율/누적 기여율 -- 화면의 `contribution_pct.toFixed(1)`과 동일."""
    if value is None:
        return None
    return f"{_quantize(value, 1)}%"


def format_range_1dp(lo: float | None, hi: float | None) -> str | None:
    """수치 범위 표기 -- `[46.97, 62.15]` 대신 `"47.0 ~ 62.2"`."""
    if lo is None or hi is None:
        return None
    return f"{format_1dp(lo)} ~ {format_1dp(hi)}"


def control_band_text(lower: float | None, upper: float | None) -> str | None:
    """관리한계 -- 단측(단조 인자, `lower`/`upper` 중 하나만 존재)인 경우
    슬래시 쌍 대신 있는 쪽만 표기한다."""
    if lower is not None and upper is not None:
        return format_range_1dp(lower, upper)
    if upper is not None:
        return f"UCL {format_1dp(upper)}"
    if lower is not None:
        return f"LCL {format_1dp(lower)}"
    return None


SHAPE_TEXT: dict[str, str] = {
    "u_shape": "양쪽 끝으로 갈수록 불량률이 오르는 U자 형태",
    "monotonic_increasing": "값이 커질수록 불량률이 오르는 형태",
    "monotonic_decreasing": "값이 커질수록 불량률이 내려가는 형태",
    "unclear": "뚜렷한 방향성이 확인되지 않는 형태",
}


def shape_text(shape: str | None) -> str | None:
    if shape is None:
        return None
    return SHAPE_TEXT.get(shape, SHAPE_TEXT["unclear"])


# 화면에 쓰는 4단계 등급(강함/보통/약함/참고)과 보고서 전용 3단계 판정
# (강함/보통/근거부족) 모두를 하나의 표로 커버한다 (spec §2-4).
CONFIDENCE_TEXT: dict[str, str] = {
    "강함": "신뢰도가 높습니다",
    "보통": "신뢰도가 보통이며 재확인이 필요합니다",
    "약함": "통계적 근거가 부족합니다",
    "참고": "통계적 근거가 부족합니다",
    "근거부족": "통계적 근거가 부족합니다",
}


def confidence_text(grade: str | None) -> str | None:
    if grade is None:
        return None
    return CONFIDENCE_TEXT.get(grade)


def chamber_interaction_text(value: bool | None) -> str | None:
    """spec §2-3: `False`/`None`인 경우는 언급할 필요가 없다 -- 필드 자체를
    만들지 않도록 `None`을 돌려준다."""
    if value:
        return "챔버에 따라 관계가 다르게 나타남"
    return None


def _annotate_factor(factor: dict[str, Any]) -> dict[str, Any]:
    out = dict(factor)

    relation = out.get("relation")
    if isinstance(relation, dict):
        relation = dict(relation)
        relation["shape_text"] = shape_text(relation.get("shape"))
        out["relation"] = relation

    control_limits = out.get("control_limits")
    if isinstance(control_limits, dict):
        control_limits = dict(control_limits)
        control_limits["band_text"] = control_band_text(control_limits.get("lcl"), control_limits.get("ucl"))
        out["control_limits"] = control_limits

    window = out.get("window")
    if isinstance(window, dict):
        window = dict(window)
        window["range_text"] = format_range_1dp(window.get("lo"), window.get("hi"))
        out["window"] = window

    if out.get("eps2") is not None:
        out["eps2_text"] = format_3dp(out["eps2"])
    if out.get("p_value") is not None:
        out["p_value_text"] = format_4dp(out["p_value"])
    if out.get("q_value") is not None:
        out["q_value_text"] = format_4dp(out["q_value"])
    if out.get("contribution_pct") is not None:
        out["contribution_pct_text"] = format_pct1(out["contribution_pct"])
    if out.get("cumulative_pct") is not None:
        out["cumulative_pct_text"] = format_pct1(out["cumulative_pct"])

    text = chamber_interaction_text(out.get("chamber_interaction"))
    if text:
        out["chamber_interaction_text"] = text
    if out.get("chamber_interaction_p") is not None:
        out["chamber_interaction_p_text"] = format_4dp(out["chamber_interaction_p"])
    if out.get("chamber_interaction_q") is not None:
        out["chamber_interaction_q_text"] = format_4dp(out["chamber_interaction_q"])

    per_chamber = out.get("per_chamber_window")
    if isinstance(per_chamber, dict) and per_chamber:
        out["per_chamber_window_text"] = {
            chamber: format_range_1dp(win.get("lo"), win.get("hi")) for chamber, win in per_chamber.items()
        }

    if out.get("report_confidence") is not None:
        out["report_confidence_text"] = confidence_text(out["report_confidence"])
    if out.get("grade") is not None:
        out["grade_text"] = confidence_text(out["grade"])

    return out


def _annotate_alarm_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if out.get("value") is not None:
        out["value_text"] = format_1dp(out["value"])
    control_band = out.get("control_band")
    if isinstance(control_band, (list, tuple)) and len(control_band) == 2:
        out["control_band_text"] = control_band_text(control_band[0], control_band[1])
    return out


def add_display_text(context: dict[str, Any]) -> dict[str, Any]:
    """Adds `_text` siblings throughout a `build_chat_context` payload
    in-place-equivalent (returns a new dict; never mutates the input).
    Called once, in `report.build_chat_context`, so both the chat and
    report prompts (which both consume that same function's output --
    see `api/routes/chat.py::_context_user_message`) get the same
    pre-formatted values without either prompt reimplementing rounding.
    """
    out = dict(context)

    targets = out.get("targets")
    if isinstance(targets, list):
        out["targets"] = [
            {
                **target_entry,
                "factors": [_annotate_factor(factor) for factor in target_entry.get("factors", [])],
            }
            for target_entry in targets
        ]

    config_screening = out.get("config_screening")
    if isinstance(config_screening, dict):
        config_screening = dict(config_screening)
        if config_screening.get("max_observed_eps2") is not None:
            config_screening["max_observed_eps2_text"] = format_3dp(config_screening["max_observed_eps2"])
        if config_screening.get("mde_eps2") is not None:
            config_screening["mde_eps2_text"] = format_3dp(config_screening["mde_eps2"])
        out["config_screening"] = config_screening

    summary = out.get("summary")
    if isinstance(summary, dict):
        summary = dict(summary)
        for key in ("mean_yield_alarm", "mean_yield_normal", "yield_gap_pp"):
            if summary.get(key) is not None:
                summary[f"{key}_text"] = format_2dp(summary[key])
        out["summary"] = summary

    alarms = out.get("alarms")
    if isinstance(alarms, dict) and isinstance(alarms.get("records"), list):
        alarms = dict(alarms)
        alarms["records"] = [_annotate_alarm_record(record) for record in alarms["records"]]
        out["alarms"] = alarms

    return out
