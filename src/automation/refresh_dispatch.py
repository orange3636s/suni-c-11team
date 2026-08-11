"""자동 갱신 파이프라인이 발송 메시지에 붙이는 출처 한 줄.

발송 자체와 억제 로직(시간당 예산/수동 10분 간격)은
`src/notifications/yield_update_dispatch.py`의 `dispatch_yield_update`가
담당한다(refresh.py의 `_dispatch_yield_update_for_refresh` 참고). 이
모듈은 수동 업로드/폴백 모드에서 발송 본문 맨 위에 붙는 출처 표시만
만든다.
"""

from __future__ import annotations


def _source_note_for(mode: str, eval_dataset_id: str) -> str | None:
    """발송 본문 맨 위에 붙는 출처 한 줄 -- SQL 연동 자동 갱신
    경로(mode == "sql")에서는 None(표시 없음). `mode`는 manual/sql/fallback
    중 하나이고 상호 배타적이라(`src/automation/refresh.py::_resolve_source`)
    "폴백+수동이 겹치는" 경우 자체가 없다 -- 수동 오버라이드가 있으면
    항상 "manual"이 우선 선택된다."""
    if mode == "manual":
        from api.routes.datasets import get_dataset_registry

        registry = get_dataset_registry()
        summary = registry.get_summary(eval_dataset_id)
        filename = summary["original_filename"] if summary else eval_dataset_id
        return f"[수동] {filename} 업로드 결과"
    if mode == "fallback":
        return "[데모] 내장 데이터 기준 — 실제 공정 데이터가 아닙니다"
    return None
