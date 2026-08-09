"""알람 발송 오케스트레이션 (알람 알림 연동 §C-4/§C-5/§C-7) -- 신뢰도 게이트,
24시간 중복 발송 방지, 재시도까지 여기서 전부 처리한다. 호출자(API 라우트,
APScheduler 잡)는 무엇을 보낼지만 넘기고, 언제·누구에게·몇 번 보낼지는
신경 쓰지 않는다.

발송은 best-effort다 -- 실패해도 예외를 올리지 않는다. 분석 실행이나
스케줄러 잡 자체가 알림 발송 실패 때문에 죽으면 안 된다.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from api.settings import settings
from src.analysis.alarm_gbdt import DEFAULT_SENSITIVITY, DEFAULT_TARGET_YIELD
from src.notifications import senders, settings_store
from src.notifications.senders import AlarmNotificationItem, AlarmNotificationPayload
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

DEDUP_WINDOW_HOURS = 24
MAX_SEND_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
_GRADE_RANK = {"주의": 0, "위험": 1, "심각": 2}


def _send_with_retry(send_fn: Callable[[], tuple[bool, str | None]]) -> tuple[bool, str | None]:
    delay = RETRY_BASE_DELAY_SECONDS
    last_error: str | None = None
    for attempt in range(MAX_SEND_ATTEMPTS):
        ok, error = send_fn()
        if ok:
            return True, None
        last_error = error
        if attempt < MAX_SEND_ATTEMPTS - 1:
            time.sleep(delay)
            delay *= 2
    return False, last_error


def _filter_new_alarms(store: RuntimeStore, dataset_id: str, channel: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """§C-7 + D-7: 동일 (dataset, wafer, grade, channel) 조합은 24시간 내
    재발송하지 않는다 -- 채널별로 따로 추적한다. channel을 빼면 한
    채널만 성공해도 (dataset, wafer, grade)가 "발송 완료"로 찍혀,
    실패한 다른 채널은 24시간 동안 재시도되지 않는다. 등급이 이전보다
    악화된 경우(예: 주의 -> 심각)는 예외로 다시 보낸다.
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=DEDUP_WINDOW_HOURS)).isoformat()
    recent = store.recent_notifications(dataset_id, since, channel=channel)
    recent_by_wafer: dict[str, set[str]] = {}
    for entry in recent:
        recent_by_wafer.setdefault(entry["wafer_id"], set()).add(entry["grade"])

    result = []
    for item in candidates:
        already_sent = recent_by_wafer.get(item["lot_wafer_id"])
        if already_sent:
            best_sent_rank = max(_GRADE_RANK.get(g, -1) for g in already_sent)
            if _GRADE_RANK.get(item["grade"], -1) <= best_sent_rank:
                continue  # 이미 보냈고 등급도 악화되지 않았다 -- 스킵
        result.append(item)
    return result


def _build_payload(
    dataset_label: str,
    items: list[dict[str, Any]],
    reliability_grade: str,
    reliability_score: int,
    dashboard_url: str | None,
    source_note: str | None = None,
    *,
    target_yield: float = DEFAULT_TARGET_YIELD,
    sensitivity: float = DEFAULT_SENSITIVITY,
) -> AlarmNotificationPayload:
    grade_counts: dict[str, int] = {}
    for item in items:
        grade_counts[item["grade"]] = grade_counts.get(item["grade"], 0) + 1
    return AlarmNotificationPayload(
        dataset_label=dataset_label,
        timestamp_label=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        items=[
            AlarmNotificationItem(
                lot_wafer_id=item["lot_wafer_id"],
                risk_percentile=float(item["risk_percentile"]),
                grade=item["grade"],
                reason=item["reason"],
            )
            for item in sorted(items, key=lambda a: a["risk_percentile"])
        ],
        grade_counts=grade_counts,
        reliability_grade=reliability_grade,
        reliability_score=reliability_score,
        dashboard_url=dashboard_url,
        source_note=source_note,
        target_yield=target_yield,
        sensitivity=sensitivity,
    )


def _send_to_all_channels(
    store: RuntimeStore,
    *,
    dataset_id: str,
    dataset_label: str,
    candidates: list[dict[str, Any]],
    reliability_grade: str,
    reliability_score: int,
    dashboard_url: str | None,
    source_note: str | None = None,
    target_yield: float = DEFAULT_TARGET_YIELD,
    sensitivity: float = DEFAULT_SENSITIVITY,
) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]], set[tuple[str, str]]]:
    """D-7: 채널마다 자기 24시간 dedupe 기준으로 자기만의 발송 대상을
    다시 추린다 -- 채널 하나가 이미 성공한 (wafer, grade)라도 다른
    채널은 아직 못 보냈을 수 있으므로 채널 전체가 같은 `to_send`를
    공유하면 안 된다. 반환하는 (attempted_keys, sent_keys) 중
    attempted_keys는 채널별로 새로 시도한(성공 여부 무관) (wafer, grade)
    전체이고, sent_keys는 그중 실제로 성공한 것만이다 -- "24시간 내
    이미 발송됨"과 "이번엔 시도했지만 실패함"을 구분하려면 둘 다
    필요하다(둘 다 비었을 때만 진짜로 "새로 보낼 게 없었다"이다).
    """
    results: dict[str, dict[str, Any]] = {}
    attempted_keys: set[tuple[str, str]] = set()
    sent_keys: set[tuple[str, str]] = set()

    def _dispatch_one(channel: str, send_fn: Callable[[AlarmNotificationPayload], tuple[bool, str | None]]) -> None:
        to_send = _filter_new_alarms(store, dataset_id, channel, candidates)
        if not to_send:
            results[channel] = {"ok": True, "error": None, "sent_count": 0}
            return
        attempted_keys.update((item["lot_wafer_id"], item["grade"]) for item in to_send)
        payload = _build_payload(
            dataset_label, to_send, reliability_grade, reliability_score, dashboard_url, source_note,
            target_yield=target_yield, sensitivity=sensitivity,
        )
        ok, error = _send_with_retry(lambda: send_fn(payload))
        results[channel] = {"ok": ok, "error": error, "sent_count": len(to_send) if ok else 0}
        if ok:
            store.record_notifications_sent(
                dataset_id, [(item["lot_wafer_id"], item["grade"]) for item in to_send], channel=channel
            )
            sent_keys.update((item["lot_wafer_id"], item["grade"]) for item in to_send)

    slack = settings_store.get_slack(store)
    if slack:
        _dispatch_one("slack", lambda payload: senders.send_slack_alarm(slack["webhook_url"], payload))

    telegram = settings_store.get_telegram(store)
    if telegram and settings.telegram_bot_token:
        _dispatch_one(
            "telegram",
            lambda payload: senders.send_telegram_alarm(settings.telegram_bot_token, telegram["chat_id"], payload),  # type: ignore[arg-type]
        )

    gmail = settings_store.get_gmail(store)
    if gmail and gmail.get("verified") and settings.smtp_host and settings.smtp_user and settings.smtp_password and settings.smtp_from_email:
        _dispatch_one(
            "gmail",
            lambda payload: senders.send_gmail_alarm(
                smtp_host=settings.smtp_host,  # type: ignore[arg-type]
                smtp_port=settings.smtp_port,
                smtp_user=settings.smtp_user,  # type: ignore[arg-type]
                smtp_password=settings.smtp_password,  # type: ignore[arg-type]
                from_email=settings.smtp_from_email,  # type: ignore[arg-type]
                to_email=gmail["email"],
                payload=payload,
            ),
        )

    return results, attempted_keys, sent_keys


def dispatch_alarm_notifications(
    store: RuntimeStore,
    *,
    trigger: str,
    dataset_id: str,
    dataset_label: str,
    alarms: list[dict[str, Any]],
    reliability_grade: str,
    reliability_score: int,
    dashboard_url: str | None = None,
    source_note: str | None = None,
    target_yield: float = DEFAULT_TARGET_YIELD,
    sensitivity: float = DEFAULT_SENSITIVITY,
) -> dict[str, Any]:
    """`alarms`의 각 원소는 {lot_wafer_id, risk_percentile, grade, reason}
    형태다 (알람 판정 GBDT 전환 §A-3의 알람 목록 항목과 동일한 필드).

    `source_note`(EB그룹)는 수동 업로드/폴백(데모) 모드에서 발송할 때
    본문 맨 위에 붙는 출처 한 줄이다 -- SQL 연동 자동 갱신 경로에서는
    생략(None)해 아무 표시도 하지 않는다.

    `target_yield`/`sensitivity`(GD-2)는 이 알람 목록을 실제로 판정한
    기준값이다 -- 발송 메시지에 판정 컷과 함께 그대로 실린다. 호출부가
    넘기지 않으면(기존 호출부와의 호환) 기본값(88.0/0.2)을 쓴다 -- 단,
    그러면 메시지에 실제 판정 기준과 다른 값이 표시될 수 있으므로 알람
    목록을 낸 것과 같은 target/sensitivity를 반드시 함께 넘겨야 한다.

    `trigger`는 호출부가 어느 경로로 불렀는지(`settings_store.TIMING_ON_ANALYSIS`
    "분석 실행 직후", `TIMING_DAILY_9AM` 매일 09:00, `TIMING_DAILY_13` 매일
    13:00 중 하나)를 밝힌다 -- A-6/DF그룹: 저장된 발송 시점 설정
    (`conditions["timing"]`, 배열)에 이 값이 포함되지 않으면 스킵한다.
    이전에는 이 값이 저장만 되고 어느 발송 경로도 읽지 않아 "매일 9시만"
    선택해도 분석 직후 발송이 나갔다.

    반환값은 항상 dict -- 발송 성공/실패와 무관하게 예외를 올리지 않는다.
    """
    try:
        # §C-5: 신뢰도 낮음이면 알람이 발생해도 발송을 건너뛴다 (핵심 요구사항).
        if reliability_grade == "낮음":
            logger.info("알림 발송 스킵: dataset=%s 신뢰도 낮음", dataset_id)
            return {"skipped": True, "reason": "분석 신뢰도 낮음"}

        conditions = settings_store.get_conditions(store)
        if trigger not in (conditions.get("timing") or []):
            return {"skipped": True, "reason": "발송 시점 설정과 일치하지 않음"}

        target_grades = set(conditions.get("grades") or [])
        candidates = [a for a in alarms if a.get("grade") in target_grades]
        if not candidates:
            return {"skipped": True, "reason": "발송 대상 등급의 알람 없음"}

        # D-7: 채널마다 자기 24시간 dedupe 기준으로 자기 대상을 다시 추리므로
        # (한 채널의 성공이 다른 채널의 재시도를 막지 않도록), 여기서는 더
        # 이상 "채널 공통 to_send" 하나로 앞질러 스킵하지 않는다 -- 연결된
        # 채널이 하나도 없거나(results 비어 있음), 모든 채널이 새로 보낼
        # 게 없었을 때(attempted_keys 비어 있음 -- 시도 자체가 없었다는
        # 뜻이라 "이미 발송됨"과 "시도했지만 실패"를 구분할 수 있다)만
        # 스킵으로 본다.
        results, attempted_keys, sent_keys = _send_to_all_channels(
            store,
            dataset_id=dataset_id,
            dataset_label=dataset_label,
            candidates=candidates,
            reliability_grade=reliability_grade,
            reliability_score=reliability_score,
            dashboard_url=dashboard_url,
            source_note=source_note,
            target_yield=target_yield,
            sensitivity=sensitivity,
        )

        if not results:
            return {"skipped": True, "reason": "연결된 채널 없음"}

        if not attempted_keys:
            return {"skipped": True, "reason": "24시간 내 이미 발송됨 (등급 변화 없음)"}

        return {"skipped": False, "sent_count": len(sent_keys), "results": results}
    except Exception:
        logger.exception("알림 발송 중 오류: dataset=%s", dataset_id)
        return {"skipped": True, "reason": "발송 중 오류 발생"}
