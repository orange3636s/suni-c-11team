from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from api.routes.analysis import _auc_gate, _cached_reliability, _dataframe_or_404, _hydrated_targets_or_409, compute_alarm_notification_items
from api.routes.datasets import get_dataset_registry
from api.schemas.notify import (
    ConditionsSaveRequest,
    DispatchRequest,
    DispatchResponse,
    GmailConnectRequest,
    NotificationSettingsSummary,
    SendTestResponse,
    SlackConnectRequest,
    SlackTestRequest,
    TelegramVerifyRequest,
)
from api.settings import settings
from src.analysis import alarm_gbdt
from src.notifications import dispatch, senders, settings_store, telegram_bot
from src.runtime.app_state import get_latest_state
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notify", tags=["notify"])


def _store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


@router.get("/settings", response_model=NotificationSettingsSummary)
def get_notification_settings() -> dict[str, Any]:
    return settings_store.get_settings_summary(_store())


@router.post("/slack", response_model=NotificationSettingsSummary)
def connect_slack(body: SlackConnectRequest) -> dict[str, Any]:
    if not settings_store.is_valid_slack_webhook_url(body.webhook_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Slack Webhook URL은 https://{settings_store.SLACK_WEBHOOK_DOMAIN}/... 형식이어야 합니다.",
        )
    store = _store()
    settings_store.save_slack(store, webhook_url=body.webhook_url, channel=body.channel)
    return settings_store.get_settings_summary(store)


@router.post("/slack/test", response_model=SendTestResponse)
def test_slack(body: SlackTestRequest) -> dict[str, Any]:
    webhook_url = body.webhook_url
    if webhook_url is None:
        # D-3: 이미 연결된 채널의 "테스트 발송" -- 저장된 값을 쓴다.
        record = settings_store.get_slack(_store())
        if not record:
            return {"ok": False, "error": "연결된 Slack 채널이 없습니다."}
        webhook_url = record["webhook_url"]
    if not settings_store.is_valid_slack_webhook_url(webhook_url):
        return {"ok": False, "error": f"Slack Webhook URL은 https://{settings_store.SLACK_WEBHOOK_DOMAIN}/... 형식이어야 합니다."}
    ok, error = senders.send_slack_test(webhook_url)
    return {"ok": ok, "error": error}


@router.post("/telegram/verify", response_model=NotificationSettingsSummary)
def verify_telegram(body: TelegramVerifyRequest) -> dict[str, Any]:
    resolved = telegram_bot.resolve_code(body.code)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="코드가 만료되었거나 올바르지 않습니다. 봇에게 /start를 다시 보내주세요.",
        )
    store = _store()
    settings_store.save_telegram(store, chat_id=resolved["chat_id"], username=resolved["username"])
    return settings_store.get_settings_summary(store)


@router.post("/telegram/test", response_model=SendTestResponse)
def test_telegram() -> dict[str, Any]:
    if not settings.telegram_bot_token:
        return {"ok": False, "error": "Telegram 봇이 서버에 설정되지 않았습니다."}
    record = settings_store.get_telegram(_store())
    if not record:
        return {"ok": False, "error": "연결된 Telegram 계정이 없습니다."}
    ok, error = senders.send_telegram_test(settings.telegram_bot_token, record["chat_id"])
    return {"ok": ok, "error": error}


@router.post("/gmail", response_model=NotificationSettingsSummary)
def connect_gmail(body: GmailConnectRequest, request: Request) -> dict[str, Any]:
    if not settings_store.is_valid_email(body.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="올바른 이메일 주소를 입력하세요.")
    store = _store()
    token = settings_store.start_gmail_verification(store, email=body.email)
    if settings.smtp_host and settings.smtp_user and settings.smtp_password and settings.smtp_from_email:
        # A-7: 명시적으로 설정된 게 없으면 이 요청을 받은 API 서버 자신의
        # 오리진을 쓴다 -- verify 라우트는 FastAPI에만 있고 Next.js에는
        # 대응하는 rewrite가 없으므로, 프런트엔드 오리진을 기본값으로
        # 쓰면 메일 링크가 항상 Next 404로 간다.
        verify_base = settings.notify_verify_base_url or str(request.base_url).rstrip("/")
        verify_url = f"{verify_base.rstrip('/')}/api/notify/gmail/verify?token={token}"
        ok, error = senders.send_gmail(
            smtp_host=settings.smtp_host,
            smtp_port=settings.smtp_port,
            smtp_user=settings.smtp_user,
            smtp_password=settings.smtp_password,
            from_email=settings.smtp_from_email,
            to_email=body.email,
            subject="[SUNI] 이메일 알림 연결 확인",
            html_body=f"<p>아래 링크를 클릭하면 SUNI 알람 이메일 알림 연결이 완료됩니다.</p><p><a href='{verify_url}'>이메일 연결 확인하기</a></p>",
        )
        if not ok:
            logger.warning("인증 메일 발송 실패: %s", error)
    else:
        logger.warning("SMTP가 설정되지 않아 인증 메일을 발송하지 못했습니다 (email=%s)", body.email)
    return settings_store.get_settings_summary(store)


@router.get("/gmail/verify", response_class=HTMLResponse)
def verify_gmail(token: str) -> str:
    ok = settings_store.verify_gmail(_store(), token=token)
    title = "이메일 연결 완료" if ok else "인증 실패"
    message = (
        "SUNI 알람 이메일 알림 연결이 완료되었습니다. 이 창을 닫고 대시보드로 돌아가세요."
        if ok
        else "인증 링크가 올바르지 않거나 이미 사용되었습니다. 대시보드에서 다시 연결해 주세요."
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
    <style>body{{font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#f4f5f7}}
    .card{{background:#fff;padding:32px 40px;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.08);max-width:420px;text-align:center}}
    h1{{font-size:18px;margin:0 0 12px}}p{{color:#555;font-size:14px;line-height:1.6}}</style></head>
    <body><div class="card"><h1>{title}</h1><p>{message}</p></div></body></html>"""


@router.post("/gmail/test", response_model=SendTestResponse)
def test_gmail() -> dict[str, Any]:
    if not (settings.smtp_host and settings.smtp_user and settings.smtp_password and settings.smtp_from_email):
        return {"ok": False, "error": "메일 발송이 서버에 설정되지 않았습니다."}
    record = settings_store.get_gmail(_store())
    if not record or not record.get("verified"):
        return {"ok": False, "error": "인증이 완료된 이메일이 없습니다."}
    ok, error = senders.send_gmail_test(
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        smtp_user=settings.smtp_user,
        smtp_password=settings.smtp_password,
        from_email=settings.smtp_from_email,
        to_email=record["email"],
    )
    return {"ok": ok, "error": error}


@router.post("/conditions", response_model=NotificationSettingsSummary)
def save_conditions(body: ConditionsSaveRequest) -> dict[str, Any]:
    store = _store()
    try:
        settings_store.save_conditions(store, grades=body.grades, timing=body.timing)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return settings_store.get_settings_summary(store)


@router.delete("/{channel}", response_model=NotificationSettingsSummary)
def disconnect_channel(channel: str) -> dict[str, Any]:
    if channel not in ("slack", "telegram", "gmail"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="알 수 없는 채널입니다.")
    store = _store()
    settings_store.disconnect(store, channel)  # type: ignore[arg-type]
    return settings_store.get_settings_summary(store)


def _target_sensitivity_from_payload(payload: dict[str, Any] | None) -> tuple[float, float]:
    """A-3: 수율 예측 탭(알람 삼각형과 같은 화면)에 저장된 목표 수율·
    민감도를 읽어온다 -- 발송이 이 값을 무시하고 항상 기본값(88.0/0.2)을
    쓰면 알림 이력·원인 분석·발송 세 곳의 판정 기준이 어긋난다. 저장된
    적이 없으면(키 없음) 기존 기본값을 그대로 쓴다.
    """
    payload = payload or {}
    target = payload.get("targetYield", alarm_gbdt.DEFAULT_TARGET_YIELD)
    sensitivity = payload.get("sensitivity", alarm_gbdt.DEFAULT_SENSITIVITY)
    return target, sensitivity


@router.post("/dispatch", response_model=DispatchResponse)
def dispatch_now(body: DispatchRequest) -> dict[str, Any]:
    """§C-4 "분석 실행 직후" 발송 트리거 -- 원인 분석 실행이 끝난 직후
    프런트엔드가 fire-and-forget으로 호출한다. 알람/신뢰도는 이미
    lru_cache된 계산을 재사용하므로 대개 즉시 끝난다."""
    store = _store()
    alarms_state = get_latest_state(store).get("alarms")
    target, sensitivity = _target_sensitivity_from_payload((alarms_state or {}).get("payload"))
    auc_lo, gate_passed = _auc_gate(body.train_dataset, body.eval_dataset)
    if not gate_passed:
        reason = (
            f"AUC 하한 {auc_lo:.3f}가 발송 기준 {alarm_gbdt.AUC_GATE:.2f} 미만입니다."
            if auc_lo is not None else "AUC 하한을 산출할 수 없어 외부 알림을 차단했습니다."
        )
        store.record_refresh_dispatch(
            new_alarm_count=0, blocked_reason=reason, summarized=False, channels={}, source="manual_dispatch"
        )
        return {"skipped": True, "reason": reason, "sent_count": None, "results": None}
    items = compute_alarm_notification_items(body.train_dataset, body.eval_dataset, target=target, sensitivity=sensitivity)
    if items is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.")

    reliability = _cached_reliability(body.train_dataset, body.eval_dataset)
    registry = get_dataset_registry()
    summary = registry.get_summary(body.eval_dataset)
    dataset_label = summary["original_filename"] if summary else body.eval_dataset
    active = store.active_model() or {}

    result = dispatch.dispatch_alarm_notifications(
        store,
        trigger=settings_store.TIMING_ON_ANALYSIS,
        dataset_id=body.eval_dataset,
        dataset_label=dataset_label,
        alarms=items,
        reliability_grade=reliability["grade"],
        reliability_score=reliability["total_score"],
        dashboard_url=body.dashboard_url,
        target_yield=target,
        sensitivity=sensitivity,
        model_version=active.get("active_model_id") or "",
        criteria_version=alarm_gbdt.ALARM_DECISION_VERSION,
    )
    _dispatch_yield_update_for_manual_analysis(
        store,
        train_dataset=body.train_dataset,
        eval_dataset=body.eval_dataset,
        dataset_label=dataset_label,
        model_label=active.get("active_model_id"),
        dashboard_url=body.dashboard_url,
    )
    return result


@router.post("/yield-update/dispatch", response_model=DispatchResponse)
def dispatch_yield_update_now(body: DispatchRequest) -> dict[str, Any]:
    """YD: 수율 예측 화면의 "알림 전송" 버튼 -- 사용자가 직접 눌러
    지금 보낸다. `/dispatch`(옛 알람 발송)와 달리 AUC 게이트·목표
    수율/민감도와 무관하다 -- 억제 규칙(신규분만·시간당 예산·수동
    최소 간격 10분)은 `dispatch_yield_update`가 그대로 적용한다."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.analysis.yield_prediction import build_yield_prediction_table
    from src.notifications.yield_update_dispatch import TRIGGER_MANUAL_BUTTON, dispatch_yield_update
    from src.notifications.yield_update_senders import build_yield_update_payload

    store = _store()
    train_df = _dataframe_or_404(body.train_dataset)
    eval_df = _dataframe_or_404(body.eval_dataset)
    hydrated = _hydrated_targets_or_409(body.eval_dataset)
    registry = get_dataset_registry()
    table = build_yield_prediction_table(
        train_df,
        eval_df,
        hydrated.dataframe,
        dataset_id=body.eval_dataset,
        train_dataset_id=body.train_dataset,
        train_dataset_version=registry.content_version(body.train_dataset),
    )

    summary = registry.get_summary(body.eval_dataset)
    dataset_label = summary["original_filename"] if summary else body.eval_dataset
    active = store.active_model() or {}
    timestamp_label = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H:%M")
    payload = build_yield_update_payload(
        table,
        dataset_label=dataset_label,
        timestamp_label=timestamp_label,
        model_label=active.get("active_model_id"),
        dashboard_url=body.dashboard_url,
    )
    result = dispatch_yield_update(store, payload, trigger=TRIGGER_MANUAL_BUTTON)
    if result.get("skipped"):
        return {"skipped": True, "reason": result.get("reason"), "sent_count": None, "results": None}
    channel_results = result.get("results") or {}
    sent_count = sum(1 for item in channel_results.values() if item.get("ok"))
    return {"skipped": False, "reason": None, "sent_count": sent_count, "results": channel_results}


def _dispatch_yield_update_for_manual_analysis(
    store: RuntimeStore,
    *,
    train_dataset: str,
    eval_dataset: str,
    dataset_label: str,
    model_label: str | None,
    dashboard_url: str | None,
) -> None:
    """VE-1: "분석 실행 직후" 발송 트리거 -- 알람과 별개로 수율 예측
    갱신도 같은 시점에 발송한다(conditions["timing"]에 on_analysis가
    없으면 yield_update_dispatch가 스킵한다). 실패해도 알람 발송 결과
    응답에는 영향을 주지 않는다(best-effort)."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from src.analysis.yield_prediction import build_yield_prediction_table
        from src.notifications.yield_update_dispatch import TRIGGER_MANUAL, dispatch_yield_update
        from src.notifications.yield_update_senders import build_yield_update_payload

        train_df = _dataframe_or_404(train_dataset)
        eval_df = _dataframe_or_404(eval_dataset)
        hydrated = _hydrated_targets_or_409(eval_dataset)
        table = build_yield_prediction_table(
            train_df,
            eval_df,
            hydrated.dataframe,
            dataset_id=eval_dataset,
            train_dataset_id=train_dataset,
            train_dataset_version=get_dataset_registry().content_version(train_dataset),
        )
        timestamp_label = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H:%M")
        payload = build_yield_update_payload(
            table,
            dataset_label=dataset_label,
            timestamp_label=timestamp_label,
            model_label=model_label,
            dashboard_url=dashboard_url,
        )
        dispatch_yield_update(store, payload, trigger=TRIGGER_MANUAL)
    except Exception:
        logger.exception("수율 예측 갱신(분석 실행 직후) 발송 처리 실패")


def _run_scheduled_dispatch_job(trigger: str, *, label: str) -> None:
    """APScheduler가 매일 정해진 시각에 호출하는 정기 발송 잡의 공통
    본문 -- 09:00/13:00 둘 다 이 함수를 쓴다(DF그룹). 가장 최근에 저장된
    알람 조회(사전 알람 로그 탭에서 마지막으로 조회한 train/eval
    데이터셋 쌍)를 기준으로 재계산해 발송한다 -- 저장된 조회가 없으면
    (한 번도 조회한 적이 없으면) 조용히 건너뛴다.

    "신규분만" 정책(DF그룹)은 여기서 별도로 구현하지 않는다 --
    `dispatch.dispatch_alarm_notifications`가 이미 채널별 24시간 dedupe로
    "직전 발송에서 이미 보낸(등급도 그대로인) 항목"을 걸러내므로, 전체
    후보를 그대로 넘기기만 하면 자연히 신규분만 나간다. 09:00과 13:00이
    4~20시간 간격이라 이 dedupe 창(24시간) 안에 항상 들어온다.
    """
    store = _store()
    try:
        latest = get_latest_state(store)
        alarms_state = latest.get("alarms")
        if not alarms_state:
            logger.info("%s 알림 발송 스킵: 저장된 알람 조회 없음", label)
            return
        train_dataset = alarms_state.get("train_dataset")
        eval_dataset = alarms_state.get("eval_dataset")
        if not train_dataset or not eval_dataset:
            return
        # A-3: 데이터셋만 읽고 target/sensitivity는 버리던 버그 -- 같은
        # alarms_state.payload에서 함께 읽어야 원인 분석 삼각형과 같은
        # 기준으로 판정한다.
        target, sensitivity = _target_sensitivity_from_payload(alarms_state.get("payload"))

        auc_lo, gate_passed = _auc_gate(train_dataset, eval_dataset)
        if not gate_passed:
            reason = (
                f"AUC 하한 {auc_lo:.3f}가 발송 기준 {alarm_gbdt.AUC_GATE:.2f} 미만입니다."
                if auc_lo is not None else "AUC 하한을 산출할 수 없어 외부 알림을 차단했습니다."
            )
            store.record_refresh_dispatch(
                new_alarm_count=0, blocked_reason=reason, summarized=False, channels={}, source=label
            )
            return

        items = compute_alarm_notification_items(train_dataset, eval_dataset, target=target, sensitivity=sensitivity)
        if items is None:
            return
        reliability = _cached_reliability(train_dataset, eval_dataset)
        registry = get_dataset_registry()
        summary = registry.get_summary(eval_dataset)
        dataset_label = summary["original_filename"] if summary else eval_dataset
        active = store.active_model() or {}

        dispatch.dispatch_alarm_notifications(
            store,
            trigger=trigger,
            dataset_id=eval_dataset,
            dataset_label=dataset_label,
            alarms=items,
            reliability_grade=reliability["grade"],
            reliability_score=reliability["total_score"],
            target_yield=target,
            sensitivity=sensitivity,
            model_version=active.get("active_model_id") or "",
            criteria_version=alarm_gbdt.ALARM_DECISION_VERSION,
        )
    except Exception:
        logger.exception("%s 알림 발송 잡 실행 실패", label)


def run_daily_dispatch_job() -> None:
    """APScheduler가 매일 09:00에 호출한다 (지시서 N-2: 8시 -> 9시)."""
    _run_scheduled_dispatch_job(settings_store.TIMING_DAILY_9AM, label="09:00")


def run_daily_13_dispatch_job() -> None:
    """DF그룹: APScheduler가 매일 13:00에 호출한다. 발송 조건(등급/신뢰도
    게이트)과 신규분 판정(24시간 dedupe 재사용)은 09:00 잡과 완전히
    같다 -- trigger만 다르다."""
    _run_scheduled_dispatch_job(settings_store.TIMING_DAILY_13, label="13:00")


# H-3②: notify_sent_log는 24시간 재발송 방지 조회에만 쓰이므로 그보다 훨씬
# 오래된 행은 볼 일이 없다 -- 지우지 않으면 무한히 커진다. 발송 잡과
# 겹치지 않도록 별도 스케줄 id로 등록한다(api/main.py).
NOTIFY_LOG_RETENTION_DAYS = 30


def run_notify_log_cleanup_job() -> None:
    """APScheduler가 주기적으로 호출해 오래된 notify_sent_log 행을 지운다."""
    store = _store()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=NOTIFY_LOG_RETENTION_DAYS)).isoformat()
        deleted = store.purge_old_notification_log(older_than_iso=cutoff)
        logger.info("notify_sent_log 정리: %d건 삭제 (%d일 이전)", deleted, NOTIFY_LOG_RETENTION_DAYS)
    except Exception:
        logger.exception("notify_sent_log 정리 잡 실행 실패")
