from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from api.routes.analysis import _cached_reliability, compute_alarm_notification_items
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
from src.notifications import dispatch, senders, settings_store, telegram_bot
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
    if not settings_store.is_valid_slack_webhook_url(body.webhook_url):
        return {"ok": False, "error": f"Slack Webhook URL은 https://{settings_store.SLACK_WEBHOOK_DOMAIN}/... 형식이어야 합니다."}
    ok, error = senders.send_slack_test(body.webhook_url)
    return {"ok": ok, "error": error}


@router.post("/telegram/verify", response_model=NotificationSettingsSummary)
def verify_telegram(body: TelegramVerifyRequest) -> dict[str, Any]:
    resolved = telegram_bot.resolve_code(body.code)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="인증 코드가 올바르지 않거나 만료되었습니다. 봇에게 /start를 다시 보내 새 코드를 받으세요.",
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
def connect_gmail(body: GmailConnectRequest) -> dict[str, Any]:
    if not settings_store.is_valid_email(body.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="올바른 이메일 주소를 입력하세요.")
    store = _store()
    token = settings_store.start_gmail_verification(store, email=body.email)
    if settings.smtp_host and settings.smtp_user and settings.smtp_password and settings.smtp_from_email:
        verify_url = f"{settings.notify_verify_base_url.rstrip('/')}/api/notify/gmail/verify?token={token}"
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


@router.post("/dispatch", response_model=DispatchResponse)
def dispatch_now(body: DispatchRequest) -> dict[str, Any]:
    """§C-4 "분석 실행 직후" 발송 트리거 -- 원인 분석 실행이 끝난 직후
    프런트엔드가 fire-and-forget으로 호출한다. 알람/신뢰도는 이미
    lru_cache된 계산을 재사용하므로 대개 즉시 끝난다."""
    items = compute_alarm_notification_items(body.train_dataset, body.eval_dataset)
    if items is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.")

    reliability = _cached_reliability(body.train_dataset)
    registry = get_dataset_registry()
    summary = registry.get_summary(body.train_dataset)
    dataset_label = summary["original_filename"] if summary else body.train_dataset

    result = dispatch.dispatch_alarm_notifications(
        _store(),
        dataset_id=body.train_dataset,
        dataset_label=dataset_label,
        alarms=items,
        reliability_grade=reliability["grade"],
        reliability_score=reliability["total_score"],
        dashboard_url=body.dashboard_url,
    )
    return result


def run_daily_dispatch_job() -> None:
    """APScheduler가 매일 08:00에 호출한다 (spec §C-4 "매일 오전 8시").
    n8n 같은 별도 서비스 없이 FastAPI 프로세스 안에서 도는 잡이다. 가장
    최근에 저장된 알람 조회(사전 알람 로그 탭에서 마지막으로 조회한
    train/eval 데이터셋 쌍)를 기준으로 재계산해 발송한다 -- 저장된 조회가
    없으면(한 번도 조회한 적이 없으면) 조용히 건너뛴다.
    """
    from src.runtime.app_state import get_latest_state

    store = _store()
    try:
        latest = get_latest_state(store)
        alarms_state = latest.get("alarms")
        if not alarms_state:
            logger.info("일일 알림 발송 스킵: 저장된 알람 조회 없음")
            return
        train_dataset = alarms_state.get("train_dataset")
        eval_dataset = alarms_state.get("eval_dataset")
        if not train_dataset or not eval_dataset:
            return

        items = compute_alarm_notification_items(train_dataset, eval_dataset)
        if items is None:
            return
        reliability = _cached_reliability(train_dataset)
        registry = get_dataset_registry()
        summary = registry.get_summary(train_dataset)
        dataset_label = summary["original_filename"] if summary else train_dataset

        dispatch.dispatch_alarm_notifications(
            store,
            dataset_id=train_dataset,
            dataset_label=dataset_label,
            alarms=items,
            reliability_grade=reliability["grade"],
            reliability_score=reliability["total_score"],
        )
    except Exception:
        logger.exception("일일 알림 발송 잡 실행 실패")
