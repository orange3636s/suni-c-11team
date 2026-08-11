from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from api.routes.analysis import _dataframe_or_404, _hydrated_targets_or_409
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
from src.notifications import senders, settings_store, telegram_bot
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


def _dispatch_yield_update(
    store: RuntimeStore,
    *,
    train_dataset: str,
    eval_dataset: str,
    trigger: str,
    dashboard_url: str | None = None,
) -> dict[str, Any]:
    """공유 헬퍼 -- 수율 예측 화면(alerts/page.tsx)이 쓰는 것과 같은
    `build_yield_prediction_table`로 `YieldUpdatePayload`를 만들고
    `dispatch_yield_update`를 호출한다. `/dispatch`(분석 실행 직후),
    `/yield-update/dispatch`(YD 버튼), 09:00/13:00 스케줄 잡이 모두 이
    구성을 공유한다 -- 호출부마다 따로 만들지 않는다. 데이터셋을 찾을
    수 없으면 `_dataframe_or_404`/`_hydrated_targets_or_409`가 그대로
    HTTPException을 올린다(호출부가 각자의 정책대로 처리한다: 라우트는
    전파, 스케줄 잡은 삼킨다)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.analysis.yield_prediction import build_yield_prediction_table
    from src.notifications.yield_update_dispatch import dispatch_yield_update
    from src.notifications.yield_update_senders import build_yield_update_payload

    train_df = _dataframe_or_404(train_dataset)
    eval_df = _dataframe_or_404(eval_dataset)
    hydrated = _hydrated_targets_or_409(eval_dataset)
    registry = get_dataset_registry()
    table = build_yield_prediction_table(
        train_df,
        eval_df,
        hydrated.dataframe,
        dataset_id=eval_dataset,
        train_dataset_id=train_dataset,
        train_dataset_version=registry.content_version(train_dataset),
    )
    summary = registry.get_summary(eval_dataset)
    dataset_label = summary["original_filename"] if summary else eval_dataset
    active = store.active_model() or {}
    timestamp_label = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H:%M")
    payload = build_yield_update_payload(
        table,
        dataset_label=dataset_label,
        timestamp_label=timestamp_label,
        model_label=active.get("active_model_id"),
        dashboard_url=dashboard_url,
    )
    return dispatch_yield_update(store, payload, trigger=trigger)


def _dispatch_response(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("skipped"):
        return {"skipped": True, "reason": result.get("reason"), "sent_count": None, "results": None}
    channel_results = result.get("results") or {}
    sent_count = sum(1 for item in channel_results.values() if item.get("ok"))
    return {"skipped": False, "reason": None, "sent_count": sent_count, "results": channel_results}


@router.post("/dispatch", response_model=DispatchResponse)
def dispatch_now(body: DispatchRequest) -> dict[str, Any]:
    """§C-4 "분석 실행 직후" 발송 트리거 -- 원인 분석 실행이 끝난 직후
    프런트엔드가 fire-and-forget으로 호출한다. 옛 알람 등급/AUC 게이트
    파이프라인은 폐기됐다 -- 수율 예측 갱신 파이프라인(TRIGGER_MANUAL)
    으로 발송하며, 저장된 발송 시점 설정에 '분석 실행 직후'가 포함됐을
    때만 실제로 보낸다(dispatch_yield_update 내부 판단)."""
    from src.notifications.yield_update_dispatch import TRIGGER_MANUAL

    store = _store()
    result = _dispatch_yield_update(
        store,
        train_dataset=body.train_dataset,
        eval_dataset=body.eval_dataset,
        trigger=TRIGGER_MANUAL,
        dashboard_url=body.dashboard_url,
    )
    return _dispatch_response(result)


@router.post("/yield-update/dispatch", response_model=DispatchResponse)
def dispatch_yield_update_now(body: DispatchRequest) -> dict[str, Any]:
    """YD: 수율 예측 화면의 "알림 전송" 버튼 -- 사용자가 직접 눌러
    지금 보낸다. `/dispatch`(분석 실행 직후)와 달리 발송 시점 설정과
    무관하게 시도한다 -- 억제 규칙(시간당 예산·수동 최소 간격 10분)은
    `dispatch_yield_update`가 그대로 적용한다."""
    from src.notifications.yield_update_dispatch import TRIGGER_MANUAL_BUTTON

    store = _store()
    result = _dispatch_yield_update(
        store,
        train_dataset=body.train_dataset,
        eval_dataset=body.eval_dataset,
        trigger=TRIGGER_MANUAL_BUTTON,
        dashboard_url=body.dashboard_url,
    )
    return _dispatch_response(result)


def _run_scheduled_dispatch_job(trigger: str, *, label: str) -> None:
    """APScheduler가 매일 정해진 시각에 호출하는 정기 발송 잡의 공통
    본문 -- 09:00/13:00 둘 다 이 함수를 쓴다(DF그룹). 가장 최근 자동
    갱신 스냅샷(`run_refresh_pipeline`이 매 주기마다 갱신)의
    train/eval 데이터셋 쌍을 기준으로 수율 예측 갱신을 다시 만들어
    발송한다.

    이전에는 `GET /api/state/alarms`에 저장된(사전 알람 로그 탭이
    쓰던) train/eval 쌍을 읽었으나, 그 저장 경로를 부르는 화면이 이제
    없어 이 값이 항상 비어 있었다 -- 두 잡 모두 실제로는 한 번도
    발송하지 못하는 상태였다(Task D 검증 중 확인). 자동 갱신 스냅샷은
    APScheduler `auto_refresh` 잡이 주기적으로 채우므로 항상 최신값을
    가진다.
    """
    store = _store()
    try:
        snapshot = store.get_refresh_snapshot_status().get("snapshot")
        source = (snapshot or {}).get("source") or {}
        train_dataset = source.get("train_dataset")
        eval_dataset = source.get("eval_dataset")
        if not train_dataset or not eval_dataset:
            logger.info("%s 알림 발송 스킵: 저장된 자동 갱신 스냅샷 없음", label)
            return
        _dispatch_yield_update(store, train_dataset=train_dataset, eval_dataset=eval_dataset, trigger=trigger)
    except Exception:
        logger.exception("%s 알림 발송 잡 실행 실패", label)


def run_daily_dispatch_job() -> None:
    """APScheduler가 매일 09:00에 호출한다 (지시서 N-2: 8시 -> 9시)."""
    from src.notifications.yield_update_dispatch import TRIGGER_DAILY_9AM

    _run_scheduled_dispatch_job(TRIGGER_DAILY_9AM, label="09:00")


def run_daily_13_dispatch_job() -> None:
    """DF그룹: APScheduler가 매일 13:00에 호출한다. trigger만 09:00
    잡과 다르다."""
    from src.notifications.yield_update_dispatch import TRIGGER_DAILY_13

    _run_scheduled_dispatch_job(TRIGGER_DAILY_13, label="13:00")


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
