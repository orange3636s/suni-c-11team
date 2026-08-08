"""Tests for src/notifications/telegram_bot.py's in-memory code exchange
(spec §C-3 Telegram: bots can't message a user first, so a /start-issued
code is the only way to learn a chat_id)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from src.notifications import telegram_bot


def test_register_and_resolve_code_roundtrip():
    code = telegram_bot.register_start("555", "@tester")
    resolved = telegram_bot.resolve_code(code)
    assert resolved == {"chat_id": "555", "username": "@tester"}


def test_resolve_code_is_single_use():
    code = telegram_bot.register_start("555", "@tester")
    assert telegram_bot.resolve_code(code) is not None
    assert telegram_bot.resolve_code(code) is None


def test_resolve_unknown_code_returns_none():
    assert telegram_bot.resolve_code("000000") is None


def test_expired_code_is_pruned(monkeypatch):
    code = telegram_bot.register_start("777", None)
    # Force it into the past without waiting 10 real minutes.
    telegram_bot._pending_codes[code].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert telegram_bot.resolve_code(code) is None


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_polling_loop_does_not_advance_offset_past_a_failed_update(monkeypatch):
    """D-7 회귀: handler가 예외를 던지면 그 update_id 이하로 offset을
    전진시키면 안 된다 -- 안 그러면 Telegram이 그 /start를 다시는 보내
    주지 않아 영영 사라진다."""
    requests_seen: list[dict | None] = []
    stop_event = asyncio.Event()

    batch_one = [{"update_id": 1, "message": {"text": "/start", "chat": {"id": 111}}}]

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, params=None):
            requests_seen.append(params)
            if len(requests_seen) == 1:
                return _FakeResponse({"ok": True, "result": batch_one})
            # Second poll: confirm no offset was carried past the failed
            # update, then end the loop.
            stop_event.set()
            return _FakeResponse({"ok": True, "result": []})

    async def _boom(bot_token, update):
        raise RuntimeError("simulated handler failure")

    monkeypatch.setattr(telegram_bot.httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient())
    monkeypatch.setattr(telegram_bot, "_handle_update", _boom)

    asyncio.run(telegram_bot.run_polling_loop("fake-token", stop_event))

    assert len(requests_seen) == 2
    assert "offset" not in requests_seen[0]
    # The failed update's id was never used to advance the offset -- the
    # second poll must not request updates starting after it.
    assert "offset" not in requests_seen[1]


def test_polling_loop_advances_offset_only_after_successful_handling(monkeypatch):
    requests_seen: list[dict | None] = []
    handled: list[int] = []
    stop_event = asyncio.Event()

    batch_one = [{"update_id": 5, "message": {"text": "/start", "chat": {"id": 111}}}]

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, params=None):
            requests_seen.append(params)
            if len(requests_seen) == 1:
                return _FakeResponse({"ok": True, "result": batch_one})
            stop_event.set()
            return _FakeResponse({"ok": True, "result": []})

    async def _ok(bot_token, update):
        handled.append(update["update_id"])

    monkeypatch.setattr(telegram_bot.httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient())
    monkeypatch.setattr(telegram_bot, "_handle_update", _ok)

    asyncio.run(telegram_bot.run_polling_loop("fake-token", stop_event))

    assert handled == [5]
    assert requests_seen[1]["offset"] == 6
