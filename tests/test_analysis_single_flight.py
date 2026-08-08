"""Unit tests for api.routes.analysis._single_flight -- B-4: heavy
lru_cache'd GBDT fits (bootstrap ensemble, transfer-AUC folds,
reliability) must not run twice concurrently for the same (train, eval)
key when a scheduler thread and a request thread race on a cold cache.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache

from api.routes.analysis import _single_flight


def test_concurrent_calls_with_same_key_run_underlying_fn_once():
    """Mirrors real usage (`@_single_flight` stacked on `@lru_cache` in
    analysis.py) -- `_single_flight` alone only serializes concurrent
    calls, it doesn't memoize; the dedup-to-one-call property only holds
    once the wrapped function is also cached."""
    call_count = 0
    lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()

    @lru_cache(maxsize=8)
    def slow(key: str) -> str:
        nonlocal call_count
        with lock:
            call_count += 1
        started.set()
        release.wait(timeout=5)
        return f"result-{key}"

    wrapped = _single_flight(slow)
    results: list[str] = []

    def caller():
        results.append(wrapped("same-key"))

    t1 = threading.Thread(target=caller)
    t1.start()
    assert started.wait(timeout=5)  # t1 is inside `slow` now

    t2 = threading.Thread(target=caller)
    t2.start()
    time.sleep(0.05)  # give t2 a chance to (wrongly) start its own call
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert call_count == 1
    assert results == ["result-same-key", "result-same-key"]


def test_different_keys_do_not_block_each_other():
    entered = threading.Event()
    release = threading.Event()

    def slow(key: str) -> str:
        if key == "blocker":
            entered.set()
            release.wait(timeout=5)
        return key

    wrapped = _single_flight(slow)
    result_holder: dict[str, str] = {}

    def block_caller():
        result_holder["blocker"] = wrapped("blocker")

    t1 = threading.Thread(target=block_caller)
    t1.start()
    assert entered.wait(timeout=5)

    # A different key must not wait behind the blocked "blocker" key.
    other_result = wrapped("other")
    assert other_result == "other"

    release.set()
    t1.join(timeout=5)
    assert result_holder["blocker"] == "blocker"
