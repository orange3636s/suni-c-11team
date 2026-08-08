"use client";

import { useEffect, useState } from "react";
import { getApiBaseUrl } from "@/lib/api";

export type ApiStatus = "checking" | "online" | "offline";

// 모듈 전역 싱글턴 -- Header와 Sidebar가 각자 폴링하면 /health를 이중으로
// 두들기게 되므로, 폴링은 첫 구독자가 한 번만 시작하고 이후 구독자는 같은
// 인터벌의 결과를 구독만 한다.
let currentStatus: ApiStatus = "checking";
const listeners = new Set<(status: ApiStatus) => void>();
let pollingStarted = false;

function broadcast(status: ApiStatus) {
  currentStatus = status;
  listeners.forEach((listener) => listener(status));
}

function startPolling() {
  if (pollingStarted) return;
  pollingStarted = true;
  let controller: AbortController | null = null;

  const check = async () => {
    controller?.abort();
    controller = new AbortController();
    const timeout = window.setTimeout(() => controller?.abort(), 5000);
    broadcast("checking");
    try {
      const response = await fetch(`${getApiBaseUrl()}/health`, { signal: controller.signal, cache: "no-store" });
      const body = (await response.json()) as { status?: string };
      broadcast(response.ok && body.status === "ok" ? "online" : "offline");
    } catch {
      broadcast("offline");
    } finally {
      window.clearTimeout(timeout);
    }
  };

  void check();
  window.setInterval(() => void check(), 30_000);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") void check();
  });
}

/** Header의 API Status 배지와 Sidebar 하단의 연결 상태 점이 공유하는 훅 --
 * 두 곳 모두 이 훅을 구독하지만 실제 /health 폴링은 앱 전체에서 한 번만
 * 돈다(모듈 전역 인터벌). */
export function useApiStatus(): ApiStatus {
  // 초기값은 useState 이니셜라이저가 마운트 시점의 currentStatus를 그대로
  // 잡는다 -- 이후 변화는 listeners 구독으로만 반영하면 되므로, 이펙트
  // 안에서 다시 setStatus(currentStatus)를 부를 필요가 없다(그 호출은
  // "이펙트 본문에서 곧장 setState" 린트 규칙에도 걸린다).
  const [status, setStatus] = useState(currentStatus);
  useEffect(() => {
    startPolling();
    listeners.add(setStatus);
    return () => {
      listeners.delete(setStatus);
    };
  }, []);
  return status;
}
