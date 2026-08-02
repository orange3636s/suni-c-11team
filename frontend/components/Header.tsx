"use client";

import { useEffect, useState } from "react";

type ApiStatus = "checking" | "online" | "offline";

function apiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";
}

export default function Header() {
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");

  useEffect(() => {
    function updateCurrentTime() {
      setCurrentTime(new Date());
    }

    updateCurrentTime();
    const intervalId = window.setInterval(updateCurrentTime, 1000);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    let disposed = false;
    let controller: AbortController | null = null;
    const check = async () => {
      controller?.abort(); controller = new AbortController();
      const timeout = window.setTimeout(() => controller?.abort(), 5000);
      setApiStatus("checking");
      try {
        const response = await fetch(`${apiBaseUrl()}/health`, { signal: controller.signal, cache: "no-store" });
        const body = await response.json() as { status?: string };
        if (!disposed) setApiStatus(response.ok && body.status === "ok" ? "online" : "offline");
      } catch {
        if (!disposed) setApiStatus("offline");
      } finally { window.clearTimeout(timeout); }
    };
    void check();
    const interval = window.setInterval(() => void check(), 30_000);
    const visible = () => { if (document.visibilityState === "visible") void check(); };
    document.addEventListener("visibilitychange", visible);
    return () => { disposed = true; controller?.abort(); window.clearInterval(interval); document.removeEventListener("visibilitychange", visible); };
  }, []);

  const currentDate = currentTime
    ? [
        currentTime.getFullYear(),
        String(currentTime.getMonth() + 1).padStart(2, "0"),
        String(currentTime.getDate()).padStart(2, "0"),
      ].join("-")
    : "---- -- --";
  const currentClock = currentTime
    ? [
        String(currentTime.getHours()).padStart(2, "0"),
        String(currentTime.getMinutes()).padStart(2, "0"),
        String(currentTime.getSeconds()).padStart(2, "0"),
      ].join(":")
    : "--:--:--";

  return (
    <header className="topHeader">
      <div className="headerContext">
        <h1>제조 공정 불량 예측 &amp; 원인 분석 AI</h1>
      </div>
      <div className="headerMeta" aria-label="현재 시각">
        <div className={`apiStatus apiStatus-${apiStatus}`} title={apiStatus === "online" ? "API 서버가 정상적으로 연결되어 있습니다." : apiStatus === "offline" ? "API 서버에 연결할 수 없습니다." : "API 연결 상태를 확인하고 있습니다."}>
          <span aria-hidden="true" /><strong>API Status</strong><small>{apiStatus === "online" ? "정상" : apiStatus === "offline" ? "연결 끊김" : "확인 중"}</small>
        </div>
        <div className="headerStatusGroup currentTimeGroup">
          <span className="headerStatusLabel">Current Time</span>
          <time dateTime={currentTime?.toISOString()}>
            <span>{currentDate}</span>
            <strong>{currentClock}</strong>
          </time>
        </div>
      </div>
    </header>
  );
}
