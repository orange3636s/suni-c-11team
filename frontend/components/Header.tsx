"use client";

import { useEffect, useState } from "react";

import { getApiHealth } from "@/lib/api";
import StatusBadge from "@/components/StatusBadge";

type ApiState = "확인 중" | "연결됨" | "연결 실패";

export default function Header() {
  const [apiState, setApiState] = useState<ApiState>("확인 중");
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [currentTime, setCurrentTime] = useState<Date | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function checkApi() {
      try {
        const result = await getApiHealth();
        if (isMounted) {
          setApiState(result.status === "ok" ? "연결됨" : "연결 실패");
          setModelReady(result.model_directory_ready ?? null);
        }
      } catch {
        if (isMounted) {
          setApiState("연결 실패");
        }
      }
    }

    void checkApi();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    function updateCurrentTime() {
      setCurrentTime(new Date());
    }

    updateCurrentTime();
    const intervalId = window.setInterval(updateCurrentTime, 1000);
    return () => window.clearInterval(intervalId);
  }, []);

  const apiStatusClass =
    apiState === "연결됨"
      ? "normal"
      : apiState === "연결 실패"
        ? "danger"
        : "warning";
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
        <h1>제조 공정 불량 예측 &amp; 불량 원인 분석 AI</h1>
      </div>
      <div className="headerMeta" aria-label="시스템 연결 상태">
        <div className="headerStatusGroup">
          <span className="headerStatusLabel">API Status</span>
          <StatusBadge
            label={apiState}
            tone={
              apiStatusClass === "normal"
                ? "success"
                : apiStatusClass === "danger"
                  ? "danger"
                  : "warning"
            }
          />
        </div>
        <div className="headerStatusGroup">
          <span className="headerStatusLabel">Model Status</span>
          <StatusBadge
            label={
              modelReady === true
                ? "Ready"
                : modelReady === false
                  ? "Not Ready"
                  : "확인 중"
            }
            tone={
              modelReady === true
                ? "success"
                : modelReady === false
                  ? "danger"
                  : "neutral"
            }
          />
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
