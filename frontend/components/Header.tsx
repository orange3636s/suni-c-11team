"use client";

import { useEffect, useState } from "react";

export default function Header() {
  const [currentTime, setCurrentTime] = useState<Date | null>(null);

  useEffect(() => {
    function updateCurrentTime() {
      setCurrentTime(new Date());
    }

    updateCurrentTime();
    const intervalId = window.setInterval(updateCurrentTime, 1000);
    return () => window.clearInterval(intervalId);
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
