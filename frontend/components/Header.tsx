"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { usePanelState } from "@/components/PanelStateProvider";
import SuniAvatar from "@/components/SuniAvatar";
import { useApiStatus } from "@/lib/useApiStatus";
import { useIsMobileLayout, useIsTabBarLayout } from "@/lib/useMediaQuery";

export default function Header() {
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  // Sidebar의 연결 상태 점과 같은 훅을 구독한다 -- 폴링은 앱 전체에서
  // 한 번만 돈다 (frontend/lib/useApiStatus.ts).
  const apiStatus = useApiStatus();
  const headerRef = useRef<HTMLElement>(null);
  const { aiPanelOpen, setAiPanelOpen } = usePanelState();
  // ≤1023px: AiPanel no longer renders its own always-visible floating
  // toggle circle once it's an overlay/full-screen drawer (spec §B-5) --
  // this header button is the only way to open it there (spec §B-7:
  // "SUNI 버튼은 항상 표시"). ≥1024px keeps using AiPanel's own circle,
  // unchanged, so this button stays hidden there.
  const isTabBarLayout = useIsTabBarLayout();
  const isMobileLayout = useIsMobileLayout();

  // --header-height drives both the sticky Y1~Y5 segment's `top` offset
  // and the sidebar/AI panel's clearance below the header -- measuring
  // the real rendered box (border included) instead of hardcoding a
  // guess is what keeps the segment from sticking a few pixels too high
  // and rendering partly behind the header (spec §1-3).
  useEffect(() => {
    const node = headerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const setHeightVar = () => {
      const height = node.getBoundingClientRect().height;
      if (height > 0) document.documentElement.style.setProperty("--header-height", `${Math.ceil(height)}px`);
    };
    setHeightVar();
    const observer = new ResizeObserver(setHeightVar);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

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
    <header className="topHeader" ref={headerRef}>
      <div className="headerContext">
        {/* SUNI C brand mark -- separate from the sidebar's character logo
            (SuniAvatar), not a replacement for it. Links to the model
            training tab like any header logo would. ≤767px: the full
            wordmark image gives way to the compact character mark alone
            (spec §B-7: "로고는 767px 이하에서 아이콘만") -- SuniAvatar is
            the only square icon-shaped brand asset in the project;
            there's no separate icon-only crop of the wordmark to use. */}
        <Link href="/training" className="headerLogoLink" aria-label="SUNI C - 모델 학습으로 이동">
          {isMobileLayout ? (
            <SuniAvatar size={28} />
          ) : (
            <Image src="/suni-c-logo.png" alt="SUNI C" width={122} height={32} unoptimized priority className="headerLogo" />
          )}
        </Link>
        <h1>{isMobileLayout ? "불량 예측 & 원인 분석" : "제조 공정 불량 예측 & 원인 분석 AI"}</h1>
      </div>
      <div className="headerMeta" aria-label="현재 시각">
        {!isMobileLayout && (
          <>
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
          </>
        )}
        {isTabBarLayout && (
          <button
            type="button"
            className="headerSuniButton"
            onClick={() => setAiPanelOpen(true)}
            aria-label={aiPanelOpen ? "SUNI 패널 열림" : "SUNI 열기"}
            aria-expanded={aiPanelOpen}
          >
            <SuniAvatar size={22} />
          </button>
        )}
      </div>
    </header>
  );
}
