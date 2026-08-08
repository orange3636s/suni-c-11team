"use client";

import { Menu } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { usePanelState } from "@/components/PanelStateProvider";
import SuniAvatar from "@/components/SuniAvatar";
import { useApiStatus } from "@/lib/useApiStatus";
import { useIsMobileLayout, useIsTabBarLayout } from "@/lib/useMediaQuery";

// 헤더를 "지금 어느 데이터를 보고 있나"를 보여주는 상태 바로 전환 (지시서
// M-2) -- 신규 API 조회를 만들지 않고 사이드바 하단 상태 블록이 이미 쓰는
// 스냅샷 컨텍스트(AnalysisStateProvider의 snapshot, J그룹 자동 갱신
// 파이프라인 산출물)를 그대로 재사용한다.
function formatHeaderClock(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export default function Header() {
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  // Sidebar의 연결 상태 점과 같은 훅을 구독한다 -- 폴링은 앱 전체에서
  // 한 번만 돈다 (frontend/lib/useApiStatus.ts). SOURCE 항목의 title
  // 툴팁으로 흡수되어 더 이상 별도 배지를 그리지 않는다.
  const apiStatus = useApiStatus();
  const { snapshot } = useAnalysisState();
  const headerRef = useRef<HTMLElement>(null);
  const { aiPanelOpen, setAiPanelOpen, setSidebarDrawerOpen } = usePanelState();
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
    <>
    <header className="topHeader" ref={headerRef}>
      <div className="headerContext">
        {/* 모바일 반응형 패치 S-1: ≤767px에서는 <Sidebar>가 오프캔버스
            드로어로 바뀌어 DOM에서 항상 보이지 않으므로, 여는 트리거가
            헤더에 하나는 있어야 한다 -- .headerSuniButton과 같은
            아이콘 버튼 시각 패턴을 그대로 따른다. */}
        {isMobileLayout && (
          <button
            type="button"
            className="headerMenuButton"
            onClick={() => setSidebarDrawerOpen(true)}
            aria-label="메뉴 열기"
            aria-haspopup="dialog"
          >
            <Menu size={16} strokeWidth={1.5} aria-hidden="true" />
          </button>
        )}
        {/* SUNI C brand mark -- separate from the sidebar's character logo
            (SuniAvatar), not a replacement for it. Links to 모니터링 홈 like
            any header logo would (지시서 I: 모델 학습은 더 이상 탭이 아니라
            사이드바 하단의 팝업이라 로고 링크가 될 수 없다). ≤767px: the full
            wordmark image gives way to the compact character mark alone
            (spec §B-7: "로고는 767px 이하에서 아이콘만") -- SuniAvatar is
            the only square icon-shaped brand asset in the project;
            there's no separate icon-only crop of the wordmark to use. */}
        <Link href="/monitoring" className="headerLogoLink" aria-label="SUNI C - 모니터링 홈으로 이동">
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
            <div className="headerContextStrip" aria-label="현재 데이터 컨텍스트">
              {snapshot && (
                <>
                  {/* 모바일 반응형 패치 S-1: 768~1023px(태블릿) 구간에서는
                      SOURCE/LAST RUN만 남기고 EVAL/WAFERS를 줄인다 -- 이
                      두 항목만 새 클래스로 구분해 globals.css의
                      1023px 티어에서 숨긴다. */}
                  <div className="headerContextItem headerContextEval">
                    <span className="headerContextKey">EVAL</span>
                    <span className="headerContextValue">{snapshot.source.eval_dataset}</span>
                  </div>
                  <div className="headerContextItem headerContextWafers">
                    <span className="headerContextKey">WAFERS</span>
                    <span className="headerContextValue">{snapshot.source.row_count.toLocaleString()}</span>
                  </div>
                  <div className="headerContextItem headerContextLastRun">
                    <span className="headerContextKey">LAST RUN</span>
                    <span className="headerContextValue">{formatHeaderClock(snapshot.created_at) ?? "-"}</span>
                  </div>
                </>
              )}
              <div
                className="headerContextItem headerContextSource"
                title={
                  snapshot
                    ? snapshot.source.mode === "sql"
                      ? "SQL 데이터소스에 연결되어 있습니다."
                      : "SQL 미연결 -- 데모(폴백) 데이터를 보고 있습니다."
                    : apiStatus === "online"
                      ? "API 서버가 정상적으로 연결되어 있습니다."
                      : apiStatus === "offline"
                        ? "API 서버에 연결할 수 없습니다."
                        : "API 연결 상태를 확인하고 있습니다."
                }
              >
                <span className="headerContextKey">SOURCE</span>
                <span
                  className={`headerContextValue headerContextSource-${
                    snapshot ? (snapshot.source.mode === "sql" ? "online" : "fallback") : apiStatus === "offline" ? "offline" : "checking"
                  }`}
                >
                  <span className="headerContextDot" aria-hidden="true" />
                  {snapshot ? (snapshot.source.mode === "sql" ? "SQL" : "데모") : apiStatus === "offline" ? "연결 끊김" : "확인 중"}
                </span>
              </div>
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
    {/* 모바일 반응형 패치 S-1: ≤767px에서는 .headerContextStrip 전체가
        렌더되지 않으므로(위 !isMobileLayout 분기), 데이터셋/행 수/최종
        실행 시각/연결 상태가 화면에서 완전히 사라진다 -- 그 정보가
        생존하는 유일한 자리로 헤더 바로 아래 한 줄짜리 요약 바를 둔다.
        헤더가 이미 읽는 snapshot/apiStatus를 그대로 재사용하고, 새
        API 조회는 하지 않는다. <header> 바깥(형제)에 둬서 그 안의
        ResizeObserver가 재는 --header-height(52px)에는 영향을 주지
        않는다. 479px 이하에서는 LAST RUN을 추가로 뺀다(globals.css). */}
    {isMobileLayout && (
      <div className="headerMobileContextBar" aria-label="현재 데이터 컨텍스트 (요약)">
        {snapshot?.source.eval_dataset && (
          <span className="headerMobileContextItem">{snapshot.source.eval_dataset}</span>
        )}
        {snapshot && (
          <span className="headerMobileContextItem">{snapshot.source.row_count.toLocaleString()} wf</span>
        )}
        {snapshot?.created_at && formatHeaderClock(snapshot.created_at) && (
          <span className="headerMobileContextItem headerMobileContextLastRun">{formatHeaderClock(snapshot.created_at)}</span>
        )}
        <span
          className={`headerMobileContextItem headerContextSource-${
            snapshot ? (snapshot.source.mode === "sql" ? "online" : "fallback") : apiStatus === "offline" ? "offline" : "checking"
          }`}
        >
          <span className="headerContextDot" aria-hidden="true" />
          {snapshot ? (snapshot.source.mode === "sql" ? "SQL" : "데모") : apiStatus === "offline" ? "연결 끊김" : "확인 중"}
        </span>
      </div>
    )}
    </>
  );
}
