"use client";

import type { ReactNode } from "react";
import { useEffect, useRef } from "react";
import AiPanel from "@/components/ai-panel/AiPanel";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import Header from "@/components/Header";
import MobileTabBar from "@/components/MobileTabBar";
import { usePanelState } from "@/components/PanelStateProvider";
import SettingsPanel from "@/components/SettingsPanel";
import Sidebar, { type NavigationLabel } from "@/components/Sidebar";
import TrainingPanel from "@/components/TrainingPanel";
import { useIsMobileLayout, useIsTabBarLayout } from "@/lib/useMediaQuery";

export default function DashboardShell({
  activeItem,
  children,
}: {
  activeItem: NavigationLabel;
  children: ReactNode;
}) {
  const {
    sidebarCollapsed,
    setSidebarCollapsed,
    sidebarDrawerOpen,
    setSidebarDrawerOpen,
    aiPanelOpen,
    setAiPanelOpen,
    settingsPanelOpen,
    setSettingsPanelOpen,
    trainingPanelOpen,
    setTrainingPanelOpen,
  } = usePanelState();
  // 모바일 반응형 패치 S-1 (셸 아키텍처 개편) -- 이전에는 ≤1023px에서
  // <Sidebar>가 DOM에서 통째로 사라지고 <MobileTabBar>만 남았다. 지금은
  // 3단계다:
  //   1024px+          <Sidebar> 펼침/접힘은 사용자가 그대로 토글.
  //   768px~1023px     <Sidebar>는 계속 마운트되지만 진입 시 기본
  //                     접힘(rail)으로 시작한다. <MobileTabBar> 없음.
  //   ≤767px           <Sidebar>는 오프캔버스 드로어(mode="drawer")로
  //                     전환되고, <MobileTabBar>가 주 내비게이션이 된다.
  const isTabBarLayout = useIsTabBarLayout();
  const isMobileLayout = useIsMobileLayout();
  // D-2: 모든 페이지에서 보이도록 셸 레벨에서 한 번만 렌더한다 -- 각
  // 페이지가 개별적으로 처리하면 배너를 빠뜨리는 페이지가 생긴다.
  const { degraded, retryHydration } = useAnalysisState();

  // 태블릿 진입 시 기본 접힘 (지시서: "기존 접힘 기능이 있으므로 초기
  // 상태만 바꾼다") -- sidebarCollapsed는 이미 사용자가 직접 토글하는
  // 기존 상태라, 여기서는 768~1023px 밴드에 "새로 진입"하는 순간에만
  // 한 번 true로 밀어준다. 이후 사용자가 다시 펼쳐도 이 effect가 다시
  // 강제로 접지 않는다(같은 좁은 세션 동안은 한 번만 적용) -- 1024px+로
  // 나갔다가 다시 좁아지면 "새 세션"으로 보고 다시 적용한다.
  const appliedTabletDefaultRef = useRef(false);
  const prevTabBarRef = useRef(false);
  useEffect(() => {
    const enteredTabBar = isTabBarLayout && !prevTabBarRef.current;
    if (enteredTabBar && !isMobileLayout && !appliedTabletDefaultRef.current) {
      setSidebarCollapsed(true);
      appliedTabletDefaultRef.current = true;
    }
    if (!isTabBarLayout) {
      // 1024px+로 돌아가면 "좁은 세션"이 끝난 것으로 보고 리셋한다.
      appliedTabletDefaultRef.current = false;
    }
    prevTabBarRef.current = isTabBarLayout;
  }, [isTabBarLayout, isMobileLayout, setSidebarCollapsed]);

  // ≤767px 드로어의 Escape 닫기 -- Sidebar.tsx의 테마 메뉴가 이미 쓰는
  // 것과 같은 패턴(문서 레벨 keydown 리스너, 열려 있을 때만 붙인다).
  useEffect(() => {
    if (!(isMobileLayout && sidebarDrawerOpen)) return;
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setSidebarDrawerOpen(false);
    }
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isMobileLayout, sidebarDrawerOpen, setSidebarDrawerOpen]);

  return (
    <div
      className="dashboardGrid"
      data-sidebar={sidebarCollapsed ? "collapsed" : "open"}
      data-ai={aiPanelOpen ? "open" : "closed"}
    >
      {!isMobileLayout && (
        <Sidebar
          activeItem={activeItem}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((value) => !value)}
        />
      )}
      {isMobileLayout && (
        <Sidebar
          activeItem={activeItem}
          mode="drawer"
          drawerOpen={sidebarDrawerOpen}
          onCloseDrawer={() => setSidebarDrawerOpen(false)}
        />
      )}
      {isMobileLayout && sidebarDrawerOpen && (
        <div
          className="sidebarDrawerBackdrop"
          onClick={() => setSidebarDrawerOpen(false)}
          role="presentation"
          aria-hidden="true"
        />
      )}
      <div className="contentShell">
        <Header />
        {degraded && (
          <div className="degradedStateBanner" role="alert">
            <span>이전 결과 복원 실패 — 학습·원인 분석·알림 기록이 일시적으로 보이지 않을 수 있습니다.</span>
            <button type="button" className="button" onClick={retryHydration}>다시 시도</button>
          </div>
        )}
        {isMobileLayout && <MobileTabBar activeItem={activeItem} />}
        <main className="mainContent uploadPage">{children}</main>
      </div>
      <AiPanel open={aiPanelOpen} onToggle={() => setAiPanelOpen((value) => !value)} />
      <SettingsPanel open={settingsPanelOpen} onClose={() => setSettingsPanelOpen(false)} />
      <TrainingPanel open={trainingPanelOpen} onClose={() => setTrainingPanelOpen(false)} />
    </div>
  );
}
