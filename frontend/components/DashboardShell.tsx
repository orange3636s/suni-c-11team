"use client";

import type { ReactNode } from "react";
import AiPanel from "@/components/ai-panel/AiPanel";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import Header from "@/components/Header";
import MobileTabBar from "@/components/MobileTabBar";
import { usePanelState } from "@/components/PanelStateProvider";
import SettingsPanel from "@/components/SettingsPanel";
import Sidebar, { type NavigationLabel } from "@/components/Sidebar";
import TrainingPanel from "@/components/TrainingPanel";
import { useIsTabBarLayout } from "@/lib/useMediaQuery";

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
    aiPanelOpen,
    setAiPanelOpen,
    settingsPanelOpen,
    setSettingsPanelOpen,
    trainingPanelOpen,
    setTrainingPanelOpen,
  } = usePanelState();
  // ≤1023px: the left sidebar becomes a horizontal tab bar (spec §B-3) --
  // a conditional render, not a CSS hide, so the collapse toggle button
  // and its bug (collapsing pushed content off-screen at narrow widths --
  // see globals.css's `@media (max-width: 1280px)` specificity fix)
  // simply don't exist in this DOM at all below 1024px.
  const isTabBarLayout = useIsTabBarLayout();
  // D-2: 모든 페이지에서 보이도록 셸 레벨에서 한 번만 렌더한다 -- 각
  // 페이지가 개별적으로 처리하면 배너를 빠뜨리는 페이지가 생긴다.
  const { degraded, retryHydration } = useAnalysisState();

  return (
    <div
      className="dashboardGrid"
      data-sidebar={sidebarCollapsed ? "collapsed" : "open"}
      data-ai={aiPanelOpen ? "open" : "closed"}
    >
      {!isTabBarLayout && (
        <Sidebar
          activeItem={activeItem}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((value) => !value)}
        />
      )}
      <div className="contentShell">
        <Header />
        {degraded && (
          <div className="degradedStateBanner" role="alert">
            <span>이전 결과 복원 실패 — 학습·원인 분석·알림 이력이 일시적으로 보이지 않을 수 있습니다.</span>
            <button type="button" className="button" onClick={retryHydration}>다시 시도</button>
          </div>
        )}
        {isTabBarLayout && <MobileTabBar activeItem={activeItem} />}
        <main className="mainContent uploadPage">{children}</main>
      </div>
      <AiPanel open={aiPanelOpen} onToggle={() => setAiPanelOpen((value) => !value)} />
      <SettingsPanel open={settingsPanelOpen} onClose={() => setSettingsPanelOpen(false)} />
      <TrainingPanel open={trainingPanelOpen} onClose={() => setTrainingPanelOpen(false)} />
    </div>
  );
}
