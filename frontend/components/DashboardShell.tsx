"use client";

import type { ReactNode } from "react";
import AiPanel from "@/components/ai-panel/AiPanel";
import Header from "@/components/Header";
import MobileTabBar from "@/components/MobileTabBar";
import { usePanelState } from "@/components/PanelStateProvider";
import Sidebar, { type NavigationLabel } from "@/components/Sidebar";
import { useIsTabBarLayout } from "@/lib/useMediaQuery";

export default function DashboardShell({
  activeItem,
  children,
}: {
  activeItem: NavigationLabel;
  children: ReactNode;
}) {
  const { sidebarCollapsed, setSidebarCollapsed, aiPanelOpen, setAiPanelOpen } = usePanelState();
  // ≤1023px: the left sidebar becomes a horizontal tab bar (spec §B-3) --
  // a conditional render, not a CSS hide, so the collapse toggle button
  // and its bug (collapsing pushed content off-screen at narrow widths --
  // see globals.css's `@media (max-width: 1280px)` specificity fix)
  // simply don't exist in this DOM at all below 1024px.
  const isTabBarLayout = useIsTabBarLayout();

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
        {isTabBarLayout && <MobileTabBar activeItem={activeItem} />}
        <main className="mainContent uploadPage">{children}</main>
      </div>
      <AiPanel open={aiPanelOpen} onToggle={() => setAiPanelOpen((value) => !value)} />
    </div>
  );
}
