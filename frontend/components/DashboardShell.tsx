"use client";

import { useState, type ReactNode } from "react";
import AiPanel from "@/components/ai-panel/AiPanel";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";

type NavigationLabel = "모델 학습" | "원인 분석" | "사전 알람 로그";

export default function DashboardShell({
  activeItem,
  children,
}: {
  activeItem: NavigationLabel;
  children: ReactNode;
}) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [aiPanelOpen, setAiPanelOpen] = useState(false);

  return (
    <div
      className="dashboardGrid"
      data-sidebar={sidebarCollapsed ? "collapsed" : "open"}
      data-ai={aiPanelOpen ? "open" : "closed"}
    >
      <Sidebar
        activeItem={activeItem}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((value) => !value)}
      />
      <div className="contentShell">
        <Header />
        <main className="mainContent uploadPage">{children}</main>
      </div>
      <AiPanel open={aiPanelOpen} onToggle={() => setAiPanelOpen((value) => !value)} />
    </div>
  );
}
