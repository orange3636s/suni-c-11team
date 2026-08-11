"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { usePanelState } from "@/components/PanelStateProvider";
import { formatSidebarDot, navigationItems, SidebarStatusDot, type NavigationLabel } from "@/components/Sidebar";

/** Replaces the left sidebar at ≤1023px -- a floating navy pill bar, same
 * visual language as the sidebar (same background var, same radius),
 * sticky under the header. Text-only tabs (no icons -- 좁은 폭에서
 * 아이콘은 공간만 차지한다), horizontally scrollable, never wraps.
 *
 * Rendered instead of <Sidebar> conditionally, never CSS-hidden, so there is no
 * collapse toggle here at all: the concept doesn't exist for a tab bar.
 */
export default function MobileTabBar({ activeItem }: { activeItem: NavigationLabel }) {
  const activeRef = useRef<HTMLAnchorElement>(null);
  const {
    settingsPanelOpen,
    setSettingsPanelOpen,
    trainingPanelOpen,
    setTrainingPanelOpen,
    analysisPanelOpen,
    setAnalysisPanelOpen,
  } = usePanelState();
  const { snapshot, training, notifications } = useAnalysisState();
  // 데스크톱 사이드바(Sidebar.tsx)와 같은 판정 규칙 -- 컴포넌트만
  // 다시 쓰고(SidebarStatusDot) 로직은 여기서 다시 계산한다(이 탭바는
  // 사이드바 대신 렌더되므로 같은 훅 인스턴스를 공유할 수 없다).
  const analysisStatus: "connected" | "offline" | "error" =
    snapshot && snapshot.errors.length > 0 ? "error" : snapshot ? "connected" : "offline";
  const analysisStatusLabel = { connected: "분석 완료", offline: "미실행", error: "분석 실패" }[analysisStatus];
  const trainingStatus: "connected" | "offline" = training ? "connected" : "offline";
  const trainingStatusLabel = training
    ? `수동 학습 · ${training.performance?.source_filename ?? "-"} · ${formatSidebarDot(training.createdAt)}`
    : "내장 데이터로 학습됨";
  const connectedChannelNames = [
    notifications.slack.connected ? "Slack" : null,
    notifications.telegram.connected ? "Telegram" : null,
    notifications.gmail.connected ? "Gmail" : null,
  ].filter((name): name is string => name != null);
  const automationEnabled = notifications.automation.enabled;
  const automationErrored = notifications.automation.last_run_status === "error";
  const notificationStatus: "connected" | "offline" | "error" = automationErrored
    ? "error"
    : connectedChannelNames.length > 0 && automationEnabled
      ? "connected"
      : "offline";
  const notificationStatusLabel = automationErrored
    ? "자동화 실행 오류"
    : connectedChannelNames.length > 0 && automationEnabled
      ? `${connectedChannelNames.join(" · ")} 연결됨 · 자동화 켜짐`
      : connectedChannelNames.length > 0
        ? `${connectedChannelNames.join(" · ")} 연결됨 · 자동화 꺼짐`
        : "연결된 채널 없음";

  // Keeps the selected tab in view if the bar has scrolled (spec: "선택된
  // 항목이 화면 밖이면 자동으로 스크롤해 보이게 한다").
  useEffect(() => {
    activeRef.current?.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
  }, [activeItem]);

  return (
    <nav className="mobileTabBar" aria-label="주요 메뉴">
      <div className="mobileTabBarScroll">
        {navigationItems.map((item) => {
          const isActive = item.label === activeItem;
          return (
            <Link
              key={item.label}
              ref={isActive ? activeRef : undefined}
              href={item.href}
              className="mobileTab"
              aria-current={isActive ? "page" : undefined}
            >
              {item.label}
              <i className="mobileTabStatusDot" aria-hidden="true" />
            </Link>
          );
        })}
        {/* 좁은 폭에서는 사이드바 자체가 없으니 설정도
            이 가로 탭바에 편입한다 (페이지 이동이 아니라 패널을 여는
            버튼이라 다른 탭들과 달리 <Link>가 아니다). */}
        {/* 데스크톱 사이드바 하단의 모델 학습 진입점을 좁은 폭에서도
            열 수 있어야 한다 -- 이 탭바에는 사이드바 자체가 없으므로
            설정과 같은 방식으로 편입한다. 모델 학습과 모델 분석은
            사이드바에서와 마찬가지로 별개 버튼이다. */}
        <button
          type="button"
          className="mobileTab mobileTabButton"
          aria-haspopup="dialog"
          aria-expanded={trainingPanelOpen}
          title={`모델 학습 (${trainingStatusLabel})`}
          onClick={() => setTrainingPanelOpen((value) => !value)}
        >
          모델 학습
          <SidebarStatusDot status={trainingStatus} label={trainingStatusLabel} />
        </button>
        <button
          type="button"
          className="mobileTab mobileTabButton"
          aria-haspopup="dialog"
          aria-expanded={analysisPanelOpen}
          title={`모델 분석 (${analysisStatusLabel})`}
          onClick={() => setAnalysisPanelOpen((value) => !value)}
        >
          모델 분석
          <SidebarStatusDot status={analysisStatus} label={analysisStatusLabel} />
        </button>
        <button
          type="button"
          className="mobileTab mobileTabButton"
          aria-haspopup="dialog"
          aria-expanded={settingsPanelOpen}
          title={`알림·자동화 설정 (${notificationStatusLabel})`}
          onClick={() => setSettingsPanelOpen((value) => !value)}
        >
          알림·자동화 설정
          <SidebarStatusDot status={notificationStatus} label={notificationStatusLabel} />
        </button>
      </div>
    </nav>
  );
}
