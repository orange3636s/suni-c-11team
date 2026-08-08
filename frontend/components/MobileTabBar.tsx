"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { navigationItems, type NavigationLabel } from "@/components/Sidebar";
import { usePanelState } from "@/components/PanelStateProvider";

/** Replaces the left sidebar at ≤1023px (spec: JSON 보고서 버튼 제거 ·
 * 모바일 레이아웃 전환 §B-3) -- a floating navy pill bar, same visual
 * language as the sidebar (same background var, same radius), sticky
 * under the header. Text-only tabs (no icons -- spec: "좁은 폭에서 아이콘은
 * 공간만 차지한다"), horizontally scrollable, never wraps.
 *
 * Rendered instead of <Sidebar> (conditional, not CSS-hidden -- spec
 * §B-3: "조건에 렌더링으로 처리한다. CSS로 숨기지 마라"), so there is no
 * collapse toggle here at all: the concept doesn't exist for a tab bar.
 */
export default function MobileTabBar({ activeItem }: { activeItem: NavigationLabel }) {
  const activeRef = useRef<HTMLAnchorElement>(null);
  const { settingsPanelOpen, setSettingsPanelOpen, trainingPanelOpen, setTrainingPanelOpen } = usePanelState();

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
        {/* 설정 패널 신설 §A-2: 좁은 폭에서는 사이드바 자체가 없으니 설정도
            이 가로 탭바에 편입한다 (페이지 이동이 아니라 패널을 여는
            버튼이라 다른 탭들과 달리 <Link>가 아니다). */}
        {/* 지시서 L-1: 데스크톱 사이드바 하단의 모델 학습 진입점을 좁은
            폭에서도 열 수 있어야 한다 -- 이 탭바에는 사이드바 자체가
            없으므로 설정과 같은 방식으로 편입한다. */}
        <button
          type="button"
          className="mobileTab mobileTabButton"
          aria-haspopup="dialog"
          aria-expanded={trainingPanelOpen}
          onClick={() => setTrainingPanelOpen((value) => !value)}
        >
          모델 학습·자동화
        </button>
        <button
          type="button"
          className="mobileTab mobileTabButton"
          aria-haspopup="dialog"
          aria-expanded={settingsPanelOpen}
          onClick={() => setSettingsPanelOpen((value) => !value)}
        >
          알림 설정
        </button>
      </div>
    </nav>
  );
}
