"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { navigationItems, type NavigationLabel } from "@/components/Sidebar";

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

  // Keeps the selected tab in view if the bar has scrolled (spec: "선택된
  // 항목이 화면 밖이면 자동으로 스크롤해 보이게 한다").
  useEffect(() => {
    activeRef.current?.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
  }, [activeItem]);

  return (
    <nav className="mobileTabBar" aria-label="주요 메뉴" role="tablist">
      <div className="mobileTabBarScroll">
        {navigationItems.map((item) => {
          const isActive = item.label === activeItem;
          return (
            <Link
              key={item.label}
              ref={isActive ? activeRef : undefined}
              href={item.href}
              className="mobileTab"
              role="tab"
              aria-selected={isActive}
              aria-current={isActive ? "page" : undefined}
            >
              {item.label}
              <i className="mobileTabStatusDot" aria-hidden="true" />
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
