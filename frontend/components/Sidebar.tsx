"use client";

import { Monitor, Moon, Settings, Sun } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { usePanelState } from "@/components/PanelStateProvider";
import SuniAvatar from "@/components/SuniAvatar";
import { type ThemePreference, useTheme } from "@/components/ThemeProvider";

// Exported so MobileTabBar (≤1023px horizontal tab bar, same nav set) can
// share one source of truth instead of a second hardcoded list drifting
// out of sync with this one.
export const navigationItems = [
  { label: "모델 학습", href: "/training", icon: "model" },
  { label: "원인 분석", href: "/root-cause", icon: "analysis" },
  { label: "사전 알람 로그", href: "/alerts", icon: "alert" },
] as const;

export type NavigationLabel = (typeof navigationItems)[number]["label"];

type SidebarProps = {
  activeItem?: NavigationLabel;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
};

export default function Sidebar({ activeItem = "모델 학습", collapsed = false, onToggleCollapse }: SidebarProps) {
  const { theme, setTheme } = useTheme();
  const { settingsPanelOpen, setSettingsPanelOpen } = usePanelState();
  const [themeMenuOpen, setThemeMenuOpen] = useState(false);
  const themeMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!themeMenuOpen) return;

    function closeThemeMenu(event: MouseEvent) {
      if (!themeMenuRef.current?.contains(event.target as Node)) {
        setThemeMenuOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setThemeMenuOpen(false);
    }

    document.addEventListener("mousedown", closeThemeMenu);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", closeThemeMenu);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [themeMenuOpen]);

  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebarSurface">
        <button
          type="button"
          className="shellLogoBlock"
          onClick={onToggleCollapse}
          aria-label={collapsed ? "메뉴 펼치기" : "메뉴 접기"}
          aria-expanded={!collapsed}
          title={collapsed ? "펼치기" : undefined}
        >
          <SuniAvatar size={collapsed ? 32 : 28} />
          {!collapsed && <span className="shellLogoBlockTitle">써니C 11팀</span>}
          {/* Collapsed rail: the chevron is unmounted entirely, not just
              hidden -- the logo itself is the only (and now sole) way to
              expand, so a "collapsed" affordance chevron pointing the
              wrong way (there's nothing left to collapse further) no
              longer makes sense here (spec §1-1). Expanded state keeps
              its `<` chevron unchanged. */}
          {!collapsed && (
            <span className="shellLogoBlockChevron" aria-hidden="true">
              <ChevronIcon direction="left" />
            </span>
          )}
        </button>

        {collapsed ? (
          <nav aria-label="주요 메뉴" className="railNav">
            {navigationItems.map((item) => {
              const isActive = item.label === activeItem;
              return (
                <Link
                  key={item.label}
                  href={item.href}
                  className={`railNavItem ${isActive ? "active" : ""}`}
                  aria-current={isActive ? "page" : undefined}
                  aria-label={item.label}
                  title={item.label}
                >
                  <NavIcon name={item.icon} />
                </Link>
              );
            })}
          </nav>
        ) : (
          <nav aria-label="주요 메뉴">
            <ul className="navigationList">
              {navigationItems.map((item) => {
                const isActive = item.label === activeItem;
                return (
                  <li key={item.label}>
                    <Link
                      className={`navigationItem ${isActive ? "active" : ""}`}
                      href={item.href}
                      aria-current={isActive ? "page" : undefined}
                    >
                      <NavIcon name={item.icon} />
                      <span>{item.label}</span>
                      <i className="menuStatusDot ready" aria-label="사용 가능" />
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        )}

        {/* 설정 패널 신설 §A-2: Theme + 설정 2분할. 접힘 상태에서도 항상
            렌더링하되(§A-2 체크리스트 4번: "접힘 상태에서 아이콘만 표시"),
            .sidebar.collapsed 쪽 CSS가 라벨/셰브론을 숨기고 세로로 쌓는다 --
            분기를 늘리는 대신 같은 마크업을 CSS로만 다르게 보이게 한다. */}
        <div className="sidebarFooter">
          <div className="themeTriggerCol" ref={themeMenuRef}>
            {themeMenuOpen && (
              <div className="themeMenu" role="menu" aria-label="Theme 선택">
                <strong>Theme</strong>
                <div className="themeOptions">
                  {(
                    [
                      ["system", "System", Monitor],
                      ["light", "Light", Sun],
                      ["dark", "Dark", Moon],
                    ] as [ThemePreference, string, typeof Monitor][]
                  ).map(([value, label, Icon]) => (
                    <button
                      key={value}
                      type="button"
                      className={theme === value ? "active" : ""}
                      onClick={() => {
                        setTheme(value);
                        setThemeMenuOpen(false);
                      }}
                      role="menuitemradio"
                      aria-checked={theme === value}
                    >
                      <Icon aria-hidden="true" className="themeOptionIcon" />
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <button
              className="themeToggle themeTrigger"
              type="button"
              aria-expanded={themeMenuOpen}
              aria-label="Theme 선택"
              aria-haspopup="menu"
              title="Theme 선택"
              onClick={() => setThemeMenuOpen((open) => !open)}
            >
              <span className="themeTriggerIcon" aria-hidden="true"><ThemeIcon theme={theme} /></span>
              <span className="themeTriggerLabel">Theme</span>
              <ChevronDown />
            </button>
          </div>
          <button
            type="button"
            className="themeToggle settingsTrigger"
            aria-label="설정"
            aria-haspopup="dialog"
            aria-expanded={settingsPanelOpen}
            title="설정"
            onClick={() => setSettingsPanelOpen((open) => !open)}
          >
            <span className="themeTriggerIcon" aria-hidden="true"><Settings size={18} /></span>
            <span className="themeTriggerLabel">설정</span>
          </button>
        </div>
      </div>
    </aside>
  );
}

function NavIcon({ name }: { name: string }) {
  const paths: Record<string, React.ReactNode> = {
    model: <><path d="M12 2v4" /><path d="M12 18v4" /><path d="M4.93 4.93l2.83 2.83" /><path d="M16.24 16.24l2.83 2.83" /><path d="M2 12h4" /><path d="M18 12h4" /><path d="M4.93 19.07l2.83-2.83" /><path d="M16.24 7.76l2.83-2.83" /><circle cx="12" cy="12" r="4" /></>,
    analysis: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /><path d="M8 11h6" /><path d="M11 8v6" /></>,
    alert: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" /><path d="M10 21h4" /></>,
  };
  return <svg className="navigationIcon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

// 설정 패널 신설 §B: 이모지(🖥️☀️🌙) 대신 lucide-react 컴포넌트를 쓴다 --
// OS/브라우저마다 이모지 모양이 달라지는 문제가 없다.
function ThemeIcon({ theme }: { theme: ThemePreference }) {
  const Icon = theme === "dark" ? Moon : theme === "system" ? Monitor : Sun;
  return <Icon className="themeIcon" aria-hidden="true" />;
}

function ChevronDown() {
  return <svg className="themeChevron" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="m7 10 5 5 5-5" /></svg>;
}

function ChevronIcon({ direction }: { direction: "left" | "right" }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d={direction === "left" ? "m15 6-6 6 6 6" : "m9 6 6 6-6 6"} />
    </svg>
  );
}
