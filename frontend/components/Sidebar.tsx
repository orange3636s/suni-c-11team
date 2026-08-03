"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { type ThemePreference, useTheme } from "@/components/ThemeProvider";

const navigationItems = [
  { label: "모델 학습", href: "/training", icon: "model" },
  { label: "원인 분석", href: "/root-cause", icon: "analysis" },
  { label: "사전 알람 로그", href: "/alerts", icon: "alert" },
] as const;

type NavigationLabel = (typeof navigationItems)[number]["label"];

type SidebarProps = {
  activeItem?: NavigationLabel;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
};

const THEME_CYCLE: ThemePreference[] = ["system", "light", "dark"];

export default function Sidebar({ activeItem = "모델 학습", collapsed = false, onToggleCollapse }: SidebarProps) {
  const { theme, setTheme } = useTheme();
  const [themeMenuOpen, setThemeMenuOpen] = useState(false);
  const themeMenuRef = useRef<HTMLDivElement>(null);

  function cycleTheme() {
    const next = THEME_CYCLE[(THEME_CYCLE.indexOf(theme) + 1) % THEME_CYCLE.length];
    setTheme(next);
  }

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
      <div className="sidebarTop">
        <Link className="brand" href="/" aria-label="SK하이닉스 수율 분석 대시보드 홈">
          <Image
            className="brandLogo"
            src="/sk-suni-c-5-character.png"
            alt="SK SUNI C 5기"
            width={150}
            height={150}
            unoptimized
            priority
          />
          {!collapsed && <strong className="brandTitle">써니C 11팀</strong>}
        </Link>
        {onToggleCollapse && (
          <button
            type="button"
            className="sidebarCollapseButton"
            onClick={onToggleCollapse}
            aria-label={collapsed ? "사이드바 펼치기" : "사이드바 접기"}
            aria-expanded={!collapsed}
          >
            <ChevronDown style={{ transform: collapsed ? "rotate(-90deg)" : "rotate(90deg)" }} />
          </button>
        )}
      </div>

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
                  title={collapsed ? item.label : undefined}
                >
                  <NavIcon name={item.icon} />
                  {!collapsed && <span>{item.label}</span>}
                  <i className="menuStatusDot ready" aria-label="사용 가능" />
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="sidebarFooter" ref={themeMenuRef}>
        {collapsed ? (
          <button
            type="button"
            className="themeToggle themeTrigger collapsed"
            aria-label={`Theme: ${theme}. 클릭하여 전환`}
            title={`Theme: ${theme}`}
            onClick={cycleTheme}
          >
            <ThemeIcon theme={theme} />
          </button>
        ) : (
          <>
        {themeMenuOpen && (
          <div className="themeMenu" role="menu" aria-label="Theme 선택">
            <strong>Theme</strong>
            <div className="themeOptions">
              {(
                [
                  ["system", "System"],
                  ["light", "Light"],
                  ["dark", "Dark"],
                ] as [ThemePreference, string][]
              ).map(([value, label]) => (
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
          </>
        )}
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

function ThemeIcon({ theme }: { theme: ThemePreference }) {
  const path = theme === "dark"
    ? <path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z" />
    : theme === "system"
      ? <><rect x="3" y="4" width="18" height="13" rx="2" /><path d="M8 21h8M12 17v4" /></>
      : <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" /></>;
  return <svg className="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{path}</svg>;
}

function ChevronDown({ style }: { style?: React.CSSProperties }) {
  return <svg className="themeChevron" style={style} viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="m7 10 5 5 5-5" /></svg>;
}
