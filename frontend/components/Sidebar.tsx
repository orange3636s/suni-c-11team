"use client";

import { Database, Monitor, Moon, Settings, Sun } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { usePanelState } from "@/components/PanelStateProvider";
import SuniAvatar from "@/components/SuniAvatar";
import { type ThemePreference, useTheme } from "@/components/ThemeProvider";

// Exported so MobileTabBar (≤1023px horizontal tab bar, same nav set) can
// share one source of truth instead of a second hardcoded list drifting
// out of sync with this one.
export const navigationItems = [
  { label: "모니터링", href: "/monitoring", icon: "monitor" },
  { label: "원인 분석", href: "/root-cause", icon: "analysis" },
  { label: "알림 이력", href: "/alerts", icon: "alert" },
  { label: "즐겨찾기", href: "/favorites", icon: "star" },
] as const;

export type NavigationLabel = (typeof navigationItems)[number]["label"];

// 접힘 상태 화면 모드 드롭다운 크기 추정치 (spec §D-2/§D-3) -- 실제 렌더
// 전에는 높이를 알 수 없으므로, flip 여부를 결정하는 데 쓸 넉넉한 상한.
const THEME_MENU_MAX_HEIGHT = 170;
const THEME_MENU_WIDTH = 140;
const VIEWPORT_EDGE_PADDING = 8;

type SidebarProps = {
  activeItem?: NavigationLabel;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
};

export default function Sidebar({ activeItem = "모니터링", collapsed = false, onToggleCollapse }: SidebarProps) {
  const { theme, setTheme } = useTheme();
  const { settingsPanelOpen, setSettingsPanelOpen, trainingPanelOpen, setTrainingPanelOpen } = usePanelState();
  const [themeMenuOpen, setThemeMenuOpen] = useState(false);
  const themeMenuRef = useRef<HTMLDivElement>(null);
  // 접힘 상태 전용 (spec §D-2) -- 트리거 아이콘의 실제 위치를 읽어 패널을
  // document.body에 포털로 띄운다. .sidebarSurface가 overflow:hidden이라
  // (좁은 rail 폭에 갇혀) 펼침 상태처럼 컨테이너 안에 absolute로 두면
  // 잘린다. 펼침 상태는 그대로 컨테이너 내부에 렌더돼 기존 동작을
  // 건드리지 않는다.
  const themeTriggerButtonRef = useRef<HTMLButtonElement>(null);
  const themePortalRef = useRef<HTMLDivElement>(null);
  const [themeMenuPos, setThemeMenuPos] = useState<{ left: number; top?: number; bottom?: number } | null>(null);

  useEffect(() => {
    if (!themeMenuOpen) return;

    function isOutside(target: Node) {
      if (themeMenuRef.current?.contains(target)) return false;
      if (themePortalRef.current?.contains(target)) return false;
      return true;
    }

    function closeThemeMenu(event: MouseEvent) {
      if (isOutside(event.target as Node)) setThemeMenuOpen(false);
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

  function toggleThemeMenu() {
    if (collapsed && !themeMenuOpen) {
      const rect = themeTriggerButtonRef.current?.getBoundingClientRect();
      if (rect) {
        // 아이콘 우측에 띄운다 (spec §D-2) -- 접힌 rail 폭이 좁아 아래로
        // 띄우면 화면을 벗어난다. 기본은 right-end: 패널의 아래 끝을
        // 버튼 아래 끝에 맞추고 위로 펼친다 -- 버튼이 사이드바 하단에
        // 있으므로 아래로 펼치면 화면을 벗어나기 때문이다. 위쪽 공간이
        // 부족할 때만 아래로 반전한다 (spec §D-3 flip).
        const spaceAbove = rect.bottom;
        const spaceBelow = window.innerHeight - rect.top;
        const flipDown = spaceAbove < THEME_MENU_MAX_HEIGHT && spaceBelow > spaceAbove;
        let left = rect.right + 8;
        // 화면 경계에서 8px 안쪽으로 밀어 넣는다 (spec §D-3 shift).
        left = Math.min(left, window.innerWidth - THEME_MENU_WIDTH - VIEWPORT_EDGE_PADDING);
        left = Math.max(left, VIEWPORT_EDGE_PADDING);
        if (flipDown) {
          const top = Math.min(rect.top, window.innerHeight - THEME_MENU_MAX_HEIGHT - VIEWPORT_EDGE_PADDING);
          setThemeMenuPos({ left, top: Math.max(top, VIEWPORT_EDGE_PADDING) });
        } else {
          const bottom = Math.max(window.innerHeight - rect.bottom, VIEWPORT_EDGE_PADDING);
          setThemeMenuPos({ left, bottom });
        }
      }
    }
    setThemeMenuOpen((open) => !open);
  }

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
          {/* 지시서 L-1: 모델 학습 진입점 -- 설정 버튼과 동일한 패턴
              (.themeToggle 공유, 접힘 시 아이콘만). 화면 모드 위에 둔다. */}
          <button
            type="button"
            className={`themeToggle trainingTrigger ${collapsed ? "railIconButton" : ""}`}
            aria-label="모델 학습"
            aria-haspopup="dialog"
            aria-expanded={trainingPanelOpen}
            title="모델 학습"
            onClick={() => setTrainingPanelOpen((open) => !open)}
          >
            <span className="themeTriggerIcon" aria-hidden="true"><Database size={18} /></span>
            <span className="themeTriggerLabel">모델 학습</span>
          </button>
          {/* 지시서 Q: 사이드바 하단 순서 -- 모델 학습 / 알림 설정 / 화면
              모드. */}
          <button
            type="button"
            className={`themeToggle settingsTrigger ${collapsed ? "railIconButton" : ""}`}
            aria-label="알림 설정"
            aria-haspopup="dialog"
            aria-expanded={settingsPanelOpen}
            title="알림 설정"
            onClick={() => setSettingsPanelOpen((open) => !open)}
          >
            <span className="themeTriggerIcon" aria-hidden="true"><Settings size={18} /></span>
            <span className="themeTriggerLabel">알림 설정</span>
          </button>
          <div className="themeTriggerCol" ref={themeMenuRef}>
            {/* 펼침 상태는 기존 그대로 컨테이너 내부에 렌더 (spec §D-2 대상은
                접힘 상태뿐이라 여기는 건드리지 않는다). */}
            {themeMenuOpen && !collapsed && (
              <div className="themeMenu" role="menu" aria-label="화면 모드 선택">
                <strong>화면 모드</strong>
                <ThemeOptionsList theme={theme} onSelect={(value) => { setTheme(value); setThemeMenuOpen(false); }} />
              </div>
            )}
            <button
              ref={themeTriggerButtonRef}
              className={`themeToggle themeTrigger ${collapsed ? "railIconButton" : ""}`}
              type="button"
              aria-expanded={themeMenuOpen}
              aria-label="화면 모드 선택"
              aria-haspopup="menu"
              title="화면 모드 선택"
              onClick={toggleThemeMenu}
            >
              <span className="themeTriggerIcon" aria-hidden="true"><ThemeIcon theme={theme} /></span>
              <span className="themeTriggerLabel">화면 모드</span>
              <ChevronDown />
            </button>
          </div>
        </div>
      </div>
      {/* 접힘 상태 전용 포털 (spec §D-2) -- document.body에 그려 .sidebarSurface의
          overflow:hidden을 벗어난다. 위치는 themeTriggerButtonRef의
          getBoundingClientRect()로 매 오픈마다 계산한다. */}
      {themeMenuOpen && collapsed && themeMenuPos && createPortal(
        <div
          ref={themePortalRef}
          className="themeMenuPortal"
          role="menu"
          aria-label="화면 모드 선택"
          style={{ left: themeMenuPos.left, top: themeMenuPos.top, bottom: themeMenuPos.bottom }}
        >
          <strong>화면 모드</strong>
          <ThemeOptionsList theme={theme} onSelect={(value) => { setTheme(value); setThemeMenuOpen(false); }} />
        </div>,
        document.body,
      )}
    </aside>
  );
}

// 펼침 상태(컨테이너 내부)와 접힘 상태(포털)가 옵션 목록 마크업을 공유한다.
function ThemeOptionsList({ theme, onSelect }: { theme: ThemePreference; onSelect: (value: ThemePreference) => void }) {
  return (
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
          onClick={() => onSelect(value)}
          role="menuitemradio"
          aria-checked={theme === value}
        >
          <Icon aria-hidden="true" className="themeOptionIcon" />
          {label}
        </button>
      ))}
    </div>
  );
}

function NavIcon({ name }: { name: string }) {
  const paths: Record<string, React.ReactNode> = {
    monitor: <><rect x="2" y="4" width="20" height="13" rx="2" /><path d="M8 21h8" /><path d="M12 17v4" /><path d="m6 12 3-3 3 2 4-5" /></>,
    analysis: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /><path d="M8 11h6" /><path d="M11 8v6" /></>,
    alert: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" /><path d="M10 21h4" /></>,
    star: <path d="M12 2.5 15.09 9l7.16.6-5.45 4.73L18.5 21 12 17.27 5.5 21l1.7-6.67L1.75 9.6 8.91 9 12 2.5Z" />,
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
