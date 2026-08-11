"use client";

import { Activity, Database, Monitor, Moon, Settings, Sun } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { usePanelState } from "@/components/PanelStateProvider";
import SuniAvatar from "@/components/SuniAvatar";
import { type ThemePreference, useTheme } from "@/components/ThemeProvider";

// Exported so MobileTabBar (≤1023px horizontal tab bar, same nav set) can
// share one source of truth instead of a second hardcoded list drifting
// out of sync with this one.
export const navigationItems = [
  { label: "모니터링 홈", href: "/monitoring", icon: "monitor" },
  // WH: Config별 트리맵 -- 모니터링 홈에서 분리한 설비 구성 트리맵 전용 탭.
  { label: "Config별 트리맵", href: "/config-treemap", icon: "treemap" },
  { label: "원인 분석", href: "/root-cause", icon: "analysis" },
  { label: "수율 예측", href: "/alerts", icon: "alert" },
  { label: "즐겨찾기", href: "/favorites", icon: "star" },
] as const;

export type NavigationLabel = (typeof navigationItems)[number]["label"];

// ME-2: 세 하단 버튼(모델 학습·모델 분석·자동화·알림 설정)이 공유하는
// 상태 점 -- 색·크기·aria-label을 한 곳에서만 정의해 각자 구현하다
// 갈리는 일을 막는다(지시서 "하지 말 것: 각자 구현하지 마라").
export function SidebarStatusDot({ status, label }: { status: "connected" | "offline" | "error"; label: string }) {
  return <span className={`sidebarStatusDot ${status === "connected" ? "" : status}`} aria-label={label} />;
}

export function formatSidebarDot(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "-";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// U-4: 화면 모드는 은유 아이콘(☀/☾) 대신 글자로 말한다 -- 계측 도구에서는
// 상태를 텍스트로 표시하는 편이 정확하다.
const THEME_LABELS: Record<ThemePreference, string> = {
  system: "시스템",
  light: "라이트",
  dark: "다크",
};

// 접힘 상태 화면 모드 드롭다운 크기 추정치 (spec §D-2/§D-3) -- 실제 렌더
// 전에는 높이를 알 수 없으므로, flip 여부를 결정하는 데 쓸 넉넉한 상한.
const THEME_MENU_MAX_HEIGHT = 170;
const THEME_MENU_WIDTH = 140;
const VIEWPORT_EDGE_PADDING = 8;

type SidebarProps = {
  activeItem?: NavigationLabel;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
  // 모바일 반응형 패치 S-1: ≤767px에서는 DashboardShell이 이 컴포넌트를
  // "drawer" 모드로 렌더한다 -- 아이콘 레일(collapsed)이 아니라 오프캔버스
  // 전체 패널(설정 전용 접근)이다. "shell" 모드(기본값)는 기존 데스크톱/
  // 태블릿 사이드바 그대로다.
  mode?: "shell" | "drawer";
  drawerOpen?: boolean;
  onCloseDrawer?: () => void;
};

export default function Sidebar({
  activeItem = "모니터링 홈",
  collapsed = false,
  onToggleCollapse,
  mode = "shell",
  drawerOpen = false,
  onCloseDrawer,
}: SidebarProps) {
  const isDrawer = mode === "drawer";
  const { theme, setTheme } = useTheme();
  const {
    settingsPanelOpen,
    setSettingsPanelOpen,
    trainingPanelOpen,
    setTrainingPanelOpen,
    analysisPanelOpen,
    setAnalysisPanelOpen,
  } = usePanelState();
  const { snapshot, training, notifications } = useAnalysisState();
  // ME-2: 세 하단 버튼(모델 학습·모델 분석·알림 설정) 모두 같은 점을
  // 쓴다 -- 오류(있으면)를 연결 여부보다 먼저 본다는 우선순위도 셋이
  // 같다.
  const analysisStatus: "connected" | "offline" | "error" =
    snapshot && snapshot.errors.length > 0 ? "error" : snapshot?.source.mode === "sql" ? "connected" : "offline";
  // QE: 헤더의 SOURCE 항목(연결 상태 배지)이 제거되면서, 이 점의 툴팁이
  // "SQL에 연결돼 있는지"를 확인할 수 있는 유일한 자리가 됐다 -- 상태
  // 이름뿐 아니라 마지막 스캔 시각/오류 사유까지 함께 보여준다.
  const analysisStatusLabel =
    analysisStatus === "connected"
      ? `SQL 연결됨 · 마지막 스캔 ${formatSidebarDot(snapshot?.created_at)}`
      : analysisStatus === "error"
        ? `연결 오류 — ${snapshot?.errors[0] ?? "알 수 없는 오류"}`
        : "SQL 미연결 · 내장 데이터 사용 중";

  // ME-2: 모델 학습 -- 이 세션에서 수동 업로드로 학습을 실행한 적이
  // 있으면(TrainingState가 채워진다) 그 파일·시각을 보여주고, 없으면
  // 콜드 스타트가 쓴 내장 train.csv가 여전히 활성 모델이라는 뜻이다.
  // 학습 실패는 팝업이 닫히면 사라지는 일시적 폼 상태라 여기(항상 보이는
  // 점)로 끌어올릴 지속 상태가 없다 -- 두 상태만 구분한다.
  const trainingStatus: "connected" | "offline" = training ? "connected" : "offline";
  const trainingStatusLabel = training
    ? `수동 학습 · ${training.performance?.source_filename ?? "-"} · ${formatSidebarDot(training.createdAt)}`
    : "내장 데이터로 학습됨";

  // ME-2: 알림 설정 -- 채널 하나라도 연결되어 있으면 초록, 아니면 회색.
  // 인증 만료·발송 실패의 지속 상태는 저장되지 않아(NotificationSettingsSummary가
  // 마지막 발송 실패 사유를 담지 않는다) 구분하지 않는다.
  const connectedChannelNames = [
    notifications.slack.connected ? "Slack" : null,
    notifications.telegram.connected ? "Telegram" : null,
    notifications.gmail.connected ? "Gmail" : null,
  ].filter((name): name is string => name != null);
  const notificationStatus: "connected" | "offline" = connectedChannelNames.length > 0 ? "connected" : "offline";
  const notificationStatusLabel =
    connectedChannelNames.length > 0 ? `${connectedChannelNames.join(" · ")} 연결됨` : "연결된 채널 없음";
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
    <aside
      className={`sidebar ${collapsed ? "collapsed" : ""}`}
      data-mode={isDrawer ? "drawer" : undefined}
      data-open={isDrawer ? (drawerOpen ? "true" : "false") : undefined}
    >
      <div className="sidebarSurface">
        <button
          type="button"
          className="shellLogoBlock"
          onClick={isDrawer ? onCloseDrawer : onToggleCollapse}
          aria-label={isDrawer ? "메뉴 닫기" : collapsed ? "메뉴 펼치기" : "메뉴 접기"}
          aria-expanded={isDrawer ? drawerOpen : !collapsed}
          title={isDrawer ? "닫기" : collapsed ? "펼치기" : undefined}
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
                      {/* U-2: 펼침 상태는 텍스트만 -- 활성 상태는 이미 왼쪽
                          2px 인디케이터(.navigationItem.active::before)가
                          표시하므로 아이콘이 추가 정보를 주지 않는다.
                          항목이 4개뿐이고 라벨이 짧아 텍스트만으로 계측기
                          다운 인상이 더 강하다. 접힘 상태(railNav, 위)는
                          라벨이 없으므로 아이콘을 유지한다. */}
                      <span>{item.label}</span>
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
          {/* AD그룹: 예전에는 여기 CHAMPION/SNAPSHOT/SQL 연결 세 줄이
              있었다 -- 모델 ID가 길어 200px 사이드바를 넘쳤고, 정보의
              소속도 모델 학습·자동화다(TrainingPanel.tsx로 이전).
              QE: 헤더의 SOURCE 항목(연결 상태 배지)이 중복이라 제거된
              뒤로는, 아래 "모델 분석·자동화" 버튼의 상태 점 툴팁이
              연결 상태를 보여주는 유일한 자리다. */}
          {/* ME-2: 세 버튼(모델 학습·모델 분석·자동화·알림 설정) 모두
              상태 점을 붙인다(RA-1 시절에는 분석에만 있었다) -- 같은
              SidebarStatusDot 컴포넌트를 재사용해 색·크기가 갈리지
              않는다. 순서: 모델 학습 → 모델 분석·자동화 → 알림 설정 →
              화면 모드. */}
          <button
            type="button"
            className={`themeToggle trainingTrigger ${collapsed ? "railIconButton" : ""}`}
            aria-label="모델 학습"
            aria-haspopup="dialog"
            aria-expanded={trainingPanelOpen}
            title={`모델 학습 (${trainingStatusLabel})`}
            onClick={() => setTrainingPanelOpen((open) => !open)}
          >
            <span className="themeTriggerIcon" aria-hidden="true"><Database size={16} strokeWidth={1.5} /></span>
            <span className="themeTriggerLabel">모델 학습</span>
            <SidebarStatusDot status={trainingStatus} label={trainingStatusLabel} />
          </button>
          <button
            type="button"
            className={`themeToggle analysisTrigger ${collapsed ? "railIconButton" : ""}`}
            aria-label="모델 분석·자동화"
            aria-haspopup="dialog"
            aria-expanded={analysisPanelOpen}
            title={`모델 분석·자동화 (${analysisStatusLabel})`}
            onClick={() => setAnalysisPanelOpen((open) => !open)}
          >
            <span className="themeTriggerIcon" aria-hidden="true"><Activity size={16} strokeWidth={1.5} /></span>
            <span className="themeTriggerLabel">모델 분석·자동화</span>
            <SidebarStatusDot status={analysisStatus} label={analysisStatusLabel} />
          </button>
          <button
            type="button"
            className={`themeToggle settingsTrigger ${collapsed ? "railIconButton" : ""}`}
            aria-label="알림 설정"
            aria-haspopup="dialog"
            aria-expanded={settingsPanelOpen}
            title={`알림 설정 (${notificationStatusLabel})`}
            onClick={() => setSettingsPanelOpen((open) => !open)}
          >
            <span className="themeTriggerIcon" aria-hidden="true"><Settings size={16} strokeWidth={1.5} /></span>
            <span className="themeTriggerLabel">알림 설정</span>
            <SidebarStatusDot status={notificationStatus} label={notificationStatusLabel} />
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
              {/* 접힘 상태(라벨 없음)에서만 아이콘이 보인다 -- CSS
                  (.sidebar:not(.collapsed) .themeTriggerIcon{display:none}).
                  펼침 상태는 현재 값을 글자로 말한다 (U-4). */}
              <span className="themeTriggerIcon" aria-hidden="true"><ThemeIcon theme={theme} /></span>
              <span className="themeTriggerLabel">화면 모드 · {THEME_LABELS[theme]}</span>
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
// U-4: 이 드롭다운은 트리거가 접혀 있어도 항상 뜨는 별도 팝업이라 글자
// 넣을 공간이 있다 -- 은유 아이콘(🖥️/☀️/🌙) 대신 이름 텍스트만 쓴다.
function ThemeOptionsList({ theme, onSelect }: { theme: ThemePreference; onSelect: (value: ThemePreference) => void }) {
  return (
    <div className="themeOptions">
      {(["system", "light", "dark"] as ThemePreference[]).map((value) => (
        <button
          key={value}
          type="button"
          className={theme === value ? "active" : ""}
          onClick={() => onSelect(value)}
          role="menuitemradio"
          aria-checked={theme === value}
        >
          {THEME_LABELS[value]}
        </button>
      ))}
    </div>
  );
}

function NavIcon({ name }: { name: string }) {
  const paths: Record<string, React.ReactNode> = {
    monitor: <><rect x="2" y="4" width="20" height="13" rx="2" /><path d="M8 21h8" /><path d="M12 17v4" /><path d="m6 12 3-3 3 2 4-5" /></>,
    treemap: <><rect x="3" y="3" width="10" height="9" rx="1" /><rect x="15" y="3" width="6" height="5" rx="1" /><rect x="15" y="10" width="6" height="11" rx="1" /><rect x="3" y="14" width="10" height="7" rx="1" /></>,
    analysis: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /><path d="M8 11h6" /><path d="M11 8v6" /></>,
    alert: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" /><path d="M10 21h4" /></>,
    star: <path d="M12 2.5 15.09 9l7.16.6-5.45 4.73L18.5 21 12 17.27 5.5 21l1.7-6.67L1.75 9.6 8.91 9 12 2.5Z" />,
  };
  return <svg className="navigationIcon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

// 설정 패널 신설 §B: 이모지(🖥️☀️🌙) 대신 lucide-react 컴포넌트를 쓴다 --
// OS/브라우저마다 이모지 모양이 달라지는 문제가 없다. U-4: 펼침 상태는
// 텍스트("화면 모드 · 시스템")로 말하므로 이 아이콘은 접힘 상태(라벨을
// 놓을 자리가 없는 40px 레일)에서만 보인다.
function ThemeIcon({ theme }: { theme: ThemePreference }) {
  const Icon = theme === "dark" ? Moon : theme === "system" ? Monitor : Sun;
  return <Icon className="themeIcon" aria-hidden="true" size={16} strokeWidth={1.5} />;
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
