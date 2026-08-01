"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  type ThemePreference,
  useTheme,
} from "@/components/ThemeProvider";
import { getApiHealth, getModels } from "@/lib/api";

const navigationItems = [
  { label: "개요", href: "/", icon: "overview" },
  { label: "데이터 전처리", href: "/upload", icon: "upload" },
  { label: "모델 학습", href: "/training", icon: "model" },
  { label: "수율 예측", href: "/prediction", icon: "trend" },
  { label: "원인 분석", href: "/root-cause", icon: "analysis" },
  { label: "분석 보고서", href: "/report", icon: "report" },
  { label: "자동화 상태", href: "/automation", icon: "automation" },
  { label: "사전 알람 로그", href: "/#alerts", icon: "alert" },
  { label: "모델 모니터링", href: "/#monitoring", icon: "monitor" },
];

type SidebarProps = {
  activeItem?:
    | "개요"
    | "데이터 전처리"
    | "모델 학습"
    | "수율 예측"
    | "원인 분석"
    | "분석 보고서"
    | "자동화 상태";
};

type MenuStatus = {
  tone: "green" | "yellow" | "gray" | "red";
  label: string;
};

export default function Sidebar({ activeItem = "개요" }: SidebarProps) {
  const { theme, setTheme } = useTheme();
  const [themeMenuOpen, setThemeMenuOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState<{
    api: "loading" | "ready" | "error";
    models: "loading" | "ready" | "empty" | "error";
  }>({ api: "loading", models: "loading" });
  const themeMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let mounted = true;

    void Promise.allSettled([getApiHealth(), getModels()]).then(
      ([healthResult, modelsResult]) => {
        if (!mounted) return;
        setSystemStatus({
          api:
            healthResult.status === "fulfilled" &&
            healthResult.value.status === "ok"
              ? "ready"
              : "error",
          models:
            modelsResult.status === "rejected"
              ? "error"
              : modelsResult.value.models.length
                ? "ready"
                : "empty",
        });
      },
    );

    return () => {
      mounted = false;
    };
  }, []);

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

  function getMenuStatus(label: string): MenuStatus {
    if (systemStatus.api === "loading") {
      return { tone: "gray", label: "상태 확인 중" };
    }
    if (systemStatus.api === "error") {
      return { tone: "red", label: "API Error" };
    }
    if (label === "분석 보고서") {
      return { tone: "gray", label: "아직 실행 안함" };
    }
    if (label === "모델 학습") {
      if (systemStatus.models === "loading") {
        return { tone: "yellow", label: "모델 확인 중" };
      }
      if (systemStatus.models === "error") {
        return { tone: "red", label: "Model Error" };
      }
      if (systemStatus.models === "empty") {
        return { tone: "yellow", label: "Model 없음" };
      }
      return { tone: "green", label: "Model Ready" };
    }
    if (["수율 예측", "원인 분석"].includes(label)) {
      return systemStatus.models === "ready"
        ? { tone: "green", label: "Ready" }
        : { tone: "yellow", label: "Model 필요" };
    }
    return { tone: "green", label: "Ready" };
  }

  return (
    <aside className="sidebar">
      <Link
        className="brand"
        href="/"
        aria-label="써니C 11팀 홈으로 이동"
      >
        <Image
          className="brandLogo"
          src="/sk-hynix-logo.png"
          alt="SK hynix"
          width={88}
          height={43}
          priority
        />
        <strong className="brandTitle">써니C 11팀</strong>
      </Link>

      <nav aria-label="주요 메뉴">
        <ul className="navigationList">
          {navigationItems.map((item) => {
            const isActive = item.label === activeItem;
            const status = getMenuStatus(item.label);
            return (
              <li key={item.label}>
                <Link
                  className={`navigationItem ${isActive ? "active" : ""}`}
                  href={item.href}
                  aria-current={isActive ? "page" : undefined}
                >
                  <NavIcon name={item.icon} />
                  <span>{item.label}</span>
                  <span
                    className={`menuStatusDot ${status.tone}`}
                    data-tooltip={status.label}
                    aria-label={status.label}
                    role="status"
                  />
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="sidebarFooter" ref={themeMenuRef}>
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
                  role="menuitemradio"
                  aria-checked={theme === value}
                  onClick={() => {
                    setTheme(value);
                    setThemeMenuOpen(false);
                  }}
                >
                  <span className="themeOptionLabel">
                    <ThemeIcon theme={value} />
                    <span>{label}</span>
                  </span>
                  {theme === value && <span aria-hidden="true">✓</span>}
                </button>
              ))}
            </div>
          </div>
        )}
        <button
          className="themeTriggerArea"
          type="button"
          aria-label={`Theme: ${theme[0].toUpperCase() + theme.slice(1)}`}
          aria-haspopup="menu"
          aria-expanded={themeMenuOpen}
          onClick={() => setThemeMenuOpen((open) => !open)}
        >
          <span className="themeTrigger" aria-hidden="true">
            <ThemeIcon theme={theme} />
          </span>
          <span className="sidebarFooterCopy">
            <strong>Theme</strong>
            <span>{theme[0].toUpperCase() + theme.slice(1)}</span>
          </span>
          <svg
            className="themeChevron"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="m4 6 4 4 4-4" />
          </svg>
        </button>
      </div>
    </aside>
  );
}

function ThemeIcon({ theme }: { theme: ThemePreference }) {
  if (theme === "light") {
    return (
      <svg className="themeIcon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" />
      </svg>
    );
  }

  if (theme === "dark") {
    return (
      <svg className="themeIcon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M20.4 14.6A8.5 8.5 0 0 1 9.4 3.6 8.5 8.5 0 1 0 20.4 14.6Z" />
      </svg>
    );
  }

  return (
    <svg className="themeIcon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="3" y="4" width="18" height="13" rx="2.5" />
      <path d="M8 21h8M12 17v4" />
    </svg>
  );
}

function NavIcon({ name }: { name: string }) {
  const paths: Record<string, React.ReactNode> = {
    overview: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="2" />
        <rect x="14" y="3" width="7" height="7" rx="2" />
        <rect x="3" y="14" width="7" height="7" rx="2" />
        <rect x="14" y="14" width="7" height="7" rx="2" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4" />
        <path d="m7.5 8.5 4.5-4.5 4.5 4.5" />
        <path d="M5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5" />
      </>
    ),
    model: (
      <>
        <rect x="4" y="4" width="16" height="16" rx="4" />
        <path d="M9 9h6v6H9z" />
        <path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M20 9h2M2 15h2M20 15h2" />
      </>
    ),
    trend: (
      <>
        <path d="M4 18 10 12l4 3 6-8" />
        <path d="M15 7h5v5" />
      </>
    ),
    analysis: (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m16.5 16.5 4 4" />
        <path d="M8 12h6M11 9v6" />
      </>
    ),
    report: (
      <>
        <path d="M6 3h9l4 4v14H6z" />
        <path d="M15 3v5h4M9 12h6M9 16h6" />
      </>
    ),
    automation: (
      <>
        <path d="M6 8a7 7 0 0 1 12-2l2 2" />
        <path d="M20 4v4h-4M18 16a7 7 0 0 1-12 2l-2-2" />
        <path d="M4 20v-4h4" />
      </>
    ),
    alert: (
      <>
        <path d="M12 3 2.8 20h18.4z" />
        <path d="M12 9v4M12 17h.01" />
      </>
    ),
    monitor: (
      <>
        <rect x="3" y="4" width="18" height="13" rx="3" />
        <path d="M8 21h8M12 17v4M7 11h3l2-3 2 5 2-2h2" />
      </>
    ),
  };
  return (
    <svg
      className="navigationIcon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
