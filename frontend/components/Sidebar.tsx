"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  type ThemePreference,
  useTheme,
} from "@/components/ThemeProvider";
import {
  getAlertSummary,
  getAnalysisHistory,
  getApiHealth,
  getDashboardOverview,
  getModels,
  getPredictionHistory,
} from "@/lib/api";

const navigationItems = [
  { label: "개요", href: "/", icon: "overview" },
  { label: "모델 학습", href: "/training", icon: "model" },
  { label: "수율 예측", href: "/prediction", icon: "trend" },
  { label: "불량 원인 분석", href: "/root-cause", icon: "analysis" },
  { label: "사전 알람 로그", href: "/alerts", icon: "alert" },
  { label: "자동화 상태", href: "/automation", icon: "automation" },
];

type SidebarProps = {
  activeItem?:
    | "개요"
    | "모델 학습"
    | "수율 예측"
    | "불량 원인 분석"
    | "사전 알람 로그"
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
    alerts: "loading" | "ready" | "empty" | "error";
    predictions: "loading" | "ready" | "empty" | "error";
    analyses: "loading" | "ready" | "empty" | "error";
    overview: "loading" | "analysis" | "prediction" | "model" | "empty" | "error";
  }>({ api: "loading", models: "loading", alerts: "loading", predictions: "loading", analyses: "loading", overview: "loading" });
  const themeMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let mounted = true;

    void Promise.allSettled([
      getApiHealth(), getModels(), getAlertSummary(),
      getPredictionHistory({ limit: 1 }), getAnalysisHistory({ limit: 1 }), getDashboardOverview(),
    ]).then(
      ([healthResult, modelsResult, alertsResult, predictionsResult, analysesResult, overviewResult]) => {
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
          alerts: alertsResult.status === "rejected" ? "error" : alertsResult.value.total ? "ready" : "empty",
          predictions: predictionsResult.status === "rejected" ? "error" : predictionsResult.value.total ? "ready" : "empty",
          analyses: analysesResult.status === "rejected" ? "error" : analysesResult.value.total ? "ready" : "empty",
          overview: overviewResult.status === "rejected" ? "error" : overviewResult.value.source_type,
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
    if (label === "수율 예측") {
      if (systemStatus.predictions === "error") return { tone: "red", label: "이력 조회 실패" };
      if (systemStatus.predictions === "ready") return { tone: "green", label: "예측 이력 정상" };
      return systemStatus.models === "ready" ? { tone: "yellow", label: "모델 준비 · 예측 필요" } : { tone: "yellow", label: "Model 필요" };
    }
    if (label === "불량 원인 분석") {
      if (systemStatus.analyses === "error") return { tone: "red", label: "이력 조회 실패" };
      if (systemStatus.analyses === "ready") return { tone: "green", label: "불량 원인 분석 Ready" };
      return systemStatus.predictions === "ready" ? { tone: "yellow", label: "예측 완료 · 분석 필요" } : { tone: "yellow", label: "예측 필요" };
    }
    if (label === "개요") {
      if (systemStatus.overview === "error") return { tone: "red", label: "요약 조회 실패" };
      if (systemStatus.overview === "analysis") return { tone: "green", label: "최근 분석 기준" };
      if (systemStatus.overview === "prediction") return { tone: "yellow", label: "최근 예측 기준" };
      return { tone: "gray", label: systemStatus.overview === "model" ? "모델 준비" : "이력 없음" };
    }
    if (label === "사전 알람 로그") {
      if (systemStatus.alerts === "error") return { tone: "red", label: "조회 실패" };
      if (systemStatus.alerts === "loading") return { tone: "gray", label: "조회 중" };
      return systemStatus.alerts === "empty" ? { tone: "gray", label: "알람 없음" } : { tone: "green", label: "로그 정상" };
    }
    if (label === "자동화 상태") {
      return { tone: "yellow", label: "외부 자동화 설정 확인 필요" };
    }
    return { tone: "green", label: "Ready" };
  }

  return (
    <aside className="sidebar">
      <Link
        className="brand"
        href="/"
        aria-label="SK SUNI C 5기"
      >
        <Image
          className="brandLogo"
          src="/sk-suni-c-5-character.png"
          alt="SK SUNI C 5기"
          width={150}
          height={150}
          unoptimized
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
