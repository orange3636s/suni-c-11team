import Image from "next/image";
import Link from "next/link";

const navigationItems = [
  { label: "개요", href: "/", icon: "overview" },
  { label: "데이터 업로드", href: "/upload", icon: "upload" },
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
    | "데이터 업로드"
    | "모델 학습"
    | "수율 예측"
    | "원인 분석"
    | "분석 보고서"
    | "자동화 상태";
};

export default function Sidebar({ activeItem = "개요" }: SidebarProps) {
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
            return (
              <li key={item.label}>
                <Link
                  className={`navigationItem ${isActive ? "active" : ""}`}
                  href={item.href}
                  aria-current={isActive ? "page" : undefined}
                >
                  <NavIcon name={item.icon} />
                  <span>{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="sidebarFooter">
        <span className="sidebarStatusIcon" aria-hidden="true">
          <span />
        </span>
        <div>
          <strong>Analysis workspace</strong>
          <span>API와 모델 상태는 상단에서 확인</span>
        </div>
      </div>
    </aside>
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
