const navigationItems = [
  { label: "개요", href: "/" },
  { label: "데이터 업로드", href: "/upload" },
  { label: "모델 학습", href: "/training" },
  { label: "수율 예측", href: "/#prediction" },
  { label: "원인 분석", href: "/#root-cause" },
  { label: "사전 알람 로그", href: "/#alerts" },
  { label: "모델 모니터링", href: "/#monitoring" },
];

type SidebarProps = {
  activeItem?: "개요" | "데이터 업로드" | "모델 학습";
};

export default function Sidebar({ activeItem = "개요" }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brandMark" aria-hidden="true">
          S
        </span>
        <div>
          <strong>SEMI AI</strong>
          <span>공정 분석 시스템</span>
        </div>
      </div>

      <nav aria-label="주요 메뉴">
        <ul className="navigationList">
          {navigationItems.map((item) => {
            const isActive = item.label === activeItem;
            return (
              <li key={item.label}>
                <a
                  className={`navigationItem ${isActive ? "active" : ""}`}
                  href={item.href}
                  aria-current={isActive ? "page" : undefined}
                >
                  <span className="navigationDot" aria-hidden="true" />
                  {item.label}
                </a>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="sidebarFooter">
        <span className="statusDot normal" aria-hidden="true" />
        분석 환경 준비 중
      </div>
    </aside>
  );
}
