import Image from "next/image";

const navigationItems = [
  { label: "개요", href: "/" },
  { label: "데이터 업로드", href: "/upload" },
  { label: "모델 학습", href: "/training" },
  { label: "수율 예측", href: "/prediction" },
  { label: "원인 분석", href: "/root-cause" },
  { label: "분석 보고서", href: "/report" },
  { label: "자동화 상태", href: "/automation" },
  { label: "사전 알람 로그", href: "/#alerts" },
  { label: "모델 모니터링", href: "/#monitoring" },
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
      <div className="brand">
        <Image
          className="brandLogo"
          src="/sk-hynix-logo.png"
          alt="SK hynix"
          width={120}
          height={59}
          priority
        />
        <strong className="brandTitle">
          제조 공정 불량 예측
          <br />
          &amp; 원인분석 AI
        </strong>
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
