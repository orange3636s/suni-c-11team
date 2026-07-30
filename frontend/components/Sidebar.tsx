const navigationItems = [
  "개요",
  "데이터 업로드",
  "수율 예측",
  "원인 분석",
  "사전 알람 로그",
  "모델 모니터링",
];

export default function Sidebar() {
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
          {navigationItems.map((item, index) => (
            <li key={item}>
              <a
                className={`navigationItem ${index === 0 ? "active" : ""}`}
                href={index === 0 ? "#overview" : `#section-${index}`}
                aria-current={index === 0 ? "page" : undefined}
              >
                <span className="navigationDot" aria-hidden="true" />
                {item}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      <div className="sidebarFooter">
        <span className="statusDot normal" aria-hidden="true" />
        분석 환경 준비 중
      </div>
    </aside>
  );
}
