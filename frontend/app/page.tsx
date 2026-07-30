import Header from "@/components/Header";
import PipelineStep from "@/components/PipelineStep";
import Sidebar from "@/components/Sidebar";
import StatusCard from "@/components/StatusCard";

const statusCards = [
  {
    label: "분석 Wafer 수",
    value: "-",
    detail: "데이터 연결 대기",
  },
  {
    label: "평균 예측 수율",
    value: "-",
    detail: "예측 모델 준비 전",
  },
  {
    label: "위험 Wafer 수",
    value: "-",
    detail: "위험 기준 설정 대기",
    tone: "danger" as const,
  },
  {
    label: "모델 신뢰도",
    value: "준비 중",
    detail: "검증 데이터 필요",
    tone: "warning" as const,
  },
];

const pipelineSteps = [
  {
    title: "CSV 업로드",
    description: "공정 원본 데이터를 안전하게 불러옵니다.",
    status: "구현 완료" as const,
  },
  {
    title: "데이터 검증",
    description: "스키마, 결측치, 중복 여부를 확인합니다.",
    status: "구현 완료" as const,
  },
  {
    title: "데이터 전처리",
    description: "설정 기반으로 분석 가능한 형태를 준비합니다.",
    status: "구현 완료" as const,
  },
  {
    title: "머신러닝 학습",
    description: "검증된 데이터로 회귀 모델을 비교하고 저장합니다.",
    status: "구현 완료" as const,
  },
  {
    title: "수율 예측",
    description: "저장된 모델로 Wafer별 수율을 예측할 예정입니다.",
    status: "다음 단계" as const,
  },
  {
    title: "원인 후보 분석",
    description: "공정 이상에 영향을 준 후보를 제시할 예정입니다.",
    status: "개발 예정" as const,
  },
  {
    title: "n8n 알림",
    description: "검증된 위험 신호의 알림을 자동화할 예정입니다.",
    status: "개발 예정" as const,
  },
];

export default function Home() {
  return (
    <div className="appShell">
      <Sidebar />
      <div className="contentShell">
        <Header />
        <main id="overview" className="mainContent">
          <section className="intro">
            <div>
              <span className="eyebrow">제조 인텔리전스 플랫폼</span>
              <h1>제조 공정 불량 예측 및 원인 분석 AI</h1>
              <p>
                공정 데이터 검증, 수율 위험 예측,
                <br className="desktopBreak" /> 불량 원인 후보 분석 및 사전
                알림 시스템
              </p>
            </div>
            <div className="phaseBadge">
              <span className="statusDot warning" aria-hidden="true" />
              초기 환경 구성 단계
            </div>
          </section>

          <section aria-labelledby="summary-title">
            <div className="sectionHeading">
              <div>
                <span className="sectionLabel">운영 요약</span>
                <h2 id="summary-title">핵심 지표</h2>
              </div>
              <p>공정 데이터가 연결되면 지표가 자동으로 표시됩니다.</p>
            </div>
            <div className="cardGrid">
              {statusCards.map((card) => (
                <StatusCard key={card.label} {...card} />
              ))}
            </div>
          </section>

          <section className="pipelineSection" aria-labelledby="pipeline-title">
            <div className="sectionHeading">
              <div>
                <span className="sectionLabel">분석 흐름</span>
                <h2 id="pipeline-title">공정 AI 파이프라인</h2>
              </div>
              <p>각 단계는 데이터와 모델 준비 상태에 따라 활성화됩니다.</p>
            </div>
            <ol className="pipelineList">
              {pipelineSteps.map((step, index) => (
                <PipelineStep
                  key={step.title}
                  index={index + 1}
                  {...step}
                />
              ))}
            </ol>
          </section>
        </main>
      </div>
    </div>
  );
}
