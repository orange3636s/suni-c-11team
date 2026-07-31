import Header from "@/components/Header";
import EmptyState from "@/components/EmptyState";
import PipelineStep from "@/components/PipelineStep";
import Sidebar from "@/components/Sidebar";
import StatusCard from "@/components/StatusCard";
import StatusBadge from "@/components/StatusBadge";

const statusCards = [
  {
    label: "Average Predicted Yield",
    value: "-",
    unit: "%",
    detail: "예측 실행 후 평균 수율 표시",
  },
  {
    label: "Yield Loss",
    value: "-",
    unit: "%p",
    detail: "기준 수율 대비 손실",
    tone: "warning" as const,
  },
  {
    label: "High-risk LOTs",
    value: "-",
    detail: "위험 LOT 집계 전",
    tone: "danger" as const,
  },
  {
    label: "Critical Equipment",
    value: "-",
    detail: "장비별 분석 결과 필요",
    tone: "danger" as const,
  },
  {
    label: "Model R²",
    value: "-",
    detail: "Test 성능 기준",
    tone: "warning" as const,
  },
  {
    label: "RMSE",
    value: "-",
    detail: "저장 모델 평가값",
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
    description: "저장된 모델로 Wafer별 수율과 위험 상태를 예측합니다.",
    status: "구현 완료" as const,
  },
  {
    title: "원인 후보 분석",
    description: "SHAP 기반으로 공정 이상에 영향을 준 후보를 분석합니다.",
    status: "구현 완료" as const,
  },
  {
    title: "AI Report",
    description: "예측과 원인 후보를 엔지니어용 자동 보고서로 정리합니다.",
    status: "구현 완료" as const,
  },
  {
    title: "n8n Automation",
    description: "Webhook으로 통합 분석과 위험 분기를 자동 실행합니다.",
    status: "구현 완료" as const,
  },
  {
    title: "Slack Alert",
    description: "실제 알림 사용을 위해 Slack credential 설정이 필요합니다.",
    status: "설정 필요" as const,
  },
];

export default function Home() {
  return (
    <div className="appShell">
      <Sidebar />
      <div className="contentShell">
        <Header />
        <main id="overview" className="mainContent">
          <section className="riskSummaryBanner" aria-labelledby="risk-summary-title">
            <div className="riskSummaryMessage">
              <span className="riskSummaryIcon" aria-hidden="true">
                <span />
              </span>
              <div>
                <StatusBadge label="No analysis result" tone="neutral" dot={false} />
                <h2 id="risk-summary-title">현재 분석 결과가 없습니다.</h2>
                <p>
                  CSV 검증과 모델 예측을 실행하면 위험 LOT, 평균 수율,
                  주요 영향 Step을 이곳에서 빠르게 확인할 수 있습니다.
                </p>
              </div>
            </div>
            <dl className="riskSummaryMetrics">
              <div>
                <dt>위험 LOT</dt>
                <dd>-</dd>
              </div>
              <div>
                <dt>평균 예측 수율</dt>
                <dd>-</dd>
              </div>
              <div>
                <dt>주요 영향 Step</dt>
                <dd>-</dd>
              </div>
            </dl>
          </section>

          <section aria-labelledby="summary-title">
            <div className="sectionHeading">
              <div>
                <span className="sectionLabel">운영 요약</span>
                <h2 id="summary-title">핵심 지표</h2>
              </div>
              <p>공정 데이터가 연결되면 지표가 자동으로 표시됩니다.</p>
            </div>
            <div className="cardGrid kpiGrid">
              {statusCards.map((card) => (
                <StatusCard key={card.label} {...card} />
              ))}
            </div>
          </section>

          <section className="overviewLinks quickActions" aria-label="주요 기능 바로가기">
            {[
              ["데이터 업로드", "/upload"],
              ["모델 학습", "/training"],
              ["수율 예측", "/prediction"],
              ["원인 분석", "/root-cause"],
              ["분석 보고서", "/report"],
              ["자동화 상태", "/automation"],
            ].map(([label, href]) => (
              <a className="button secondary" href={href} key={href}>
                {label}
              </a>
            ))}
          </section>

          <section className="dashboardAnalysisGrid" aria-label="분석 시각화">
            <article className="surfaceCard analysisPrimary">
              <div className="sectionHeading compact">
                <div>
                  <span className="sectionLabel">Yield trend</span>
                  <h2>LOT / Wafer 수율 예측 추이</h2>
                </div>
                <StatusBadge label="데이터 필요" tone="neutral" />
              </div>
              <p className="chartDescription">
                실제값과 예측값, 위험 기준선을 동일한 축에서 비교합니다.
              </p>
              <EmptyState
                title="표시할 예측 데이터가 없습니다."
                description="수율 예측 페이지에서 모델과 CSV를 선택해 예측을 실행해 주세요."
              />
            </article>
            <article className="surfaceCard analysisSecondary">
              <div className="sectionHeading compact">
                <div>
                  <span className="sectionLabel">Yield loss drivers</span>
                  <h2>주요 수율 저하 요인</h2>
                </div>
              </div>
              <p className="chartDescription">
                SHAP 영향도 기준 상위 Step · R · D · EQ를 보여줍니다.
              </p>
              <EmptyState
                compact
                title="원인 분석 대기"
                description="SHAP 분석 결과가 생성되면 영향 방향과 크기를 표시합니다."
              />
            </article>
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
