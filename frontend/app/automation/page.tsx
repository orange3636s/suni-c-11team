"use client";

import { useEffect, useState } from "react";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { getApiHealth } from "@/lib/api";

type ApiState = "확인 중" | "연결됨" | "연결 실패";

export default function AutomationPage() {
  const webhookUrl = process.env.NEXT_PUBLIC_N8N_WEBHOOK_URL ?? "";
  const [apiState, setApiState] = useState<ApiState>("확인 중");

  useEffect(() => {
    void getApiHealth()
      .then((result) =>
        setApiState(result.status === "ok" ? "연결됨" : "연결 실패"),
      )
      .catch(() => setApiState("연결 실패"));
  }, []);

  return (
    <div className="appShell">
      <Sidebar activeItem="자동화 상태" />
      <div className="contentShell">
        <Header />
        <main className="mainContent automationPage">
          <section className="intro">
            <div>
              <span className="eyebrow">Workflow Automation</span>
              <h1>자동화 상태</h1>
              <p>
                n8n Webhook, FastAPI 통합 분석 및 Slack 알림 설정 상태를
                확인합니다.
              </p>
            </div>
          </section>

          <section className="automationStatusGrid">
            <StatusPanel
              label="n8n 연결 상태"
              value={webhookUrl ? "URL 설정됨" : "연결 전"}
              tone={webhookUrl ? "normal" : "warning"}
              detail={
                webhookUrl
                  ? "Webhook URL 환경변수가 설정되어 있습니다."
                  : "n8n Webhook URL이 설정되지 않았습니다."
              }
            />
            <StatusPanel
              label="FastAPI 분석 API"
              value={apiState}
              tone={
                apiState === "연결됨"
                  ? "normal"
                  : apiState === "연결 실패"
                    ? "danger"
                    : "warning"
              }
              detail="POST /api/analyze"
            />
            <StatusPanel
              label="Slack 알림"
              value="설정 전"
              tone="warning"
              detail="n8n에서 Slack credential 연결이 필요합니다."
            />
            <StatusPanel
              label="최근 분석 ID"
              value="-"
              detail="이번 단계에서는 n8n 실행 이력을 저장하지 않습니다."
            />
            <StatusPanel
              label="최근 알림 상태"
              value="-"
              detail="Slack 실행 결과는 n8n에서 확인합니다."
            />
          </section>

          <section className="resultCard automationGuide">
            <span className="sectionLabel">Webhook 설정</span>
            <h2>n8n 연결 안내</h2>
            <p>
              프런트 환경변수에 활성화된 n8n Webhook URL을 설정하면 이
              화면에서 설정 여부를 확인할 수 있습니다.
            </p>
            <pre>
              <code>
                NEXT_PUBLIC_N8N_WEBHOOK_URL=
                http://localhost:5678/webhook/manufacturing-ai-analysis
              </code>
            </pre>
            {webhookUrl && (
              <p className="configuredWebhook">
                설정된 URL: <strong>{webhookUrl}</strong>
              </p>
            )}
            <p>
              실제 실행 이력, Slack credential 및 운영 인증 상태는 n8n
              관리자 화면에서 확인하세요.
            </p>
          </section>
        </main>
      </div>
    </div>
  );
}

function StatusPanel({
  label,
  value,
  detail,
  tone = "normal",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "normal" | "warning" | "danger";
}) {
  return (
    <article className="statusCard automationStatusCard">
      <div className="automationStatusHeading">
        <span className={`statusDot ${tone}`} aria-hidden="true" />
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <p>{detail}</p>
    </article>
  );
}
