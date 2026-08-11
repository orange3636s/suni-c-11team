"use client";

import { Fragment, useEffect, useState } from "react";
import DashboardShell from "@/components/DashboardShell";
import { PageHeaderMeta } from "@/components/LastRunNote";
import { usePanelState } from "@/components/PanelStateProvider";
import { getNotifyHistory } from "@/lib/api";
import { formatLastRun } from "@/lib/timeFormat";
import type { NotifyHistoryItem } from "@/types/data";

// SE그룹: 발송된 알림의 내용을 다시 볼 수 있는 화면 -- 사용자가 알림을
// 받고 접속했을 때 여기서 확인한다. 발송/건너뜀 모두 최신순으로 보여주고,
// 행을 열면 발송 당시의 메시지 원문을 그대로 펼친다(재계산하지 않는다).
export default function NotifyHistoryPage() {
  const { setSettingsPanelOpen } = usePanelState();
  const [items, setItems] = useState<NotifyHistoryItem[] | null>(null);
  const [error, setError] = useState("");
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [copiedId, setCopiedId] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    getNotifyHistory(100)
      .then((response) => {
        if (!cancelled) setItems(response.items);
      })
      .catch((failure) => {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "알림 기록을 불러오지 못했습니다.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleCopy(item: NotifyHistoryItem) {
    if (!item.message_text) return;
    try {
      await navigator.clipboard.writeText(item.message_text);
      setCopiedId(item.id);
      window.setTimeout(() => setCopiedId((current) => (current === item.id ? null : current)), 2000);
    } catch {
      // best-effort -- 클립보드 접근이 막힌 환경에서는 조용히 넘어간다.
    }
  }

  return (
    <DashboardShell activeItem="알림 기록">
      <div className="rcPage">
        <section className="uploadIntro pageHeading">
          <span className="eyebrow">알림·자동화</span>
          <h1>알림 기록</h1>
          <p>
            발송된 알림과 건너뛴 이력을 최신순으로 모아 봅니다. 메시지는 발송 당시의 원문을 그대로 보관합니다(재계산하지
            않습니다).
          </p>
          {/* T9-2: 이 화면의 항목들은 각자의 발송 시점 데이터셋으로 만들어졌다
              (아래 표의 "시각" 열이 항목별 실제 발송 시각이다) -- 헤더의
              시각·파일명은 "지금 이 화면이 조회 기준으로 삼는" 현재 분석을
              가리킬 뿐이므로 "마지막 실행"이 아니라 "현재 분석 기준"이라고
              부른다. */}
          <PageHeaderMeta label="현재 분석 기준" />
        </section>

        {error && <p className="errorMessage">{error}</p>}

        {/* B-5: 원인 분석의 분석-없음 안내(analysisErrorBox)와 같은 모양으로
            통일한다 -- 버튼이 문장 안에 섞여 있던 이전 모양은 이 화면만
            달랐다. */}
        {items && items.length === 0 && (
          <div className="analysisErrorBox" role="status">
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">발송된 알림이 없습니다. 알림·자동화 설정에서 채널을 연결하고 자동화를 켜세요.</p>
            </div>
            <button type="button" className="button sm secondary" onClick={() => setSettingsPanelOpen(true)}>
              열기
            </button>
          </div>
        )}

        {items && items.length > 0 && (
          <div className="resultCard" style={{ overflowX: "auto" }}>
            <table className="dataTable">
              <thead>
                <tr>
                  <th>시각</th>
                  <th>채널</th>
                  <th>소스</th>
                  <th>건수</th>
                  <th>상태</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => {
                  const expanded = expandedId === item.id;
                  return (
                    <Fragment key={item.id}>
                      <tr
                        onClick={() => setExpandedId(expanded ? null : item.id)}
                        style={{ cursor: "pointer" }}
                        aria-expanded={expanded}
                      >
                        <td>{formatLastRun(item.sent_at)}</td>
                        <td>{item.channels.length > 0 ? item.channels.join(" · ") : "—"}</td>
                        <td>{item.dataset_label ?? "—"}</td>
                        <td>{item.status === "sent" ? `TOP ${item.sent_count > 0 ? "10" : "0"}` : "—"}</td>
                        <td>
                          {item.status === "sent" ? "발송됨" : `건너뜀 · ${item.skip_reason ?? "사유 없음"}`}
                        </td>
                      </tr>
                      {expanded && (
                        <tr>
                          <td colSpan={5}>
                            {item.message_text ? (
                              <div className="notifyHistoryMessage">
                                <div className="notifyFormActions">
                                  <button type="button" className="button secondary" onClick={() => void handleCopy(item)}>
                                    {copiedId === item.id ? "복사됨" : "복사"}
                                  </button>
                                </div>
                                <pre className="notifyHistoryMessageText">{item.message_text}</pre>
                              </div>
                            ) : (
                              <p className="emptyMessage">이 건은 실제로 발송되지 않아 보관된 메시지가 없습니다.</p>
                            )}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </DashboardShell>
  );
}
