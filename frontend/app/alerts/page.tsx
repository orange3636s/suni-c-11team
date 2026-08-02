"use client";

import { useCallback, useEffect, useState } from "react";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { getAlerts, getAlertSummary, updateAlertStatus } from "@/lib/api";
import type { AlertLogItem, AlertStatus, AlertSummary } from "@/types/data";

const EMPTY_SUMMARY: AlertSummary = { total: 0, new_count: 0, acknowledged_count: 0, resolved_count: 0, critical_count: 0, warning_count: 0, external_not_configured_count: 0 };

function display(value: number | null, suffix = ""): string {
  return value === null ? "-" : `${value.toFixed(2)}${suffix}`;
}

export default function AlertsPage() {
  const [summary, setSummary] = useState(EMPTY_SUMMARY);
  const [items, setItems] = useState<AlertLogItem[]>([]);
  const [selected, setSelected] = useState<AlertLogItem | null>(null);
  const [risk, setRisk] = useState("");
  const [status, setStatus] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("date");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true); setError("");
    const query = new URLSearchParams({ limit: "100", sort });
    if (risk) query.set("risk_level", risk);
    if (status) query.set("status", status);
    if (search) query.set("wafer_id", search);
    try {
      const [list, nextSummary] = await Promise.all([getAlerts(query.toString()), getAlertSummary()]);
      setItems(list.items); setSummary(nextSummary);
      setSelected((current) => list.items.find((item) => item.alert_id === current?.alert_id) ?? list.items[0] ?? null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "알람 로그를 불러오지 못했습니다.");
    } finally { setLoading(false); }
  }, [risk, search, sort, status]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  async function changeStatus(nextStatus: AlertStatus) {
    if (!selected) return;
    try { await updateAlertStatus(selected.alert_id, nextStatus); await load(); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : "알람 상태를 변경하지 못했습니다."); }
  }

  return <div className="appShell">
    <Sidebar activeItem="사전 알람 로그" />
    <div className="contentShell"><Header />
      <main className="mainContent dashboardPage">
        <section className="intro"><div><span className="eyebrow">Early Warning</span><h1>사전 알람 로그</h1><p>실제 수율 예측에서 발생한 Critical·Warning 이벤트만 기록합니다.</p></div></section>
        <div className="runtimeNotice">외부 자동화 미연결 · Dashboard 내부 로그만 기록 중</div>
        <section className="dashboardKpis">
          {[["전체",summary.total],["신규",summary.new_count],["Critical",summary.critical_count],["Warning",summary.warning_count],["해결",summary.resolved_count]].map(([label,value]) => <article className="statusCard" key={label}><span>{label}</span><strong>{value}</strong></article>)}
        </section>
        <section className="resultCard dashboardFilters" aria-label="알람 필터">
          <select aria-label="위험도" value={risk} onChange={(event) => setRisk(event.target.value)}><option value="">전체 위험도</option><option value="danger">Critical</option><option value="warning">Warning</option></select>
          <select aria-label="상태" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">전체 상태</option><option>New</option><option>Acknowledged</option><option>Resolved</option></select>
          <input aria-label="LOT 또는 Wafer 검색" placeholder="LOT_WAFER_ID 검색" value={search} onChange={(event) => setSearch(event.target.value)} />
          <select aria-label="정렬" value={sort} onChange={(event) => setSort(event.target.value)}><option value="date">최신순</option><option value="risk">위험도순</option></select>
          <button className="button secondary" type="button" onClick={() => void load()}>새로고침</button>
        </section>
        {error && <div className="messageBox error retryMessage" role="alert"><span><strong>사전 알람 로그를 불러오지 못했습니다.</strong><small>{error}</small></span><button className="button secondary compact" onClick={() => void load()}>다시 시도</button></div>}
        <section className="dashboardSplit">
          <div className="resultCard dashboardTableWrap">
            {loading ? <p>사전 알람 로그를 불러오는 중입니다.</p> : !items.length ? <p>기록된 사전 알람이 없습니다.</p> : <table><thead><tr><th>발생 시각</th><th>Lot / Wafer</th><th>예측 수율</th><th>Critical 확률</th><th>위험도</th><th>신뢰도</th><th>주요 Target</th><th>상태</th><th>외부 전송</th></tr></thead><tbody>{items.map((item) => <tr key={item.alert_id} className={selected?.alert_id === item.alert_id ? "selected" : ""} onClick={() => setSelected(item)}><td>{new Date(item.created_at).toLocaleString("ko-KR")}</td><td>{item.lot_wafer_id}</td><td>{display(item.predicted_y,"%")}</td><td>{display(item.critical_probability === null ? null : item.critical_probability * 100,"%")}</td><td>{item.risk_level === "danger" ? "Critical" : "Warning"}</td><td>{item.confidence ?? "-"}</td><td>{item.top_failure_target ?? "-"}</td><td>{item.status}</td><td>{item.external_delivery_status}</td></tr>)}</tbody></table>}
          </div>
          <aside className="resultCard dashboardDetail">
            {!selected ? <p>알람을 선택하면 상세 정보가 표시됩니다.</p> : <><span className="sectionLabel">Alert Detail</span><h2>{selected.lot_wafer_id}</h2><dl>{Object.entries({"Analysis ID":selected.analysis_id,"Model ID":selected.model_id,"예측 Y":display(selected.predicted_y,"%"),"Top Feature":selected.top_feature ?? "-","Top Step":selected.top_step ?? "-","Config":selected.top_equipment ?? "-","Acknowledged":selected.acknowledged_at ?? "-","Resolved":selected.resolved_at ?? "-"}).map(([label,value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl><div className="dashboardActions"><button className="button secondary" disabled={selected.status !== "New"} onClick={() => void changeStatus("Acknowledged")}>Acknowledge</button><button className="button primary" disabled={selected.status === "Resolved"} onClick={() => void changeStatus("Resolved")}>Resolve</button></div></>}
          </aside>
        </section>
        <p className="metricNotice">현재 로그는 배포 인스턴스의 임시 저장소에 보관되며 재배포 시 초기화될 수 있습니다.</p>
      </main>
    </div>
  </div>;
}
