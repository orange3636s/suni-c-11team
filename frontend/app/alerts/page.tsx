"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import { getAlarmSummary, getAlarms } from "@/lib/api";
import type { AlarmItem, AlarmListResponse, AlarmSummaryResponse } from "@/types/data";

const SEVERITY_LABEL: Record<string, string> = { low: "낮음", medium: "중간", high: "높음" };

function SeverityBadge({ severity }: { severity: string }) {
  return <span className={`severityBadge severityBadge-${severity}`}>{SEVERITY_LABEL[severity] ?? severity}</span>;
}

export default function AlertsPage() {
  const [trainDataset, setTrainDataset] = useState("train");
  const [evalDataset, setEvalDataset] = useState("test");
  const [summary, setSummary] = useState<AlarmSummaryResponse | null>(null);
  const [alarms, setAlarms] = useState<AlarmListResponse | null>(null);
  const [severityFilter, setSeverityFilter] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [summaryResponse, alarmsResponse] = await Promise.all([
        getAlarmSummary(trainDataset, evalDataset),
        getAlarms(trainDataset, evalDataset, severityFilter || undefined),
      ]);
      setSummary(summaryResponse);
      setAlarms(alarmsResponse);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "알람 로그를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [trainDataset, evalDataset, severityFilter]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const yieldGap = summary?.yield_gap;

  return (
    <DashboardShell activeItem="사전 알람 로그">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">PRE-ALERT LOG</span>
        <h1>사전 알람 로그</h1>
        <p>학습 데이터셋에서 산출한 정상범위를 평가 데이터셋에 적용해 이탈 여부를 판정합니다.</p>
      </section>

      <section className="uploadCard">
        <div className="rcControlBar" style={{ gridTemplateColumns: "minmax(200px,1fr) minmax(200px,1fr) minmax(140px,.6fr)" }}>
          <DatasetSelector label="정상범위 산출 (train)" value={trainDataset} onChange={setTrainDataset} />
          <DatasetSelector label="판정 대상 (eval)" value={evalDataset} onChange={setEvalDataset} />
          <div className="fieldGroup">
            <span>Severity</span>
            <select value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value)}>
              <option value="">전체</option>
              <option value="low">낮음</option>
              <option value="medium">중간</option>
              <option value="high">높음</option>
            </select>
          </div>
        </div>
        {error && <p className="errorMessage">{error}</p>}
      </section>

      <section className="secomKpiGrid">
        <div><span>알람 wafer</span><strong>{summary?.counts.alarm ?? "-"}</strong></div>
        <div><span>정상</span><strong>{summary?.counts.normal ?? "-"}</strong></div>
        <div><span>판정불가 (미계측)</span><strong>{summary?.counts.unmeasured ?? "-"}</strong></div>
        <div><span>알람군 평균수율</span><strong>{summary?.alarm_group_yield_avg?.toFixed(2) ?? "-"}</strong></div>
        <div><span>무알람군 평균수율</span><strong>{summary?.no_alarm_group_yield_avg?.toFixed(2) ?? "-"}</strong></div>
        <div><span>격차</span><strong>{yieldGap != null ? `${yieldGap.toFixed(2)}%p` : "-"}</strong></div>
      </section>

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">ALARMS</span>
            <h2>알람 목록 ({alarms?.total ?? 0}건)</h2>
          </div>
        </div>
        {loading && <p className="emptyMessage">불러오는 중…</p>}
        {!loading && alarms && alarms.items.length === 0 && (
          <p className="emptyMessage">조건에 맞는 알람이 없습니다.</p>
        )}
        {!loading && alarms && alarms.items.length > 0 && (
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Wafer</th><th>LOT</th><th>Step</th><th>인자</th><th>타깃</th><th>값</th><th>정상범위</th><th>이탈량</th><th>방향</th><th>심각도</th><th>실측값</th>
                </tr>
              </thead>
              <tbody>
                {alarms.items.map((item, index) => (
                  <AlarmRow key={`${item.lot_wafer_id}-${item.feature}-${index}`} item={item} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">LOT</span>
            <h2>LOT별 알람 집계 (상위)</h2>
          </div>
        </div>
        {summary && summary.top_lots.length > 0 ? (
          <div className="tableWrap">
            <table>
              <thead><tr><th>LOT</th><th>알람 건수</th></tr></thead>
              <tbody>
                {summary.top_lots.slice(0, 10).map((lot) => (
                  <tr key={lot.lot_id}><td>{lot.lot_id}</td><td>{lot.alarm_count}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="emptyMessage">알람이 발생한 LOT이 없습니다.</p>
        )}
      </section>

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>정상범위는 학습 데이터셋에서 해당 인자 자신의 분포로 산출한 IQR×1.5 관리한계(LCL~UCL)이며, 타깃(Y) 값과는 무관하게 계산됩니다. 인과관계가 아닌 통계적 이탈 판정입니다.</li>
          <li>판정불가 wafer는 선정 인자가 계측되지 않아 판정 자체가 불가능한 것이며, 정상을 의미하지 않습니다.</li>
        </ul>
      </section>
    </DashboardShell>
  );
}

function AlarmRow({ item }: { item: AlarmItem }) {
  const [lo, hi] = item.normal_range;
  const rangeText = `${lo != null ? lo.toFixed(1) : "-∞"} ~ ${hi != null ? hi.toFixed(1) : "+∞"}`;
  const rootCauseHref = `/root-cause?target=${encodeURIComponent(item.target)}&kind=all&feature=${encodeURIComponent(item.feature)}`;
  return (
    <tr>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.lot_wafer_id}</Link>
      </td>
      <td>{item.lot_id ?? "-"}</td>
      <td>{item.step}</td>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.feature}</Link>
      </td>
      <td>{item.target}</td>
      <td>{item.value.toFixed(2)}</td>
      <td title={`train에서 ${item.feature} 자체 분포의 IQR×1.5 관리한계 (Y와 무관)`}>{rangeText}</td>
      <td>{item.deviation.toFixed(2)}</td>
      <td>{item.direction === "above" ? "높음" : "낮음"}</td>
      <td><SeverityBadge severity={item.severity} /></td>
      <td>{item.actual_y != null ? item.actual_y.toFixed(2) : "-"}</td>
    </tr>
  );
}
