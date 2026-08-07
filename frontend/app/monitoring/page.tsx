"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import ConfigTreemap from "@/components/ConfigTreemap";
import DashboardShell from "@/components/DashboardShell";
import { LastRunNote } from "@/components/LastRunNote";
import MeasurementExpansionCard from "@/components/MeasurementExpansionCard";
import {
  getLatestSnapshot,
  getMeasurementQueue,
  type LotMeasurementRow,
  type MeasurementQueueData,
  type MonitoringSnapshot,
  type SignificantFactorDetail,
} from "@/lib/monitoringSource";

type ActionItem = { key: string; text: string; href: string; buttonLabel: string; note?: string };

/** SUMMARY의 "지금 할 수 있는 것 / 실험으로 확인할 것 / 확인이 필요한 것"
 * 3분류 (지시서 §4①) -- R/Eq 조정 권고는 인과 검증이 안 됐으므로 절대
 * "지금 할 수 있는 것"에 넣지 않고 "실험으로 확인할 것"에 "단정 아님"과
 * 함께 둔다. 계측 관련만 "지금 할 수 있는 것"에 들어간다. 이미 fetch된
 * snapshot/queue 데이터에서 파생만 할 뿐 별도 조회는 하지 않는다.
 */
function buildActionTriage(
  snapshot: MonitoringSnapshot,
  queue: MeasurementQueueData,
): { doNow: ActionItem[]; experiment: ActionItem[]; needsCheck: ActionItem[] } {
  const doNow: ActionItem[] = queue.lots.slice(0, 2).map((lot) => ({
    key: `lot-${lot.lotId}`,
    text: `${lot.lotId} 집중 계측 (${lot.recommendation})`,
    href: `/alerts?lot=${encodeURIComponent(lot.lotId)}`,
    buttonLabel: "실행",
  }));

  const experiment: ActionItem[] = [];
  for (const f of snapshot.significantFactors) {
    if (!f.feature || f.kind === "Config") continue;
    const direction =
      f.relationShape === "monotonic_increasing" ? "상향" : f.relationShape === "monotonic_decreasing" ? "하향" : null;
    if (!direction) continue;
    experiment.push({
      key: `exp-${f.target}-${f.feature}`,
      text: `${f.feature} ${direction} 스플릿랏 (단정 아님)`,
      href: `/root-cause?target=${encodeURIComponent(f.target)}&feature=${encodeURIComponent(f.feature)}`,
      buttonLabel: "상세",
    });
  }

  const needsCheckMap = new Map<string, ActionItem>();
  for (const f of snapshot.significantFactors) {
    if (f.unknownConfigCount <= 0 || f.step == null || !f.feature) continue;
    const key = `check-Step${f.step}`;
    if (needsCheckMap.has(key)) continue;
    needsCheckMap.set(key, {
      key,
      text: `Step${f.step} 미지 Config ${f.unknownConfigCount}건`,
      href: `/root-cause?target=${encodeURIComponent(f.target)}&feature=${encodeURIComponent(f.feature)}`,
      buttonLabel: "보기",
    });
  }

  return { doNow, experiment: experiment.slice(0, 3), needsCheck: Array.from(needsCheckMap.values()).slice(0, 3) };
}

type YieldStatus = "high" | "medium" | "low";

function classifyYieldStatus(predLo: number, predHi: number, target: number): { status: YieldStatus; label: string } {
  if (predLo >= target) return { status: "high", label: "● 양호" };
  if (predHi >= target) return { status: "medium", label: "◐ 주의" };
  return { status: "low", label: "● 위험" };
}

export default function MonitoringPage() {
  const [snapshot, setSnapshot] = useState<MonitoringSnapshot | null>(null);
  const [queue, setQueue] = useState<MeasurementQueueData>({ yieldSummary: null, lots: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setLoading(true);
      void getLatestSnapshot().then(async (snap) => {
        if (cancelled) return;
        setSnapshot(snap);
        if (!snap.hasAnalysis) {
          setLoading(false);
          return;
        }
        const queueData = await getMeasurementQueue(snap.alarmsRecord);
        if (cancelled) return;
        setQueue(queueData);
        setLoading(false);
      });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  return (
    <DashboardShell activeItem="모니터링">
      <div className="rcPage">
        <div className="pageHeading">
          <h1>모니터링</h1>
          <p>가장 최근 원인 분석 결과를 한눈에 봅니다. 이 화면 자체는 아무것도 실행하지 않습니다.</p>
        </div>

        {loading ? (
          <p className="emptyMessage">불러오는 중…</p>
        ) : !snapshot?.hasAnalysis ? (
          <section className="resultCard">
            <p className="emptyMessage">
              아직 분석 결과가 없습니다. <Link href="/root-cause">원인 분석 탭</Link>에서 분석을 실행하세요.
            </p>
          </section>
        ) : (
          <>
            <SummaryBlock snapshot={snapshot} queue={queue} />
            <section className="resultCard">
              <div className="sectionHeading compact">
                <div>
                  <span className="sectionLabel">MEASUREMENT</span>
                  <h2>추가 계측 권고</h2>
                </div>
              </div>
              <MeasurementExpansionCard data={snapshot.measurementExpansion} />
              <LotQueueTable lots={queue.lots} />
            </section>
            <ConfigTreemap datasetId={snapshot.dataset ?? "train"} />
          </>
        )}
      </div>
    </DashboardShell>
  );
}

function SummaryBlock({ snapshot, queue }: { snapshot: MonitoringSnapshot; queue: MeasurementQueueData }) {
  const targetYield = snapshot.alarmsRecord?.payload.targetYield ?? null;
  const yieldStatus =
    queue.yieldSummary && targetYield != null
      ? classifyYieldStatus(queue.yieldSummary.predLo, queue.yieldSummary.predHi, targetYield)
      : null;
  const gapLo = queue.yieldSummary && targetYield != null ? targetYield - queue.yieldSummary.predHi : null;
  const gapHi = queue.yieldSummary && targetYield != null ? targetYield - queue.yieldSummary.predLo : null;

  const triage = buildActionTriage(snapshot, queue);

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">SUMMARY</span>
          <h2>공정 현황 요약</h2>
        </div>
      </div>

      {!queue.yieldSummary || targetYield == null ? (
        <p className="sectionCaption">예측 없음 — 사전 알람 로그 탭에서 목표 수율을 설정하면 예상 구간이 표시됩니다.</p>
      ) : (
        <p className="sectionCaption">
          예상 수율 {queue.yieldSummary.predMean.toFixed(1)}% [구간 {queue.yieldSummary.predLo.toFixed(1)} – {queue.yieldSummary.predHi.toFixed(1)}]
          {" "}· 목표 {targetYield.toFixed(0)} 대비 갭{" "}
          {gapLo != null && gapHi != null
            ? gapLo <= 0 && gapHi <= 0
              ? "목표 달성"
              : `${Math.max(gapLo, 0).toFixed(1)} – ${Math.max(gapHi, 0).toFixed(1)}%p`
            : "-"}
          {yieldStatus && <strong className={`reliabilityGradeText grade-${yieldStatus.status}`}> {yieldStatus.label}</strong>}
        </p>
      )}
      <p className="sectionCaption">
        <LastRunNote createdAt={snapshot.createdAt} /> · {snapshot.dataset}
      </p>

      <h3 className="monitoringSubheading">유의 인자</h3>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>타깃</th>
              <th>인자</th>
              <th>권장구간 / 기준</th>
              <th className="numCol">이탈 · 계측</th>
              <th>등급</th>
            </tr>
          </thead>
          <tbody>
            {snapshot.significantFactors.map((f) => (
              <SignificantFactorRow key={f.target} factor={f} />
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="monitoringSubheading">지금 할 수 있는 것</h3>
      <ActionList items={triage.doNow} empty="계측 확대가 시급한 랏이 없습니다." />

      <h3 className="monitoringSubheading">실험으로 확인할 것</h3>
      <ActionList items={triage.experiment} empty="실험 후보로 제안할 인자가 없습니다." />

      <h3 className="monitoringSubheading">확인이 필요한 것</h3>
      <ActionList items={triage.needsCheck} empty="확인이 필요한 데이터 이상이 없습니다." />
    </section>
  );
}

function SignificantFactorRow({ factor }: { factor: SignificantFactorDetail }) {
  if (!factor.feature) {
    return (
      <tr>
        <td>{factor.target}</td>
        <td colSpan={4} className="emptyMessage" style={{ padding: "6px 0" }}>유의 인자 없음</td>
      </tr>
    );
  }
  return (
    <tr>
      <td>{factor.target}</td>
      <td>
        <Link href={`/root-cause?target=${encodeURIComponent(factor.target)}&feature=${encodeURIComponent(factor.feature)}`}>
          {factor.feature}
        </Link>
      </td>
      <td>{factor.rangeText ?? "-"}</td>
      <td className="numCol">{factor.deviationText ?? "-"}</td>
      <td>{factor.confidenceTier ? <ConfidenceBadge tier={factor.confidenceTier} /> : "-"}</td>
    </tr>
  );
}

function ActionList({ items, empty }: { items: ActionItem[]; empty: string }) {
  if (items.length === 0) return <p className="emptyMessage">{empty}</p>;
  return (
    <ul className="monitoringActionList">
      {items.map((item) => (
        <li key={item.key}>
          <span>{item.text}</span>
          <Link href={item.href} className="button secondary">{item.buttonLabel}</Link>
        </li>
      ))}
    </ul>
  );
}

function LotQueueTable({ lots }: { lots: LotMeasurementRow[] }) {
  if (lots.length === 0) return null;
  return (
    <div className="tableWrap monitoringLotQueue">
      <table>
        <thead>
          <tr>
            <th>랏</th>
            <th className="numCol">예측 구간</th>
            <th>분산</th>
            <th>사유</th>
            <th>계측 권고</th>
            <th aria-hidden="true" />
          </tr>
        </thead>
        <tbody>
          {lots.map((lot) => (
            <tr key={lot.lotId}>
              <td>{lot.lotId}</td>
              <td className="numCol">{lot.predLo.toFixed(1)} – {lot.predHi.toFixed(1)}%</td>
              <td>{lot.variance}</td>
              <td>{lot.reason ?? "-"}</td>
              <td>{lot.recommendation}</td>
              <td><Link href={`/alerts?lot=${encodeURIComponent(lot.lotId)}`} className="button secondary">배정</Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
