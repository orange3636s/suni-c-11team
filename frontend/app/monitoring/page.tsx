"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import ConfigTreemap from "@/components/ConfigTreemap";
import DashboardShell from "@/components/DashboardShell";
import { LastRunNote } from "@/components/LastRunNote";
import MeasurementExpansionCard from "@/components/MeasurementExpansionCard";
import {
  getLatestSnapshot,
  getMeasurementQueue,
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
): { doNow: ActionItem[]; experiment: ActionItem[]; needsCheck: ActionItem[] } {
  // 지시서 K-5: "지금 할 수 있는 것"은 랏 단위 큐 대신 계측 확대 제안의
  // 인자별 우선순위 목록(유지 아닌 것만)에서 가져온다 -- 랏 집계 로직
  // 자체를 없앴으므로 그 데이터를 다시 만들 수 없다.
  const doNow: ActionItem[] = (snapshot.measurementExpansion?.priorities ?? [])
    .filter((priority) => priority.recommendation !== "유지")
    .slice(0, 2)
    .map((priority) => ({
      key: `priority-${priority.target}-${priority.feature}`,
      text: `${priority.feature} → ${priority.target} 계측 확대 (${priority.recommendation})`,
      href: `/root-cause?target=${encodeURIComponent(priority.target)}&feature=${encodeURIComponent(priority.feature)}`,
      buttonLabel: "상세",
    }));

  const experiment: ActionItem[] = [];
  for (const f of snapshot.significantFactors) {
    if (!f.feature || f.kind === "Config") continue;
    const direction =
      f.relationShape === "monotonic_increasing" ? "상향" : f.relationShape === "monotonic_decreasing" ? "하향" : null;
    if (!direction) continue;
    experiment.push({
      key: `exp-${f.target}-${f.feature}`,
      text: `${f.feature} ${direction} SPLIT LOT (단정 아님)`,
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
      buttonLabel: "상세",
    });
  }

  return { doNow, experiment: experiment.slice(0, 3), needsCheck: Array.from(needsCheckMap.values()).slice(0, 3) };
}

type YieldStatus = "high" | "medium" | "low";

// 지시서 K-1: 상태 배지는 갭 구간(목표-예측)의 하한이 0을 넘는지로
// 판정한다 -- 점추정(predMean)으로 판정하지 않는다. gapLo = target -
// predHi(최선의 경우 갭), gapHi = target - predLo(최악의 경우 갭).
// gapLo > 0이면 최선의 경우조차 목표 미달(경보), gapHi <= 0이면 최악의
// 경우도 목표 달성(정상), 그 사이는 불확실(주의).
function classifyYieldStatus(predLo: number, predHi: number, target: number): { status: YieldStatus; label: string; icon: string } {
  const gapLo = target - predHi;
  const gapHi = target - predLo;
  if (gapHi <= 0) return { status: "high", label: "정상", icon: "●" };
  if (gapLo > 0) return { status: "low", label: "경보", icon: "●" };
  return { status: "medium", label: "주의", icon: "◐" };
}

// 지시서 K-1: 고정 스케일(80~95%) -- 갱신 때마다 관측값에 맞춰 축을
// 다시 잡으면 막대가 요동쳐 없는 변화를 만들어 보이므로 절대 값에
// 맞추지 않는다.
const YIELD_GAP_SCALE_MIN = 80;
const YIELD_GAP_SCALE_MAX = 95;

function YieldGapBar({ predLo, predHi, target }: { predLo: number; predHi: number; target: number }) {
  const span = YIELD_GAP_SCALE_MAX - YIELD_GAP_SCALE_MIN;
  const pctOf = (value: number) => Math.min(100, Math.max(0, ((value - YIELD_GAP_SCALE_MIN) / span) * 100));
  const bandLoPct = pctOf(predLo);
  const bandHiPct = pctOf(predHi);
  const targetPct = pctOf(target);
  const ticks = Array.from(
    new Set([YIELD_GAP_SCALE_MIN, predLo, predHi, target, YIELD_GAP_SCALE_MAX].map((v) => Math.round(v * 10) / 10)),
  ).sort((a, b) => a - b);

  return (
    <div className="yieldGapBar">
      <div className="yieldGapBarTrack">
        <div className="yieldGapBarBand" style={{ left: `${bandLoPct}%`, width: `${Math.max(bandHiPct - bandLoPct, 0.6)}%` }} />
        <div className="yieldGapBarBoundary" style={{ left: `${bandLoPct}%` }} />
        <div className="yieldGapBarBoundary" style={{ left: `${bandHiPct}%` }} />
        <div className="yieldGapBarTarget" style={{ left: `${targetPct}%` }} />
        <span className="yieldGapBarTargetLabel" style={{ left: `${targetPct}%` }}>목표</span>
      </div>
      <div className="yieldGapBarTicks">
        {ticks.map((tick) => (
          <span key={tick} className="yieldGapBarTick" style={{ left: `${pctOf(tick)}%` }}>{tick.toFixed(1)}</span>
        ))}
      </div>
    </div>
  );
}

export default function MonitoringPage() {
  // 지시서 K-3: 원인 분석·학습 결과가 그대로면(무효화 조건 ①②가 안
  // 일어났으면) 재조회하지 않고 캐시를 그대로 쓴다. 캐시는
  // AnalysisStateProvider에 있어 탭을 옮겼다 돌아와도(페이지 언마운트)
  // 살아남는다 -- 하드 새로고침(조건 ③)만 이 컨텍스트 자체를 초기화한다.
  const { hydrated, analysis, training, alarms, monitoringHome, setMonitoringHome } = useAnalysisState();
  // A-9: alarms.createdAt도 캐시 키에 포함해야 한다 -- 알림 이력에서
  // 목표 수율·민감도를 바꾸면(alerts/page.tsx가 alarms.createdAt을
  // 갱신) 이 화면의 SUMMARY 갭·경보 배지도 새 기준으로 다시 계산해야
  // 하는데, 이 키가 빠지면 옛 캐시가 그대로 적중해 버린다.
  const cacheKey = `${analysis?.createdAt ?? ""}|${training?.createdAt ?? ""}|${alarms?.createdAt ?? ""}`;
  const cached = monitoringHome && monitoringHome.cacheKey === cacheKey ? monitoringHome : null;

  const [snapshot, setSnapshot] = useState<MonitoringSnapshot | null>(cached?.snapshot ?? null);
  const [queue, setQueue] = useState<MeasurementQueueData>(cached?.queue ?? { yieldSummary: null });
  const [loading, setLoading] = useState(!cached);
  const [loadError, setLoadError] = useState(false);
  // 재시도 버튼이 이 값을 올려 아래 effect를 다시 돈다 -- cacheKey는
  // 바뀌지 않았으므로(분석 결과가 바뀐 게 아니라 조회만 실패한 것) 이
  // 카운터 없이는 effect 의존성이 그대로라 다시 실행되지 않는다.
  const [retryToken, setRetryToken] = useState(0);

  useEffect(() => {
    if (!hydrated) return;
    if (monitoringHome && monitoringHome.cacheKey === cacheKey) {
      // 캐시 적중 -- API를 다시 부르지 않고 캐시된 결과를 그대로 보여준다.
      setSnapshot(monitoringHome.snapshot);
      setQueue(monitoringHome.queue);
      setLoading(false);
      setLoadError(false);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setLoading(true);
      setLoadError(false);
      void getLatestSnapshot()
        .then(async (snap) => {
          if (cancelled) return;
          setSnapshot(snap);
          if (!snap.hasAnalysis) {
            setLoading(false);
            setMonitoringHome({ cacheKey, snapshot: snap, queue: { yieldSummary: null } });
            return;
          }
          const queueData = await getMeasurementQueue(snap.alarmsRecord);
          if (cancelled) return;
          setQueue(queueData);
          setLoading(false);
          setMonitoringHome({ cacheKey, snapshot: snap, queue: queueData });
        })
        .catch(() => {
          // A-9: 서버가 잠깐 죽으면 loading이 영구 true로 남아 무한
          // 로딩처럼 보였다 -- 실패도 명시적인 상태로 남기고 재시도
          // 버튼을 보여준다.
          if (cancelled) return;
          setLoading(false);
          setLoadError(true);
        });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, cacheKey, retryToken]);

  return (
    <DashboardShell activeItem="모니터링">
      <div className="rcPage">
        <div className="pageHeading">
          <h1>모니터링</h1>
          <p>가장 최근 원인 분석 결과를 한눈에 봅니다.</p>
          {/* 지시서 V: SUMMARY 카드 안에 있던 실행 시각을 페이지 상단으로
              옮겼다 -- 이력이 없으면(snapshot.createdAt이 없으면) 표시하지
              않는다. */}
          {snapshot?.createdAt && (
            <p className="sectionCaption">
              <LastRunNote createdAt={snapshot.createdAt} /> · {snapshot.dataset}
            </p>
          )}
        </div>

        {loading ? (
          <p className="emptyMessage">불러오는 중…</p>
        ) : loadError ? (
          <section className="resultCard">
            <div className="analysisErrorBox" role="alert">
              <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
              <div className="analysisErrorBody">
                <p className="analysisErrorMessage">모니터링 데이터를 불러오지 못했습니다. 서버 상태를 확인해 주세요.</p>
              </div>
              <button type="button" className="button" onClick={() => setRetryToken((token) => token + 1)}>
                다시 시도
              </button>
            </div>
          </section>
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

  const triage = buildActionTriage(snapshot);

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">SUMMARY</span>
          <h2>공정 현황 요약</h2>
        </div>
      </div>

      {!queue.yieldSummary || targetYield == null ? (
        <p className="sectionCaption">예측 없음 — 알림 기록 탭에서 목표 수율을 설정하면 예상 구간이 표시됩니다.</p>
      ) : (
        <div className="yieldGapSection">
          <div className="yieldGapHeaderRow">
            <span className="yieldGapMean">예상 수율 {queue.yieldSummary.predMean.toFixed(1)}%</span>
            <span className="yieldGapText">
              갭{" "}
              {gapLo != null && gapHi != null
                ? gapLo <= 0 && gapHi <= 0
                  ? "목표 달성"
                  : `${Math.max(gapLo, 0).toFixed(1)} – ${Math.max(gapHi, 0).toFixed(1)}%p`
                : "-"}
            </span>
            {yieldStatus && (
              <strong className={`reliabilityGradeText grade-${yieldStatus.status}`}>{yieldStatus.icon} {yieldStatus.label}</strong>
            )}
          </div>
          <YieldGapBar predLo={queue.yieldSummary.predLo} predHi={queue.yieldSummary.predHi} target={targetYield} />
        </div>
      )}

      <h3 className="monitoringSubheading">유의 인자</h3>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>타깃</th>
              <th>인자</th>
              <th>권장구간 / 기준</th>
              <th className="numCol">이탈 · 계측</th>
              <th>상관성</th>
            </tr>
          </thead>
          <tbody>
            {snapshot.significantFactors.map((f) => (
              <SignificantFactorRow key={f.target} factor={f} />
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="monitoringSubheading">실행 과제</h3>
      <ActionList items={triage.doNow} empty="계측 확대가 시급한 랏이 없습니다." />

      <h3 className="monitoringSubheading">실험 확인 대상</h3>
      <ActionList items={triage.experiment} empty="실험 후보로 제안할 인자가 없습니다." />

      <h3 className="monitoringSubheading">확인 필요 대상</h3>
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

// 지시서 K-4/P-2/CC: 세 레일(실행 과제/실험 확인 대상/확인 필요 대상)의
// 버튼 크기를 하나로 통일한다 -- 새 버튼 스타일을 만들지 않고 다른
// 화면과 공유하는 `.button.sm`을 쓴다. 셋 다 흰 배경(secondary)이고
// 라벨도 모두 "상세"로 통일했다 -- 강조색(.primary)은 쓰지 않는다
// (지시서 P-2: 세 레일 버튼이 같은 모양이어야 한다).
function ActionList({ items, empty }: { items: ActionItem[]; empty: string }) {
  if (items.length === 0) return <p className="emptyMessage">{empty}</p>;
  return (
    <ul className="monitoringActionList">
      {items.map((item) => (
        <li key={item.key}>
          <span>{item.text}</span>
          <Link href={item.href} className="button sm secondary">
            {item.buttonLabel}
          </Link>
        </li>
      ))}
    </ul>
  );
}

