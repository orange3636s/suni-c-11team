"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import ConfigTreemap from "@/components/ConfigTreemap";
import DashboardShell from "@/components/DashboardShell";
import EvidenceBand from "@/components/EvidenceBand";
import FallbackModeBadge from "@/components/FallbackModeBadge";
import FmeaTable from "@/components/FmeaTable";
import { DatasetMismatchWarning, LastRunNote } from "@/components/LastRunNote";
import MeasurementExpansionCard from "@/components/MeasurementExpansionCard";
import RecommendedActions from "@/components/RecommendedActions";
import { classifyMargin } from "@/lib/alertsClassify";
import { ApiResponseError, triggerRefresh } from "@/lib/api";
import {
  buildMonitoringSnapshot,
  getMeasurementQueue,
  type MeasurementQueueData,
  type MonitoringSnapshot,
} from "@/lib/monitoringSource";
import type { ConfigTreemapResponse } from "@/types/data";

type YieldStatus = "high" | "medium" | "low";

// 지시서 K-1: 상태 배지는 갭 구간(목표-예측)의 하한이 0을 넘는지로
// 판정한다 -- 점추정(predMean)으로 판정하지 않는다. gapLo = target -
// predHi(최선의 경우 갭), gapHi = target - predLo(최악의 경우 갭).
// gapLo > 0이면 최선의 경우조차 목표 미달(경보), gapHi <= 0이면 최악의
// 경우도 목표 달성(정상), 그 사이는 불확실. 이 함수가 받는 predLo/predHi는
// (GA그룹) 웨이퍼 conformal 여유가 아니라 집계 여유(interval_conformal_q_agg,
// 약 ±0.2%p)로 낸 구간이다 -- 웨이퍼 여유(±5.5%p 안팎)를 그대로 썼을 때는
// 이 가운데 구간(판정 보류)이 항상 다수였지만, 이는 통계적으로 틀린
// 계산이었다(평균의 불확실성을 개별값 수준으로 과대평가). 여전히 구간이
// 실제로 목표를 걸치는 경우는 존재하며 그때는 "판정 보류"가 맞다 --
// "주의"라고 부르면 조치가 필요한 신호처럼 읽혀 오해를 만든다.
function classifyYieldStatus(predLo: number, predHi: number, target: number): { status: YieldStatus; label: string; icon: string } {
  const gapLo = target - predHi;
  const gapHi = target - predLo;
  if (gapHi <= 0) return { status: "high", label: "정상", icon: "●" };
  if (gapLo > 0) return { status: "low", label: "경보", icon: "●" };
  return { status: "medium", label: "판정 보류", icon: "◐" };
}

// 지시서 K-1: 고정 스케일(80~95%) -- 갱신 때마다 관측값에 맞춰 축을
// 다시 잡으면 막대가 요동쳐 없는 변화를 만들어 보이므로 절대 값에
// 맞추지 않는다.
const YIELD_GAP_SCALE_MIN = 80;
const YIELD_GAP_SCALE_MAX = 95;

export default function MonitoringPage() {
  // 지시서 K-3: 원인 분석·학습 결과가 그대로면(무효화 조건 ①②가 안
  // 일어났으면) 재조회하지 않고 캐시를 그대로 쓴다. 캐시는
  // AnalysisStateProvider에 있어 탭을 옮겼다 돌아와도(페이지 언마운트)
  // 살아남는다 -- 하드 새로고침(조건 ③)만 이 컨텍스트 자체를 초기화한다.
  const { hydrated, analysis, training, alarms, monitoringHome, setMonitoringHome, refreshRunning } = useAnalysisState();
  // AF그룹: 모니터링은 읽기 전용이 원칙이므로 이 버튼은 자동 갱신을
  // 기다리지 않을 때의 보조 수단이다 -- 채워진 primary가 아니라 테두리
  // 버튼으로 둔다. 클릭은 주기 잡과 같은 파이프라인(POST /api/state/refresh
  // -> run_refresh_pipeline)을 1회 실행할 뿐, 이 화면이 직접 무언가를
  // 계산하지 않는다.
  const [manualRefreshPending, setManualRefreshPending] = useState(false);
  const [manualRefreshError, setManualRefreshError] = useState<string | null>(null);
  const refreshBusy = refreshRunning || manualRefreshPending;

  async function handleManualRefresh() {
    setManualRefreshError(null);
    setManualRefreshPending(true);
    try {
      await triggerRefresh();
    } catch (failure) {
      setManualRefreshError(
        failure instanceof ApiResponseError && failure.status === 409
          ? "자동 갱신이 이미 진행 중입니다."
          : "최신화를 시작하지 못했습니다.",
      );
    } finally {
      setManualRefreshPending(false);
    }
  }
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
      void buildMonitoringSnapshot(analysis, alarms)
        .then(async (snap) => {
          if (cancelled) return;
          setSnapshot(snap);
          if (!snap.hasAnalysis) {
            setLoading(false);
            setMonitoringHome({ cacheKey, snapshot: snap, queue: { yieldSummary: null }, treemap: null });
            return;
          }
          const queueData = await getMeasurementQueue(snap.alarmsRecord);
          if (cancelled) return;
          setQueue(queueData);
          setLoading(false);
          setMonitoringHome({ cacheKey, snapshot: snap, queue: queueData, treemap: null });
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
  }, [hydrated, cacheKey, retryToken, analysis, alarms]);

  // E-4: ConfigTreemap이 탭 이동으로 언마운트/리마운트돼도 마지막으로 조회한
  // 스텝 결과를 다시 조회하지 않도록, monitoringHome 캐시에 함께 들고 있는다.
  function handleTreemapDataChange(step: number, data: ConfigTreemapResponse | null) {
    if (!monitoringHome) return;
    setMonitoringHome({ ...monitoringHome, treemap: { step, data } });
  }

  return (
    <DashboardShell activeItem="모니터링">
      <div className="rcPage">
        <div className="pageHeading">
          <div className="monitoringHeadingRow">
            <h1>모니터링</h1>
            {/* AF그룹: LAST RUN이 이미 아래 줄에 표시되므로 버튼 옆에
                다시 적지 않는다. */}
            <button
              type="button"
              className="button secondary monitoringRefreshButton"
              onClick={() => void handleManualRefresh()}
              disabled={refreshBusy}
              title={refreshBusy ? "자동 갱신이 진행 중입니다" : "지금 자동 갱신 파이프라인을 1회 실행합니다"}
            >
              {refreshBusy ? "갱신 중…" : "↻ 최신화"}
            </button>
          </div>
          <p>가장 최근 원인 분석 결과를 한눈에 봅니다.</p>
          {/* 지시서 V: SUMMARY 카드 안에 있던 실행 시각을 페이지 상단으로
              옮겼다 -- 이력이 없으면(snapshot.createdAt이 없으면) 표시하지
              않는다. */}
          {snapshot?.createdAt && (
            <p className="sectionCaption">
              <LastRunNote createdAt={snapshot.createdAt} /> · {snapshot.dataset}
              <FallbackModeBadge />
            </p>
          )}
          {manualRefreshError && <p className="notifyFieldError">{manualRefreshError}</p>}
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
            {/* ID-3 배치: ① 판단 요약 ② FMEA 분석표 ③ 권고 조치
                ④ 추가 계측 권고 ⑤ 설비 구성 트리맵 */}
            <SummaryBlock snapshot={snapshot} queue={queue} />
            <FmeaTable data={snapshot.fmea} error={snapshot.fmeaError} />
            <RecommendedActions data={snapshot.fmea} />
            {/* 지시서 JB-1: MeasurementExpansionCard가 이미 자기 카드
                (resultCard, eyebrow "계측 우선순위" · 제목 "계측 확대
                제안")를 갖고 있는데 여기서 "추가 계측 권고"라는 제목만
                있는 빈 카드로 한 번 더 감싸 이중 카드가 됐었다 -- 바깥
                래퍼만 제거한다(컴포넌트 내부는 그대로, JB-1). */}
            <MeasurementExpansionCard data={snapshot.measurementExpansion} />
            {/* 계측 확대 제안 카드 바로 아래 붙는 두 캡션 -- 하나의 grid
                항목으로 묶어(.rcPage가 grid+gap이라 항목마다 별도 여백이
                생긴다) 카드에서 시각적으로 분리되지 않게 한다.
                MeasurementExpansionCard.tsx 내부는 건드리지 않는다
                (JB-1 "하지 말 것": 바깥에서만 덧붙인다). */}
            <div className="fmeaMeasurementFootnotes">
              {/* 지시서 JB-2: "수율 기여"(이 카드의 표)와 "수율 편차"
                  (FMEA 표)는 서로 다른 질문에 대한 답이라 값이 다른 게
                  정상이다 -- "더 재면 얼마나 얻나" vs "구간에 맞추면
                  얼마나 얻나". 계산을 통일하지 않고 정의만 병기한다. */}
              <p className="fmeaMetricDefinitions">
                수율 기여 — 계측 확대 시 추가로 판정되는 wafer에서 얻는 기대 이득 · 수율 편차 — 권장 구간 안팎의 평균 수율 차이(FMEA 표)
              </p>
              {/* ID-4: FMEA 표의 실익 필터(편차 ≥ 0.3%p)와 정합성을 맞추는
                  한 줄. 계측 부족(선정 인자 전부 미계측) wafer만 계측
                  확대의 실제 대상이고, 상관성 부족(실익 없다고 걸러진
                  인자에서만 근거가 나온) wafer는 계측을 늘려도 해소되지
                  않아 제외했다는 사실만 덧붙인다. */}
              {snapshot.fmea && (
                <p className="fmeaMeasurementAlignmentNote">
                  FMEA 실익 기준 정합: 계측 부족 {snapshot.fmea.measurement_shortage_wafers.toLocaleString()}장이 대상 ·
                  상관성 부족 {snapshot.fmea.correlation_shortage_wafers.toLocaleString()}장은 제외
                </p>
              )}
            </div>
            <ConfigTreemap
              datasetId={snapshot.dataset ?? "train"}
              initialStep={cached?.treemap?.step}
              initialData={cached?.treemap ?? null}
              onDataChange={handleTreemapDataChange}
            />
          </>
        )}
      </div>
    </DashboardShell>
  );
}

function SummaryBlock({ snapshot, queue }: { snapshot: MonitoringSnapshot; queue: MeasurementQueueData }) {
  const targetYield = snapshot.alarmsRecord?.payload.targetYield ?? null;
  const sensitivity = snapshot.alarmsRecord?.payload.sensitivity ?? null;
  // 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-2) -- 근거 밴드에
  // 목표선과 별개로 판정선을 그린다.
  const judgmentLine = targetYield != null && sensitivity != null ? targetYield - classifyMargin(sensitivity) : null;
  const yieldStatus =
    queue.yieldSummary && targetYield != null
      ? classifyYieldStatus(queue.yieldSummary.predLo, queue.yieldSummary.predHi, targetYield)
      : null;
  const gapLo = queue.yieldSummary && targetYield != null ? targetYield - queue.yieldSummary.predHi : null;
  const gapHi = queue.yieldSummary && targetYield != null ? targetYield - queue.yieldSummary.predLo : null;

  // E-4: SUMMARY가 쓰는 alarmsRecord(알림 기록에서 저장한 판정 결과)와
  // 현재 조회 중인 snapshot.dataset이 다르면, 서로 다른 데이터셋의
  // 숫자를 섞어 보여주는 것이다 -- 다른 화면들처럼 경고를 띄운다.
  const datasetMismatch =
    snapshot.dataset != null && snapshot.alarmsRecord != null && snapshot.alarmsRecord.eval_dataset !== snapshot.dataset;

  return (
    <section className="resultCard panel-primary">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">판단 요약</span>
          <h2>공정 현황 요약</h2>
        </div>
      </div>
      <DatasetMismatchWarning
        mismatch={datasetMismatch}
        datasets={{
          left: { label: "원인 분석", value: snapshot.dataset ?? "-" },
          right: { label: "알람 판정", value: snapshot.alarmsRecord?.eval_dataset ?? "-" },
        }}
        actions={[
          { label: "원인 분석 다시 실행", href: "/root-cause" },
          { label: "알림 기록에서 변경", href: "/alerts" },
        ]}
      />

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
          <EvidenceBand
            lo={queue.yieldSummary.predLo}
            hi={queue.yieldSummary.predHi}
            target={targetYield}
            judgmentLine={judgmentLine}
            scaleMin={YIELD_GAP_SCALE_MIN}
            scaleMax={YIELD_GAP_SCALE_MAX}
          />
        </div>
      )}
    </section>
  );
}

