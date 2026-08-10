"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import ActionBlock from "@/components/ActionBlock";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import DashboardShell from "@/components/DashboardShell";
import DataLimitationDiagnostics from "@/components/DataLimitationDiagnostics";
import FallbackModeBadge from "@/components/FallbackModeBadge";
import FmeaTable from "@/components/FmeaTable";
import { DatasetMismatchWarning, LastRunNote, TrainingAnalysisDataNote } from "@/components/LastRunNote";
import ModeLossBars from "@/components/ModeLossBars";
import SummaryCards from "@/components/SummaryCards";
import YieldHistogram from "@/components/YieldHistogram";
import { ApiResponseError, triggerRefresh } from "@/lib/api";
import {
  buildMonitoringSnapshot,
  getYieldSummary,
  type MonitoringSnapshot,
} from "@/lib/monitoringSource";

export default function MonitoringPage() {
  // 지시서 K-3: 원인 분석·학습 결과가 그대로면(무효화 조건 ①②가 안
  // 일어났으면) 재조회하지 않고 캐시를 그대로 쓴다. 캐시는
  // AnalysisStateProvider에 있어 탭을 옮겼다 돌아와도(페이지 언마운트)
  // 살아남는다 -- 하드 새로고침(조건 ③)만 이 컨텍스트 자체를 초기화한다.
  const {
    hydrated, analysis, training, alarms, monitoringHome, setMonitoringHome, refreshRunning,
    // TD-2: 컨텍스트의 자동 갱신 스냅샷 -- 상단 "분석 데이터" 파일명
    // 표기에만 쓴다.
    snapshot: refreshSnapshot,
  } = useAnalysisState();
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
  // 갱신) 이 화면의 요약도 새 기준으로 다시 계산해야 하는데, 이 키가
  // 빠지면 옛 캐시가 그대로 적중해 버린다.
  const cacheKey = `${analysis?.createdAt ?? ""}|${training?.createdAt ?? ""}|${alarms?.createdAt ?? ""}`;
  const cached = monitoringHome && monitoringHome.cacheKey === cacheKey ? monitoringHome : null;

  const [snapshot, setSnapshot] = useState<MonitoringSnapshot | null>(cached?.snapshot ?? null);
  const [yieldSummary, setYieldSummary] = useState(cached?.yieldSummary ?? null);
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
      setYieldSummary(monitoringHome.yieldSummary);
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
            setMonitoringHome({ cacheKey, snapshot: snap, yieldSummary: null });
            return;
          }
          const summary = await getYieldSummary(snap.alarmsRecord);
          if (cancelled) return;
          setYieldSummary(summary);
          setLoading(false);
          setMonitoringHome({ cacheKey, snapshot: snap, yieldSummary: summary });
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

  // WG: "데이터셋 불일치 경고"는 유지한다 -- SUMMARY가 쓰는
  // alarmsRecord(수율 예측에서 저장한 판정 결과)와 현재 조회 중인
  // snapshot.dataset이 다르면, 서로 다른 데이터셋의 숫자를 섞어 보여주는
  // 것이다. 두 값은 서로 다른 탭(원인 분석/수율 예측)이 독립적으로
  // 저장하는 상태라, 한쪽만 바꾸고 재실행하지 않으면 실제로 어긋날 수
  // 있다(스냅샷 원자성 문제가 아니라 사용자가 다른 데이터셋을 고른
  // 것이다).
  const datasetMismatch =
    !!snapshot?.dataset && !!snapshot.alarmsRecord && snapshot.alarmsRecord.eval_dataset !== snapshot.dataset;

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
          {snapshot?.createdAt && (
            <p className="sectionCaption">
              <LastRunNote createdAt={snapshot.createdAt} /> · {snapshot.dataset}
              <TrainingAnalysisDataNote
                trainFilename={training?.performance?.source_filename ?? null}
                evalFilename={refreshSnapshot?.source?.eval_dataset_filename ?? null}
              />
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
            {(snapshot.fmea?.target_provenance?.uses_predictions || snapshot.measurementExpansion?.target_provenance?.uses_predictions) && (() => {
              const provenance = snapshot.fmea?.target_provenance ?? snapshot.measurementExpansion?.target_provenance;
              return provenance ? (
                <p className="analysisDataNotice" role="note">
                  이 분석은 실측값이 없는 항목을 모델 예측값으로 보완해 계산했습니다. 예측값 기반 관계는 실제 공정 원인과 다를 수 있으므로 공정 검증과 함께 사용해 주세요. 모델 {provenance.model_version ?? provenance.model_id ?? "정보 없음"} · 예측 {provenance.predicted_target_cells.toLocaleString()}셀
                </p>
              ) : null;
            })()}

            <DatasetMismatchWarning
              mismatch={datasetMismatch}
              datasets={{
                left: { label: "원인 분석", value: snapshot.dataset ?? "-" },
                right: { label: "알람 판정", value: snapshot.alarmsRecord?.eval_dataset ?? "-" },
              }}
              actions={[
                { label: "원인 분석 다시 실행", href: "/root-cause" },
                { label: "수율 예측에서 변경", href: "/alerts" },
              ]}
            />

            {/* 작업 지시서 최종 화면 구조: ① 상단 요약 카드 4개 ② 모드별
                손실 막대 ③ 수율 분포 히스토그램 ④ FMEA 분석표 ⑤ 조치
                블록 ⑥ 데이터 한계 진단. 트리맵은 별도 탭(WH)으로 뺐다. */}
            <SummaryCards summary={yieldSummary} />
            <ModeLossBars summary={yieldSummary} />
            <YieldHistogram summary={yieldSummary} />
            <FmeaTable data={snapshot.fmea} error={snapshot.fmeaError} />
            <ActionBlock fmea={snapshot.fmea} measurementExpansion={snapshot.measurementExpansion} />
            <DataLimitationDiagnostics fmea={snapshot.fmea} />
          </>
        )}
      </div>
    </DashboardShell>
  );
}
