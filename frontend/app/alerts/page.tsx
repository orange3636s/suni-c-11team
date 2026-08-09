"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { DEFAULT_SENSITIVITY, DEFAULT_TARGET_YIELD, useAnalysisState } from "@/components/AnalysisStateProvider";
import DashboardShell from "@/components/DashboardShell";
import EvidenceBand from "@/components/EvidenceBand";
import FallbackModeBadge from "@/components/FallbackModeBadge";
import DatasetSelector from "@/components/DatasetSelector";
import {
  HScrollTableBody,
  NoSearchResults,
  TableCaption,
  TableToolbar,
  useTableSearchSort,
  type SortOption,
} from "@/components/DataTablePanel";
import { DatasetMismatchWarning, LastRunNote } from "@/components/LastRunNote";
import { usePanelState } from "@/components/PanelStateProvider";
import {
  type ClassifiedWafer,
  type ClassKey,
  type PrecisionRecallEstimate,
  CLASS_KEYS,
  CLASS_LABELS,
  classifyAll,
  classifyMargin,
  downloadAlarmsCsv,
  estimatePrecisionRecall,
  reasonFor,
  summarizeClasses,
  targetYieldMismatch,
} from "@/lib/alertsClassify";
import { activateDataset, getAlertsData, getDatasets, getReliability, saveAlarmsState } from "@/lib/api";
import { ALARM_GRADE_COLOR } from "@/lib/constants";
import type { DatasetSummary, ReliabilityResponse } from "@/types/data";

const RELIABILITY_GRADE_CLASS: Record<string, string> = { 높음: "high", 보통: "medium", 낮음: "low" };

// 알람 목록은 전체를 렌더하고 (지시서 AF: 7건 이후를 볼 방법이 없던 문제를
// 고친다) 표 영역 자체가 세로로 스크롤된다 -- 이 상수는 스크롤 없이 보이는
// 초기 행 수(HScrollTableBody의 --scroll-rows)일 뿐, 더 이상 렌더링 개수를
// 제한하지 않는다. CSV 다운로드는 항상 alarmTable.sorted 전체를 쓴다.
const ALARM_VISIBLE_ROWS = 7;

// 보정 지시서 §I-1: 근거 밴드(EvidenceBand)는 행마다 자동으로 스케일을
// 맞추면 행 간 비교가 안 되므로 알람 목록 전체에서 같은 축을 쓴다 --
// 관측값에 맞춰 조정하지 않는다.
const ALARM_YIELD_SCALE_MIN = 60;
const ALARM_YIELD_SCALE_MAX = 100;

/** spec §E-3 헤더 배지 -- 클릭하면 상세 패널이 열린다. */
function ReliabilityBadge({
  reliability,
  open,
  onToggle,
}: {
  reliability: ReliabilityResponse;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      className={`reliabilityBadge reliabilityBadge-${RELIABILITY_GRADE_CLASS[reliability.grade] ?? "medium"}`}
      onClick={onToggle}
      aria-expanded={open}
    >
      분석 신뢰도 {reliability.grade}
    </button>
  );
}

const AUC_TOOLTIP = "알람 순위가 실제 위험 순서와 얼마나 맞는지. 0.5는 무작위, 1.0은 완벽";

/** spec §E-3 상세 패널: 5개 지표 배점 내역 + 감점 사유(코드 생성) + 경험값 고지. */
function ReliabilityPanel({ reliability: r }: { reliability: ReliabilityResponse }) {
  return (
    <div className="reliabilityPanel">
      <div className="reliabilityPanelHeader">
        <span>분석 신뢰도</span>
        <strong className={`reliabilityGradeText grade-${RELIABILITY_GRADE_CLASS[r.grade] ?? "medium"}`}>
          {r.grade} {r.total_score}점
        </strong>
      </div>
      <table className="reliabilityTable">
        <tbody>
          <tr>
            <td title={AUC_TOOLTIP}>알람 순위 품질</td>
            <td className="numCol">{r.auc_lower_bound != null ? `AUC ${r.auc_lower_bound.toFixed(3)} (하한)` : "산출 불가"}</td>
            <td className="numCol">{r.auc_score} / 40</td>
          </tr>
          {r.auc_gate_message && (
            <tr className="reliabilityGateRow">
              <td colSpan={3}>{r.auc_gate_message}</td>
            </tr>
          )}
          <tr>
            <td>유의 인자 수</td>
            <td className="numCol">{r.n_significant_factors}개</td>
            <td className="numCol">{r.n_significant_score} / 25</td>
          </tr>
          <tr>
            <td>최대 설명력</td>
            <td className="numCol">{r.max_eps2 != null ? r.max_eps2.toFixed(3) : "-"}</td>
            <td className="numCol">{r.max_eps2_score} / 20</td>
          </tr>
          <tr>
            <td>표본 크기</td>
            <td className="numCol">{r.n_train.toLocaleString()}행</td>
            <td className="numCol">{r.n_train_score} / 10</td>
          </tr>
          <tr>
            <td>판정 커버리지</td>
            <td className="numCol">{r.coverage_pct != null ? `${r.coverage_pct.toFixed(1)}%` : "-"}</td>
            <td className="numCol">{r.coverage_score} / 5</td>
          </tr>
        </tbody>
      </table>
      {r.deduction_reasons.length > 0 && (
        <p className="reliabilityDeductions">감점 사유: {r.deduction_reasons.join(" ")}</p>
      )}
      {r.low_holdout_sample && (
        <p className="reliabilityDeductions">
          평가 표본이 부족해(하위 5% 표본 적음) 성능 지표의 불확실성이 큽니다.
        </p>
      )}
      <p className="reliabilityDisclaimer">{r.thresholds_disclaimer}</p>
    </div>
  );
}

function lotCompare(a: string | null, b: string | null): number {
  return (a ?? "").localeCompare(b ?? "", undefined, { numeric: true });
}

// -- 실시간 재계산용 디바운스 (spec §A-3: "10,000장 이상이면 디바운스
// 100ms를 건다. 그 미만은 필요 없다.") -- 슬라이더 자체는 항상 즉시
// 움직이고, 재분류에 쓰는 값만 큰 데이터셋에서 지연시킨다.
function useDebouncedNumber(value: number, delayMs: number): number {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    if (delayMs <= 0) {
      // 즉시 반영 -- 아래 return이 `value`를 바로 쓴다. `debounced`도 배경에서
      // 맞춰 둬야, 나중에 delayMs가 0보다 커질 때(데이터셋이 바뀌어 wafer 수가
      // 임계를 넘는 경우) 낡은 값으로 잠깐 되돌아가지 않는다.
      const id = requestAnimationFrame(() => setDebounced(value));
      return () => cancelAnimationFrame(id);
    }
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return delayMs <= 0 ? value : debounced;
}

const LARGE_DATASET_DEBOUNCE_THRESHOLD = 10_000;
const DEBOUNCE_MS = 100;

// 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4 프리셋 재정의) --
// 오탐이 실제로 나타나기 시작하는 지점(0.6)을 "균형"으로 삼는다. 세
// 값은 holdout 실측 표(target=85, test.CSV)와 대조해 정한 것이라 다른
// 값으로 바꾸려면 그 표를 다시 뽑아 확인해야 한다.
type PresetKey = "low_fp" | "balanced" | "low_fn";
const SENSITIVITY_PRESETS: Array<{ key: PresetKey; label: string; value: number }> = [
  { key: "low_fp", label: "오경보 최소", value: 0.2 },
  { key: "balanced", label: "균형", value: 0.6 },
  { key: "low_fn", label: "미탐 최소", value: 1.0 },
];

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export default function AlertsPage() {
  return (
    <Suspense fallback={null}>
      <AlertsContent />
    </Suspense>
  );
}

function AlertsContent() {
  const searchParams = useSearchParams();
  const { analysisDataset, requestChat, setTrainingPanelOpen } = usePanelState();
  // AG-1/AG-2: 새 파일을 업로드하면 활성 평가 데이터셋으로 전환하고
  // 스냅샷 파이프라인을 1회 실행한다 -- 화면별 개별 재분석은 걸지 않는다.
  const [showTrainingSuggestion, setShowTrainingSuggestion] = useState(false);
  const [activateError, setActivateError] = useState("");
  function handleDatasetUploaded(uploadedId: string, hasTargetColumns: boolean) {
    setActivateError("");
    void activateDataset(uploadedId)
      .then(() => refreshSnapshotNow())
      .catch((failure) => {
        setActivateError(failure instanceof Error ? failure.message : "활성 평가 데이터셋 전환에 실패했습니다.");
      });
    setShowTrainingSuggestion(hasTargetColumns);
  }
  const { alarms: alarmsState, setAlarms: setAlarmsState, training, hydrated, refreshSnapshotNow } = useAnalysisState();

  // 지시서 O-1: train 셀렉터를 없애고 최근 학습 모델의 데이터셋을 자동으로
  // 따른다 -- 모델 학습 팝업이 저장한 source_filename을 데이터셋 목록의
  // original_filename과 매칭한다. 매칭되는 게 없으면(학습에 쓴 파일이
  // 데이터셋으로 등록되지 않았거나 아직 학습 기록이 없으면) 기존
  // 기본값(train)으로 폴백한다.
  const [datasetList, setDatasetList] = useState<DatasetSummary[]>([]);
  const [datasetsLoaded, setDatasetsLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    getDatasets()
      .then((response) => {
        if (!cancelled) setDatasetList(response.items);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setDatasetsLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);
  const trainDataset = useMemo(() => {
    const sourceFilename = training?.performance.source_filename;
    if (sourceFilename) {
      const match = datasetList.find((item) => item.original_filename === sourceFilename);
      if (match) return match.dataset_id;
    }
    return "train";
  }, [training, datasetList]);
  const trainDatasetLabel =
    datasetList.find((item) => item.dataset_id === trainDataset)?.original_filename
    ?? training?.performance.source_filename
    ?? "train.CSV";

  const [evalDataset, setEvalDataset] = useState("test");
  const [targetYield, setTargetYield] = useState(DEFAULT_TARGET_YIELD);
  const [sensitivity, setSensitivity] = useState(DEFAULT_SENSITIVITY);
  // AA-4: DEFAULT_SENSITIVITY(0.2)가 "오경보 최소" 프리셋과 같은 값이라
  // 첫 로딩 시 그 프리셋이 선택된 상태로 보여야 한다 -- 기본값과
  // 어느 프리셋도 활성이 아닌 상태로 어긋나면 안 된다.
  const [activePreset, setActivePreset] = useState<PresetKey | null>("low_fp");
  const [gradeFilter, setGradeFilter] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [reliability, setReliability] = useState<ReliabilityResponse | null>(null);
  const [reliabilityPanelOpen, setReliabilityPanelOpen] = useState(false);
  // DG그룹: 목표 수율·민감도의 서버 저장이 실패해도 조용히 넘어가지
  // 않는다 -- 발송 판정이 화면과 다른 기준을 쓰게 될 수 있다.
  const [settingsSaveError, setSettingsSaveError] = useState(false);
  const data = alarmsState?.data ?? null;
  const datasetMismatch = Boolean(
    alarmsState && (alarmsState.trainDataset !== trainDataset || alarmsState.evalDataset !== evalDataset),
  );

  // eval을 인자로도 받는다 -- 재접속 직후 복원된 데이터셋으로 바로 조회할
  // 때, 방금 호출한 setEvalDataset은 같은 틱 안의 클로저에 아직 반영되지
  // 않으므로 state를 읽는 대신 명시적으로 넘긴다.
  const load = useCallback(
    async (overrideEval?: string) => {
      const evalDs = overrideEval ?? evalDataset;
      setLoading(true);
      setError("");
      try {
        const [dataResponse, reliabilityResponse] = await Promise.all([
          getAlertsData(trainDataset, evalDs),
          getReliability(trainDataset, evalDs).catch(() => null),
        ]);
        setAlarmsState((previous) => ({
          trainDataset,
          evalDataset: evalDs,
          createdAt: new Date().toISOString(),
          targetYield: previous?.targetYield ?? targetYield,
          sensitivity: previous?.sensitivity ?? sensitivity,
          data: dataResponse,
        }));
        setReliability(reliabilityResponse);
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : "알림 기록을 불러오지 못했습니다.");
      } finally {
        setLoading(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [trainDataset, evalDataset, setAlarmsState],
  );

  // 재접속/새로고침 + 첫 방문을 한 이펙트가 함께 처리한다 -- predictions/
  // holdout은 wafer 수만큼 커질 수 있어 서버에 저장하지 않으므로(spec:
  // alarmGradeByWaferId와 같은 원칙), 복원된 설정(있다면)을 반영한 뒤 항상
  // 새로 불러온다. trainDataset은 datasetsLoaded가 끝나야 정확히
  // 해석되므로 그것도 함께 기다린다.
  const initializedFromHydration = useRef(false);
  useEffect(() => {
    if (!hydrated || !datasetsLoaded || initializedFromHydration.current) return;
    initializedFromHydration.current = true;
    const timer = window.setTimeout(() => {
      if (alarmsState) {
        setEvalDataset(alarmsState.evalDataset);
        setTargetYield(alarmsState.targetYield);
        setSensitivity(alarmsState.sensitivity);
        setActivePreset(null);
        void load(alarmsState.evalDataset);
      } else {
        void load();
      }
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, datasetsLoaded]);

  // 목표 수율·민감도는 가볍게 서버에 저장해 둔다 (spec: 재접속 시 마지막
  // 설정을 복원) -- 슬라이더를 계속 움직이는 동안 매번 POST하지 않도록
  // 800ms 묶어서 보낸다. 재분류 자체는 이 저장과 무관하게 즉시 일어난다.
  useEffect(() => {
    if (!alarmsState) return;
    const timer = window.setTimeout(() => {
      setSettingsSaveError(false);
      void saveAlarmsState(alarmsState.trainDataset, alarmsState.evalDataset, { targetYield, sensitivity }).catch(() => {
        // DG그룹: 저장 실패를 화면에 드러낸다 -- 조용히 넘어가면 사용자는
        // 목표·민감도가 다음 접속에서도 유지되는 줄 알지만, 실제로는
        // 기본값으로 되돌아가고 자동 발송 판정 기준도 화면과 어긋난다.
        setSettingsSaveError(true);
      });
    }, 800);
    return () => window.clearTimeout(timer);
  }, [targetYield, sensitivity, alarmsState]);

  const debounceMs = (data?.total_wafers ?? 0) >= LARGE_DATASET_DEBOUNCE_THRESHOLD ? DEBOUNCE_MS : 0;
  const targetYieldForClassify = useDebouncedNumber(targetYield, debounceMs);
  const sensitivityForClassify = useDebouncedNumber(sensitivity, debounceMs);

  // 사전 알람 로그 전면 개편의 핵심 -- API를 다시 부르지 않고, 이미 받아둔
  // 원시 예측치를 목표 수율·민감도로 즉시 재분류한다 (spec §A-3). 민감도
  // 슬라이더를 실제 트레이드오프로 (spec §CA-1) -- 판정은 점추정
  // (pred_mean) 기준이라 sigma를 더 이상 넘기지 않는다.
  const classified = useMemo<ClassifiedWafer[]>(() => {
    if (!data) return [];
    return classifyAll(data.predictions, {
      target: targetYieldForClassify,
      sensitivity: sensitivityForClassify,
      gatePassed: data.auc_gate_passed,
    });
  }, [data, targetYieldForClassify, sensitivityForClassify]);

  const classSummary = useMemo(
    () => summarizeClasses(classified, data?.total_wafers ?? 0),
    [classified, data],
  );

  const alarmItems = useMemo(
    () => classified.filter((item): item is ClassifiedWafer & { grade: "심각" | "위험" | "주의" } =>
      item.grade === "심각" || item.grade === "위험" || item.grade === "주의"),
    [classified],
  );
  const gradeFilteredAlarmItems = useMemo(
    () => (gradeFilter ? alarmItems.filter((item) => item.grade === gradeFilter) : alarmItems),
    [alarmItems, gradeFilter],
  );

  // 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4) -- 슬라이더 옆
  // 정밀도·재현율 표시 전용 디바운스(150ms, 지시서 고정값). 데이터셋
  // 크기와 무관하게 항상 적용한다 -- 매 프레임 홀드아웃 1,000쌍을 다시
  // 훑는 건 아니지만, 드래그마다 갱신하면 숫자가 너무 빨리 바뀌어
  // 읽기 어렵다.
  const targetYieldForEstimate = useDebouncedNumber(targetYield, 150);
  const sensitivityForEstimate = useDebouncedNumber(sensitivity, 150);
  const precisionRecallEstimate = useMemo<PrecisionRecallEstimate | null>(() => {
    if (!data) return null;
    return estimatePrecisionRecall(data.holdout_oof_actual, data.holdout_oof_predicted, {
      target: targetYieldForEstimate,
      sensitivity: sensitivityForEstimate,
      evalAlarmCount: alarmItems.length,
    });
  }, [data, targetYieldForEstimate, sensitivityForEstimate, alarmItems.length]);

  const alarmTable = useTableSearchSort(
    gradeFilteredAlarmItems,
    (item) => `${item.lot_wafer_id} ${reasonFor(item)}`,
    ALARM_SORT_OPTIONS,
    "default",
    alarmTieBreak,
  );

  // 모니터링 홈의 랏 딥링크 (`/alerts?lot=L412`) -- lot_wafer_id가 lot_id로
  // 시작하므로 검색창에 그대로 채우면 그 랏의 wafer만 걸러진다. 한 번만
  // 반영하고, 그 뒤로는 사용자가 검색창을 직접 조작하게 둔다.
  const lotDeepLinkHandled = useRef(false);
  useEffect(() => {
    if (lotDeepLinkHandled.current) return;
    const lot = searchParams.get("lot");
    if (!lot) return;
    lotDeepLinkHandled.current = true;
    alarmTable.setSearch(lot);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  function handleTargetYieldChange(next: number) {
    setTargetYield(clamp(next, 0, 100));
  }
  function handleSensitivityChange(next: number) {
    setSensitivity(clamp(next, 0, 1));
    setActivePreset(null);
  }
  function applyPreset(preset: (typeof SENSITIVITY_PRESETS)[number]) {
    setSensitivity(preset.value);
    setActivePreset(preset.key);
  }

  // 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-2) -- 근거 밴드에
  // 목표선과 별개로 판정선(현재 민감도의 "주의" 컷)을 그린다.
  const judgmentLine = targetYield - classifyMargin(sensitivity);

  const mismatchWarning = data ? targetYieldMismatch(targetYield, data.train_y_p1, data.train_y_p99) : false;
  const nonNormalPct = data && data.total_wafers > 0
    ? ((classified.length - classSummary.정상.count) / data.total_wafers) * 100
    : 0;

  return (
    <DashboardShell activeItem="알림 기록">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">알람 판정</span>
        <div className="pageHeadingTitleRow">
          <h1>알림 기록</h1>
          {reliability && (
            <ReliabilityBadge reliability={reliability} open={reliabilityPanelOpen} onToggle={() => setReliabilityPanelOpen((v) => !v)} />
          )}
        </div>
        <p>
          목표 수율과 민감도를 조절해 wafer를 5분류로 판정합니다.
          <br />
          알람(심각·위험·주의)은 예측 수율 구간 상한이 목표보다 낮은 wafer입니다.
        </p>
        {reliability && reliabilityPanelOpen && <ReliabilityPanel reliability={reliability} />}
        {reliability && reliability.grade === "낮음" && (
          <p className="reliabilityLowWarning">
            ⚠ 이 데이터셋에서는 분석 신뢰도가 낮습니다.
            <br />
            불량률 변동이 계측 인자로 설명되지 않아 알람 정확도를 보장할 수 없습니다.
          </p>
        )}
        {reliability?.target_fallback_message && (
          <p className="analysisFallbackNotice">{reliability.target_fallback_message}</p>
        )}
        <LastRunNote createdAt={alarmsState?.createdAt} />
        <FallbackModeBadge />
      </section>

      <section className="uploadCard">
        {/* 지시서 B: 1줄(판정 대상·목표 수율·조회, 바닥선 정렬) / 2줄
            (민감도 하나만, 중앙선 정렬)로 나눈다 -- 민감도 블록이 다른
            컨트롤보다 훨씬 높아 한 줄에 몰아넣으면 카드가 비대칭이었다. */}
        <div className="alertsQueryGrid">
          <div className="alertsQueryRow1">
            <div className="alertsDatasetBlock">
              <DatasetSelector label="예측 대상" value={evalDataset} onChange={setEvalDataset} onUploaded={handleDatasetUploaded} />
              <p className="sectionCaption alertsDatasetCaption">정상범위 기준: {trainDatasetLabel}</p>
            </div>
            <TargetYieldField value={targetYield} onChange={handleTargetYieldChange} disabled={!hydrated} />
            <button type="button" className="button primary alertsQueryButton" disabled={loading} onClick={() => void load()}>
              {loading ? "조회 중…" : data ? "다시 조회" : "조회"}
            </button>
          </div>
          <div className="alertsQueryRow2">
            <SensitivityField
              value={sensitivity}
              activePreset={activePreset}
              onApplyPreset={applyPreset}
              onChange={handleSensitivityChange}
              estimate={precisionRecallEstimate}
              disabled={!hydrated}
            />
          </div>
        </div>
        {/* DG그룹: 서버에 저장된 목표·민감도를 복원하기 전에는 값을 만질
            수 없게 막는다 -- 기본값이 잠깐 보였다가 저장값으로 튀는
            혼란을 막는다. */}
        {!hydrated && <p className="sectionCaption">이전 설정을 불러오는 중…</p>}
        {settingsSaveError && (
          <p className="notifyFieldError">목표 수율·민감도 저장에 실패했습니다. 네트워크를 확인해 주세요.</p>
        )}
        <DatasetMismatchWarning mismatch={datasetMismatch} />
        {activateError && <p className="notifyFieldError">{activateError}</p>}
        {/* AG-2: Y 계열이 감지돼도 자동 학습은 걸지 않는다 -- 안내 +
            모델 학습·자동화 팝업을 여는 링크만 제공한다. */}
        {showTrainingSuggestion && (
          <p className="analysisFallbackNotice" role="status">
            이 파일에는 Y 계열이 있습니다. 학습에 사용하려면 모델 학습·자동화에서 실행하세요.{" "}
            <button type="button" className="linkButton" onClick={() => setTrainingPanelOpen(true)}>
              열기
            </button>
          </p>
        )}
        {error && <p className="errorMessage">{error}</p>}
        {data && mismatchWarning && (
          <div className="alertsMismatchWarning">
            <strong>⚠ 목표 수율 {targetYield.toFixed(1)}%가 이 데이터셋의 분포와 맞지 않습니다.</strong>
            <p>
              현재 데이터 수율은 {data.train_y_min.toFixed(1)} ~ {data.train_y_max.toFixed(1)}% 범위이며 중앙값은 {data.train_y_median.toFixed(1)}%입니다.
              <br />
              전체 wafer의 {nonNormalPct.toFixed(1)}%가 미달로 분류됩니다.
            </p>
            <button type="button" className="button secondary" onClick={() => handleTargetYieldChange(Number(data.train_y_median.toFixed(1)))}>
              중앙값으로 설정 ({data.train_y_median.toFixed(1)}%)
            </button>
          </div>
        )}
        {!data && !loading && <p className="emptyMessage">원인 분석을 실행하면 조회할 수 있습니다</p>}
      </section>

      {data && (
        <FiveClassGrid
          summary={classSummary}
          gatePassed={data.auc_gate_passed}
          aucLowerBound={data.auc_lower_bound}
          aucGateThreshold={data.auc_gate_threshold}
          coverageTarget={data.interval_coverage_target}
          coverageActual={data.interval_coverage_actual}
        />
      )}

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">알람 목록</span>
            <h2>알림 기록 ({gradeFilteredAlarmItems.length}건)</h2>
          </div>
          <div className="alertsAlarmListActions">
            {data?.auc_gate_passed && gradeFilteredAlarmItems.length > 0 && (
              <div className="fieldGroup">
                <span>등급</span>
                <select value={gradeFilter} onChange={(event) => setGradeFilter(event.target.value)}>
                  <option value="">전체</option>
                  <option value="심각">심각</option>
                  <option value="위험">위험</option>
                  <option value="주의">주의</option>
                </select>
              </div>
            )}
            {data?.auc_gate_passed && alarmTable.sorted.length > 0 && (
              <button
                type="button"
                className="button secondary"
                onClick={() => downloadAlarmsCsv(alarmTable.sorted, `alerts_${trainDataset}_${evalDataset}`)}
              >
                CSV 내려받기
              </button>
            )}
          </div>
        </div>
        {data && data.auc_gate_passed && gradeFilteredAlarmItems.length > 0 && (
          <TableToolbar
            search={alarmTable.search}
            onSearchChange={alarmTable.setSearch}
            sort={alarmTable.sort}
            onSortChange={alarmTable.setSort}
            sortOptions={ALARM_SORT_OPTIONS}
            placeholder="LOT_WF_ID · 사유 검색"
          />
        )}
        {loading && <p className="emptyMessage">불러오는 중…</p>}
        {!loading && data && !data.auc_gate_passed && (
          <div className="alarmGateBanner">
            <strong>이 데이터셋에서는 알람을 제공하지 않습니다</strong>
            <p>
              {data.auc_lower_bound != null ? (
                <>
                  알람 순위 품질(AUC)이 {data.auc_lower_bound.toFixed(3)}로 기준({data.auc_gate_threshold.toFixed(2)})에
                  미치지 못합니다.
                </>
              ) : (
                "알람 순위 품질(AUC)을 산출할 수 없습니다 (표본 부족)."
              )}
              <br />
              불량률 변동이 계측 인자로 설명되지 않아 위험 wafer를 구분할 수 없습니다. 정상·판별불가는 계속 계산됩니다.
            </p>
          </div>
        )}
        {!loading && data && data.auc_gate_passed && (
          <>
            {alarmItems.length === 0 && (
              <p className="emptyMessage">
                현재 설정을 충족하는 알람이 없습니다.
                <br />
                목표 수율이나 민감도를 조절해 다시 확인해 보세요.
                <br />
                {/* spec §BD-1: 알람 0건이 "안전"으로 읽히지 않도록 판정
                    범위를 함께 보여준다 -- 구간이 넓으면 대부분이
                    미분류로 빠지고 알람은 애초에 나오기 어렵다. */}
                판정 가능 {(data.total_wafers - classSummary.판별불가.count).toLocaleString()}장 · 미분류{" "}
                {classSummary.판별불가.count.toLocaleString()}장
              </p>
            )}
            {alarmItems.length > 0 && gradeFilteredAlarmItems.length === 0 && (
              <p className="emptyMessage">조건에 맞는 알람이 없습니다.</p>
            )}
            {gradeFilteredAlarmItems.length > 0 && alarmTable.sorted.length === 0 && (
              <NoSearchResults onClear={() => alarmTable.setSearch("")} />
            )}
            {alarmTable.sorted.length > 0 && (
              <>
                <div className="tableHideOnMobile">
                  <HScrollTableBody minWidth={760} rows={ALARM_VISIBLE_ROWS}>
                    <table className="alarmListTable">
                      <thead>
                        <tr>
                          <th className="col-wafer colNoTruncate">LOT_WF_ID</th>
                          <th className="col-lot colNoTruncate">LOT</th>
                          <th className="col-yield numCol">예측 수율 구간</th>
                          <th className="col-severity colNoTruncate">등급</th>
                          <th className="col-reason">사유</th>
                          <th className="col-explain" aria-label="해설" />
                        </tr>
                      </thead>
                      <tbody>
                        {alarmTable.sorted.map((item, index) => (
                          <AlarmRow
                            key={`${item.lot_wafer_id}-${index}`}
                            item={item}
                            targetYield={targetYield}
                            judgmentLine={judgmentLine}
                            onExplain={() => requestChat(alarmExplainMessage(item), "chat")}
                            explainDisabled={!analysisDataset}
                          />
                        ))}
                      </tbody>
                    </table>
                  </HScrollTableBody>
                </div>
                <div className="alarmCardList">
                  {alarmTable.sorted.map((item, index) => (
                    <AlarmCard
                      key={`card-${item.lot_wafer_id}-${index}`}
                      item={item}
                      targetYield={targetYield}
                      judgmentLine={judgmentLine}
                      onExplain={() => requestChat(alarmExplainMessage(item), "chat")}
                      explainDisabled={!analysisDataset}
                    />
                  ))}
                </div>
                <TableCaption total={alarmTable.sorted.length} shown={alarmTable.sorted.length} />
              </>
            )}
            <p className="tableDisclaimer">
              예측 수율 절대값은 정확도가 낮아 구간으로 표시합니다. 원인이 확정된 것은 아니며 우선 확인 대상을 좁히는 용도입니다.
            </p>
          </>
        )}
      </section>

    </DashboardShell>
  );
}

function alarmExplainMessage(item: ClassifiedWafer): string {
  return (
    `알람: ${item.lot_wafer_id} · 등급 ${item.grade} · 예측 수율 ${item.pred_lo.toFixed(1)}~${item.pred_hi.toFixed(1)}\n` +
    `사유: ${reasonFor(item)}\n` +
    "이 알람에 대해 설명해 주세요."
  );
}

const GRADE_CLASS: Record<string, string> = { 심각: "severe", 위험: "danger", 주의: "caution" };

// spec §BC-2: 계측 없이(measured=false) 매겨진 등급은 배지에 구분 표기를
// 붙여 사유 있는 알람과 구분한다 -- 옅은 테두리 + "*" 접미사.
function GradeBadge({ grade, unreasoned }: { grade: string; unreasoned?: boolean }) {
  return (
    <span
      className={`severityBadge severityBadge-grade-${GRADE_CLASS[grade] ?? "caution"}${unreasoned ? " severityBadge-unreasoned" : ""}`}
      title={unreasoned ? "선정 인자가 계측되지 않아 사유를 제시할 수 없습니다 (자동 발송 제외)" : undefined}
    >
      {grade}{unreasoned ? "*" : ""}
    </span>
  );
}

const GRADE_RANK: Record<string, number> = { 심각: 3, 위험: 2, 주의: 1 };

function alarmTieBreak(a: ClassifiedWafer, b: ClassifiedWafer): number {
  return lotCompare(a.lot_id, b.lot_id) || a.lot_wafer_id.localeCompare(b.lot_wafer_id, undefined, { numeric: true });
}

// 정렬 기본값: 등급 -> 예측 수율 오름차순 (spec §D-2).
const ALARM_SORT_OPTIONS: SortOption<ClassifiedWafer>[] = [
  {
    value: "default", label: "등급 · 예측 수율",
    compare: (a, b) => (GRADE_RANK[b.grade ?? ""] ?? 0) - (GRADE_RANK[a.grade ?? ""] ?? 0) || a.pred_mean - b.pred_mean,
  },
  { value: "yield_asc", label: "예측 수율 오름차순", compare: (a, b) => a.pred_mean - b.pred_mean },
  { value: "yield_desc", label: "예측 수율 내림차순", compare: (a, b) => b.pred_mean - a.pred_mean },
  { value: "lot_asc", label: "LOT 오름차순", compare: (a, b) => lotCompare(a.lot_id, b.lot_id) },
  { value: "lot_desc", label: "LOT 내림차순", compare: (a, b) => lotCompare(b.lot_id, a.lot_id) },
  // DA그룹: LOT 1차 오름차순, 같은 LOT 안에서는 2차로 등급 순 -- 컬럼
  // 순서가 LOT_WF_ID·LOT 우선으로 바뀌어도 기본 정렬은 등급 순 그대로
  // 유지하고(위험한 wafer가 먼저 보여야 한다), 이 옵션은 LOT별로 묶어
  // 보고 싶을 때만 선택한다.
  {
    value: "lot_grouped", label: "LOT별",
    compare: (a, b) =>
      lotCompare(a.lot_id, b.lot_id) || (GRADE_RANK[b.grade ?? ""] ?? 0) - (GRADE_RANK[a.grade ?? ""] ?? 0) || a.pred_mean - b.pred_mean,
  },
];

function AlarmRow({
  item,
  targetYield,
  judgmentLine,
  onExplain,
  explainDisabled,
}: {
  item: ClassifiedWafer;
  targetYield: number;
  judgmentLine: number;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  return (
    <tr>
      <td className="col-wafer colNoTruncate">{item.lot_wafer_id}</td>
      <td className="col-lot colNoTruncate">{item.lot_id ?? "-"}</td>
      <td className="col-yield numCol">
        {item.pred_lo.toFixed(1)} ~ {item.pred_hi.toFixed(1)}
        {/* 보정 §I-1: 근거 밴드 위치 2 -- 진짜 lo/hi가 있는 이 표로
            옮겼다. SUMMARY와 동일한 컴포넌트(EvidenceBand), 전 행이 같은
            고정 축을 쓴다(행별 자동 스케일 금지). 판정과 표시는 분리된
            개념이다(spec §CA-2) -- 구간(lo/hi)은 conformal 그대로,
            판정선(judgmentLine)만 민감도에 따라 움직인다. */}
        <EvidenceBand
          lo={item.pred_lo}
          hi={item.pred_hi}
          target={targetYield}
          judgmentLine={judgmentLine}
          scaleMin={ALARM_YIELD_SCALE_MIN}
          scaleMax={ALARM_YIELD_SCALE_MAX}
          mini
        />
      </td>
      <td className="col-severity colNoTruncate">{item.grade && <GradeBadge grade={item.grade} unreasoned={!item.measured} />}</td>
      <td className="alarmReasonCell col-reason">{reasonFor(item)}</td>
      <td className="col-explain"><ExplainButton onClick={onExplain} disabled={explainDisabled} /></td>
    </tr>
  );
}

function AlarmCard({
  item,
  targetYield,
  judgmentLine,
  onExplain,
  explainDisabled,
}: {
  item: ClassifiedWafer;
  targetYield: number;
  judgmentLine: number;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  return (
    <div className="alarmCard">
      <div className="alarmCardTopRow">
        <span className="alarmCardId">{item.lot_wafer_id}</span>
        {item.grade && <GradeBadge grade={item.grade} unreasoned={!item.measured} />}
      </div>
      <div className="alarmCardMeta">
        예측 수율 {item.pred_lo.toFixed(1)} ~ {item.pred_hi.toFixed(1)}
        {item.lot_id && ` · ${item.lot_id}`}
        <EvidenceBand
          lo={item.pred_lo}
          hi={item.pred_hi}
          target={targetYield}
          judgmentLine={judgmentLine}
          scaleMin={ALARM_YIELD_SCALE_MIN}
          scaleMax={ALARM_YIELD_SCALE_MAX}
          mini
        />
      </div>
      <div className="alarmCardStatsRow">
        <span>{reasonFor(item)}</span>
      </div>
      <div className="alarmCardActions">
        <ExplainButton onClick={onExplain} disabled={explainDisabled} />
      </div>
    </div>
  );
}

function ExplainButton({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      className="explainButton"
      onClick={onClick}
      disabled={disabled}
      title={disabled ? "원인 분석을 먼저 실행하세요" : "SUNI에게 이 건에 대해 물어보기"}
    >
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      </svg>
      해설
    </button>
  );
}

/* ===================================================================
   지시서 O-2: 목표 수율 · 민감도를 판정 대상(eval)·조회 버튼과 같은 카드로
   통합했다 -- 조절할 때마다 즉시(디바운스는 큰 데이터셋에서만)
   재계산되고, API를 다시 부르지 않는다.
   =================================================================== */

function TargetYieldField({
  value,
  onChange,
  disabled,
}: {
  value: number;
  onChange: (value: number) => void;
  // DG그룹: 서버에 저장된 값을 복원하기 전에는 조작을 막는다.
  disabled?: boolean;
}) {
  return (
    <div className="alertsSettingField alertsTargetField">
      <span className="alertsSettingLabel">목표 수율</span>
      <div className="alertsTargetInputRow" title="이 값 미만을 미달로 봅니다">
        <input
          key={value}
          type="number" step="0.1" min={0} max={100}
          defaultValue={value.toFixed(1)}
          onBlur={(event) => onChange(Number(event.target.value))}
          onKeyDown={(event) => {
            if (event.key === "Enter") onChange(Number((event.target as HTMLInputElement).value));
          }}
          disabled={disabled}
        />
        <span>%</span>
      </div>
    </div>
  );
}

// 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4) -- 이 아래로 정밀도가
// 떨어지면 주황으로 표시한다.
const PRECISION_WARNING_THRESHOLD_PCT = 95;

function SensitivityField({
  value,
  activePreset,
  onApplyPreset,
  onChange,
  estimate,
  disabled,
}: {
  value: number;
  activePreset: PresetKey | null;
  onApplyPreset: (preset: (typeof SENSITIVITY_PRESETS)[number]) => void;
  onChange: (value: number) => void;
  estimate: PrecisionRecallEstimate | null;
  // DG그룹: 서버에 저장된 값을 복원하기 전에는 조작을 막는다.
  disabled?: boolean;
}) {
  return (
    <div className="alertsSensitivityField">
      <span className="alertsSettingLabel">민감도</span>
      <div className="alertsPresetRow">
        {SENSITIVITY_PRESETS.map((preset) => (
          <button
            key={preset.key}
            type="button"
            className={`alertsPresetButton ${activePreset === preset.key ? "active" : ""}`}
            onClick={() => onApplyPreset(preset)}
            disabled={disabled}
          >
            {preset.label}
          </button>
        ))}
      </div>
      {/* 슬라이더+값 칸을 한 그룹으로 묶는다 (지시서 B-4) -- 좁은 화면에서
          줄바꿈되더라도 이 둘은 쪼개지지 않는다. */}
      <div className="alertsGaugeGroup">
        <div className="alertsGaugeWrap">
          <div className="alertsGaugeHints"><span>오경보 ↓</span><span>미탐 ↓</span></div>
          <input
            type="range" min={0} max={1} step={0.01}
            value={value}
            onChange={(event) => onChange(Number(event.target.value))}
            className="alertsGaugeSlider"
            aria-label="민감도 직접 조절"
            disabled={disabled}
          />
          <div className="alertsGaugeEnds"><span>0</span><span>1</span></div>
        </div>
        <input
          key={value}
          type="number" min={0} max={1} step={0.01}
          defaultValue={value.toFixed(2)}
          onBlur={(event) => onChange(Number(event.target.value))}
          onKeyDown={(event) => {
            if (event.key === "Enter") onChange(Number((event.target as HTMLInputElement).value));
          }}
          className="alertsSensitivityNumber"
          title="직접 입력"
          disabled={disabled}
        />
      </div>
      <SensitivityTradeoffLine estimate={estimate} />
    </div>
  );
}

/** 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4 핵심) -- 사용자가
 * 슬라이더를 밀 때 지불하는 대가(정밀도 하락)를 그 자리에서 보여준다.
 * "홀드아웃 기준 추정"을 반드시 병기한다 -- eval 실측이 아니다. */
function SensitivityTradeoffLine({ estimate }: { estimate: PrecisionRecallEstimate | null }) {
  if (!estimate) return null;
  if (estimate.oofSampleSize === 0) {
    return <p className="alertsSensitivityEstimate sectionCaption">홀드아웃 표본이 부족해 정밀도·재현율을 추정할 수 없습니다.</p>;
  }
  const precisionLow = estimate.precisionPct != null && estimate.precisionPct < PRECISION_WARNING_THRESHOLD_PCT;
  return (
    <p className="alertsSensitivityEstimate sectionCaption">
      알람 {estimate.evalAlarms.toLocaleString()}장 · 정밀도{" "}
      <span className={precisionLow ? "alertsSensitivityEstimateWarning" : undefined}>
        {estimate.precisionPct != null ? `${estimate.precisionPct.toFixed(1)}%` : "-"}
      </span>{" "}
      · 재현율 {estimate.recallPct != null ? `${estimate.recallPct.toFixed(1)}%` : "-"}
      {estimate.missedEstimate != null && <> · 놓칠 것으로 추정 {estimate.missedEstimate.toLocaleString()}장</>}
      <br />
      <span className="alertsSensitivityEstimateNote">홀드아웃 기준 추정 (n={estimate.oofSampleSize.toLocaleString()})</span>
    </p>
  );
}

/* ===================================================================
   §B 판정 결과 -- 5분류 카드.
   =================================================================== */

// C-1: 정상·판별불가는 이상 신호가 아니라 분류이므로 무채색 -- 알람
// 3등급(심각/위험/주의)만 신호색(ALARM_GRADE_COLOR)을 유지한다.
const CLASS_COLOR: Record<ClassKey, string> = {
  심각: ALARM_GRADE_COLOR.심각,
  위험: ALARM_GRADE_COLOR.위험,
  주의: ALARM_GRADE_COLOR.주의,
  정상: "var(--text-secondary)",
  판별불가: "var(--text-secondary)",
};
const ALARM_CLASS_KEYS: ClassKey[] = ["심각", "위험", "주의"];

function TriangleIcon({ color }: { color: string }) {
  return (
    <svg width="18" height="15" viewBox="0 0 18 15" aria-hidden="true">
      <polygon points="9,1 1,14 17,14" fill="none" strokeWidth="1.7" style={{ stroke: color }} />
    </svg>
  );
}
function CircleIcon({ color }: { color: string }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <ellipse cx="8" cy="8" rx="7" ry="4.5" opacity="0.85" style={{ fill: color }} />
    </svg>
  );
}

function FiveClassGrid({
  summary,
  gatePassed,
  aucLowerBound,
  aucGateThreshold,
  coverageTarget,
  coverageActual,
}: {
  summary: ReturnType<typeof summarizeClasses>;
  gatePassed: boolean;
  aucLowerBound: number | null;
  aucGateThreshold: number;
  coverageTarget: number;
  coverageActual: number | null;
}) {
  const unclassified = summary.판별불가;
  return (
    <section className="resultCard alertsClassSection">
      <div className="yieldBandCardTitle"><h3>판정 결과</h3></div>
      <p className="sectionCaption">평가 wafer 기준</p>
      <div className="alertsClassGrid">
        {CLASS_KEYS.map((key) => {
          const item = summary[key];
          const isAlarmTier = ALARM_CLASS_KEYS.includes(key);
          const color = CLASS_COLOR[key];
          const tooltip = key === "판별불가"
            ? `상관성 부족 ${item.straddleCount}장 · 계측 부족 ${item.unmeasuredCount}장`
            : undefined;
          // C-1: 게이트 미달이면 알람 3등급(심각/위험/주의)은 실제로는
          // "0장"(알람 없음)이 아니라 "판정 불가"다 -- 배너를 안 읽으면
          // 숫자만 보고 정반대로("알람 없음 = 좋음") 읽는다. 정상/판별불가는
          // 게이트와 무관하게 계속 계산되므로 그대로 둔다.
          const gated = isAlarmTier && !gatePassed;
          return (
            <div key={key} className="alertsClassCard" style={{ ["--class-color" as string]: color }} title={tooltip}>
              <div className="alertsClassCardHead">
                {isAlarmTier ? <TriangleIcon color={color} /> : <CircleIcon color={color} />}
                <strong>{CLASS_LABELS[key]}</strong>
              </div>
              {gated ? (
                <>
                  <div className="alertsClassCardValue alertsClassCardValueGated">—</div>
                  <span className="alertsClassCardGatedBadge">게이트 미달</span>
                </>
              ) : (
                <>
                  <div className="alertsClassCardValue">{item.count.toLocaleString()}장</div>
                  <div className="alertsClassCardPct">{item.pct.toFixed(1)}%</div>
                  {item.avgPredMean != null && (
                    <div className="alertsClassCardYield">평균 수율 {item.avgPredMean.toFixed(1)}</div>
                  )}
                  {/* spec §BB-1: 미분류를 하나의 카운트로 보여주되 사유별
                      내역을 함께 표시한다 -- 조치 가능 여부가 다른 두
                      사유를 구분해야 계측 우선순위 큐가 무엇을 대상으로
                      하는지 오해하지 않는다. */}
                  {key === "판별불가" && (
                    <div className="alertsClassCardBreakdown">
                      상관성 부족 {item.straddleCount.toLocaleString()} · 계측 부족 {item.unmeasuredCount.toLocaleString()}
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
      <div className="sectionCaption alertsClassGridFoot">
        <p>
          미분류는 판정 근거가 부족한 wafer입니다 (정상이 아닙니다).
          <br />
          · 상관성 부족 {unclassified.straddleCount.toLocaleString()}장 — 예측 구간이 목표 수율을 걸쳐 방향을 정할 수
          없습니다. 계측을 늘려도 해소되지 않습니다.
          <br />
          · 계측 부족 {unclassified.unmeasuredCount.toLocaleString()}장 — 선정 인자가 계측되지 않아 판정 근거가
          없습니다. 계측을 늘리면 해소됩니다.
        </p>
      </div>
      {/* spec §BA-4/§BD-1: "구간을 믿어도 되는지"가 이 시스템 신뢰도의
          근간이므로 항상 표시한다 -- 숨기거나 생략하지 않는다. */}
      <p className="sectionCaption alertsCoverageNote">
        {coverageActual != null
          ? `예측 구간 포함률 ${(coverageActual * 100).toFixed(1)}% (목표 ${(coverageTarget * 100).toFixed(0)}%)`
          : `예측 구간 목표 포함률 ${(coverageTarget * 100).toFixed(0)}% (이 평가 데이터셋은 실제 정답이 없어 실측 포함률을 검증할 수 없습니다)`}
      </p>
      {!gatePassed && (
        <p className="alertsGateNote">
          이 데이터셋에서는 알람을 제공하지 않습니다.{" "}
          {aucLowerBound != null
            ? `알람 순위 품질(AUC)이 ${aucLowerBound.toFixed(2)}로 기준(${aucGateThreshold.toFixed(2)})에 미치지 못합니다.`
            : "알람 순위 품질(AUC)을 산출할 수 없습니다."}
        </p>
      )}
    </section>
  );
}

