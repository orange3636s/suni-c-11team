"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_SENSITIVITY, DEFAULT_TARGET_YIELD, useAnalysisState } from "@/components/AnalysisStateProvider";
import DashboardShell from "@/components/DashboardShell";
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
  CLASS_KEYS,
  classifyAll,
  downloadAlarmsCsv,
  estimatePrecisionRecall,
  representativeWafer,
  summarizeClasses,
  targetYieldMismatch,
} from "@/lib/alertsClassify";
import { measurementRateDisclaimer } from "@/lib/measurementDisclaimer";
import { niceTicks } from "@/lib/niceTicks";
import { getAlertsData, getDatasetSchema, getReliability, saveAlarmsState } from "@/lib/api";
import type {
  DatasetSchemaResponse,
  FactorBand,
  MeasurementBiasSummary,
  ReliabilityResponse,
} from "@/types/data";

const RELIABILITY_GRADE_CLASS: Record<string, string> = { 높음: "high", 보통: "medium", 낮음: "low" };

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

type PresetKey = "low_fp" | "balanced" | "low_fn";
const SENSITIVITY_PRESETS: Array<{ key: PresetKey; label: string; value: number }> = [
  { key: "low_fp", label: "오경보 최소", value: 0.2 },
  { key: "balanced", label: "균형", value: 0.5 },
  { key: "low_fn", label: "미탐 최소", value: 0.8 },
];

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export default function AlertsPage() {
  const { analysisDataset, requestChat } = usePanelState();
  const { alarms: alarmsState, setAlarms: setAlarmsState, hydrated } = useAnalysisState();
  const [trainDataset, setTrainDataset] = useState("train");
  const [evalDataset, setEvalDataset] = useState("test");
  const [targetYield, setTargetYield] = useState(DEFAULT_TARGET_YIELD);
  const [sensitivity, setSensitivity] = useState(DEFAULT_SENSITIVITY);
  const [activePreset, setActivePreset] = useState<PresetKey | null>("balanced");
  const [gradeFilter, setGradeFilter] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [factorBandIndex, setFactorBandIndex] = useState(0);
  const [reliability, setReliability] = useState<ReliabilityResponse | null>(null);
  const [reliabilityPanelOpen, setReliabilityPanelOpen] = useState(false);
  const [trainSchema, setTrainSchema] = useState<DatasetSchemaResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    getDatasetSchema(trainDataset)
      .then((result) => {
        if (!cancelled) setTrainSchema(result);
      })
      .catch(() => {
        if (!cancelled) setTrainSchema(null);
      });
    return () => {
      cancelled = true;
    };
  }, [trainDataset]);

  const data = alarmsState?.data ?? null;
  const datasetMismatch = Boolean(
    alarmsState && (alarmsState.trainDataset !== trainDataset || alarmsState.evalDataset !== evalDataset),
  );

  // train/eval을 인자로도 받는다 -- 재접속 직후 복원된 데이터셋으로 바로
  // 조회할 때, 방금 호출한 setTrainDataset/setEvalDataset은 같은 틱 안의
  // 클로저에 아직 반영되지 않으므로 state를 읽는 대신 명시적으로 넘긴다.
  const load = useCallback(
    async (overrideTrain?: string, overrideEval?: string) => {
      const train = overrideTrain ?? trainDataset;
      const evalDs = overrideEval ?? evalDataset;
      setLoading(true);
      setError("");
      try {
        const [dataResponse, reliabilityResponse] = await Promise.all([
          getAlertsData(train, evalDs),
          getReliability(train, evalDs).catch(() => null),
        ]);
        setAlarmsState((previous) => ({
          trainDataset: train,
          evalDataset: evalDs,
          createdAt: new Date().toISOString(),
          targetYield: previous?.targetYield ?? targetYield,
          sensitivity: previous?.sensitivity ?? sensitivity,
          data: dataResponse,
        }));
        setReliability(reliabilityResponse);
        setFactorBandIndex(0);
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : "사전 알람 로그를 불러오지 못했습니다.");
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
  // 새로 불러온다.
  const initializedFromHydration = useRef(false);
  useEffect(() => {
    if (!hydrated || initializedFromHydration.current) return;
    initializedFromHydration.current = true;
    const timer = window.setTimeout(() => {
      if (alarmsState) {
        setTrainDataset(alarmsState.trainDataset);
        setEvalDataset(alarmsState.evalDataset);
        setTargetYield(alarmsState.targetYield);
        setSensitivity(alarmsState.sensitivity);
        setActivePreset(null);
        void load(alarmsState.trainDataset, alarmsState.evalDataset);
      } else {
        void load();
      }
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated]);

  // 목표 수율·민감도는 가볍게 서버에 저장해 둔다 (spec: 재접속 시 마지막
  // 설정을 복원) -- 슬라이더를 계속 움직이는 동안 매번 POST하지 않도록
  // 800ms 묶어서 보낸다. 재분류 자체는 이 저장과 무관하게 즉시 일어난다.
  useEffect(() => {
    if (!alarmsState) return;
    const timer = window.setTimeout(() => {
      void saveAlarmsState(alarmsState.trainDataset, alarmsState.evalDataset, { targetYield, sensitivity }).catch(() => {});
    }, 800);
    return () => window.clearTimeout(timer);
  }, [targetYield, sensitivity, alarmsState]);

  const debounceMs = (data?.total_wafers ?? 0) >= LARGE_DATASET_DEBOUNCE_THRESHOLD ? DEBOUNCE_MS : 0;
  const targetYieldForClassify = useDebouncedNumber(targetYield, debounceMs);
  const sensitivityForClassify = useDebouncedNumber(sensitivity, debounceMs);

  // 사전 알람 로그 전면 개편의 핵심 -- API를 다시 부르지 않고, 이미 받아둔
  // 원시 예측치를 목표 수율·민감도로 즉시 재분류한다 (spec §A-3).
  const classified = useMemo<ClassifiedWafer[]>(() => {
    if (!data) return [];
    return classifyAll(data.predictions, {
      target: targetYieldForClassify,
      sensitivity: sensitivityForClassify,
      sigma: data.sigma,
      gatePassed: data.auc_gate_passed,
    });
  }, [data, targetYieldForClassify, sensitivityForClassify]);

  const classSummary = useMemo(
    () => summarizeClasses(classified, data?.total_wafers ?? 0),
    [classified, data],
  );

  const precisionRecall = useMemo(
    () => estimatePrecisionRecall(data?.holdout ?? null, { target: targetYieldForClassify, sensitivity: sensitivityForClassify }),
    [data, targetYieldForClassify, sensitivityForClassify],
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

  const alarmTable = useTableSearchSort(
    gradeFilteredAlarmItems,
    (item) => `${item.lot_wafer_id} ${item.reason ?? ""}`,
    ALARM_SORT_OPTIONS,
    "default",
    alarmTieBreak,
  );

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

  const mismatchWarning = data ? targetYieldMismatch(targetYield, data.train_y_p1, data.train_y_p99) : false;
  const nonNormalPct = data && data.total_wafers > 0
    ? ((classified.length - classSummary.정상.count) / data.total_wafers) * 100
    : 0;

  return (
    <DashboardShell activeItem="사전 알람 로그">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">PRE-ALERT LOG</span>
        <div className="pageHeadingTitleRow">
          <h1>사전 알람 로그</h1>
          {reliability && (
            <ReliabilityBadge reliability={reliability} open={reliabilityPanelOpen} onToggle={() => setReliabilityPanelOpen((v) => !v)} />
          )}
        </div>
        <p>
          목표 수율과 민감도를 조절해 wafer를 5분류로 판정합니다.
          <br />
          알람(심각·위험·주의)은 예측 수율 구간의 상한이 목표보다 확실히 낮은 wafer이며, 전체 인자를 종합해 판정합니다.
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
      </section>

      <section className="uploadCard">
        <div className="rcControlBar alarmControlBar">
          <DatasetSelector label="정상범위 산출 (train)" value={trainDataset} onChange={setTrainDataset} />
          <DatasetSelector label="판정 대상 (eval)" value={evalDataset} onChange={setEvalDataset} />
          <button type="button" className="button primary" disabled={loading} onClick={() => void load()} style={{ alignSelf: "end" }}>
            {loading ? "조회 중…" : data ? "다시 조회" : "조회"}
          </button>
        </div>
        <DatasetMismatchWarning mismatch={datasetMismatch} />
        {error && <p className="errorMessage">{error}</p>}
      </section>

      {data ? (
        <SettingsBar
          targetYield={targetYield}
          onTargetYieldChange={handleTargetYieldChange}
          sensitivity={sensitivity}
          onSensitivityChange={handleSensitivityChange}
          activePreset={activePreset}
          onApplyPreset={applyPreset}
          mismatchWarning={mismatchWarning}
          nonNormalPct={nonNormalPct}
          trainYMin={data.train_y_min}
          trainYMax={data.train_y_max}
          trainYMedian={data.train_y_median}
          alarmCount={classSummary.심각.count + classSummary.위험.count + classSummary.주의.count}
          normalCount={classSummary.정상.count}
          undecidableCount={classSummary.판별불가.count}
          precisionPct={precisionRecall.precisionPct}
          recallPct={precisionRecall.recallPct}
        />
      ) : (
        <section className="uploadCard alertsSettingsSkeleton">
          <p className="emptyMessage">{loading ? "불러오는 중…" : "원인 분석을 실행하면 조회할 수 있습니다"}</p>
        </section>
      )}

      {data && (
        <FiveClassGrid
          summary={classSummary}
          gatePassed={data.auc_gate_passed}
          aucLowerBound={data.auc_lower_bound}
          aucGateThreshold={data.auc_gate_threshold}
        />
      )}

      {data && <JudgmentPrincipleChart summary={classSummary} target={targetYield} />}

      {data && data.factor_bands.length > 0 ? (
        <FactorYieldBandCard
          bands={data.factor_bands}
          activeIndex={Math.min(factorBandIndex, data.factor_bands.length - 1)}
          onChange={setFactorBandIndex}
        />
      ) : data ? (
        <section className="resultCard yieldBandCard">
          <div className="yieldBandCardTitle"><h3>인자별 불량률 (%)</h3></div>
          <p className="emptyMessage">강함·보통 등급 인자가 없어 인자별 불량률을 표시할 수 없습니다.</p>
        </section>
      ) : null}

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">ALARMS</span>
            <h2>알람 목록 ({gradeFilteredAlarmItems.length}건)</h2>
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
            placeholder="Wafer ID · 사유 검색"
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
                  <HScrollTableBody minWidth={760}>
                    <table>
                      <thead>
                        <tr>
                          <th className="col-wafer colNoTruncate">Wafer</th>
                          <th>예측 수율 구간</th>
                          <th className="col-severity colNoTruncate">등급</th>
                          <th>사유</th>
                          <th>LOT</th>
                          <th aria-label="해설" />
                        </tr>
                      </thead>
                      <tbody>
                        {alarmTable.sorted.map((item, index) => (
                          <AlarmRow
                            key={`${item.lot_wafer_id}-${index}`}
                            item={item}
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
                      onExplain={() => requestChat(alarmExplainMessage(item), "chat")}
                      explainDisabled={!analysisDataset}
                    />
                  ))}
                </div>
                <TableCaption total={alarmTable.sorted.length} shown={Math.min(10, alarmTable.sorted.length)} />
              </>
            )}
            <p className="tableDisclaimer">
              예측 수율 절대값은 정확도가 낮아 구간으로 표시합니다. 불량의 원인으로 확정된 것은 아니며, 우선 확인 대상을 좁히는 용도입니다.
            </p>
          </>
        )}
      </section>

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>{measurementRateDisclaimer(trainSchema)}</li>
          {data?.measurement_bias && <li>{describeMeasurementBias(data.measurement_bias)}</li>}
          <li>알람은 관측 데이터의 통계적 연관성에 기반하며 인과관계를 의미하지 않습니다.</li>
          <li>정밀도·재현율은 학습 데이터 홀드아웃 기준 추정치입니다. 평가 데이터의 정답을 모르므로 실측이 아닙니다.</li>
          <li>분석 신뢰도가 낮은 데이터셋에서는 알람 정확도를 보장할 수 없습니다.</li>
          <li>판별불가(미계측) wafer는 선정 인자가 계측되지 않아 판정 자체가 불가능한 것이며, 정상을 의미하지 않습니다.</li>
        </ul>
      </section>
    </DashboardShell>
  );
}

function alarmExplainMessage(item: ClassifiedWafer): string {
  return (
    `알람: ${item.lot_wafer_id} · 등급 ${item.grade} · 예측 수율 ${item.pred_lo.toFixed(1)}~${item.pred_hi.toFixed(1)}\n` +
    `사유: ${item.reason ?? "-"}\n` +
    "이 알람에 대해 설명해 주세요."
  );
}

const GRADE_CLASS: Record<string, string> = { 심각: "severe", 위험: "danger", 주의: "caution" };

function GradeBadge({ grade }: { grade: string }) {
  return <span className={`severityBadge severityBadge-grade-${GRADE_CLASS[grade] ?? "caution"}`}>{grade}</span>;
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
];

function AlarmRow({
  item,
  onExplain,
  explainDisabled,
}: {
  item: ClassifiedWafer;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  return (
    <tr>
      <td className="col-wafer colNoTruncate">{item.lot_wafer_id}</td>
      <td className="numCol">{item.pred_lo.toFixed(1)} ~ {item.pred_hi.toFixed(1)}</td>
      <td className="col-severity colNoTruncate">{item.grade && <GradeBadge grade={item.grade} />}</td>
      <td className="alarmReasonCell">{item.reason ?? "-"}</td>
      <td>{item.lot_id ?? "-"}</td>
      <td><ExplainButton onClick={onExplain} disabled={explainDisabled} /></td>
    </tr>
  );
}

function AlarmCard({
  item,
  onExplain,
  explainDisabled,
}: {
  item: ClassifiedWafer;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  return (
    <div className="alarmCard">
      <div className="alarmCardTopRow">
        <span className="alarmCardId">{item.lot_wafer_id}</span>
        {item.grade && <GradeBadge grade={item.grade} />}
      </div>
      <div className="alarmCardMeta">
        예측 수율 {item.pred_lo.toFixed(1)} ~ {item.pred_hi.toFixed(1)}
        {item.lot_id && ` · ${item.lot_id}`}
      </div>
      <div className="alarmCardStatsRow">
        <span>{item.reason ?? "-"}</span>
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
   §A 설정 바 -- 목표 수율 · 민감도. 조절할 때마다 즉시(디바운스는 큰
   데이터셋에서만) 재계산되고, API를 다시 부르지 않는다.
   =================================================================== */

function SettingsBar({
  targetYield,
  onTargetYieldChange,
  sensitivity,
  onSensitivityChange,
  activePreset,
  onApplyPreset,
  mismatchWarning,
  nonNormalPct,
  trainYMin,
  trainYMax,
  trainYMedian,
  alarmCount,
  normalCount,
  undecidableCount,
  precisionPct,
  recallPct,
}: {
  targetYield: number;
  onTargetYieldChange: (value: number) => void;
  sensitivity: number;
  onSensitivityChange: (value: number) => void;
  activePreset: PresetKey | null;
  onApplyPreset: (preset: (typeof SENSITIVITY_PRESETS)[number]) => void;
  mismatchWarning: boolean;
  nonNormalPct: number;
  trainYMin: number;
  trainYMax: number;
  trainYMedian: number;
  alarmCount: number;
  normalCount: number;
  undecidableCount: number;
  precisionPct: number | null;
  recallPct: number | null;
}) {
  return (
    <section className="resultCard alertsSettingsBar">
      <div className="alertsSettingsRow">
        <div className="alertsSettingField alertsTargetField">
          <span className="alertsSettingLabel">목표 수율</span>
          <div className="alertsTargetInputRow">
            <input
              key={targetYield}
              type="number" step="0.1" min={0} max={100}
              defaultValue={targetYield.toFixed(1)}
              onBlur={(event) => onTargetYieldChange(Number(event.target.value))}
              onKeyDown={(event) => {
                if (event.key === "Enter") onTargetYieldChange(Number((event.target as HTMLInputElement).value));
              }}
            />
            <span>%</span>
          </div>
          <span className="alertsSettingHint">이 값 미만을 미달로 봅니다</span>
        </div>

        <div className="alertsSettingField alertsSensitivityField">
          <span className="alertsSettingLabel">민감도</span>
          <div className="alertsPresetRow">
            {SENSITIVITY_PRESETS.map((preset) => (
              <button
                key={preset.key}
                type="button"
                className={`alertsPresetButton ${activePreset === preset.key ? "active" : ""}`}
                onClick={() => onApplyPreset(preset)}
              >
                {preset.label}
              </button>
            ))}
          </div>
          <div className="alertsGaugeRow">
            <div className="alertsGaugeWrap">
              <div className="alertsGaugeHints"><span>오경보 ↓</span><span>미탐 ↓</span></div>
              <input
                type="range" min={0} max={1} step={0.01}
                value={sensitivity}
                onChange={(event) => onSensitivityChange(Number(event.target.value))}
                className="alertsGaugeSlider"
                aria-label="민감도 직접 조절"
              />
              <div className="alertsGaugeEnds"><span>0</span><span>1</span></div>
            </div>
            <input
              key={sensitivity}
              type="number" min={0} max={1} step={0.01}
              defaultValue={sensitivity.toFixed(2)}
              onBlur={(event) => onSensitivityChange(Number(event.target.value))}
              onKeyDown={(event) => {
                if (event.key === "Enter") onSensitivityChange(Number((event.target as HTMLInputElement).value));
              }}
              className="alertsSensitivityNumber"
              title="직접 입력"
            />
          </div>
        </div>
      </div>

      {mismatchWarning && (
        <div className="alertsMismatchWarning">
          <strong>⚠ 목표 수율 {targetYield.toFixed(1)}%가 이 데이터셋의 분포와 맞지 않습니다.</strong>
          <p>
            현재 데이터 수율은 {trainYMin.toFixed(1)} ~ {trainYMax.toFixed(1)}% 범위이며 중앙값은 {trainYMedian.toFixed(1)}%입니다.
            <br />
            전체 wafer의 {nonNormalPct.toFixed(1)}%가 미달로 분류됩니다.
          </p>
          <button type="button" className="button secondary" onClick={() => onTargetYieldChange(Number(trainYMedian.toFixed(1)))}>
            중앙값으로 설정 ({trainYMedian.toFixed(1)}%)
          </button>
        </div>
      )}

      <div className="alertsPreviewRow">
        <span className="alertsPreviewLabel">현재 설정 기준 예상</span>
        <div className="alertsPreviewStats">
          <div className="alertsPreviewStat">
            <strong>{alarmCount.toLocaleString()}건</strong>
            <span>알람</span>
          </div>
          <div className="alertsPreviewStat" title="학습 데이터 검증 결과에 기반한 추정치입니다">
            <strong>{precisionPct != null ? `${precisionPct.toFixed(0)}%` : "-"}</strong>
            <span>정밀도</span>
          </div>
          <div className="alertsPreviewStat" title="학습 데이터 검증 결과에 기반한 추정치입니다">
            <strong>{recallPct != null ? `${recallPct.toFixed(0)}%` : "-"}</strong>
            <span>재현율</span>
          </div>
          <div className="alertsPreviewStat">
            <strong>{normalCount.toLocaleString()}장</strong>
            <span>정상</span>
          </div>
          <div className="alertsPreviewStat">
            <strong>{undecidableCount.toLocaleString()}장</strong>
            <span>판별불가</span>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ===================================================================
   §B 판정 결과 -- 5분류 카드.
   =================================================================== */

const CLASS_COLOR: Record<ClassKey, string> = {
  심각: "#DC2626",
  위험: "#EA580C",
  주의: "#CA8A04",
  정상: "#0D9668",
  판별불가: "#9CA3AF",
};
const ALARM_CLASS_KEYS: ClassKey[] = ["심각", "위험", "주의"];

function TriangleIcon({ color }: { color: string }) {
  return (
    <svg width="18" height="15" viewBox="0 0 18 15" aria-hidden="true">
      <polygon points="9,1 1,14 17,14" fill="none" stroke={color} strokeWidth="1.7" />
    </svg>
  );
}
function CircleIcon({ color }: { color: string }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <ellipse cx="8" cy="8" rx="7" ry="4.5" fill={color} opacity="0.85" />
    </svg>
  );
}

function FiveClassGrid({
  summary,
  gatePassed,
  aucLowerBound,
  aucGateThreshold,
}: {
  summary: ReturnType<typeof summarizeClasses>;
  gatePassed: boolean;
  aucLowerBound: number | null;
  aucGateThreshold: number;
}) {
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
            ? `구간 걸침 ${item.straddleCount}장 · 미계측 ${item.unmeasuredCount}장`
            : undefined;
          return (
            <div key={key} className="alertsClassCard" style={{ ["--class-color" as string]: color }} title={tooltip}>
              <div className="alertsClassCardHead">
                {isAlarmTier ? <TriangleIcon color={color} /> : <CircleIcon color={color} />}
                <strong>{key}</strong>
              </div>
              <div className="alertsClassCardValue">{item.count.toLocaleString()}장</div>
              <div className="alertsClassCardPct">{item.pct.toFixed(1)}%</div>
              {item.avgPredMean != null && (
                <div className="alertsClassCardYield">평균 수율 {item.avgPredMean.toFixed(1)}</div>
              )}
              {isAlarmTier && <div className="alertsClassCardNote">알람 로그 기록</div>}
            </div>
          );
        })}
      </div>
      <p className="sectionCaption alertsClassGridFoot">
        심각·위험·주의는 알람 로그에 기록됩니다. 알림 발송 대상은 설정에서 지정합니다.
        <br />
        판별불가는 예측 구간이 목표 수율을 가로질러 판단을 유보한 wafer입니다.
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

/* ===================================================================
   §C 판정 원리 -- 분류별 대표 사례의 예측 구간을 목표와 비교하는 그림.
   =================================================================== */

function JudgmentPrincipleChart({
  summary,
  target,
}: {
  summary: ReturnType<typeof summarizeClasses>;
  target: number;
}) {
  const rows = useMemo(() => {
    return (["심각", "위험", "주의", "판별불가", "정상"] as ClassKey[])
      .map((key) => ({ key, rep: representativeWafer(summary[key].items) }))
      .filter((row): row is { key: ClassKey; rep: NonNullable<typeof row.rep> } => row.rep != null);
  }, [summary]);

  if (rows.length === 0) return null;

  const values = rows.flatMap((row) => [row.rep.pred_lo, row.rep.pred_hi]);
  values.push(target);
  const domainLoRaw = Math.min(...values);
  const domainHiRaw = Math.max(...values);
  const pad = (domainHiRaw - domainLoRaw) * 0.08 || 1;
  const domainLo = domainLoRaw - pad;
  const domainHi = domainHiRaw + pad;
  const span = domainHi - domainLo || 1;
  const pctOf = (value: number) => ((value - domainLo) / span) * 100;

  return (
    <section className="resultCard alertsPrincipleCard">
      <div className="yieldBandCardTitle"><h3>판정 원리</h3></div>
      <p className="sectionCaption">예측 수율 구간을 목표와 비교합니다</p>
      <div className="alertsPrincipleBody">
        <div className="alertsPrincipleChart">
          <div className="alertsPrincipleTargetLine" style={{ left: `${pctOf(target)}%` }} />
          <span className="alertsPrincipleTargetLabel" style={{ left: `${pctOf(target)}%` }}>목표 {target.toFixed(1)}</span>
          <div className="alertsPrincipleRows">
            {rows.map(({ key, rep }) => {
              const loPct = pctOf(rep.pred_lo);
              const hiPct = pctOf(rep.pred_hi);
              const meanPct = pctOf(rep.pred_mean);
              return (
                <div key={key} className="alertsPrincipleRow">
                  <span className="alertsPrincipleRowLabel" style={{ color: CLASS_COLOR[key] }}>{key}</span>
                  <div className="alertsPrincipleTrack">
                    <div
                      className="alertsPrincipleBar"
                      style={{ left: `${loPct}%`, width: `${Math.max(hiPct - loPct, 0.5)}%`, background: CLASS_COLOR[key] }}
                    />
                    <div className="alertsPrincipleMean" style={{ left: `${meanPct}%`, background: CLASS_COLOR[key] }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div className="alertsPrincipleRules">
          <strong>판정 규칙</strong>
          <dl>
            <div><dt>알람</dt><dd>구간 상한 &lt; 목표</dd></div>
            <div><dt>정상</dt><dd>구간 하한 ≥ 목표</dd></div>
            <div><dt>판별불가</dt><dd>구간이 목표를 가로지름</dd></div>
            <div><dt>판별불가</dt><dd>선정 인자 미계측</dd></div>
          </dl>
        </div>
      </div>
      <p className="sectionCaption">민감도를 올리면 알람 판정선이 목표에 가까워져 더 많은 wafer가 알람으로 분류됩니다.</p>
    </section>
  );
}

/* ===================================================================
   §D-1 인자별 불량률 -- 인자 값 구간별 불량률(5분류와 무관). 용어만
   경고선/최적 구간 기준으로 갱신했다.
   =================================================================== */

type BandTone = "out" | "outrec" | "inrec";
type BandSegment = { tone: BandTone; label: string; widthPct: number };
type BandLine = { tone: "control" | "recommended"; pct: number };
type BandTick = { pct: number; label: string };

// 사전 알람 로그 전면 개편 (spec §D-1) -- "관리한계 이탈" -> "경고선 초과",
// "권장구간 밖/안" -> "최적 구간 밖/내".
const TONE_LABEL: Record<BandTone, string> = { out: "경고선 초과", outrec: "최적 구간 밖", inrec: "최적 구간 내" };

function formatTickValue(value: number): string {
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1);
}

function formatDeltaPP(delta: number): string {
  if (!Number.isFinite(delta)) return "-";
  const sign = delta > 0 ? "+" : delta < 0 ? "−" : "";
  return `${sign}${Math.abs(delta).toFixed(2)}%p`;
}

function ArrowIcon() {
  return (
    <svg width="20" height="12" viewBox="0 0 20 12" fill="none" aria-hidden="true">
      <path d="M1 6h15" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 1.5 17 6l-5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function BandTrack({
  segments,
  lines,
  ticks,
  lclPct,
  uclPct,
  recLoPct,
  recHiPct,
}: {
  segments: BandSegment[];
  lines: BandLine[];
  ticks?: BandTick[];
  lclPct: number | null;
  uclPct: number | null;
  recLoPct: number | null;
  recHiPct: number | null;
}) {
  return (
    <>
      <div className="yieldBandLabelsRow">
        {lclPct != null && <span className="yieldBandBoundaryLabel" style={{ left: `${lclPct}%` }}>경고선</span>}
        {recLoPct != null && recHiPct != null && (
          <span className="yieldBandRecommendedArrow" style={{ left: `${(recLoPct + recHiPct) / 2}%` }}>
            ◀─ 최적 구간 ─▶
          </span>
        )}
        {uclPct != null && <span className="yieldBandBoundaryLabel" style={{ left: `${uclPct}%` }}>경고선</span>}
      </div>
      <div className="yieldBandTrack">
        {segments.map((seg, index) => (
          <div key={index} className={`yieldBandSeg seg-${seg.tone}`} style={{ width: `${Math.max(seg.widthPct, 0)}%` }}>
            <span>{seg.label}</span>
          </div>
        ))}
        {lines.map((line, index) => (
          <div key={index} className={`yieldBandLine line-${line.tone}`} style={{ left: `${line.pct}%` }} />
        ))}
      </div>
      {ticks && (
        <div className="yieldBandTicks">
          {ticks.map((tick) => (
            <span key={tick.label} className="yieldBandTick" style={{ left: `${tick.pct}%` }}>{tick.label}</span>
          ))}
        </div>
      )}
    </>
  );
}

function BandStatsRow({
  stats,
  arrowTone,
}: {
  stats: { name: string; tone: BandTone; value: number | null; count: number; pct: number }[];
  arrowTone: "yield" | "defect";
}) {
  return (
    <div className="yieldBandStats">
      {stats.map((stat, index) => (
        <div key={stat.tone} style={{ display: "contents" }}>
          <div className={`yieldBandStatCol seg-${stat.tone}`}>
            <span className="yieldBandStatName">{stat.name}</span>
            <strong className="yieldBandStatValue">{stat.value != null ? stat.value.toFixed(2) : "-"}</strong>
            <span className="yieldBandStatSub">{stat.count.toLocaleString()}장 · {stat.pct.toFixed(1)}%</span>
          </div>
          {index < stats.length - 1 && stat.value != null && stats[index + 1].value != null && (
            <div className={`yieldBandArrowGap tone-${arrowTone}`}>
              <ArrowIcon />
              <span>{formatDeltaPP(stats[index + 1].value! - stat.value!)}</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function FactorYieldBandCard({
  bands,
  activeIndex,
  onChange,
}: {
  bands: FactorBand[];
  activeIndex: number;
  onChange: (index: number) => void;
}) {
  const band = bands[activeIndex] ?? bands[0];
  if (!band) return null;

  const pad = (band.x_max - band.x_min) * 0.04 || 1;
  const domainLoRaw = Math.min(band.x_min, band.lcl ?? band.x_min);
  const domainHiRaw = Math.max(band.x_max, band.ucl ?? band.x_max);
  const domainLo = domainLoRaw - pad;
  const domainHi = domainHiRaw + pad;
  const span = domainHi - domainLo || 1;
  const pctOf = (v: number) => ((v - domainLo) / span) * 100;

  const lclPct = band.lcl != null ? pctOf(band.lcl) : null;
  const uclPct = band.ucl != null ? pctOf(band.ucl) : null;
  const hasRecommended = band.recommended_lo != null && band.recommended_hi != null;
  const recLoPct = hasRecommended ? pctOf(band.recommended_lo as number) : null;
  const recHiPct = hasRecommended ? pctOf(band.recommended_hi as number) : null;

  const segments: BandSegment[] = [];
  const lines: BandLine[] = [];
  let leftEdge = 0;
  if (lclPct != null) {
    segments.push({ tone: "out", label: TONE_LABEL.out, widthPct: lclPct - leftEdge });
    lines.push({ tone: "control", pct: lclPct });
    leftEdge = lclPct;
  }
  const rightEdge = uclPct ?? 100;
  if (recLoPct != null && recHiPct != null) {
    segments.push({ tone: "outrec", label: TONE_LABEL.outrec, widthPct: recLoPct - leftEdge });
    lines.push({ tone: "recommended", pct: recLoPct });
    segments.push({ tone: "inrec", label: TONE_LABEL.inrec, widthPct: recHiPct - recLoPct });
    lines.push({ tone: "recommended", pct: recHiPct });
    leftEdge = recHiPct;
  }
  segments.push({ tone: "outrec", label: TONE_LABEL.outrec, widthPct: rightEdge - leftEdge });
  if (uclPct != null) {
    lines.push({ tone: "control", pct: uclPct });
    segments.push({ tone: "out", label: TONE_LABEL.out, widthPct: 100 - uclPct });
  }

  const ticks: BandTick[] = niceTicks([domainLo, domainHi], 6).map((value) => ({
    pct: pctOf(value),
    label: formatTickValue(value),
  }));

  const totalMeasured = band.out_of_control.count + band.out_of_recommended.count + band.in_recommended.count;
  const statsOf = (count: number) => (totalMeasured > 0 ? (count / totalMeasured) * 100 : 0);
  const stats = [
    { name: TONE_LABEL.out, tone: "out" as const, value: band.out_of_control.mean_defect_rate, count: band.out_of_control.count, pct: statsOf(band.out_of_control.count) },
    { name: TONE_LABEL.outrec, tone: "outrec" as const, value: band.out_of_recommended.mean_defect_rate, count: band.out_of_recommended.count, pct: statsOf(band.out_of_recommended.count) },
    { name: TONE_LABEL.inrec, tone: "inrec" as const, value: band.in_recommended.mean_defect_rate, count: band.in_recommended.count, pct: statsOf(band.in_recommended.count) },
  ];

  return (
    <section className="resultCard yieldBandCard">
      <div className="yieldBandCardHeader">
        <div className="yieldBandCardTitle">
          <h3>인자별 불량률 (%)</h3>
          <label className="factorSelectField">
            <select
              className="factorSelect"
              value={activeIndex}
              onChange={(event) => onChange(Number(event.target.value))}
            >
              {bands.map((item, index) => (
                <option key={`${item.feature}-${item.target}`} value={index}>
                  {item.feature} → {item.target} · {item.confidence_tier === "strong" ? "강함" : "보통"}
                </option>
              ))}
            </select>
          </label>
        </div>
        <span className="yieldBandCardMeta">낮을수록 좋음</span>
      </div>
      <BandTrack segments={segments} lines={lines} ticks={ticks} lclPct={lclPct} uclPct={uclPct} recLoPct={recLoPct} recHiPct={recHiPct} />
      <BandStatsRow stats={stats} arrowTone="defect" />
    </section>
  );
}

/** 인자별 계측 편향 검정 결과를 사람이 읽을 문장으로 바꾼다 (spec 문구 전수
 * 검토 §A-7). */
function describeMeasurementBias(bias: MeasurementBiasSummary | null): string | null {
  if (!bias) return null;
  if (bias.significant_count === 0) {
    return "계측 대상 선정에 따른 편향은 관측되지 않았습니다.";
  }
  const directionWord = { low: "낮게", high: "높게", mixed: "다르게" }[bias.direction ?? "mixed"];
  const scope =
    bias.significant_count === bias.tested_count
      ? `선정 인자 ${bias.significant_count}개 모두에서`
      : `선정 인자 ${bias.tested_count}개 중 ${bias.significant_count}개에서`;
  return (
    `계측 대상이 무작위로 선정되지 않았을 가능성이 있습니다. ${scope} 계측된 wafer의 불량률이 ` +
    `미계측 wafer보다 ${directionWord} 관측되었습니다. 이 분석 결과를 미계측 wafer로 ` +
    `일반화할 때는 주의가 필요합니다.`
  );
}
