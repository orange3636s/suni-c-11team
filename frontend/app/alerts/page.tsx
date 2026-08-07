"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
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
import { measurementRateDisclaimer } from "@/lib/measurementDisclaimer";
import { niceTicks } from "@/lib/niceTicks";
import { getAlarmSummary, getAlarms, getDatasetSchema, getReliability, saveAlarmsState } from "@/lib/api";
import type {
  AlarmGrade,
  AlarmItem,
  AlarmListResponse,
  AlarmSummaryResponse,
  DatasetSchemaResponse,
  FactorBand,
  MeasurementBiasSummary,
  ReliabilityResponse,
} from "@/types/data";

// 알람 판정 GBDT 전환 (spec §A) -- 관리한계 이탈량이 아니라 부트스트랩
// 앙상블 예측 수율 기준 등급이다. "개선 권고" 등급은 삭제됐다 (spec 알람
// 신뢰도 게이트 §B-1: 정밀도가 무작위 수준과 다르지 않았다).
const GRADE_LABEL: Record<AlarmGrade, string> = { 심각: "심각", 위험: "위험", 주의: "주의" };
const GRADE_RANK: Record<AlarmGrade, number> = { 심각: 3, 위험: 2, 주의: 1 };

function alarmExplainMessage(item: AlarmItem): string {
  return (
    `알람: ${item.lot_wafer_id} · 등급 ${item.grade} · 위험 순위 하위 ${item.risk_percentile.toFixed(1)}%\n` +
    `사유: ${item.reason}\n` +
    "이 알람에 대해 설명해 주세요."
  );
}

const GRADE_CLASS: Record<AlarmGrade, string> = { 심각: "severe", 위험: "danger", 주의: "caution" };

function GradeBadge({ grade }: { grade: AlarmGrade }) {
  return <span className={`severityBadge severityBadge-grade-${GRADE_CLASS[grade] ?? "caution"}`}>{GRADE_LABEL[grade] ?? grade}</span>;
}

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
          {/* 알람 신뢰도 게이트 §D-3 -- AUC 항목에 게이트 통과 여부를 덧붙인다. */}
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

/** 알람 신뢰도 게이트 안내 (spec 알람 신뢰도 게이트 §A-3) -- 빈 목록만
 * 보여주면 기능 오류로 오해하므로, 게이트가 발동한 이유와 신뢰도 등급을
 * 함께 보여준다. `alarms.auc_lower_bound`가 null이면(표본 부족 등)
 * 구체적인 AUC 수치 없이 안내한다. */
function GateBanner({
  alarms,
  reliability,
}: {
  alarms: AlarmListResponse;
  reliability: ReliabilityResponse | null;
}) {
  return (
    <div className="alarmGateBanner">
      <strong>이 데이터셋에서는 알람을 제공하지 않습니다</strong>
      <p>
        {alarms.auc_lower_bound != null ? (
          <>
            알람 순위 품질(AUC)이 {alarms.auc_lower_bound.toFixed(3)}로 기준({alarms.auc_gate_threshold.toFixed(2)})에
            미치지 못합니다.
          </>
        ) : (
          "알람 순위 품질(AUC)을 산출할 수 없습니다 (표본 부족)."
        )}
        <br />
        불량률 변동이 계측 인자로 설명되지 않아 위험 wafer를 구분할 수 없습니다.
      </p>
      {reliability && (
        <p className="alarmGateBannerScore">
          분석 신뢰도 <strong className={`reliabilityGradeText grade-${RELIABILITY_GRADE_CLASS[reliability.grade] ?? "medium"}`}>
            {reliability.grade}
          </strong>{" "}
          ({reliability.total_score}점)
        </p>
      )}
    </div>
  );
}

function lotCompare(a: string | null, b: string | null): number {
  return (a ?? "").localeCompare(b ?? "", undefined, { numeric: true });
}

function alarmTieBreak(a: AlarmItem, b: AlarmItem): number {
  return lotCompare(a.lot_id, b.lot_id) || a.lot_wafer_id.localeCompare(b.lot_wafer_id, undefined, { numeric: true });
}

const ALARM_SORT_OPTIONS: SortOption<AlarmItem>[] = [
  { value: "risk", label: "위험 순위", compare: (a, b) => a.risk_percentile - b.risk_percentile },
  { value: "grade", label: "등급", compare: (a, b) => (GRADE_RANK[b.grade] ?? 0) - (GRADE_RANK[a.grade] ?? 0) },
  { value: "lot_asc", label: "LOT 오름차순", compare: (a, b) => lotCompare(a.lot_id, b.lot_id) },
  { value: "lot_desc", label: "LOT 내림차순", compare: (a, b) => lotCompare(b.lot_id, a.lot_id) },
];

export default function AlertsPage() {
  const { analysisDataset, requestChat } = usePanelState();
  // 사전 알람 결과 상태 유지 (spec: 학습·분석 결과 상태 유지) -- summary/
  // alarms live in the shared context now, so tab switching renders
  // instantly with no refetch, and a reload restores
  // the last-fetched result from the server (spec §3-1: 산점도 좌표가
  // 없는 다른 두 결과와 달리, 이 결과는 애초에 좌표를 담지 않으므로
  // 복원이 항상 완전하다 -- 별도의 배경 재요청이 필요 없다).
  const { alarms: alarmsState, setAlarms: setAlarmsState, hydrated } = useAnalysisState();
  const [trainDataset, setTrainDataset] = useState("train");
  const [evalDataset, setEvalDataset] = useState("test");
  const [gradeFilter, setGradeFilter] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [factorBandIndex, setFactorBandIndex] = useState(0);
  // 종합 신뢰성 등급 (spec §E, 알람 신뢰도 게이트 §D-3) -- 이제 (train,eval)
  // 쌍에 매인 값이다 (AUC가 선택된 eval로의 전이 성능이므로) -- alarmsState와
  // 별도로 두고, 조회 버튼을 누를 때마다 같은 쌍으로 다시 가져온다.
  const [reliability, setReliability] = useState<ReliabilityResponse | null>(null);
  const [reliabilityPanelOpen, setReliabilityPanelOpen] = useState(false);
  // 해석 시 한계의 계측률 문구는 데이터셋마다 다르므로 (spec §E-3: "계측률
  // 하드코딩" -> "실측값") train 데이터셋 스키마를 불러와 반영한다 --
  // 원인 분석 탭의 analysisSchema와 같은 패턴.
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

  const summary = alarmsState?.summary ?? null;
  const alarms = alarmsState?.alarms ?? null;
  // 셀렉터(train/eval) 중 하나라도 표시 중인 결과와 다르면 경고 (spec
  // §4-3/§5-3) -- 자동으로 결과를 지우거나 다시 불러오지 않는다.
  const datasetMismatch = Boolean(
    alarmsState && (alarmsState.trainDataset !== trainDataset || alarmsState.evalDataset !== evalDataset),
  );

  // 심각성 필터는 이미 불러온 alarms.items를 클라이언트에서 거를 뿐, 서버를
  // 다시 부르지 않는다 -- 필터를 바꿔도 "탭 전환 시 API 재호출 금지"와
  // 같은 원칙(불필요한 네트워크 요청 최소화)을 지키면서 즉시 반응한다.
  const gradeFilteredAlarmItems = useMemo(() => {
    const items = alarms?.items ?? [];
    return gradeFilter ? items.filter((item) => item.grade === gradeFilter) : items;
  }, [alarms, gradeFilter]);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [summaryResponse, alarmsResponse, reliabilityResponse] = await Promise.all([
        getAlarmSummary(trainDataset, evalDataset),
        getAlarms(trainDataset, evalDataset),
        // 신뢰성 등급은 이제 (train,eval) 쌍의 함수다 -- 실패해도 알람
        // 자체는 계속 보여줘야 하므로 별도로 처리한다 (spec §E: 배지가
        // 없다고 알람 목록까지 막으면 안 된다).
        getReliability(trainDataset, evalDataset).catch(() => null),
      ]);
      setAlarmsState({
        trainDataset,
        evalDataset,
        createdAt: new Date().toISOString(),
        summary: summaryResponse,
        alarms: alarmsResponse,
      });
      setReliability(reliabilityResponse);
      setFactorBandIndex(0);
      // 조회 성공 직후 저장 (spec §3-4) -- 실패해도 방금 불러온 결과는
      // 이미 화면에 반영되어 있다 (spec §3-2).
      void saveAlarmsState(trainDataset, evalDataset, {
        summary: summaryResponse,
        alarms: alarmsResponse,
      }).catch(() => {});
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "알람 로그를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [trainDataset, evalDataset, setAlarmsState]);

  // 재접속/새로고침, 그리고 탭을 옮겼다 돌아온 경우 모두 이 마운트
  // 이펙트가 처리한다 (spec §4-2/§4-3) -- 셀렉터를 컨텍스트의 결과에 맞춰
  // 한 번만 동기화한다.
  const syncedFromRestore = useRef(false);
  useEffect(() => {
    if (!hydrated || syncedFromRestore.current) return;
    syncedFromRestore.current = true;
    if (!alarmsState) return;
    const timer = window.setTimeout(() => {
      setTrainDataset(alarmsState.trainDataset);
      setEvalDataset(alarmsState.evalDataset);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [hydrated, alarmsState]);

  // Fallback only: nothing restored/cached yet -- load once with
  // whatever the default train/eval selection is, matching this page's
  // original always-auto-load behavior for a genuinely first visit.
  // Never refires just because a selector changes afterward (spec §5-3:
  // "결과를 자동으로 지우지 마라" / "사용자가 실행 버튼을 눌러야 갱신된다"),
  // and never on a tab revisit (checklist §탭 이동 #4).
  const autoLoaded = useRef(false);
  useEffect(() => {
    if (!hydrated || autoLoaded.current) return;
    autoLoaded.current = true;
    if (alarmsState) return;
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, alarmsState]);

  const alarmTable = useTableSearchSort(
    gradeFilteredAlarmItems,
    (item) => `${item.lot_wafer_id} ${item.reason}`,
    ALARM_SORT_OPTIONS,
    "risk",
    alarmTieBreak,
  );

  const factorBands = summary?.factor_bands ?? [];
  const activeFactorBand = factorBands[factorBandIndex] ?? factorBands[0] ?? null;

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
        {/* 탭 상단 캡션 (spec 알람 신뢰도 게이트 §D-1). */}
        <p>
          알람은 예측 수율이 통계적으로 확실하게 낮은 wafer이며, 전체 인자를 종합해 판정합니다.
          <br />
          최적 구간은 원인 분석에서 채택된 권장 관리 범위입니다.
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
          <div className="fieldGroup">
            <span>등급</span>
            <select value={gradeFilter} onChange={(event) => setGradeFilter(event.target.value)}>
              <option value="">전체</option>
              <option value="심각">심각</option>
              <option value="위험">위험</option>
              <option value="주의">주의</option>
            </select>
          </div>
          <button type="button" className="button primary" disabled={loading} onClick={() => void load()} style={{ alignSelf: "end" }}>
            {loading ? "조회 중…" : summary ? "다시 조회" : "조회"}
          </button>
        </div>
        <DatasetMismatchWarning mismatch={datasetMismatch} />
        {error && <p className="errorMessage">{error}</p>}
      </section>

      {summary ? (
        <>
          {/* 집계 카드 3-tile (spec 알람 신뢰도 게이트 §B-4): 알람/정상/판정불가.
              "개선 권고" tile은 삭제됐다 -- 해당 GBDT 등급 자체가 삭제됐다. */}
          <section className="alarmSummaryGrid">
            <AlarmSummaryCard
              label="알람 wafer"
              value={`${summary.counts.alarm}장`}
              aux={pct(summary.counts.alarm, summary.measured_wafers)}
              tone="highlight"
              title="예측 수율 신뢰구간 상한이 하위 15% 분위수 이하 (판정 가능 기준 비율, 신뢰도 게이트 통과 시)"
            />
            <AlarmSummaryCard
              label="정상 wafer"
              value={`${summary.counts.normal}장`}
              aux={pct(summary.counts.normal, summary.measured_wafers)}
              tone="good"
              title="알람이 아닌 판정 가능 wafer (판정 가능 기준 비율)"
            />
            <AlarmSummaryCard
              label="판정불가 (미계측) wafer"
              value={`${summary.counts.unmeasured}장`}
              aux={pct(summary.counts.unmeasured, summary.total_wafers)}
              tone="faint"
              title="선정 인자가 하나도 계측되지 않아 판정할 수 없는 wafer"
            />
          </section>
          <p className="alarmSummaryCaption">전체 {summary.total_wafers.toLocaleString()}장 기준</p>

          <ConceptYieldBandCard summary={summary} />

          {activeFactorBand ? (
            <FactorYieldBandCard
              bands={factorBands}
              activeIndex={factorBands.indexOf(activeFactorBand)}
              onChange={setFactorBandIndex}
            />
          ) : (
            // 강함·보통 등급 인자가 0개인 데이터셋 (spec §E-2, 예: killing_event).
            <section className="resultCard yieldBandCard">
              <div className="yieldBandCardTitle"><h3>인자별 불량률 (%)</h3></div>
              <p className="emptyMessage">강함·보통 등급 인자가 없어 인자별 불량률을 표시할 수 없습니다.</p>
            </section>
          )}

          <UnmeasuredCard summary={summary} />
        </>
      ) : (
        <section className="alarmSummaryGrid">
          {Array.from({ length: 3 }).map((_, index) => (
            <div className="alarmSummaryCard skeleton" key={index}>
              <div className="alarmSkeletonLine label" />
              <div className="alarmSkeletonLine value" />
            </div>
          ))}
        </section>
      )}
      {!summary && <p className="alarmSummaryNote">원인 분석을 실행하면 집계됩니다</p>}

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">ALARMS</span>
            <h2>예측 기반 알람 목록 ({gradeFilteredAlarmItems.length}건)</h2>
          </div>
          {alarms && alarms.auc_gate_passed && gradeFilteredAlarmItems.length > 0 && (
            <TableToolbar
              search={alarmTable.search}
              onSearchChange={alarmTable.setSearch}
              sort={alarmTable.sort}
              onSortChange={alarmTable.setSort}
              sortOptions={ALARM_SORT_OPTIONS}
              placeholder="Wafer ID · 사유 검색"
            />
          )}
        </div>
        {loading && <p className="emptyMessage">불러오는 중…</p>}
        {/* 알람 신뢰도 게이트 (spec 알람 신뢰도 게이트 §A-3) -- 게이트
            미달이면 테이블 자체를 렌더링하지 않고 안내만 보여준다. */}
        {!loading && alarms && !alarms.auc_gate_passed && (
          <GateBanner alarms={alarms} reliability={reliability} />
        )}
        {!loading && alarms && alarms.auc_gate_passed && (
          <>
            {alarms.alarm_share_warning && (
              <p className="alarmShareWarning">
                알람이 {alarms.alarm_total}건으로 평가 대상의{" "}
                {alarms.evaluated_total > 0 ? ((alarms.alarm_total / alarms.evaluated_total) * 100).toFixed(1) : "0"}%입니다.
                공정 전반의 점검이 필요할 수 있습니다.
              </p>
            )}
            {alarms.total === 0 && (
              <p className="emptyMessage">
                기준을 충족하는 알람이 없습니다.
                <br />
                예측 수율이 통계적으로 확실하게 낮은 wafer가 발견되지 않았습니다.
              </p>
            )}
            {alarms.total > 0 && gradeFilteredAlarmItems.length === 0 && (
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
                          <th>위험 순위</th>
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
                {/* ≤767px: 카드형 전환 (spec §B-6) -- 같은 alarmTable.sorted를
                    공유하므로 데이터/정렬/검색 상태가 둘 사이에서 갈라질 일이 없다. */}
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
              알람은 학습 데이터로 학습한 예측 모델이 최종 수율(Y)을 낮게 예측한 wafer입니다.
              예측이 불안정한(신뢰구간이 넓은) wafer는 오히려 더 보수적으로(안전한 쪽으로) 판단해 알람에 포함될 수 있습니다.
              불량의 원인으로 확정된 것은 아니며, 우선 확인 대상을 좁히는 용도입니다.
            </p>
          </>
        )}
      </section>

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          {/* 계측률 하드코딩 -> 실측값 (spec §E-3) -- 데이터셋별로 최대
              4.7배까지 차이 나므로 고정 문구를 쓰지 않는다. */}
          <li>{measurementRateDisclaimer(trainSchema)}</li>
          {summary?.measurement_bias && (
            <li>{describeMeasurementBias(summary.measurement_bias)}</li>
          )}
          <li>알람은 관측 데이터의 통계적 연관성에 기반하며 인과관계를 의미하지 않습니다.</li>
          {/* 임계·기준이 경험값임을 명시 (spec 알람 신뢰도 게이트 §D-2). */}
          <li>알람 판정 임계(예측 수율 하위 15% 분위수)와 신뢰도 기준(AUC 0.65)은 내장 데이터셋에서 설정한 경험값이며 절대 기준이 아닙니다.</li>
          <li>분석 신뢰도가 낮은 데이터셋에서는 알람 정확도를 보장할 수 없습니다.</li>
          <li>판정불가 wafer는 선정 인자가 계측되지 않아 판정 자체가 불가능한 것이며, 정상을 의미하지 않습니다.</li>
          <li>최적 구간은 관리한계(LCL/UCL) 안쪽으로 clamp되며, clamp 결과 구간이 사라지면 해당 인자는 최적 구간 판정에서 제외됩니다.</li>
        </ul>
      </section>
    </DashboardShell>
  );
}

function pct(count: number, total: number): string {
  if (total <= 0) return "0.0%";
  return `${((count / total) * 100).toFixed(1)}%`;
}

function AlarmSummaryCard({
  label,
  value,
  aux,
  tone = "default",
  title,
}: {
  label: string;
  value: string;
  aux?: string;
  tone?: "default" | "highlight" | "good" | "faint";
  title?: string;
}) {
  return (
    <div className={`alarmSummaryCard ${tone !== "default" ? `tone-${tone}` : ""}`} title={title}>
      <span className="alarmSummaryLabel" title={label}>{label}</span>
      <div className="alarmSummaryValueRow">
        <strong className="alarmSummaryValue">{value}</strong>
        {aux && <span className="alarmSummaryAux">{aux}</span>}
      </div>
    </div>
  );
}

/* ===================================================================
   카드①②: 구간별 수율/불량률 밴드. 공통 세그먼트+세로선+수치 레이아웃을
   두 카드가 공유하고, 개념도(균등 폭)와 실제 축(값 기준 폭)만 갈린다.
   =================================================================== */

// "out"/"outrec"/"inrec"는 인자별 불량률(카드②)의 관리한계/권장구간 톤이고,
// "alarm"/"optimalOut"/"normal"은 구간별 평균 수율(카드①, spec 알람 신뢰도
// 게이트 §C-1)의 톤이다 -- 라벨·색이 서로 다르므로 하나로 합치지 않는다.
type BandTone = "out" | "outrec" | "inrec" | "alarm" | "optimalOut" | "normal";
type BandSegment = { tone: BandTone; label: string; widthPct: number };
type BandLine = { tone: "control" | "recommended"; pct: number };
type BandTick = { pct: number; label: string };

const TONE_LABEL: Record<"out" | "outrec" | "inrec", string> = { out: "이탈", outrec: "권장 밖", inrec: "권장 내" };

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
        {lclPct != null && <span className="yieldBandBoundaryLabel" style={{ left: `${lclPct}%` }}>LCL</span>}
        {recLoPct != null && recHiPct != null && (
          <span
            className="yieldBandRecommendedArrow"
            style={{ left: `${(recLoPct + recHiPct) / 2}%` }}
          >
            ◀─ 권장구간 ─▶
          </span>
        )}
        {uclPct != null && <span className="yieldBandBoundaryLabel" style={{ left: `${uclPct}%` }}>UCL</span>}
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
  pctDenominatorLabel,
}: {
  stats: { name: string; tone: BandTone; value: number | null; count: number; pct: number }[];
  arrowTone: "yield" | "defect";
  // 이 %의 분모가 무엇인지 (spec 문구 전수 검토 §A-6-1) -- 요약 카드 상단은
  // 전체 wafer 기준, 이 행은 판정 완료(계측) wafer 기준으로 분모가 서로
  // 다르므로, 같은 count(예: alarm 225장)가 화면마다 다른 %로 보일 때
  // 혼동하지 않도록 분모를 명시한다.
  pctDenominatorLabel?: string;
}) {
  return (
    <div className="yieldBandStats">
      {stats.map((stat, index) => (
        <div key={stat.tone} style={{ display: "contents" }}>
          <div className={`yieldBandStatCol seg-${stat.tone}`}>
            <span className="yieldBandStatName">{stat.name}</span>
            <strong className="yieldBandStatValue">{stat.value != null ? stat.value.toFixed(2) : "-"}</strong>
            <span className="yieldBandStatSub">
              {stat.count.toLocaleString()}장 · {stat.pct.toFixed(1)}%{pctDenominatorLabel ? ` (${pctDenominatorLabel})` : ""}
            </span>
          </div>
          {/* 알람 0건일 때(spec §C-4) 알람 구간 값이 없다 -- 그 구간으로
              들어오는/나가는 화살표는 그리지 않는다("화살표는 두 번째
              구간부터"). 인접한 두 구간 모두 값이 있을 때만 그린다. */}
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

// 구간별 평균 수율 카드 전용 라벨·색 (spec 알람 신뢰도 게이트 §C-1/§C-3) --
// "채택된 권장구간"(SPC 또는 ML) 기준으로 재정의되어 "이탈/권장 밖/권장 내"
// 라는 관리한계 이탈 개념이 더 이상 맞지 않는다. 인자별 불량률(카드②,
// FactorYieldBandCard)은 여전히 각 인자 자신의 관리한계·권장구간 기준이라
// 공유 TONE_LABEL을 그대로 쓴다 -- 이 카드만 별도 톤·라벨을 쓴다.
const CONCEPT_TONE_LABEL: Record<"alarm" | "optimalOut" | "normal", string> = {
  alarm: "알람 구간",
  optimalOut: "최적 구간 밖",
  normal: "정상",
};

/** 카드①: 구간별 평균 수율 -- 개념도. 예측 수율(위험) 낮음→높음 한 방향
 * 순서라 (spec §C-1) 기존의 좌우 대칭 5구간(관리한계 이탈이 양쪽에 있는
 * 그림)이 아니라 알람 구간 → 최적 구간 밖 → 정상 3구간을 순서대로
 * 그린다. 폭은 개념도라 균등하다(실제 값 축이 아니다). 알람 구간은
 * 신뢰도 게이트가 막히면(spec §A) "해당 없음"이 된다 -- 값이 없다고
 * 오류로 처리하지 않는다(spec §C-4: "알람 0건은 정상 상태다"). */
function ConceptYieldBandCard({ summary }: { summary: AlarmSummaryResponse }) {
  const segments: BandSegment[] = [
    { tone: "alarm", label: CONCEPT_TONE_LABEL.alarm, widthPct: 100 / 3 },
    { tone: "optimalOut", label: CONCEPT_TONE_LABEL.optimalOut, widthPct: 100 / 3 },
    { tone: "normal", label: CONCEPT_TONE_LABEL.normal, widthPct: 100 / 3 },
  ];
  const lines: BandLine[] = [
    { tone: "recommended", pct: 100 / 3 },
    { tone: "recommended", pct: 200 / 3 },
  ];

  const measured = summary.measured_wafers;
  const stats = [
    {
      name: CONCEPT_TONE_LABEL.alarm,
      tone: "alarm" as const,
      value: summary.band_counts.alarm > 0 ? summary.band_yield.alarm : null,
      count: summary.band_counts.alarm,
      pct: measured > 0 ? (summary.band_counts.alarm / measured) * 100 : 0,
    },
    {
      name: CONCEPT_TONE_LABEL.optimalOut,
      tone: "optimalOut" as const,
      value: summary.band_yield.out_of_recommended,
      count: summary.band_counts.out_of_recommended,
      pct: measured > 0 ? (summary.band_counts.out_of_recommended / measured) * 100 : 0,
    },
    {
      name: CONCEPT_TONE_LABEL.normal,
      tone: "normal" as const,
      value: summary.band_yield.in_recommended,
      count: summary.band_counts.in_recommended,
      pct: measured > 0 ? (summary.band_counts.in_recommended / measured) * 100 : 0,
    },
  ];

  // 총 격차 (spec §C-3/§C-4) -- 값이 있는 첫 구간과 마지막 구간의 차이.
  // 알람 0건이면 최적 구간 밖 -> 정상 두 구간만으로 계산된다.
  const valuedStats = stats.filter((s) => s.value != null);
  const totalGap =
    valuedStats.length >= 2
      ? (valuedStats[valuedStats.length - 1].value as number) - (valuedStats[0].value as number)
      : null;

  return (
    <section className="resultCard yieldBandCard">
      <div className="yieldBandCaptionRow">
        <div className="yieldBandCardTitle"><h3>구간별 평균 수율 (%)</h3></div>
        <span className="yieldBandCardMeta">{measured.toLocaleString()}장 판정 가능</span>
      </div>
      <BandTrack segments={segments} lines={lines} lclPct={null} uclPct={null} recLoPct={null} recHiPct={null} />
      <BandStatsRow stats={stats} arrowTone="yield" pctDenominatorLabel="판정 가능 기준" />
      {totalGap != null && (
        <p className="yieldBandTotalGap">총 격차 {formatDeltaPP(totalGap)}</p>
      )}
      {/* 최적 구간 밖 하단 안내 (spec §D-2/구간별 평균 수율 §C-1). */}
      <p className="sectionCaption">
        최적 구간 밖은 알람을 제외한 wafer 중 원인 분석에서 채택된 권장구간(SPC 또는 ML) 밖인 경우입니다.
      </p>
    </section>
  );
}

/** 카드②: 인자별 불량률 -- 선택된 인자의 실제 값 축(실측 min~max) 기준.
 * 단조 인자(하한 없음)는 왼쪽 이탈 영역을 그리지 않는다. */
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
    segments.push({ tone: "out", label: "이탈", widthPct: lclPct - leftEdge });
    lines.push({ tone: "control", pct: lclPct });
    leftEdge = lclPct;
  }
  const rightEdge = uclPct ?? 100;
  if (recLoPct != null && recHiPct != null) {
    segments.push({ tone: "outrec", label: "권장 밖", widthPct: recLoPct - leftEdge });
    lines.push({ tone: "recommended", pct: recLoPct });
    segments.push({ tone: "inrec", label: "권장 내", widthPct: recHiPct - recLoPct });
    lines.push({ tone: "recommended", pct: recHiPct });
    leftEdge = recHiPct;
  }
  segments.push({ tone: "outrec", label: "권장 밖", widthPct: rightEdge - leftEdge });
  if (uclPct != null) {
    lines.push({ tone: "control", pct: uclPct });
    segments.push({ tone: "out", label: "이탈", widthPct: 100 - uclPct });
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

/** 카드③: 판정불가 -- 톤을 낮춘다(보조 텍스트만, 큰 숫자 없음). 위 카드들과
 * 분리해 90.24가 "최고 구간"으로 오독되지 않게 한다. 비율은 표기하지
 * 않는다(분모가 다르다). p값은 데이터셋별로 갱신되며, 검정이 불가능하면
 * 그 문장을 생략한다. */
/** 인자별 계측 편향 검정 결과를 사람이 읽을 문장으로 바꾼다 (spec 문구 전수
 * 검토 §A-7) -- 전체 wafer를 뭉뚱그린 이전 집계 검정("R/D 계측이 하나도
 * 없는 wafer" vs "하나 이상 계측된 wafer")은 인자별로 보면 실제로 존재하는
 * 편향을 평균 내 지워버릴 수 있어(실측: train.CSV에서 집계 검정은
 * p=0.74로 "편향 없음"이었지만 선정 인자 5개는 전부 q<0.0001로 유의했다),
 * 백엔드가 이미 인자별로 재검정한 요약(`MeasurementBiasSummary`)을 그대로
 *문장으로 옮긴다 -- 여기서 다시 계산하지 않는다. */
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

function UnmeasuredCard({ summary }: { summary: AlarmSummaryResponse }) {
  const avgYield = summary.band_yield.unmeasured;
  const biasText = describeMeasurementBias(summary.measurement_bias);
  return (
    <section className="unmeasuredCard">
      <p>
        <strong>
          판정불가 {summary.counts.unmeasured.toLocaleString()}장{avgYield != null ? ` · 평균 수율 ${avgYield.toFixed(2)} (%)` : ""}
        </strong>
      </p>
      <p>선정 인자가 계측되지 않아 판정할 수 없는 wafer입니다.</p>
      {biasText && <p>{biasText}</p>}
    </section>
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

function AlarmRow({
  item,
  onExplain,
  explainDisabled,
}: {
  item: AlarmItem;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  return (
    <tr>
      <td className="col-wafer colNoTruncate">{item.lot_wafer_id}</td>
      <td className="numCol">하위 {item.risk_percentile.toFixed(1)}%</td>
      <td className="col-severity colNoTruncate"><GradeBadge grade={item.grade} /></td>
      <td className="alarmReasonCell">{item.reason}</td>
      <td>{item.lot_id ?? "-"}</td>
      <td><ExplainButton onClick={onExplain} disabled={explainDisabled} /></td>
    </tr>
  );
}

/** ≤767px row-to-card conversion (spec §B-6) -- same data as AlarmRow,
 * compressed for a 375px screen. CSS (.tableHideOnMobile / .alarmCardList)
 * decides which of the two renders; both stay mounted so no extra
 * fetch/state is needed. */
function AlarmCard({
  item,
  onExplain,
  explainDisabled,
}: {
  item: AlarmItem;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  return (
    <div className="alarmCard">
      <div className="alarmCardTopRow">
        <span className="alarmCardId">{item.lot_wafer_id}</span>
        <GradeBadge grade={item.grade} />
      </div>
      <div className="alarmCardMeta">
        위험 순위 하위 {item.risk_percentile.toFixed(1)}%
        {item.lot_id && ` · ${item.lot_id}`}
      </div>
      <div className="alarmCardStatsRow">
        <span>{item.reason}</span>
      </div>
      <div className="alarmCardActions">
        <ExplainButton onClick={onExplain} disabled={explainDisabled} />
      </div>
    </div>
  );
}
