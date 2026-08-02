"use client";

import { useMemo, useState } from "react";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import StatusBadge from "@/components/StatusBadge";
import CsvUploadPanel from "@/components/CsvUploadPanel";
import {
  analyzeRelationships,
} from "@/lib/api";
import type {
  CategoricalStatistic,
  LotCauseItem,
  LotFeatureImportanceItem,
  LotWaferItem,
  RelationshipAnalysisResponse,
  RelationshipFeature,
} from "@/types/data";

type WorkspaceTab = "yield" | "lot" | "wafer" | "relationships";
type LotFeatureGroup = "all" | "r" | "d" | "config";

const TABS: Array<[WorkspaceTab, string]> = [
  ["yield", "Y 수율 분석"],
  ["lot", "Lot별 원인"],
  ["wafer", "Wafer 상세"],
  ["relationships", "공정 관계"],
];
const ANALYSIS_OPTIONS = { max_rows: 500, top_n: 20, per_wafer_top_n: 8 };

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("ko-KR", { maximumFractionDigits: digits })
    : "-";
}

function waferKey(wafer: LotWaferItem): string {
  return String(wafer.identifier ?? wafer.wafer_id ?? wafer.wafer_slot ?? "");
}

function compactFeatureName(feature: string | null | undefined): string {
  if (!feature) return "데이터 없음";
  return feature.replace(/_frequency$/i, "").replace(/_freq$/i, "");
}

export default function RootCausePage() {
  const [file, setFile] = useState<File | null>(null);
  const [correlation, setCorrelation] = useState<"pearson" | "spearman">("spearman");
  const [result, setResult] = useState<RelationshipAnalysisResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<WorkspaceTab>("yield");
  const [selectedLotId, setSelectedLotId] = useState<string | null>(null);
  const [selectedWaferId, setSelectedWaferId] = useState<string | null>(null);
  const [lotGroup, setLotGroup] = useState<LotFeatureGroup>("all");
  const [lotSearch, setLotSearch] = useState("");
  const [stepFilter, setStepFilter] = useState<number | "all">("all");

  const lots = useMemo(() => result?.lot_analysis?.lots ?? [], [result]);
  const riskLots = useMemo(
    () => [...lots]
      .sort((a, b) => (a.ranking_yield ?? Number.POSITIVE_INFINITY) - (b.ranking_yield ?? Number.POSITIVE_INFINITY))
      .slice(0, 20),
    [lots],
  );
  const selectedLot = useMemo(
    () => lots.find((lot) => lot.lot_id === selectedLotId) ?? riskLots[0] ?? lots[0] ?? null,
    [lots, riskLots, selectedLotId],
  );
  const selectedWafer = useMemo(
    () => selectedLot?.wafer_list.find((wafer) => waferKey(wafer) === selectedWaferId) ?? null,
    [selectedLot, selectedWaferId],
  );
  const dangerWafers = useMemo(
    () => lots.flatMap((lot) => lot.wafer_list.map((wafer) => ({ lot, wafer })))
      .filter(({ wafer }) => wafer.risk_level === "danger")
      .sort((a, b) => (a.wafer.predicted_yield ?? a.wafer.predicted_value ?? 100) - (b.wafer.predicted_yield ?? b.wafer.predicted_value ?? 100)),
    [lots],
  );
  const shownLots = useMemo(() => {
    const query = lotSearch.trim().toLowerCase();
    return query ? lots.filter((lot) => lot.lot_id.toLowerCase().includes(query)) : lots;
  }, [lots, lotSearch]);

  function setQuerySelection(lotId: string, waferId?: string) {
    const params = new URLSearchParams(window.location.search);
    params.set("lot_id", lotId);
    if (waferId) params.set("wafer_id", waferId);
    else params.delete("wafer_id");
    window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
  }

  function selectLot(lotId: string) {
    setSelectedLotId(lotId);
    setSelectedWaferId(null);
    setQuerySelection(lotId);
  }

  function openWafer(lot: LotCauseItem, wafer: LotWaferItem) {
    const key = waferKey(wafer);
    setSelectedLotId(lot.lot_id);
    setSelectedWaferId(key);
    setTab("wafer");
    setQuerySelection(lot.lot_id, key);
  }

  async function runAnalysis() {
    if (!file) return;
    setRunning(true);
    setError(null);
    try {
      const response = await analyzeRelationships(
        file,
        ANALYSIS_OPTIONS,
        correlation,
        "wafer_observed_only",
        { warning_threshold: 90, danger_threshold: 85 },
      );
      setResult(response);
      setSelectedLotId(null);
      setSelectedWaferId(null);
      setTab("yield");
      window.history.replaceState({}, "", window.location.pathname);
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : "원인 분석 중 오류가 발생했습니다.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="appShell">
      <Sidebar activeItem="원인 분석" />
      <div className="mainArea">
        <Header />
        <main className="pageContent rcPage">
          <section className="pageHeading compactHeading">
            <div><span className="eyebrow">ROOT CAUSE WORKSPACE</span><h1>원인 분석</h1><p>최신 Y 모델로 Lot 위험, Wafer 기여도와 공정 관계를 추적합니다.</p></div>
            {result && <StatusBadge label={`${result.target} · ${lots.length} Lots`} tone="success" />}
          </section>

          <section className="resultCard rcControlBar" aria-label="분석 조건">
            <CsvUploadPanel id="root-cause-file" file={file} onFileSelect={(selected) => setFile(selected ?? null)} compact title="CSV 파일을 선택해 주세요." description="활성 Y 모델로 원인 분석을 실행합니다." />
            <label className="fieldGroup rcSmallField"><span>상관</span><select value={correlation} onChange={(event) => setCorrelation(event.target.value as typeof correlation)}><option value="spearman">Spearman</option><option value="pearson">Pearson</option></select></label>
            <button className="button primary" type="button" disabled={!file || running} onClick={() => void runAnalysis()}>{running ? "분석 중…" : "원인 분석 실행"}</button>
          </section>

          {error && <div className="errorMessage" role="alert">{error}</div>}

          <nav className="workspaceTabs rcTabs" aria-label="원인 분석 워크스페이스">
            {TABS.map(([value, label]) => <button key={value} type="button" className={tab === value ? "active" : ""} onClick={() => setTab(value)}>{label}</button>)}
          </nav>

          {!result ? (
            <section className="resultCard rcEmpty"><h2>분석할 데이터를 선택해 주세요</h2><p>서버에 저장된 최신 Y 모델로 공정 원인을 분석합니다.</p></section>
          ) : tab === "yield" ? (
            <TargetView result={result} dangerWafers={dangerWafers} onWaferClick={openWafer} />
          ) : tab === "lot" ? (
            <LotView lots={shownLots} riskLots={riskLots} selectedLot={selectedLot} search={lotSearch} onSearch={setLotSearch} onSelect={selectLot} group={lotGroup} onGroup={setLotGroup} onWaferClick={openWafer} />
          ) : tab === "wafer" ? (
            <WaferView lots={lots} selectedLot={selectedLot} selectedWafer={selectedWafer} onSelectLot={selectLot} onSelectWafer={openWafer} />
          ) : (
            <RelationshipsView result={result} stepFilter={stepFilter} onStepFilter={setStepFilter} />
          )}

        </main>
      </div>
    </div>
  );
}

function TargetView({ result, dangerWafers, onWaferClick }: { result: RelationshipAnalysisResponse; dangerWafers: Array<{ lot: LotCauseItem; wafer: LotWaferItem }>; onWaferClick: (lot: LotCauseItem, wafer: LotWaferItem) => void }) {
  const explanation = result.explanation;
  const risk = result.analysis_result?.risk;
  return <div className="rcGrid">
    <section className="resultCard rcSpan2"><div className="sectionHeader"><div><span className="sectionLabel">Final yield</span><h2>최종 수율 Y 원인 분석</h2></div><StatusBadge label="정답 레이블: Y" tone="neutral" /></div><p className="microcopy">활성 Y 모델의 기여도를 직접 사용합니다. 음수 기여는 수율 악화, 양수 기여는 수율 개선을 뜻합니다.</p></section>
    <section className="resultCard"><span className="sectionLabel">Risk mix</span><h2>Wafer 위험 분포</h2><div className="rcRiskCounts"><span className="normal">정상 <strong>{risk?.normal_count ?? 0}</strong></span><span className="warning">주의 <strong>{risk?.warning_count ?? 0}</strong></span><span className="danger">위험 <strong>{risk?.critical_count ?? 0}</strong></span></div></section>
    <section className="resultCard"><span className="sectionLabel">SHAP coverage</span><h2>설명 범위</h2><strong className="rcHeroNumber">{formatNumber(explanation?.analysis_summary.analyzed_rows, 0)}</strong><p>전체 {formatNumber(explanation?.analysis_summary.total_rows, 0)} Wafer · {explanation?.explanation_method ?? "-"}</p></section>
    <section className="resultCard rcSpan2"><div className="sectionHeader"><div><span className="sectionLabel">Global importance</span><h2>상위 영향 Feature</h2></div></div><FeatureTable features={(explanation?.global_importance ?? []).slice(0, 12).map((item) => ({ feature: item.feature, display_name: compactFeatureName(item.feature), group: item.parameter_type, score: item.mean_abs_shap, direction: item.direction, valid_count: null } as RelationshipFeature))} /></section>
    <section className="resultCard rcSpan2"><span className="sectionLabel">Danger wafers</span><h2>위험 Wafer Top 5</h2>{dangerWafers.length ? <div className="rcChipList">{dangerWafers.slice(0, 5).map(({ lot, wafer }) => <button type="button" key={`${lot.lot_id}-${waferKey(wafer)}`} onClick={() => onWaferClick(lot, wafer)}><strong>{lot.lot_id}</strong><span>{waferKey(wafer)} · {formatNumber(wafer.predicted_yield ?? wafer.predicted_value)}%</span></button>)}</div> : <p className="emptyMessage">위험 Wafer가 없습니다.</p>}</section>
  </div>;
}

function LotView({ lots, riskLots, selectedLot, search, onSearch, onSelect, group, onGroup, onWaferClick }: { lots: LotCauseItem[]; riskLots: LotCauseItem[]; selectedLot: LotCauseItem | null; search: string; onSearch: (value: string) => void; onSelect: (lotId: string) => void; group: LotFeatureGroup; onGroup: (group: LotFeatureGroup) => void; onWaferClick: (lot: LotCauseItem, wafer: LotWaferItem) => void }) {
  const lowSampleConfigs = selectedLot?.low_sample_config ?? [];
  const ranked = selectedLot?.feature_importance[group] ?? [];
  const pareto = selectedLot?.pareto[group] ?? [];
  const stepTotals = Object.entries(ranked.reduce<Record<string, number>>((accumulator, item) => {
    const step = item.step || "unknown";
    accumulator[step] = (accumulator[step] ?? 0) + Math.max(item.adverse_contribution ?? 0, 0);
    return accumulator;
  }, {})).sort((left, right) => right[1] - left[1]).slice(0, 10);
  const riskWafers = [...(selectedLot?.wafer_list ?? [])]
    .filter((wafer) => wafer.risk_level === "danger" || wafer.risk_level === "warning")
    .sort((left, right) => (left.ranking_yield ?? left.predicted_yield ?? left.predicted_value ?? 100) - (right.ranking_yield ?? right.predicted_yield ?? right.predicted_value ?? 100))
    .slice(0, 5);
  const failBits = Object.entries(selectedLot?.fail_bit_count_averages ?? {});
  return <div className="rcLotLayout">
    <aside className="resultCard rcLotList"><div className="sectionHeader"><div><span className="sectionLabel">Lots · 수율 오름차순</span><h2>Lot 선택</h2></div><span>{lots.length}</span></div><input className="rcSearch" value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Lot 검색" aria-label="Lot 검색" /><div className="rcLotScroll">{lots.map((lot) => <button type="button" key={lot.lot_id} className={selectedLot?.lot_id === lot.lot_id ? "active" : ""} onClick={() => onSelect(lot.lot_id)}><span><strong>{lot.lot_id}</strong><small>{lot.wafer_count ?? 0} wafers · {lot.ranking_basis === "actual_y" ? "실제" : "예측"}</small></span><b>{formatNumber(lot.ranking_yield)}%</b></button>)}</div></aside>
    <div className="rcLotMain">
      <section className="resultCard"><div className="sectionHeader"><div><span className="sectionLabel">Bottom 20 risk lots + selection</span><h2>위험 Lot 수율 분포</h2></div>{selectedLot && <StatusBadge label={`${selectedLot.lot_id} · ${selectedLot.ranking_basis === "actual_y" ? "실제 Y 기준" : "예측 Y 기준"}`} tone="warning" />}</div><LotBoxPlot lots={riskLots} selectedLot={selectedLot} onSelect={onSelect} /></section>
      {selectedLot ? <>
        <section className="resultCard"><div className="rcMetricStrip"><div><span>실제 평균 Y</span><strong>{formatNumber(selectedLot.average_actual_yield)}%</strong></div><div><span>예측 평균 Y</span><strong>{formatNumber(selectedLot.average_predicted_yield)}%</strong></div><div><span>전체 대비</span><strong>{formatNumber(selectedLot.difference_from_overall)}%p</strong></div><div><span>수율 손실</span><strong>{formatNumber(selectedLot.yield_loss)}%p</strong></div><div><span>최소 / 최대</span><strong>{formatNumber(selectedLot.minimum_yield)} / {formatNumber(selectedLot.maximum_yield)}</strong></div><div><span>표준편차</span><strong>{formatNumber(selectedLot.yield_standard_deviation)}</strong></div><div><span>Critical / Warning</span><strong>{selectedLot.critical_wafer_count ?? 0} / {selectedLot.warning_wafer_count ?? 0}</strong></div><div><span>주요 Fail</span><strong>{selectedLot.top_failure_target ?? "-"}</strong></div></div>{failBits.length > 0 && <div className="rcChipList rcBitSummary">{failBits.map(([name, value]) => <span key={name}><strong>{name}</strong> 평균 {formatNumber(value)}</span>)}</div>}</section>
        <section className="resultCard"><div className="sectionHeader"><div><span className="sectionLabel">Lot diagnosis</span><h2>{selectedLot.lot_id} 원인 순위</h2></div><div className="rcSegmented">{(["all", "r", "d", "config"] as LotFeatureGroup[]).map((value) => <button key={value} type="button" className={group === value ? "active" : ""} onClick={() => onGroup(value)}>{value === "config" ? "Config" : value.toUpperCase()}</button>)}</div></div>{lowSampleConfigs.length > 0 && <div className="rcLowSample"><StatusBadge label={`표본 부족 Config ${lowSampleConfigs.length}개`} tone="warning" /><span>sample_count &lt; 5는 공식 순위·Pareto·유의성 결론에서 제외했습니다.</span><div className="tableWrap"><table><thead><tr><th>Step</th><th>Config</th><th>Lot 평균 Y</th><th>동일 Config 평균 Y</th><th>전체 평균 Y</th><th>수율 차이</th><th>평균 / 합계 기여</th><th>판정</th></tr></thead><tbody>{lowSampleConfigs.map((item) => <tr key={item.feature}><td>{item.step}</td><td>{compactFeatureName(item.display_name || item.feature)}</td><td>{formatNumber(item.lot_mean_value)}</td><td>{formatNumber(item.overall_mean_value)}</td><td>{formatNumber(item.overall_yield)}</td><td>{formatNumber(item.mean_difference)}%p</td><td>{formatNumber(item.adverse_contribution, 4)} / {formatNumber(item.total_adverse_contribution, 4)}</td><td><StatusBadge label="표본 부족" tone="warning" /></td></tr>)}</tbody></table></div></div>}<ContributionBars features={ranked} /><LotFeatureTable features={ranked} /><ParetoBars items={pareto} />{stepTotals.length > 0 && <div className="rcPareto"><h3>주요 Step Top 10</h3>{stepTotals.map(([step, value]) => <div key={step}><span>{step}</span><i><b style={{ width: `${Math.max(2, (value / stepTotals[0][1]) * 100)}%` }} /></i><em>{formatNumber(value, 4)}</em></div>)}</div>}</section>
        <section className="resultCard"><span className="sectionLabel">Wafer drill-down</span><h2>위험 Wafer Top 5</h2>{riskWafers.length ? <div className="rcWaferRows">{riskWafers.map((wafer) => <button type="button" key={waferKey(wafer)} onClick={() => onWaferClick(selectedLot, wafer)}><StatusBadge label={wafer.risk_level ?? "unknown"} tone={wafer.risk_level === "danger" ? "danger" : "warning"} /><strong>{waferKey(wafer)}</strong><span>{formatNumber(wafer.predicted_yield ?? wafer.predicted_value)}%</span><small>{compactFeatureName(wafer.top_feature)}</small></button>)}</div> : <p className="emptyMessage">위험 또는 주의 Wafer가 없습니다.</p>}</section>
      </> : <section className="resultCard"><p className="emptyMessage">Lot을 선택해 주세요.</p></section>}
    </div>
  </div>;
}

function WaferView({ lots, selectedLot, selectedWafer, onSelectLot, onSelectWafer }: { lots: LotCauseItem[]; selectedLot: LotCauseItem | null; selectedWafer: LotWaferItem | null; onSelectLot: (lotId: string) => void; onSelectWafer: (lot: LotCauseItem, wafer: LotWaferItem) => void }) {
  return <div className="rcGrid"><section className="resultCard"><label className="fieldGroup"><span>Lot</span><select value={selectedLot?.lot_id ?? ""} onChange={(event) => onSelectLot(event.target.value)}>{lots.map((lot) => <option key={lot.lot_id}>{lot.lot_id}</option>)}</select></label>{selectedLot && <div className="rcWaferNav">{selectedLot.wafer_list.map((wafer) => <button key={waferKey(wafer)} className={selectedWafer && waferKey(selectedWafer) === waferKey(wafer) ? "active" : ""} type="button" onClick={() => onSelectWafer(selectedLot, wafer)}>{waferKey(wafer)}<span>{formatNumber(wafer.predicted_yield ?? wafer.predicted_value)}%</span></button>)}</div>}</section><section className="resultCard rcSpan3">{selectedWafer ? <><div className="sectionHeader"><div><span className="sectionLabel">Wafer detail</span><h2>{waferKey(selectedWafer)}</h2></div><StatusBadge label={selectedWafer.risk_level ?? "unknown"} tone={selectedWafer.risk_level === "danger" ? "danger" : selectedWafer.risk_level === "warning" ? "warning" : "success"} /></div><div className="rcMetricStrip"><div><span>예측 수율</span><strong>{formatNumber(selectedWafer.predicted_yield ?? selectedWafer.predicted_value)}%</strong></div><div><span>신뢰도</span><strong>{formatNumber(selectedWafer.confidence)}</strong></div><div><span>Top Step</span><strong>{selectedWafer.top_step ?? "-"}</strong></div><div><span>Top Config</span><strong>{compactFeatureName(selectedWafer.top_config)}</strong></div></div><div className="rcCauseCallout"><span>가장 큰 불리 기여</span><strong>{compactFeatureName(selectedWafer.top_feature)}</strong><p>이 항목은 선택 Wafer의 국소 SHAP 기여를 요약합니다. 인과관계로 단정하지 않습니다.</p></div></> : <p className="emptyMessage">왼쪽에서 Wafer를 선택하거나 위험 Wafer 바로가기를 눌러 주세요.</p>}</section></div>;
}

function RelationshipsView({ result, stepFilter, onStepFilter }: { result: RelationshipAnalysisResponse; stepFilter: number | "all"; onStepFilter: (value: number | "all") => void }) {
  const paths = result.relationship_paths.filter((path) => stepFilter === "all" || path.step === stepFilter);
  const rankings = result.rankings.shap.all.length ? result.rankings.shap.all : result.rankings.correlation.all;
  const configStatistics = result.statistics.categorical.filter((item) => {
    if (stepFilter === "all") return true;
    return Number(item.feature.match(/^Step(\d+)/i)?.[1]) === stepFilter;
  });
  return <div className="rcGrid">
    <section className="resultCard rcSpan2"><div className="sectionHeader"><div><span className="sectionLabel">R · D · Config</span><h2>공정 영향 순위</h2></div><select value={stepFilter} onChange={(event) => onStepFilter(event.target.value === "all" ? "all" : Number(event.target.value))}><option value="all">전체 Step</option>{result.available_steps.map((step) => <option key={step} value={step}>Step {step}</option>)}</select></div><FeatureTable features={rankings.filter((item) => stepFilter === "all" || item.step === stepFilter).slice(0, 20)} /></section>
    <section className="resultCard rcSpan2"><span className="sectionLabel">Relationship paths</span><h2>공정 관계 경로</h2>{paths.length ? <div className="rcPathList">{paths.slice(0, 20).map((path) => <article key={`${path.rank}-${path.step}-${path.defect}`}><span>Step {path.step}</span><strong>{path.response ?? "R"} → {path.defect} → {result.target}</strong><small>{path.equipment ? `Config ${path.equipment} · ` : ""}{path.interpretation}</small><StatusBadge label={path.confidence} tone={path.confidence === "sufficient" ? "success" : path.confidence === "caution" ? "warning" : "neutral"} /></article>)}</div> : <p className="emptyMessage">선택 조건의 관계 경로가 없습니다.</p>}</section>
    <section className="resultCard rcSpan4"><div className="sectionHeader"><div><span className="sectionLabel">Categorical statistics</span><h2>Config 관계·통계</h2></div><StatusBadge label="최소 표본 n=5" tone="neutral" /></div><p className="microcopy">원본 Config Category에 범주형 검정을 적용합니다. Frequency Encoding 숫자에는 Pearson/Spearman을 적용하지 않습니다.</p><ConfigStatistics items={configStatistics} methods={result.statistics.methods} /></section>
    {result.caveats.length > 0 && <section className="resultCard rcSpan4"><span className="sectionLabel">Caveats</span><h2>해석 주의사항</h2><ul className="rcNotes">{result.caveats.map((note) => <li key={note}>{note}</li>)}</ul></section>}
  </div>;
}

function ConfigStatistics({ items, methods }: { items: CategoricalStatistic[]; methods: string[] }) {
  if (!items.length) return <p className="emptyMessage">선택 조건에서 분석 가능한 Config 컬럼이 없습니다.</p>;
  return <div className="rcConfigStats">{items.map((item) => <article key={item.feature}>
    <div className="sectionHeader"><div><h3>{item.feature}</h3><p>사용 분석법: {methods.filter((method) => ["anova", "welch_anova", "kruskal", "fdr", "effect_size"].includes(method)).join(" · ")}</p></div><div className="rcChipList"><span>Category {item.category_count}</span><span>유효 {item.eligible_category_count ?? 0}</span>{Boolean(item.insufficient_category_count) && <StatusBadge label={`표본 부족 ${item.insufficient_category_count}`} tone="warning" />}</div></div>
    <div className="tableWrap"><table><thead><tr><th>검정</th><th>Statistic</th><th>p-value</th><th>FDR</th><th>Effect size</th><th>사용 표본</th><th>제외 표본</th></tr></thead><tbody>{([['ANOVA', item.anova], ['Welch ANOVA', item.welch_anova], ['Kruskal', item.kruskal]] as const).map(([name, test]) => <tr key={name}><th>{name}</th><td>{formatNumber(test.statistic, 5)}</td><td>{formatNumber(test.p_value, 5)}</td><td>{formatNumber(test.fdr_p_value, 5)}</td><td>{formatNumber(item.effect_size, 5)}</td><td>{item.valid_count}</td><td>{item.excluded_low_sample_count ?? 0}</td></tr>)}</tbody></table></div>
    <div className="tableWrap"><table><thead><tr><th>Config Category</th><th>n</th><th>평균</th><th>중앙값</th><th>Q1 / Q3</th><th>전체 대비</th><th>판정</th></tr></thead><tbody>{(item.category_summary ?? []).map((category) => <tr key={category.category}><td>{category.category}</td><td>{category.count}</td><td>{formatNumber(category.mean)}</td><td>{formatNumber(category.median)}</td><td>{formatNumber(category.q1)} / {formatNumber(category.q3)}</td><td>{formatNumber(category.difference_from_overall)}%p</td><td>{category.sample_warning ? <StatusBadge label="표본 부족" tone="warning" /> : <StatusBadge label="통계 포함" tone="success" />}</td></tr>)}</tbody></table></div>
  </article>)}</div>;
}

function FeatureTable({ features }: { features: RelationshipFeature[] }) {
  return features.length ? <div className="tableWrap"><table><thead><tr><th>#</th><th>Feature</th><th>그룹</th><th>점수</th><th>방향</th><th>n</th></tr></thead><tbody>{features.map((item, index) => <tr key={`${item.feature}-${index}`}><td>{item.rank ?? index + 1}</td><td>{item.display_name || compactFeatureName(item.feature)}</td><td>{item.group === "EQ" ? "Config" : item.group}</td><td>{formatNumber(item.score ?? item.mean_abs_shap, 4)}</td><td>{item.direction}</td><td>{formatNumber(item.valid_count, 0)}</td></tr>)}</tbody></table></div> : <p className="emptyMessage">표시할 Feature가 없습니다.</p>;
}

function LotFeatureTable({ features }: { features: LotFeatureImportanceItem[] }) {
  return features.length ? <div className="tableWrap"><table><thead><tr><th>#</th><th>Step</th><th>Feature / Config</th><th>Lot / 비교 평균</th><th>평균 불리 기여</th><th>합계 불리 기여</th><th>개선 기여</th><th>표본 / Coverage</th><th>관련 Target</th></tr></thead><tbody>{features.slice(0, 20).map((item, index) => <tr key={item.feature}><td>{index + 1}</td><td>{item.step}</td><td>{compactFeatureName(item.display_name || item.feature)}</td><td>{formatNumber(item.lot_mean_value)} / {formatNumber(item.overall_mean_value)}<small className="rcCellMeta">{item.group === "Config" ? `전체 Y ${formatNumber(item.overall_yield)} · 수율차 ${formatNumber(item.mean_difference)}%p` : `Δ ${formatNumber(item.mean_difference)}`}</small></td><td>{formatNumber(item.adverse_contribution, 4)}</td><td title="선택 Lot 내 Wafer 기여 합계">{formatNumber(item.total_adverse_contribution, 4)}</td><td>{formatNumber(item.improvement_contribution, 4)}<small className="rcCellMeta">합계 {formatNumber(item.total_improvement_contribution, 4)}</small></td><td>{item.sample_count}<small className="rcCellMeta">{formatNumber(item.coverage * 100, 1)}%</small></td><td>{item.related_failure_target ?? "-"}<small className="rcCellMeta">{item.related_fail_bit_count_target ? `${item.related_fail_bit_count_target} ${formatNumber(item.related_fail_bit_count_average)}` : ""}</small></td></tr>)}</tbody></table></div> : <p className="emptyMessage">공식 순위에 포함할 Feature가 없습니다.</p>;
}

function ContributionBars({ features }: { features: LotFeatureImportanceItem[] }) {
  const items = features.slice(0, 12);
  const maximum = Math.max(...items.map((item) => Math.max(item.adverse_contribution ?? 0, 0)), 0);
  if (!items.length || maximum <= 0) return null;
  return <div className="rcPareto"><h3>평균 수율 악화 기여</h3>{items.map((item) => <div key={item.feature}><span>{compactFeatureName(item.display_name || item.feature)}</span><i><b style={{ width: `${Math.max(2, (Math.max(item.adverse_contribution, 0) / maximum) * 100)}%` }} /></i><em>{formatNumber(item.adverse_contribution, 4)}</em></div>)}</div>;
}

function ParetoBars({ items }: { items: LotCauseItem["pareto"]["all"] }) {
  if (!items.length) return null;
  return <div className="rcPareto"><h3>Pareto 기여</h3>{items.slice(0, 12).map((item) => <div key={item.feature}><span>{compactFeatureName(item.display_name || item.feature)}</span><i><b style={{ width: `${Math.max(2, Math.min(100, item.share * 100))}%` }} /></i><em>{formatNumber(item.cumulative_share * 100, 1)}%</em></div>)}</div>;
}

function quartiles(values: number[]): [number, number, number] {
  if (!values.length) return [0, 0, 0];
  const sorted = [...values].sort((a, b) => a - b);
  const at = (p: number) => { const position = (sorted.length - 1) * p; const low = Math.floor(position); const high = Math.ceil(position); return sorted[low] + (sorted[high] - sorted[low]) * (position - low); };
  return [at(0.25), at(0.5), at(0.75)];
}

function LotBoxPlot({ lots, selectedLot, onSelect }: { lots: LotCauseItem[]; selectedLot: LotCauseItem | null; onSelect: (lotId: string) => void }) {
  const plotLots = selectedLot && !lots.some((lot) => lot.lot_id === selectedLot.lot_id) ? [...lots, selectedLot] : lots;
  if (!plotLots.length) return <p className="emptyMessage">Lot 분포 데이터가 없습니다.</p>;
  const width = Math.max(720, plotLots.length * 42 + 80);
  const y = (value: number) => 20 + (100 - Math.max(0, Math.min(100, value))) * 2.2;
  return <div className="rcPlotScroll"><svg className="rcBoxPlot" viewBox={`0 0 ${width} 275`} role="img" aria-label="위험 Lot 수율 박스 플롯">
    {[0, 25, 50, 75, 100].map((tick) => <g key={tick}><line x1="52" x2={width - 10} y1={y(tick)} y2={y(tick)} /><text x="45" y={y(tick) + 4}>{tick}</text></g>)}
    {plotLots.map((lot, index) => {
      const values = lot.wafer_list.map((wafer) => wafer.ranking_yield ?? wafer.actual_yield ?? wafer.predicted_yield ?? wafer.predicted_value).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
      const fallback = lot.ranking_yield ?? lot.average_predicted_yield ?? 0;
      const samples = values.length ? values : [fallback];
      const [q1, median, q3] = quartiles(samples);
      const iqr = q3 - q1;
      const lowerFence = q1 - 1.5 * iqr;
      const upperFence = q3 + 1.5 * iqr;
      const inliers = samples.filter((value) => value >= lowerFence && value <= upperFence);
      const whiskerMin = Math.min(...(inliers.length ? inliers : samples));
      const whiskerMax = Math.max(...(inliers.length ? inliers : samples));
      const outliers = samples.filter((value) => value < lowerFence || value > upperFence);
      const x = 72 + index * 42;
      const active = selectedLot?.lot_id === lot.lot_id;
      return <g key={lot.lot_id} className={active ? "active" : ""} onClick={() => onSelect(lot.lot_id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onSelect(lot.lot_id); }} role="button" tabIndex={0}><line className="whisker" x1={x} x2={x} y1={y(whiskerMax)} y2={y(whiskerMin)} /><rect x={x - 10} y={y(q3)} width="20" height={Math.max(2, y(q1) - y(q3))} rx="3" /><line className="median" x1={x - 10} x2={x + 10} y1={y(median)} y2={y(median)} />{outliers.map((value, outlierIndex) => <circle className="outlier" key={`${value}-${outlierIndex}`} cx={x} cy={y(value)} r="2.5" />)}<text className="lotLabel" transform={`translate(${x + 3} 266) rotate(-55)`}>{lot.lot_id}</text></g>;
    })}
  </svg></div>;
}
