"use client";

import { useEffect, useState } from "react";
import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { createModelUpdate, getActiveModel, getCumulativeDataStatus, getModelUpdate, ingestProcessData, type CumulativeDataStatus } from "@/lib/api";

const EMPTY: CumulativeDataStatus = { dataset_version: "-", total_rows: 0, total_lots: 0, labeled_rows: 0, pending_label_rows: 0, conflict_rows: 0, new_labeled_rows_since_active_model: 0, new_lots_since_active_model: 0, retraining_required: false };

export default function TrainingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<CumulativeDataStatus>(EMPTY);
  const [active, setActive] = useState<Record<string, unknown> | null>(null);
  const [message, setMessage] = useState("");
  const [job, setJob] = useState<Record<string, unknown> | null>(null);
  const refresh = async () => { const [status, model] = await Promise.all([getCumulativeDataStatus(), getActiveModel()]); setData(status); setActive(model.active_model); };
  useEffect(() => { const timer = window.setTimeout(() => { void refresh().catch((error: unknown) => setMessage(error instanceof Error ? error.message : "현황을 불러오지 못했습니다.")); }, 0); return () => clearTimeout(timer); }, []);
  useEffect(() => { if (!job?.job_id || ["completed", "failed"].includes(String(job.status))) return; const timer = window.setTimeout(async () => { try { const next = await getModelUpdate(String(job.job_id)); setJob(next); if (["completed", "failed"].includes(String(next.status))) void refresh(); } catch (error) { setMessage(error instanceof Error ? error.message : "갱신 상태를 확인하지 못했습니다."); } }, 2000); return () => clearTimeout(timer); }, [job, job?.job_id, job?.status]);
  async function ingest() { if (!file) return; try { const result = await ingestProcessData(file); setMessage(`입력 완료: 신규 ${result.inserted_rows ?? 0} · Label 갱신 ${result.updated_label_rows ?? 0} · 중복 ${result.duplicate_rows ?? 0} · 충돌 ${result.conflict_rows ?? 0}`); setFile(null); await refresh(); } catch (error) { setMessage(error instanceof Error ? error.message : "데이터 입력에 실패했습니다."); } }
  async function update() { try { setJob(await createModelUpdate()); setMessage(""); } catch (error) { setMessage(error instanceof Error ? error.message : "모델 갱신을 시작하지 못했습니다."); } }
  return <div className="appShell"><Sidebar activeItem="모델 갱신" /><div className="contentShell"><Header /><main className="mainContent uploadPage"><section className="uploadIntro"><span className="eyebrow">머신러닝</span><h1>모델 갱신</h1><p>누적 공정 데이터를 안전하게 Upsert하고, 후보 모델이 Champion보다 우수할 때만 승격합니다.</p></section>
    <section className="resultCard"><h2>누적 데이터 현황</h2><div className="trainingSummaryGrid">{[["전체 Wafer",data.total_rows],["전체 Lot",data.total_lots],["Label 완료",data.labeled_rows],["Label 대기",data.pending_label_rows],["충돌 데이터",data.conflict_rows],["신규 Label",data.new_labeled_rows_since_active_model]].map(([label,value])=><div key={String(label)}><span>{label}</span><strong>{value}</strong></div>)}</div><p>Dataset {data.dataset_version} · 신규 Lot {data.new_lots_since_active_model} · {data.retraining_required ? "모델 갱신 권장" : "갱신 기준 미도달"}</p></section>
    <section className="uploadCard"><h2>데이터 추가</h2><CsvUploadPanel id="ingest-file" file={file} onFileSelect={(selected) => setFile(selected ?? null)} compact /><div className="uploadActions"><button className="button primary" disabled={!file} onClick={ingest}>누적 데이터 입력</button></div></section>
    <section className="resultCard"><h2>현재 활성 모델</h2>{active ? <p>{String(active.active_model_id)} · 승격 {String(active.promoted_at ?? "-")} · {String(active.pipeline_version ?? "-")}</p> : <p>활성 모델이 없습니다. 누적 Label 데이터를 등록한 뒤 모델 갱신을 실행해 주세요.</p>}<button className="button primary" onClick={update} disabled={Boolean(job && !["completed", "failed"].includes(String(job.status)))}>모델 갱신</button>{job && <p role="status">{String(job.stage ?? job.status)} · {String(job.progress ?? 0)}% {job.promotion_result ? `· ${String(job.promotion_result)}` : ""}</p>}</section>
    {message && <p className="messageBox" role="status">{message}</p>}
  </main></div></div>;
}
