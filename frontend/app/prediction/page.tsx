"use client";
import { useState } from "react";
import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { downloadPredictions, predictCsv } from "@/lib/api";
import type { PredictionResponse } from "@/types/data";

export default function PredictionPage() {
  const [file, setFile] = useState<File | null>(null); const [result, setResult] = useState<PredictionResponse | null>(null); const [error, setError] = useState(""); const [running, setRunning] = useState(false);
  const thresholds = { warning_threshold: 90, danger_threshold: 85 };
  async function run() { if (!file) return; setRunning(true); setError(""); setResult(null); try { setResult(await predictCsv(file, null, thresholds)); } catch (e) { setError(e instanceof Error ? e.message : "수율 예측에 실패했습니다."); } finally { setRunning(false); } }
  async function download() { if (!file) return; try { const blob=await downloadPredictions(file,null,thresholds); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url;a.download="current_predictions.csv";a.click();URL.revokeObjectURL(url); } catch(e) { setError(e instanceof Error ? e.message : "다운로드에 실패했습니다."); } }
  return <div className="appShell"><Sidebar activeItem="수율 예측"/><div className="contentShell"><Header/><main className="mainContent uploadPage"><section className="uploadIntro"><span className="eyebrow">PREDICTION</span><h1>수율 예측</h1><p>현재 활성 Champion 모델로 선택한 공정 데이터의 수율을 예측합니다.</p></section><section className="uploadCard"><h2>예측 데이터 선택</h2><CsvUploadPanel id="prediction-file" file={file} onFileSelect={(item)=>setFile(item??null)} compact/><div className="uploadActions"><button className="button primary" disabled={!file||running} onClick={run}>{running?"수율 예측 중…":"수율 예측 실행"}</button>{result&&<button className="button secondary" onClick={download}>현재 결과 다운로드</button>}</div>{error&&<p className="errorMessage" role="alert">{error}</p>}</section>{result&&<section className="resultCard"><h2>현재 예측 결과</h2><div className="trainingSummaryGrid"><div><span>평균 수율</span><strong>{result.summary.average_prediction.toFixed(2)}%</strong></div><div><span>전체 Wafer</span><strong>{result.summary.total_rows}</strong></div><div><span>Critical</span><strong>{result.summary.danger_count}</strong></div><div><span>Warning</span><strong>{result.summary.warning_count}</strong></div></div><div className="tableWrap"><table><thead><tr><th>Wafer</th><th>예측 수율</th><th>위험도</th></tr></thead><tbody>{result.predictions.map((row,index)=><tr key={String(row[result.identifier_column]??index)}><td>{String(row[result.identifier_column]??"-")}</td><td>{typeof row.predicted_Y==="number"?`${row.predicted_Y.toFixed(2)}%`:"-"}</td><td>{String(row.risk_level??"-")}</td></tr>)}</tbody></table></div></section>}</main></div></div>;
}
