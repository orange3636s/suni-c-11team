"use client";

import { useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { usePanelState } from "@/components/PanelStateProvider";
import { deactivateDataset } from "@/lib/api";

/** SC-2: 업로드/DB 불러오기로 등록된 분석 데이터가 있는 동안 모든 화면
 * 상단에 뜬다(DashboardShell이 셸 레벨에서 한 번만 렌더 -- 위
 * BootstrapStatusBanner/degradedStateBanner와 같은 원칙). "한 번 등록되면
 * 다시 바꿀 때까지 유지된다"는 사실과, 내장 데이터로 되돌리는 경로를
 * 항상 보여준다. 되돌리기는 등록만 지울 뿐(SC-3과 분리) 분석을 자동으로
 * 다시 실행하지 않는다 -- 새 결과를 보려면 모델 분석에서 [분석 시작]을
 * 눌러야 한다.
 */
export default function ManualModeBanner() {
  const { manualEvalOverride, refreshSnapshotNow } = useAnalysisState();
  const { setAnalysisPanelOpen } = usePanelState();
  const [reverting, setReverting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!manualEvalOverride) return null;

  async function handleRevert() {
    setError(null);
    setReverting(true);
    try {
      await deactivateDataset();
      refreshSnapshotNow();
    } catch {
      setError("되돌리지 못했습니다.");
    } finally {
      setReverting(false);
    }
  }

  return (
    <div className="manualModeBanner" role="status">
      <span>
        분석 데이터: {manualEvalOverride.filename}
        {error && <span className="manualModeBannerError"> — {error}</span>}
      </span>
      <button type="button" className="button secondary" onClick={() => void handleRevert()} disabled={reverting}>
        내장 데이터로 되돌리기
      </button>
      <button type="button" className="button secondary" onClick={() => setAnalysisPanelOpen(true)}>
        모델 분석 열기
      </button>
    </div>
  );
}
