"use client";

import { useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { ApiResponseError, deactivateDataset } from "@/lib/api";

/** AG-3/AG-4: 업로드로 활성화된 수동 평가 데이터셋이 있는 동안 모든
 * 화면 상단에 뜬다(DashboardShell이 셸 레벨에서 한 번만 렌더 -- 위
 * BootstrapStatusBanner/degradedStateBanner와 같은 원칙). 자동 갱신이
 * 이 데이터셋을 덮어쓰지 않는다는 사실과, 원래대로 되돌리는 경로를
 * 항상 보여준다.
 */
export default function ManualModeBanner() {
  const { manualEvalOverride, refreshRunning, refreshSnapshotNow } = useAnalysisState();
  const [reverting, setReverting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!manualEvalOverride) return null;

  async function handleRevert() {
    setError(null);
    setReverting(true);
    try {
      await deactivateDataset();
      refreshSnapshotNow();
    } catch (failure) {
      setError(
        failure instanceof ApiResponseError && failure.status === 409
          ? "자동 갱신이 진행 중입니다. 잠시 후 다시 시도하세요."
          : "복귀하지 못했습니다.",
      );
    } finally {
      setReverting(false);
    }
  }

  return (
    <div className="manualModeBanner" role="status">
      <span>
        수동 · {manualEvalOverride.filename} · 자동 갱신 일시 중지 중{refreshRunning ? " · 분석 중…" : ""}
        {error && <span className="manualModeBannerError"> — {error}</span>}
      </span>
      <button type="button" className="button secondary" onClick={() => void handleRevert()} disabled={reverting || refreshRunning}>
        자동 갱신으로 복귀
      </button>
    </div>
  );
}
