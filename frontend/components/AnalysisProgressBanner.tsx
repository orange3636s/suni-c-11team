"use client";

import { useAnalysisState } from "@/components/AnalysisStateProvider";

/** SF-3: "모델 분석"([분석 시작])이 실행 중이면 네 화면(모니터링 홈·
 * Config별 트리맵·원인 분석·수율 예측) 모두 같은 진행 표시를 본다 --
 * 어느 화면에 있든 같은 상태를 보여준다는 원칙에 따라 각 페이지 상단에
 * 동일하게 삽입한다. 오류 배너가 아니라 진행 표시만 보여준다("하지 말
 * 것": 자동 복구 상황을 오류 배너로 알리지 마라).
 */
export default function AnalysisProgressBanner() {
  const { refreshRunning, analysisProgress } = useAnalysisState();

  if (!refreshRunning) return null;

  const label = analysisProgress
    ? `분석 진행 중… (${analysisProgress.index}/${analysisProgress.total}) ${analysisProgress.stage}`
    : "분석 진행 중…";

  return (
    <p className="analysisProgressBanner" role="status">
      {label}
    </p>
  );
}
