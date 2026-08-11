"use client";

import { useEffect, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { triggerRefresh } from "@/lib/api";
import type { AnalysisProgress } from "@/types/data";

/** T7-2: "· 100,000행 · 약 2분 예상" -- 대기 시간을 아는 것만으로 "끊겼다"는
 * 오해가 줄어든다(작업지시). 값이 없으면(구버전 서버 등) 아무것도 붙이지
 * 않는다. */
function formatRowsAndEta(progress: AnalysisProgress): string {
  const parts: string[] = [];
  if (progress.row_count != null) parts.push(`${progress.row_count.toLocaleString()}행`);
  if (progress.estimated_seconds != null) {
    const minutes = Math.round(progress.estimated_seconds / 60);
    parts.push(minutes >= 1 ? `약 ${minutes}분 예상` : `약 ${progress.estimated_seconds}초 예상`);
  }
  return parts.length > 0 ? ` · ${parts.join(" · ")}` : "";
}

/** SF-3: "모델 분석"([분석 시작])이 실행 중이면 네 화면(모니터링 홈·
 * Config별 트리맵·원인 분석·수율 예측) 모두 같은 진행 표시를 본다 --
 * 어느 화면에 있든 같은 상태를 보여준다는 원칙에 따라 각 페이지 상단에
 * 동일하게 삽입한다. 오류 배너가 아니라 진행 표시만 보여준다("하지 말
 * 것": 자동 복구 상황을 오류 배너로 알리지 마라).
 *
 * 작업지시 T5/T8-3/T7-3: `analysisProgress.heartbeat_at`이 60초 넘게
 * 갱신되지 않으면(서버 프로세스가 죽었거나 멈춘 것으로 본다) "중단됨"으로
 * 전환하고 재시도 버튼을 붙인다 -- 지금까지는 프로세스가 죽으면 DB에 남은
 * "진행 중… (3/8)" 표시가 영원히 그대로였다.
 */
const STALE_HEARTBEAT_MS = 60_000;

export default function AnalysisProgressBanner() {
  const { refreshRunning, analysisProgress, snapshot } = useAnalysisState();
  const [now, setNow] = useState(() => Date.now());
  const [retryMessage, setRetryMessage] = useState<string | null>(null);

  const heartbeatAt = analysisProgress?.heartbeat_at ?? null;

  useEffect(() => {
    if (!refreshRunning || !heartbeatAt) return;
    const timer = window.setInterval(() => setNow(Date.now()), 5_000);
    return () => window.clearInterval(timer);
  }, [refreshRunning, heartbeatAt]);

  if (!refreshRunning) {
    // T8-4: 완료된 분석에 부분 실패(errors)가 남아 있으면 계속 보여준다 --
    // 자동 복구가 아니라 실제로 무언가 실패했다는 뜻이라 오류 배너가 맞다.
    if (snapshot?.errors && snapshot.errors.length > 0) {
      return (
        <p className="analysisProgressBanner analysisErrorsBanner" role="alert">
          ⚠ 마지막 분석에서 일부가 실패했습니다: {snapshot.errors.join(" · ")}
        </p>
      );
    }
    return null;
  }

  const heartbeatAgeMs = heartbeatAt ? now - new Date(heartbeatAt).getTime() : null;
  const isStale = heartbeatAgeMs != null && heartbeatAgeMs > STALE_HEARTBEAT_MS;

  if (isStale) {
    const stageLabel = analysisProgress ? ` (${analysisProgress.stage})` : "";
    return (
      <p className="analysisProgressBanner analysisProgressBannerStale" role="alert">
        분석이 중단되었습니다{stageLabel}
        <button
          type="button"
          className="button sm secondary"
          onClick={() => {
            setRetryMessage(null);
            void triggerRefresh()
              .then(() => setRetryMessage("다시 시작했습니다."))
              .catch(() => setRetryMessage("아직 이전 실행이 정리되지 않았습니다. 잠시 후 다시 시도해 주세요."));
          }}
        >
          다시 시작
        </button>
        {retryMessage && <span className="analysisProgressBannerRetryNote"> {retryMessage}</span>}
      </p>
    );
  }

  const label = analysisProgress
    ? `분석 진행 중… (${analysisProgress.index}/${analysisProgress.total}) ${analysisProgress.stage}${formatRowsAndEta(analysisProgress)}`
    : "분석 진행 중…";

  return (
    <p className="analysisProgressBanner" role="status">
      {label}
    </p>
  );
}
