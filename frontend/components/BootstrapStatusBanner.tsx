"use client";

import { useAnalysisState } from "@/components/AnalysisStateProvider";

/** W-4: 첫 기동 부트스트랩(스냅샷이 아직 없을 때 서버가 1회 학습+분석을
 * 돌리는 동안) 진행 상태 -- DashboardShell이 셸 레벨에서 한 번만
 * 렌더한다(위 .degradedStateBanner와 같은 원칙: 페이지마다 따로 넣으면
 * 빠뜨리는 화면이 생긴다). 스냅샷이 이미 있으면(bootstrapStatus가 null이거나
 * status가 "done") 아무것도 그리지 않는다 -- 빈 화면·무한 로딩 대신
 * 기존 카드들의 빈 상태 문구가 그 자리를 대신한다.
 */
export default function BootstrapStatusBanner() {
  const { bootstrapStatus, refreshSnapshotNow } = useAnalysisState();
  if (!bootstrapStatus || bootstrapStatus.status === "done") return null;

  if (bootstrapStatus.status === "failed") {
    return (
      <div className="bootstrapStateBanner error" role="alert">
        <span>
          첫 분석에 실패했습니다{bootstrapStatus.error ? ` — ${bootstrapStatus.error}` : ""}. 원인 분석 화면의
          &lsquo;다시 분석&rsquo;으로 다시 시도할 수 있습니다.
        </span>
        <button type="button" className="button" onClick={refreshSnapshotNow}>
          다시 확인
        </button>
      </div>
    );
  }

  // 진행 단계(stage)를 알면 붙이고, 모르면 지시서 원칙대로 가짜 진행률
  // 없이 "첫 분석 진행 중"만 보여준다.
  const label = bootstrapStatus.stage
    ? `첫 분석 진행 중 · ${bootstrapStatus.stage} · 완료 후 자동으로 표시됩니다`
    : "첫 분석 진행 중 · 완료 후 자동으로 표시됩니다";

  return (
    <div className="bootstrapStateBanner" role="status">
      <span>{label}</span>
    </div>
  );
}
