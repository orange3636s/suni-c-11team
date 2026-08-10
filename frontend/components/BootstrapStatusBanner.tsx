"use client";

import { useAnalysisState } from "@/components/AnalysisStateProvider";

// 작업 지시서 WK-2: "N/M단계" 표시 -- api/main.py `_run_bootstrap`이
// `store.set_bootstrap_status`에 싣는 stage 문자열과 순서가 같아야 한다.
// 그 3개가 실제 파이프라인의 전부다(가짜 진행률을 만들지 않는다는 이
// 코드베이스의 원칙과 정면으로 부딪히지 않도록, 실재하지 않는 4번째
// 단계를 지어내 "N/4"로 표시하지 않는다 -- 진짜 단계 수(3)를 그대로
// 쓴다).
const BOOTSTRAP_STAGES = ["데이터 확인 중", "학습 중", "평가 · 원인분석 중"];

/** W-4: 첫 기동 부트스트랩(스냅샷이 아직 없을 때 서버가 1회 학습+분석을
 * 돌리는 동안) 진행 상태 -- DashboardShell이 셸 레벨에서 한 번만
 * 렌더한다(위 .degradedStateBanner와 같은 원칙: 페이지마다 따로 넣으면
 * 빠뜨리는 화면이 생긴다). 스냅샷이 이미 있으면(bootstrapStatus가 null이거나
 * status가 "done") 아무것도 그리지 않는다 -- 빈 화면·무한 로딩 대신
 * 기존 카드들의 빈 상태 문구가 그 자리를 대신한다.
 */
export default function BootstrapStatusBanner() {
  const { bootstrapStatus } = useAnalysisState();
  if (!bootstrapStatus || bootstrapStatus.status === "done") return null;

  // NE-3: SQL 미연결로 내장 데이터를 쓰는 것은 정상 경로이므로 그 자체를
  // 실패로 취급하지 않는다 -- 여기 도달하는 "failed"는 내장 데이터
  // 분석 자체가 깨진, 사용자가 조치할 수 없는 데이터·코드 문제다. 재시도
  // 버튼을 두지 않는다(눌러도 같은 이유로 다시 실패한다).
  if (bootstrapStatus.status === "failed") {
    return (
      <div className="bootstrapStateBanner error" role="alert">
        <span>내장 데이터 분석에 실패했습니다. 서버 로그를 확인하세요.</span>
      </div>
    );
  }

  // 진행 단계(stage)를 알면 실제 단계 번호(N/M)를 붙이고, 모르면
  // 지시서 원칙대로 가짜 진행률 없이 "첫 분석 진행 중"만 보여준다.
  const stageIndex = bootstrapStatus.stage ? BOOTSTRAP_STAGES.indexOf(bootstrapStatus.stage) : -1;
  const stagePrefix = stageIndex >= 0 ? `(${stageIndex + 1}/${BOOTSTRAP_STAGES.length}단계) ` : "";
  const label = bootstrapStatus.stage
    ? `첫 분석 진행 중 · ${stagePrefix}${bootstrapStatus.stage} · 완료 후 자동으로 표시됩니다`
    : "첫 분석 진행 중 · 완료 후 자동으로 표시됩니다";

  return (
    <div className="bootstrapStateBanner" role="status">
      <span>{label}</span>
    </div>
  );
}
