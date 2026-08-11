import type { LatestAnalysisRecord } from "@/types/data";

// 저장된 원인 분석 스냅샷(state/latest의 analysis)의 응답 형태·내용
// 규칙이 바뀔 때마다 올린다. 백엔드의 봉투
// schema_version(src/runtime/app_state.py STATE_SCHEMA_VERSION)과는
// 의도적으로 별개다: 그쪽이 버전 불일치를 서버에서 필터링해 버리면
// (record -> null) 프론트는 "분석을 실행한 적이 없다"와 "스냅샷이 낡아
// 쓸 수 없다"를 구분할 수 없어 조용히 빈 화면이 된다. 이 버전은 payload
// 안에 실려 그대로 내려오므로, 프론트가 직접 비교해 사유를 안내할 수 있다.
export const ANALYSIS_SNAPSHOT_VERSION = 3;

/** 복원 후보 레코드가 지금 화면에 써도 되는지 판정한다 -- 원인 분석과
 * 모니터링 홈이 이 함수를 함께 import해 같은 기준으로 판단한다. 부분
 * 복원은 하지 않는다: false면 호출자는 레코드를 통째로 버려야 한다.
 *
 * - 버전이 다르면(다른 버전으로 저장된 스냅샷, 또는 필드 자체가 없어
 *   undefined) 쓰지 않는다.
 * - `trainingCreatedAt`이 주어졌고 분석 스냅샷보다 더 최신이면(분석 이후
 *   모델이 재학습됨) 쓰지 않는다 -- 재학습 전 모델로 계산한 결과를
 *   보여주면 안 된다.
 */
export function isAnalysisSnapshotUsable(
  record: LatestAnalysisRecord,
  trainingCreatedAt: string | null,
): boolean {
  if (record.payload.snapshotVersion !== ANALYSIS_SNAPSHOT_VERSION) return false;
  if (trainingCreatedAt != null && record.created_at < trainingCreatedAt) return false;
  return true;
}
