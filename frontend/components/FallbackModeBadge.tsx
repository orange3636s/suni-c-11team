"use client";

import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { formatLastRun } from "@/lib/timeFormat";

const BUNDLED_LABEL: Record<string, string> = { train: "train.CSV", test: "test.CSV" };

/** J-6: SQL 미연결 상태에서 내장 CSV(train/test)를 쓰고 있다는 사실을
 * 네 화면 모두에 표기한다 -- 옵션으로 끌 수 없다(지시서: "데모 데이터를
 * 실데이터로 오인하는 것이 이 시스템에서 가장 비싼 오해"). 경고색이
 * 아니라 중립색을 쓴다 -- 오류가 아니라 상태이기 때문이다. 스냅샷이
 * 아직 없으면(최초 기동, 첫 갱신 전) 아무것도 표시하지 않는다 -- 아직
 * 어느 모드인지 알 수 없다.
 */
export default function FallbackModeBadge() {
  const { snapshot } = useAnalysisState();
  if (!snapshot) return null;
  const { source } = snapshot;

  if (source.mode === "fallback") {
    const trainLabel = BUNDLED_LABEL[source.train_dataset] ?? source.train_dataset;
    const evalLabel = BUNDLED_LABEL[source.eval_dataset] ?? source.eval_dataset;
    return (
      <span className="fallbackModeBadge" title="SQL 데이터 소스가 연결되지 않아 내장 데이터로 동작 중입니다.">
        데모 데이터 · SQL 미연결 (학습 {trainLabel} → 평가 {evalLabel})
      </span>
    );
  }
  return (
    <span className="fallbackModeBadge fallbackModeBadge-sql">
      SQL · 마지막 수집 {formatLastRun(snapshot.created_at)}
    </span>
  );
}
