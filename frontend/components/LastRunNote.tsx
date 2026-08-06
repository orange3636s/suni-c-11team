"use client";

import { formatLastRun, isStaleResult } from "@/lib/timeFormat";

/** "마지막 실행 2026-08-06 08:05 · 하루가 지났습니다" (spec §5-1/§5-2) --
 * never disappears on its own; a 24h-stale result still renders, just
 * with the extra note appended. */
export function LastRunNote({ createdAt }: { createdAt: string | null | undefined }) {
  if (!createdAt) return null;
  return (
    <span className="lastRunNote">
      마지막 실행 {formatLastRun(createdAt)}
      {isStaleResult(createdAt) && <span className="lastRunStale"> · 하루가 지났습니다</span>}
    </span>
  );
}

/** 셀렉터가 가리키는 데이터셋과 화면에 표시 중인 결과의 데이터셋이 다를 때
 * (spec §5-3) -- 결과를 자동으로 지우지 않고 경고만 띄운다. */
export function DatasetMismatchWarning({ mismatch }: { mismatch: boolean }) {
  if (!mismatch) return null;
  return (
    <p className="datasetMismatchWarning" role="alert">
      ⚠ 선택한 데이터셋과 다른 결과입니다. 다시 실행해 주세요.
    </p>
  );
}
