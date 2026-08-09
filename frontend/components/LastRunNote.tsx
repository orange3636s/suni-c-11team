"use client";

import Link from "next/link";
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
 * (spec §5-3) -- 결과를 자동으로 지우지 않고 경고만 띄운다.
 *
 * HF그룹: 모니터링 SUMMARY처럼 "어느 두 값이 갈렸는지"를 구체적으로 알
 * 수 있는 화면은 `datasets`/`actions`를 넘겨 그 정보와 이동 버튼을 함께
 * 보여준다 -- "다시 실행해 주세요"만으로는 무엇을 다시 실행해야 하는지
 * 알 수 없었다. 넘기지 않으면(알림 기록·원인 분석처럼 비교 대상이
 * 이 컴포넌트만으로는 명확하지 않은 화면) 기존의 일반 문구로 그대로
 * 동작한다 -- 그 화면들의 동작은 바뀌지 않는다. */
export function DatasetMismatchWarning({
  mismatch,
  datasets,
  actions,
}: {
  mismatch: boolean;
  datasets?: { left: { label: string; value: string }; right: { label: string; value: string } };
  actions?: { label: string; href: string }[];
}) {
  if (!mismatch) return null;
  if (!datasets) {
    return (
      <p className="datasetMismatchWarning" role="alert">
        ⚠ 선택한 데이터셋과 다른 결과입니다. 다시 실행해 주세요.
      </p>
    );
  }
  return (
    <div className="datasetMismatchWarning datasetMismatchWarningRich" role="alert">
      <strong>⚠ 분석과 알람이 서로 다른 데이터셋 기준입니다</strong>
      <p className="datasetMismatchWarningDetail">
        {datasets.left.label}: {datasets.left.value} · {datasets.right.label}: {datasets.right.value}
      </p>
      {actions && actions.length > 0 && (
        <div className="datasetMismatchWarningActions">
          {actions.map((action) => (
            <Link key={action.href} href={action.href} className="button sm secondary">
              {action.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
