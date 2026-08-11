"use client";

import Link from "next/link";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { formatLastRun, isStaleResult } from "@/lib/timeFormat";

/** "마지막 실행 2026-08-06 08:05 · 하루가 지났습니다" --
 * never disappears on its own; a 24h-stale result still renders, just
 * with the extra note appended. `label`은 기본 "마지막 실행"이지만, 알림
 * 기록처럼 이 시각이 "지금 화면의 기준"이지 "이 항목이 실행된 시각"이
 * 아닌 화면은 T9-2에 따라 "현재 분석 기준"으로 바꿔 부른다(개별 항목은
 * 각자의 발송 시각을 따로 표시하므로 헤더와 혼동되지 않는다). */
export function LastRunNote({
  createdAt,
  label = "마지막 실행",
}: {
  createdAt: string | null | undefined;
  label?: string;
}) {
  if (!createdAt) return null;
  return (
    <span className="lastRunNote">
      {label} {formatLastRun(createdAt)}
      {isStaleResult(createdAt) && <span className="lastRunStale"> · 하루가 지났습니다</span>}
    </span>
  );
}

/** 모니터링·원인 분석·수율 예측 세 화면 상단, "마지막 실행" 옆에
 * 학습/분석 데이터 파일명을 보여준다 -- 둘 다 이미 앱 전역에서
 * AnalysisStateProvider가 한 번만 불러온 값(training.performance.
 * source_filename / snapshot.source.eval_dataset_filename)이라 이 컴포넌트
 * 자체는 API를 부르지 않는다. 세 화면이 각자 구현을 복붙하지 않도록
 * 이 파일 하나로 공유한다. 파일명만 있는 그대로 보여준다. */
export function TrainingAnalysisDataNote({
  trainFilename,
  evalFilename,
}: {
  trainFilename: string | null;
  evalFilename: string | null;
}) {
  if (!trainFilename && !evalFilename) return null;
  return (
    <span className="trainingAnalysisDataNote">
      {trainFilename && (
        <span className="trainingAnalysisDataNoteItem" title={trainFilename}>
          훈련 데이터 {trainFilename}
        </span>
      )}
      {trainFilename && evalFilename && " · "}
      {evalFilename && (
        <span className="trainingAnalysisDataNoteItem" title={evalFilename}>
          분석 데이터 {evalFilename}
        </span>
      )}
    </span>
  );
}

/** 작업지시 T9: 여섯 화면(모니터링/원인분석/수율예측/Config별 트리맵/
 * 알림 기록/즐겨찾기) 제목 아래에 "마지막 실행 14:10 · 훈련 데이터
 * train_config.csv · 분석 데이터 test_remove_y.CSV"를 동일한 구분자·순서로
 * 보여준다. 데이터셋 id(train/test)는 파일명과 중복이라 따로 표시하지
 * 않는다. 세 화면이 이미 이 조합을 각자 미세하게 다르게 조립하고 있어
 * (구분자 유무, 데이터셋 id 표시 여부) 하나로 합친다 -- `AnalysisStateProvider`에서
 * 직접 읽으므로 호출부는 prop을 넘기지 않는다(새 API 호출 없음).
 * `created_at`이 없으면(분석 미실행) 아무것도 렌더하지 않는다 -- 빈
 * 구분자만 남는 것을 막는다. `label`(T9-2)은 알림 기록처럼 "마지막
 * 실행"이 아니라 "현재 분석 기준"으로 불러야 하는 화면을 위한 오버라이드
 * -- 그 화면들의 개별 항목은 각자의 실제 발송 시각을 따로 표시한다. */
export function PageHeaderMeta({ label }: { label?: string } = {}) {
  const { snapshot, training } = useAnalysisState();
  if (!snapshot?.created_at) return null;
  return (
    <p className="sectionCaption pageHeaderMeta">
      <LastRunNote createdAt={snapshot.created_at} label={label} />
      <TrainingAnalysisDataNote
        trainFilename={training?.performance?.source_filename ?? null}
        evalFilename={snapshot.source.eval_dataset_filename}
      />
    </p>
  );
}

/** 셀렉터가 가리키는 데이터셋과 화면에 표시 중인 결과의 데이터셋이 다를 때
 * -- 결과를 자동으로 지우지 않고 경고만 띄운다.
 *
 * 모니터링 SUMMARY처럼 "어느 두 값이 갈렸는지"를 구체적으로 알
 * 수 있는 화면은 `datasets`/`actions`를 넘겨 그 정보와 이동 버튼을 함께
 * 보여준다 -- "다시 실행해 주세요"만으로는 무엇을 다시 실행해야 하는지
 * 알 수 없다. 넘기지 않으면(수율 예측·원인 분석처럼 비교 대상이
 * 이 컴포넌트만으로는 명확하지 않은 화면) 일반 문구만 보여준다. */
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
