"use client";

// RD-2: 원인분석의 "분석 대상"·알림기록의 "예측 대상" 셀렉터를 선택
// 컨트롤에서 표시 라벨로 바꾼다 -- 데이터는 모델 분석이 정하고, 화면은
// 그 값을 보여주기만 한다. [변경] 버튼은 모델 분석 팝업을 연다(자체
// 데이터셋 선택/업로드 UI를 두지 않는다 -- RD-1).
export default function CurrentDatasetLabel({
  label,
  datasetId,
  onOpenAnalysisPanel,
}: {
  label: string;
  datasetId: string;
  onOpenAnalysisPanel: () => void;
}) {
  return (
    <div className="fieldGroup">
      <span>{label}</span>
      <div className="currentDatasetLabel">
        <span className="currentDatasetLabelValue" title={datasetId}>
          {datasetId} <span className="currentDatasetLabelHint">(모델 분석)</span>
        </span>
        <button type="button" className="button sm secondary" onClick={onOpenAnalysisPanel}>
          변경
        </button>
      </div>
    </div>
  );
}
