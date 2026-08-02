type OperationProgressProps = {
  message: string;
  timeLabel: "학습 시간" | "추론 시간";
  formattedElapsed: string;
  detail?: string;
};

export default function OperationProgress({
  message,
  timeLabel,
  formattedElapsed,
  detail,
}: OperationProgressProps) {
  return (
    <span className="operationProgress">
      <span className="operationProgressMessage">{message}</span>
      <span className="operationProgressTime">
        <span aria-hidden="true">·</span> {timeLabel} {formattedElapsed}
      </span>
      {detail && <span className="operationProgressDetail">{detail}</span>}
    </span>
  );
}
