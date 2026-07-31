type PipelineStepProps = {
  index: number;
  title: string;
  description: string;
  status: "구현 완료" | "다음 단계" | "개발 예정" | "설정 필요";
};

export default function PipelineStep({
  index,
  title,
  description,
  status,
}: PipelineStepProps) {
  return (
    <li className="pipelineStep">
      <div className="stepNumber">{String(index).padStart(2, "0")}</div>
      <div className="stepContent">
        <strong>{title}</strong>
        <span>{description}</span>
      </div>
      <span
        className={`stepStatus ${
          status === "구현 완료"
            ? "complete"
            : status === "다음 단계"
              ? "next"
              : status === "설정 필요"
                ? "next"
              : ""
        }`}
      >
        {status}
      </span>
    </li>
  );
}
