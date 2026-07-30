type PipelineStepProps = {
  index: number;
  title: string;
  description: string;
};

export default function PipelineStep({
  index,
  title,
  description,
}: PipelineStepProps) {
  return (
    <li className="pipelineStep">
      <div className="stepNumber">{String(index).padStart(2, "0")}</div>
      <div className="stepContent">
        <strong>{title}</strong>
        <span>{description}</span>
      </div>
      <span className="stepStatus">준비 중</span>
    </li>
  );
}
