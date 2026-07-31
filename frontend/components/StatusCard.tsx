type StatusCardProps = {
  label: string;
  value: string;
  detail: string;
  tone?: "default" | "warning" | "danger" | "normal";
  unit?: string;
};

export default function StatusCard({
  label,
  value,
  detail,
  tone = "default",
  unit,
}: StatusCardProps) {
  return (
    <article className={`statusCard ${tone}`}>
      <div className="cardHeader">
        <span>{label}</span>
        <span className={`cardIndicator ${tone}`} aria-hidden="true" />
      </div>
      <div className="cardMetric">
        <strong className="cardValue">{value}</strong>
        {unit && <span>{unit}</span>}
      </div>
      <p>{detail}</p>
    </article>
  );
}
