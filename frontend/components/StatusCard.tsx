type StatusCardProps = {
  label: string;
  value: string;
  detail: string;
  tone?: "default" | "warning" | "danger" | "normal";
};

export default function StatusCard({
  label,
  value,
  detail,
  tone = "default",
}: StatusCardProps) {
  return (
    <article className={`statusCard ${tone}`}>
      <div className="cardHeader">
        <span>{label}</span>
        <span className={`cardIndicator ${tone}`} aria-hidden="true" />
      </div>
      <strong className="cardValue">{value}</strong>
      <p>{detail}</p>
    </article>
  );
}
