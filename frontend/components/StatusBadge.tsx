type StatusBadgeProps = {
  label: string;
  tone?: "success" | "info" | "warning" | "danger" | "neutral";
  dot?: boolean;
};

export default function StatusBadge({
  label,
  tone = "neutral",
  dot = true,
}: StatusBadgeProps) {
  return (
    <span className={`statusBadge ${tone}`}>
      {dot && <span className="statusBadgeDot" aria-hidden="true" />}
      {label}
    </span>
  );
}
