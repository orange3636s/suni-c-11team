type EmptyStateProps = {
  title: string;
  description: string;
  compact?: boolean;
};

export default function EmptyState({
  title,
  description,
  compact = false,
}: EmptyStateProps) {
  return (
    <div
      className={`emptyState ${compact ? "compact" : ""}`}
      role="status"
      aria-live="polite"
    >
      <span className="emptyStateMark" aria-hidden="true">
        <span />
      </span>
      <div>
        <strong>{title}</strong>
        <p>{description}</p>
      </div>
    </div>
  );
}
