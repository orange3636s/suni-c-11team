import type { DashboardSectionState as SectionState } from "@/types/data";


type DashboardSectionStateProps = {
  state: Exclude<SectionState, "ready">;
  message: string;
  onRetry?: () => void;
  compact?: boolean;
};

export default function DashboardSectionState({
  state,
  message,
  onRetry,
  compact = false,
}: DashboardSectionStateProps) {
  if (state === "loading") {
    return (
      <div className={`dashboardSectionState loading${compact ? " compact" : ""}`} aria-live="polite">
        <span className="dashboardSkeletonLine wide" />
        <span className="dashboardSkeletonLine" />
        <span className="srOnly">{message}</span>
      </div>
    );
  }
  return (
    <div className={`dashboardSectionState ${state}${compact ? " compact" : ""}`}>
      <span>{message}</span>
      {state === "error" && onRetry && (
        <button className="button secondary" type="button" onClick={onRetry}>다시 시도</button>
      )}
    </div>
  );
}
