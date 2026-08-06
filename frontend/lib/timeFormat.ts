/** "마지막 실행" timestamp formatting -- 오늘이면 시각만, 어제 이전이면
 * 날짜까지 (spec: 학습·분석 결과 상태 유지 §5-1). */
export function formatLastRun(iso: string | null | undefined): string {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "-";
  const pad = (n: number) => String(n).padStart(2, "0");
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
  const now = new Date();
  const isToday =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  if (isToday) return time;
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${time}`;
}

/** 24시간이 지난 결과에 안내를 붙인다 (spec §5-2) -- 지우지는 않는다, 표시만. */
export function isStaleResult(iso: string | null | undefined): boolean {
  if (!iso) return false;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return false;
  return Date.now() - date.getTime() > 24 * 60 * 60 * 1000;
}
