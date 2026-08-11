import type { DatasetSchemaResponse } from "@/types/data";

// 계측률이 높으면 "일부만 대상"이라는 표현이 과장이므로 문구를 바꾼다.
// 60% 기준과 문구 전환 규칙은 src/analysis/report.py의
// _measurement_rate_limitation과 같은 값으로 유지해야 한다. 학습/원인분석
// 탭이 같은 문구를 쓰므로 여기 하나로 모은다.
const HIGH_MEASUREMENT_RATE_THRESHOLD = 60;

export function measurementRateDisclaimer(schema: DatasetSchemaResponse | null): string {
  const r = schema?.r_measurement_rate ?? null;
  const d = schema?.d_measurement_rate ?? null;
  const parts: string[] = [];
  if (r != null) parts.push(`Response는 전체의 ${r.toFixed(1)}%`);
  if (d != null) parts.push(`Defect는 전체의 ${d.toFixed(1)}%`);
  const rates = [r, d].filter((value): value is number => value != null);
  const scope =
    rates.length > 0 && rates.every((value) => value >= HIGH_MEASUREMENT_RATE_THRESHOLD)
      ? "계측된 wafer를 분석 대상으로 합니다"
      : "계측된 wafer만 대상으로 합니다";
  const observed = parts.length > 0 ? ` ${parts.join(", ")}에서 관측되었습니다.` : "";
  return `이 분석은 해당 인자가 ${scope}.${observed} 미계측 wafer로의 일반화는 보장되지 않습니다.`;
}
