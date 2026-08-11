// E-6: 여러 파일에 중복 선언돼 있던 상수를 여기 하나로 모은다. 각 값이
// 어디서 왔고 왜 이 모양인지는 아래 각 항목의 주석 참고.
import { factorAxisLabel, targetAxisLabel } from "@/lib/chartLabels";
import type { CategoricalScatterResponse } from "@/types/data";

// 화면 스크리닝이 항상 다루는 5개 수율 타깃 (monitoringSource.ts / root-cause/page.tsx 중복).
export const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

// Config 값 형식 검증 전용(캡처 그룹 없음) -- monitoringSource.ts가 쓰던 용도.
export const CONFIG_FORMAT_RE = /^Step\d+_Model\d+_EQ[A-Z]_CH\d+$/;

export type ConfigParts = { step: number; model: string; eq: string; chamber: string };
const CONFIG_PARSE_RE = /^Step(\d+)_(Model\d+)_(EQ[A-Z])_(CH\d+)$/;

/** `Step16_Model2_EQB_CH3` -> step/model/eq/chamber 4계층 분해. 매치
 * 실패(형식이 다른 미지 Config)는 호출부에서 "미상"/null 그룹으로
 * 모은다 -- 조용히 버리지 않는다. */
export function parseConfig(config: string): ConfigParts | null {
  const m = CONFIG_PARSE_RE.exec(config);
  if (!m) return null;
  return { step: Number(m[1]), model: m[2], eq: m[3], chamber: m[4] };
}

/** 발산형(빨강=낮음/초록=높음) 팔레트 -- Treemap 색 스케일, 비교 모달의
 * "최적 중심/구간 평균 불량률" 범례 스와치가 공유한다. dataviz 스킬의
 * validate_palette.js로 확인된 값(라이트 CVD ΔE 8.6, 다크 6.5). */
export const DIVERGING_GREEN = { light: "#059669", dark: "#34D399" };
export const DIVERGING_RED = { light: "#DC2626", dark: "#F87171" };

/** 카테고리형(Config) 인자의 Box Plot 스펙 -- 원인 분석 전체 화면과
 * 즐겨찾기 썸네일이 같은 모양을 쓰되 축 제목 유무가 다르다: 썸네일은
 * 160px 높이라 제목을 넣으면 잘리므로 생략하고(J-3), 전체 화면은
 * factorAxisLabel/targetAxisLabel로 축 의미를 명시한다. 이 차이는
 * 의도된 것이므로 `compact` 옵션으로 명시적으로 고른다. */
export function buildCategoricalSpec(data: CategoricalScatterResponse, options?: { compact?: boolean }) {
  const x = data.groups.flatMap((group) => group.values.map(() => group.category));
  const y = data.groups.flatMap((group) => group.values);
  const compact = options?.compact ?? false;
  return {
    data: [
      { type: "box", x, y, boxpoints: "outliers", marker: { color: "#1D4ED8" }, line: { color: "#1D4ED8" } },
    ],
    layout: compact
      ? { xaxis: { tickangle: 0 }, yaxis: {}, margin: { t: 10, b: 40, l: 40, r: 10 } }
      : {
          xaxis: { title: { text: factorAxisLabel(data.axis.x_label) }, tickangle: 0 },
          yaxis: { title: { text: targetAxisLabel(data.axis.y_label) } },
          margin: { t: 20, b: 90 },
        },
  };
}
