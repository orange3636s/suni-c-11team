"use client";

// HA그룹: Scatter/Box/Pareto 세 뷰가 각자 다른 컴포넌트에서 "해석"과
// "신뢰도" 카드를 따로 렌더하던 것(스타일이 갈라지는 원인)을 하나로
// 합친다 -- 세 뷰 모두 이 컴포넌트 하나만 쓴다. 카드를 두 개 쌓지
// 않는다: 카드 하나 안에 행(row)을 여러 개 둔다.

export type InterpretationRow = { label: "해석" | "신뢰도"; text: string };

export default function InterpretationCard({ rows }: { rows: InterpretationRow[] }) {
  if (rows.length === 0) return null;
  return (
    <div className="interpretationCard">
      {rows.map((row) => (
        <div className="interpretationRow" key={row.label}>
          <span className="interpretationLabel">{row.label}</span>
          <p className="interpretationBody">{row.text}</p>
        </div>
      ))}
    </div>
  );
}
