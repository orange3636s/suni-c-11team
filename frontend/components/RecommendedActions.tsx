"use client";

import { buildRecommendedActions } from "@/lib/fmeaActions";
import type { FmeaTablePayload } from "@/types/data";

function badgeClass(strength: string): string {
  if (strength === "확정") return "badge badge-green";
  if (strength === "실험 후보") return "badge badge-amber";
  return "badge badge-neutral";
}

/** 권고 조치 표 (모니터링 홈, 지시서 IC) -- FMEA 표에서 자동 도출한다
 * (하드코딩 없음). 정렬은 RPN이 아니라 실익(수율 편차) 순 -- 두 순위가
 * 다를 때는 실익을 따른다. "제안하지 않음" 두 행은 조건과 무관하게
 * 항상 맨 끝에 붙는다. */
export default function RecommendedActions({ data }: { data: FmeaTablePayload | null }) {
  const rows = buildRecommendedActions(data);

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">조치 우선순위</span>
          <h2>권고 조치</h2>
        </div>
      </div>
      <p className="sectionCaption">실익(수율 편차) 순 · FMEA 분석표에서 자동 도출</p>

      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th className="numCol">#</th>
              <th>조치</th>
              <th>대상</th>
              <th>기대 효과</th>
              <th>근거 강도</th>
              <th>비고</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key}>
                <td className="numCol">{row.order}</td>
                <td className="data">{row.action}</td>
                <td>{row.target}</td>
                <td>{row.expectedEffect}</td>
                <td>
                  <span className={badgeClass(row.strength)}>{row.strength}</span>
                </td>
                <td className="fmeaActionNote">{row.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
