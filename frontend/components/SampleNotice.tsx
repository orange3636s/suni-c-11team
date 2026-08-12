"use client";

import type { SampleInfo } from "@/types/data";

/** 20,000행을 넘는 데이터셋은 인자 순위/히트맵/트리맵 계산에
 * 로트 단위 표본을 쓴다 -- 숫자를 조용히 표본으로 바꾸고 말 안 하는 것이
 * 가장 하면 안 되는 일이므로, 표본을 쓴 화면에는 항상 이 한 줄을 띄운다.
 * `sampleInfo`가 없거나 `is_sampled`가 false면 아무것도 렌더하지 않는다
 * (전량 기준일 때는 굳이 "전량입니다"를 매번 말하지 않는다). */
export default function SampleNotice({ sampleInfo }: { sampleInfo: SampleInfo | null | undefined }) {
  if (!sampleInfo || !sampleInfo.is_sampled) return null;
  const lotSuffix = sampleInfo.lot_count != null ? `(로트 ${sampleInfo.lot_count.toLocaleString()}개)` : "";
  return (
    <p className="sampleNotice" role="note">
      {sampleInfo.original_rows.toLocaleString()}행 중 {sampleInfo.sampled_rows.toLocaleString()}행{lotSuffix} 표본으로
      분석했습니다 · 수율 예측은 전량 기준
    </p>
  );
}
