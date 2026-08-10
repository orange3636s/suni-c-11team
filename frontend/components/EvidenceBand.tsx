"use client";

import { useEffect, useRef, useState } from "react";

// HP-HMI 디자인의 시그니처 컴포넌트 -- 모델이 낸 모든 수치 아래에 같은
// 형태의 구간 트랙을 깔아 "점이 아니라 범위"임을 화면 문법으로 반복한다
// (지시서 §E, 보정 지시서 §I-1). 원래 모니터링 SUMMARY 전용
// `YieldGapBar`였던 것을 수율 예측 알람 목록에서도 그대로 재사용하기
// 위해 스케일(scaleMin/scaleMax)과 목표선(target) 유무를 props로 뺐다 --
// 계산 로직 자체는 그대로다.
//
// mini=true(계측 우선순위류의 96px 미니 자리)에서는 지시서대로 라벨을
// 아예 렌더하지 않는다 -- 라벨 충돌 배치 로직은 mini에서 쓰이지 않는다.

// F-2: 이 미만으로 두 틱 라벨의 x좌표(%)가 가까우면 텍스트가 겹쳐
// "90.791.0"처럼 붙어 보인다 -- 실측(스크린샷)한 겹침 사례가 2%p
// 간격이었으므로, 4~5자 라벨 폭을 감안해 여유 있게 잡는다.
const TICK_OVERLAP_THRESHOLD_PCT = 7;
// HE-2: GA그룹 이후 집계 구간(SUMMARY)은 폭이 밴드 전체의 몇 % 수준으로
// 좁아질 수 있다 -- 이 미만이면 양끝 숫자를 따로 찍는 게 의미가 없다
// (거의 같은 자리에 찍혀 겹친다). "88.9 – 89.3" 대신 "88.9~89.3" 하나로
// 합친다.
const MERGE_INTERVAL_THRESHOLD_PCT = 2;
// 등폭(--font-data) 9px(--fs-nano) 기준 대략적인 문자 폭 근사치(px) --
// 실측(getBoundingClientRect)은 렌더 이후에나 가능해 최초 레이아웃
// 판단에는 쓸 수 없으므로 문자 수 × 이 값으로 근사한다.
const CHAR_WIDTH_PX = 5.5;
const LABEL_PADDING_PX = 6; // 라벨 사이 최소 여백

export type EvidenceBandProps = {
  /** 구간 하한/상한 (모델 추정치 -- 스케일과 같은 단위). */
  lo: number;
  hi: number;
  /** 목표선. 없으면 목표 tick/라벨을 그리지 않는다. */
  target?: number | null;
  /** 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-2) -- 현재 민감도의
   * "주의" 컷(가장 느슨한 알람 컷) 위치. 목표선과 별개로 그린다 --
   * 목표선은 "모델이 아는 만큼의 불확실성"과 무관하게 고정된 기준이고,
   * 판정선은 "사용자가 고른 위험 감수 수준"이라 서로 다른 정보다. 없으면
   * 그리지 않는다. */
  judgmentLine?: number | null;
  /** 고정 스케일 -- 관측값에 맞춰 자동 조정하지 않는다(지시서 원칙). */
  scaleMin: number;
  scaleMax: number;
  /** 96px 내외, 라벨 없는 축소형(계측 우선순위류). */
  mini?: boolean;
  /** 목표선 텍스트 라벨("목표" 등). 기본값 "목표". */
  targetLabel?: string;
  /** 판정선 텍스트 라벨. 기본값 "판정선". */
  judgmentLabel?: string;
};

// HE-1: 라벨 우선순위 -- 겹치면 이 순서대로 지킨다(숫자가 클수록 우선).
// 목표선은 항상 보여야 하는 고정 기준, 판정선은 사용자가 방금 조절한
// 값이라 그다음, 구간 양끝은 실측/추정값, 축 눈금(스케일 경계)이 가장
// 덜 중요하다(스케일 자체는 화면에 고정 표기돼 있어 다른 곳에서도 읽힌다).
const PRIORITY = { target: 4, judgment: 3, interval: 2, axis: 1 } as const;

type BelowTrackLabel = {
  key: string;
  text: string;
  pct: number;
  priority: number;
  outOfRange: "low" | "high" | null;
  variant: "judgment" | "tick";
};

type PlacedLabel = BelowTrackLabel & { row: 0 | 1 };

function useTrackWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(w);
    });
    observer.observe(el);
    setWidth(el.getBoundingClientRect().width);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function labelWidthPx(text: string): number {
  return text.length * CHAR_WIDTH_PX;
}

/** HE-1: 트랙 아래쪽에 놓이는 모든 라벨(판정선 + 눈금)을 한 좌표계에서
 * 함께 배치한다 -- 이전에는 판정선(절대 위치 span)과 눈금(별도 겹침
 * 회피 로직)이 서로의 존재를 모른 채 따로 그려져 "판정선89.2"처럼
 * 붙어 보였다. 우선순위가 낮은 라벨부터 접어 처리한다: ① 2단 배치로
 * 분리 시도 → ② 그래도 겹치면 우선순위 낮은 쪽을 생략(제목 속성으로
 * 값은 유지). */
function layoutBelowTrackLabels(labels: BelowTrackLabel[], trackWidthPx: number): { placed: PlacedLabel[]; dropped: BelowTrackLabel[] } {
  if (trackWidthPx <= 0) return { placed: labels.map((l) => ({ ...l, row: 0 })), dropped: [] };
  // 우선순위 높은 라벨부터 자리를 먼저 잡는다 -- 낮은 우선순위가 밀려나야
  // 하므로.
  const ordered = [...labels].sort((a, b) => b.priority - a.priority || a.pct - b.pct);
  const rows: PlacedLabel[][] = [[], []];
  const dropped: BelowTrackLabel[] = [];

  function fitsRow(row: PlacedLabel[], candidate: BelowTrackLabel): boolean {
    const candidateLeftPx = (candidate.pct / 100) * trackWidthPx - labelWidthPx(candidate.text) / 2;
    const candidateRightPx = (candidate.pct / 100) * trackWidthPx + labelWidthPx(candidate.text) / 2;
    return row.every((placed) => {
      const placedLeftPx = (placed.pct / 100) * trackWidthPx - labelWidthPx(placed.text) / 2;
      const placedRightPx = (placed.pct / 100) * trackWidthPx + labelWidthPx(placed.text) / 2;
      return candidateLeftPx >= placedRightPx + LABEL_PADDING_PX || candidateRightPx <= placedLeftPx - LABEL_PADDING_PX;
    });
  }

  for (const label of ordered) {
    if (fitsRow(rows[0], label)) {
      rows[0].push({ ...label, row: 0 });
    } else if (fitsRow(rows[1], label)) {
      rows[1].push({ ...label, row: 1 });
    } else {
      // HE-1 3순위: 두 줄로도 못 넣으면 우선순위가 낮은 라벨을 생략한다
      // (추측이 아니라 실제로 자리가 없을 때만 -- title 속성으로 값은
      // 그대로 남긴다, 호버로 확인 가능).
      dropped.push(label);
    }
  }
  return { placed: [...rows[0], ...rows[1]], dropped };
}

export default function EvidenceBand({
  lo, hi, target = null, judgmentLine = null, scaleMin, scaleMax, mini = false,
  targetLabel = "목표", judgmentLabel = "판정선",
}: EvidenceBandProps) {
  const [trackRef, trackWidthPx] = useTrackWidth();
  const span = scaleMax - scaleMin;
  // 범위 밖 값은 경계에 클램프하되, 라벨에 화살표(←/→)를 붙여 "표시
  // 위치 ≠ 실제 값"임을 알린다 (mini에는 라벨이 없어 해당 없음).
  const pctOf = (value: number) => Math.min(100, Math.max(0, ((value - scaleMin) / span) * 100));
  const bandLoPct = pctOf(Math.min(lo, hi));
  const bandHiPct = pctOf(Math.max(lo, hi));
  const targetPct = target != null ? pctOf(target) : null;
  const judgmentPct = judgmentLine != null ? pctOf(judgmentLine) : null;

  if (mini) {
    return (
      <div className="evidenceBand mini">
        <div className="evidenceBandTrack">
          <div
            className="evidenceBandFill"
            style={{ left: `${bandLoPct}%`, width: `${Math.max(bandHiPct - bandLoPct, 0.6)}%` }}
          />
          {judgmentPct != null && <div className="evidenceBandJudgmentTick" style={{ left: `${judgmentPct}%` }} />}
          {targetPct != null && <div className="evidenceBandTargetTick" style={{ left: `${targetPct}%` }} />}
        </div>
      </div>
    );
  }

  // HE-2: 구간이 스케일 전체 폭의 MERGE_INTERVAL_THRESHOLD_PCT% 미만이면
  // 양끝을 "lo~hi" 하나로 합친다 -- 따로 찍어도 거의 같은 자리라 못 읽는다.
  const intervalWidthPct = bandHiPct - bandLoPct;
  const mergeInterval = intervalWidthPct < MERGE_INTERVAL_THRESHOLD_PCT && Math.min(lo, hi) !== Math.max(lo, hi);

  const belowTrackLabels: BelowTrackLabel[] = [];
  if (mergeInterval) {
    const loVal = Math.round(Math.min(lo, hi) * 10) / 10;
    const hiVal = Math.round(Math.max(lo, hi) * 10) / 10;
    belowTrackLabels.push({
      key: "interval-merged",
      text: `${loVal.toFixed(1)}~${hiVal.toFixed(1)}`,
      pct: (bandLoPct + bandHiPct) / 2,
      priority: PRIORITY.interval,
      outOfRange: null,
      variant: "tick",
    });
  } else {
    for (const raw of [Math.min(lo, hi), Math.max(lo, hi)]) {
      const value = Math.round(raw * 10) / 10;
      belowTrackLabels.push({
        key: `interval-${value}`,
        text: value.toFixed(1),
        pct: pctOf(value),
        priority: PRIORITY.interval,
        outOfRange: value < scaleMin ? "low" : value > scaleMax ? "high" : null,
        variant: "tick",
      });
    }
  }
  for (const raw of [scaleMin, scaleMax]) {
    const value = Math.round(raw * 10) / 10;
    belowTrackLabels.push({
      key: `axis-${value}`,
      text: value.toFixed(1),
      pct: pctOf(value),
      priority: PRIORITY.axis,
      outOfRange: null,
      variant: "tick",
    });
  }
  if (target != null) {
    const value = Math.round(target * 10) / 10;
    belowTrackLabels.push({
      key: `target-tick-${value}`,
      text: value.toFixed(1),
      pct: pctOf(value),
      priority: PRIORITY.target,
      outOfRange: value < scaleMin ? "low" : value > scaleMax ? "high" : null,
      variant: "tick",
    });
  }
  if (judgmentPct != null && judgmentLine != null) {
    // HE-3: "판정선"만 있으면 근처 눈금 숫자와 붙었을 때 "판정선89.2"로
    // 읽힌다 -- 값을 라벨 자체에 함께 적어 하나의 완결된 문자열로 만든다.
    belowTrackLabels.push({
      key: "judgment",
      text: `${judgmentLabel} ${judgmentLine.toFixed(1)}`,
      pct: judgmentPct,
      priority: PRIORITY.judgment,
      outOfRange: null,
      variant: "judgment",
    });
  }
  // 중복 좌표(예: 목표와 구간 상한이 같은 값)는 하나만 남긴다.
  const dedupedLabels = Array.from(new Map(belowTrackLabels.map((l) => [`${l.pct.toFixed(2)}|${l.variant}`, l])).values());
  const { placed, dropped } = layoutBelowTrackLabels(dedupedLabels, trackWidthPx);

  return (
    <div className="evidenceBand">
      <div className="evidenceBandTrack" ref={trackRef}>
        <div
          className="evidenceBandFill"
          style={{ left: `${bandLoPct}%`, width: `${Math.max(bandHiPct - bandLoPct, 0.6)}%` }}
        />
        <div className="evidenceBandBoundary" style={{ left: `${bandLoPct}%` }} />
        <div className="evidenceBandBoundary" style={{ left: `${bandHiPct}%` }} />
        {judgmentPct != null && <div className="evidenceBandJudgmentTick" style={{ left: `${judgmentPct}%` }} />}
        {targetPct != null && target != null && (
          <>
            <div className="evidenceBandTargetTick" style={{ left: `${targetPct}%` }} />
            {/* HE-3: 목표선도 값을 함께 적는다("목표 91.0") -- 판정선과
                같은 원칙, 트랙 위쪽에 있어 아래쪽 라벨 그룹과는 애초에
                겹치지 않는다(디자인상 의도적 분리, 위 CSS 주석 참고). */}
            <span className="evidenceBandTargetLabel" style={{ left: `${targetPct}%` }}>{targetLabel} {target.toFixed(1)}</span>
          </>
        )}
      </div>
      <div className="evidenceBandTicks">
        {placed.map((label) => (
          <span
            key={label.key}
            className={[
              label.variant === "judgment" ? "evidenceBandJudgmentLabel" : "evidenceBandTick",
              label.row === 1 ? "evidenceBandTickRow2" : "",
            ].filter(Boolean).join(" ")}
            style={{ left: `${label.pct}%` }}
          >
            {label.outOfRange === "low" && "← "}
            {label.text}
            {label.outOfRange === "high" && " →"}
          </span>
        ))}
        {/* HE-1 3순위: 자리가 없어 생략된 라벨 -- 위치는 점으로 남기고
            값은 title(네이티브 툴팁)로만 제공한다. */}
        {dropped.map((label) => (
          <span
            key={`dropped-${label.key}`}
            className="evidenceBandTickDropped"
            style={{ left: `${label.pct}%` }}
            title={label.text}
            aria-label={label.text}
          />
        ))}
      </div>
    </div>
  );
}
