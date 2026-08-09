"use client";

// HP-HMI 디자인의 시그니처 컴포넌트 -- 모델이 낸 모든 수치 아래에 같은
// 형태의 구간 트랙을 깔아 "점이 아니라 범위"임을 화면 문법으로 반복한다
// (지시서 §E, 보정 지시서 §I-1). 원래 모니터링 SUMMARY 전용
// `YieldGapBar`였던 것을 알림 기록 알람 목록에서도 그대로 재사용하기
// 위해 스케일(scaleMin/scaleMax)과 목표선(target) 유무를 props로 뺐다 --
// 계산 로직 자체는 그대로다.
//
// mini=true(계측 우선순위류의 96px 미니 자리)에서는 지시서대로 라벨을
// 아예 렌더하지 않는다 -- 라벨 충돌 배치 로직은 mini에서 쓰이지 않는다.

// F-2: 이 미만으로 두 틱 라벨의 x좌표(%)가 가까우면 텍스트가 겹쳐
// "90.791.0"처럼 붙어 보인다 -- 실측(스크린샷)한 겹침 사례가 2%p
// 간격이었으므로, 4~5자 라벨 폭을 감안해 여유 있게 잡는다.
const TICK_OVERLAP_THRESHOLD_PCT = 7;

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

export default function EvidenceBand({
  lo, hi, target = null, judgmentLine = null, scaleMin, scaleMax, mini = false,
  targetLabel = "목표", judgmentLabel = "판정선",
}: EvidenceBandProps) {
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

  const rawValues = [scaleMin, lo, hi, scaleMax];
  if (target != null) rawValues.push(target);
  const rawTicks = Array.from(new Set(rawValues.map((v) => Math.round(v * 10) / 10))).sort((a, b) => a - b);

  // 인접한 라벨이 임계 미만으로 가까우면 두 줄로 번갈아 배치한다(생략하지
  // 않는다 -- 전부 실제 값이라 하나를 지우면 정보 손실이다).
  const lastRowPct: [number, number] = [-Infinity, -Infinity];
  const ticks = rawTicks.map((value) => {
    const pct = pctOf(value);
    const row = pct - lastRowPct[0] >= TICK_OVERLAP_THRESHOLD_PCT ? 0 : pct - lastRowPct[1] >= TICK_OVERLAP_THRESHOLD_PCT ? 1 : 0;
    lastRowPct[row] = pct;
    const outOfRange = value < scaleMin ? "low" : value > scaleMax ? "high" : null;
    return { value, pct, row, outOfRange };
  });

  return (
    <div className="evidenceBand">
      <div className="evidenceBandTrack">
        <div
          className="evidenceBandFill"
          style={{ left: `${bandLoPct}%`, width: `${Math.max(bandHiPct - bandLoPct, 0.6)}%` }}
        />
        <div className="evidenceBandBoundary" style={{ left: `${bandLoPct}%` }} />
        <div className="evidenceBandBoundary" style={{ left: `${bandHiPct}%` }} />
        {judgmentPct != null && (
          <>
            <div className="evidenceBandJudgmentTick" style={{ left: `${judgmentPct}%` }} />
            <span className="evidenceBandJudgmentLabel" style={{ left: `${judgmentPct}%` }}>{judgmentLabel}</span>
          </>
        )}
        {targetPct != null && (
          <>
            <div className="evidenceBandTargetTick" style={{ left: `${targetPct}%` }} />
            <span className="evidenceBandTargetLabel" style={{ left: `${targetPct}%` }}>{targetLabel}</span>
          </>
        )}
      </div>
      <div className="evidenceBandTicks">
        {ticks.map((tick) => (
          <span
            key={tick.value}
            className={tick.row === 1 ? "evidenceBandTick evidenceBandTickRow2" : "evidenceBandTick"}
            style={{ left: `${tick.pct}%` }}
          >
            {tick.outOfRange === "low" && "← "}
            {tick.value.toFixed(1)}
            {tick.outOfRange === "high" && " →"}
          </span>
        ))}
      </div>
    </div>
  );
}
