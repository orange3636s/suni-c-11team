// 산점도 위 "구간 평균 불량률" 추세를 F 검정으로 차수를 고른 1·2차 다항
// 곡선으로 그린다. ScatterChart/CompareAcrossConfigsModal/
// CompareAcrossTargetsModal 세 곳이 이 한 모듈을 공유한다. 순수 프런트
// 렌더링 오버레이다 -- data.points 위에 그릴 뿐, 최적 중심·권장 구간·알람
// 판정(백엔드 src/analysis/*)에는 전혀 관여하지 않는다.

export type CurveFitResult = {
  degree: 1 | 2;
  /** [a, b] for degree 1 (y = a + bx), [a, b, c] for degree 2 (y = a + bx + cx^2). */
  coeffs: number[];
  /** R² of the ADOPTED degree. */
  r2: number;
  r2_linear: number;
  /** null if the quadratic fit wasn't attempted/valid (e.g. singular matrix). */
  r2_quadratic: number | null;
  fStatistic: number | null;
  pValue: number | null;
  /** Observed x min/max -- evaluateCurve never extrapolates beyond this. */
  domain: [number, number];
  /** Non-null means: don't draw a curve, fall back to the stepped
   * bin-average line. */
  fallbackReason: string | null;
};

// -- 선형대수: Gauss-Jordan 소거(부분 피벗) ---------------------------------
// 2x2(1차)/3x3(2차) 정규방정식을 풀 뿐이라 외부 라이브러리 없이 짧은
// 소거법으로 충분하다. 피벗이 거의 0이면(특이 행렬) null을 돌려준다 --
// 호출부가 이를 "수치적으로 퇴화" 폴백 사유로 취급한다.
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivotRow = col;
    let maxAbs = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(M[r][col]) > maxAbs) {
        maxAbs = Math.abs(M[r][col]);
        pivotRow = r;
      }
    }
    if (maxAbs < 1e-10) return null;
    if (pivotRow !== col) {
      const tmp = M[col];
      M[col] = M[pivotRow];
      M[pivotRow] = tmp;
    }
    const pivot = M[col][col];
    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const factor = M[r][col] / pivot;
      if (factor === 0) continue;
      for (let c = col; c <= n; c += 1) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// -- F(1,df2) 분포의 생존함수(꼬리 확률) -----------------------------------
// F(1,df2)의 제곱근은 Student's-t(df2)이므로, 정칙화 불완전 베타 함수로
// 계산한다: P(F(1,df2) > F) = I_{df2/(df2+F)}(df2/2, 1/2). 아래
// regularizedIncompleteBeta/betacf/gammaln은 Numerical Recipes의 표준
// 연분수 알고리즘을 그대로 옮긴 것 -- jStat 등 여러 통계 라이브러리가 쓰는
// 것과 같은 알고리즘이다.
function gammaln(xx: number): number {
  const cof = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ];
  let x = xx;
  let y = xx;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j += 1) {
    y += 1;
    ser += cof[j] / y;
  }
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

function betacf(x: number, a: number, b: number): number {
  const MAXIT = 200;
  const EPS = 3e-9;
  const FPMIN = 1e-30;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** Regularized incomplete beta I_x(a, b), x in [0, 1]. */
function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(
    gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x),
  );
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(x, a, b)) / a;
  }
  return 1 - (bt * betacf(1 - x, b, a)) / b;
}

/** P(F(1, df2) > F) via the regularized incomplete beta function. */
function fSurvival1(F: number, df2: number): number {
  if (F <= 0) return 1;
  if (df2 <= 0) return NaN;
  return regularizedIncompleteBeta(df2 / (df2 + F), df2 / 2, 0.5);
}

function degenerate(
  points: { x: number; y: number }[],
  domain: [number, number],
  reason: string,
): CurveFitResult {
  const meanY = points.length ? points.reduce((sum, p) => sum + p.y, 0) / points.length : 0;
  return {
    degree: 1,
    coeffs: [meanY, 0],
    r2: 0,
    r2_linear: 0,
    r2_quadratic: null,
    fStatistic: null,
    pValue: null,
    domain,
    fallbackReason: reason,
  };
}

/** OLS-fits a 1st and (if valid) 2nd degree polynomial to `points`, picks
 * the degree via an F-test comparing the nested models, and reports
 * whether the caller should fall back to the stepped bin-average line
 * instead of drawing a curve at all. Pure function -- no fetch, no
 * mutation, safe to call from a `useMemo`. */
export function fitDefectRateCurve(points: { x: number; y: number }[]): CurveFitResult {
  const n = points.length;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const domain: [number, number] = n > 0 ? [Math.min(...xs), Math.max(...xs)] : [0, 0];

  if (n < 30) return degenerate(points, domain, `표본 부족 (n=${n})`);
  const distinctX = new Set(xs).size;
  if (distinctX < 5) return degenerate(points, domain, `x 고유값 부족 (${distinctX}개)`);

  let Sx = 0, Sx2 = 0, Sx3 = 0, Sx4 = 0, Sy = 0, Sxy = 0, Sx2y = 0;
  for (let i = 0; i < n; i += 1) {
    const x = xs[i];
    const y = ys[i];
    const x2 = x * x;
    const x3 = x2 * x;
    const x4 = x3 * x;
    Sx += x; Sx2 += x2; Sx3 += x3; Sx4 += x4;
    Sy += y; Sxy += x * y; Sx2y += x2 * y;
  }

  const linearSol = solveLinearSystem([[n, Sx], [Sx, Sx2]], [Sy, Sxy]);
  if (!linearSol) return degenerate(points, domain, "적합 실패 (특이 행렬)");
  const [aLin, bLin] = linearSol;

  const meanY = Sy / n;
  let rssLinear = 0;
  let tss = 0;
  for (let i = 0; i < n; i += 1) {
    const yHat = aLin + bLin * xs[i];
    rssLinear += (ys[i] - yHat) ** 2;
    tss += (ys[i] - meanY) ** 2;
  }
  const r2Linear = tss > 0 ? 1 - rssLinear / tss : 0;

  const quadSol = solveLinearSystem(
    [[n, Sx, Sx2], [Sx, Sx2, Sx3], [Sx2, Sx3, Sx4]],
    [Sy, Sxy, Sx2y],
  );

  let r2Quadratic: number | null = null;
  let fStatistic: number | null = null;
  let pValue: number | null = null;
  let adoptQuadratic = false;
  let quadCoeffs: number[] = [aLin, bLin, 0];

  if (quadSol) {
    const [aQ, bQ, cQ] = quadSol;
    quadCoeffs = [aQ, bQ, cQ];
    let rssQuad = 0;
    for (let i = 0; i < n; i += 1) {
      const yHat = aQ + bQ * xs[i] + cQ * xs[i] * xs[i];
      rssQuad += (ys[i] - yHat) ** 2;
    }
    r2Quadratic = tss > 0 ? 1 - rssQuad / tss : 0;
    const df2 = n - 3;
    if (df2 > 0) {
      if (rssQuad <= 1e-12) {
        // Perfect (or near-perfect) quadratic fit -- F is effectively
        // infinite; treat as maximally significant rather than dividing by
        // ~0 (which would otherwise produce Infinity/NaN noise).
        fStatistic = Number.POSITIVE_INFINITY;
        pValue = 0;
      } else {
        const F = ((rssLinear - rssQuad) / 1) / (rssQuad / df2);
        fStatistic = F;
        pValue = fSurvival1(F, df2);
      }
      // p < 0.01 AND 2차 계수(c) > 0(아래로 볼록, 즉 U자) 일 때만
      // 2차를 채택한다. c <= 0("가운데가 최악")은 불량률 곡선의 "권장
      // 구간" 개념과 모순되므로 통계적으로 유의해도 버린다.
      if (pValue !== null && pValue < 0.01 && cQ > 0) adoptQuadratic = true;
    }
  }

  const degree: 1 | 2 = adoptQuadratic ? 2 : 1;
  const coeffs = adoptQuadratic ? quadCoeffs : [aLin, bLin];
  const r2 = adoptQuadratic ? (r2Quadratic as number) : r2Linear;

  if (r2 < 0.02) {
    return {
      degree, coeffs, r2, r2_linear: r2Linear, r2_quadratic: r2Quadratic,
      fStatistic, pValue, domain,
      fallbackReason: `설명력 낮음 (R²=${r2.toFixed(2)})`,
    };
  }

  return {
    degree, coeffs, r2, r2_linear: r2Linear, r2_quadratic: r2Quadratic,
    fStatistic, pValue, domain, fallbackReason: null,
  };
}

/** Evaluates the fitted curve at `x`, clamped to the observed domain --
 * never extrapolates beyond where data was actually seen. */
export function evaluateCurve(fit: CurveFitResult, x: number): number {
  const [lo, hi] = fit.domain;
  const clamped = Math.min(Math.max(x, lo), hi);
  if (fit.degree === 2) {
    const [a, b, c] = fit.coeffs;
    return a + b * clamped + c * clamped * clamped;
  }
  const [a, b] = fit.coeffs;
  return a + b * clamped;
}
