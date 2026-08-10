"""FMEA(고장모드영향분석) 분석표 산출 -- 모니터링 홈의 '유의 인자' 표와
실행 과제/실험 확인 대상/확인 필요 대상 3개 레일을 대체한다.

작업 지시서 WE: 행 선정은 더 이상 RPN 상위 N개가 아니다 -- 타깃별 파레토
기여율(selector.py의 contribution_pct)이 CORE_FACTOR_CONTRIBUTION_MIN
이상인 인자를 전부 남긴다(타깃당 0개일 수도, 2개 이상일 수도 있다 --
지금 데이터에서는 타깃당 1개씩 5행). YG: 이 임계는 원래 20%였고 "하지
말 것: 임계를 낮추지 마라"는 주석이 있었다 -- 작업 지시서가 명시적으로
10%로 낮추고 화면 전체에 통일하라고 요구해 그 결정을 src/analysis/
thresholds.py의 단일 상수로 대체했다(폐기 배경은 docs/decisions.md).
S(심각도)·O(발생도)·D(검출도)·RPN은 더 이상 계산하지 않는다
(선정에도 표시에도 안 쓴다). "구간 내/외 평균 Y"를 구하려면 원본 데이터가
필요하므로 이 계산은 전부 여기(백엔드)에서 끝내고, 프런트는 표시만 한다
(지시서 IA-5).

지시서 KA-1: 타깃 컬럼(Y1~Y5)은 "불량률"이지 수율이 아니다 -- 이 모듈이
산출하는 expected_defect_rate_pct/defect_rate_deviation_pct는 그 불량률
기준이다. 진짜 수율(최종 Y 컬럼) 기준 값은 별도로 expected_yield_pct에
담는다 -- 둘을 합치지 않는다. 정렬은 defect_rate_deviation_pct(불량률
편차) 내림차순이다(RPN을 없앴으므로 실익 순이 자연스럽다, 지시서 WE-4).

Config(범주형) 인자는 후보에서 제외한다 -- 수치형 권장구간 개념이 없고,
README/챗봇 컨텍스트 문서가 이미 밝힌 대로 약 600건 검정에서 FDR 통과 0건이라
"검증하고 배제한" 상태다(잠재 원인으로 등재하지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.control_range import compute_control_range
from src.analysis.data_limitations import worst_decile_measurement_rate
from src.analysis.measurement_expansion import _measured_any_mask
from src.analysis.recommendations import compute_factor_recommendation
from src.analysis.screening.schema import FINAL_YIELD_COLUMN
from src.analysis.screening.selector import ParetoFactor, _row_to_factor
from src.analysis.thresholds import CORE_FACTOR_CONTRIBUTION_MIN

# WE-2: 선정 기준 -- 타깃별 파레토 기여율(contribution_pct)이 이 값
# 이상인 인자를 전부 남긴다. RPN 기반 top_n 선정을 대체한다. YG/ZF-1:
# src/analysis/thresholds.py가 유일한 소스다.
MIN_CONTRIBUTION_PCT = CORE_FACTOR_CONTRIBUTION_MIN
MIN_MNAR_SAMPLE = 30  # 이보다 표본이 적은 쪽은 MNAR 갭을 신뢰할 수 없어 None 처리

DETECTION_METHOD_LABELS: dict[str, str] = {
    "R": "In-line 샘플 계측",
    "D": "Defect 검사",
}


def _mnar_gap_pp(eval_df: pd.DataFrame, feature: str) -> float | None:
    """계측 여부가 무작위가 아님(MNAR)을 인자 단위로 드러낸다 -- 이 인자가
    계측된 wafer군과 미계측 wafer군의 최종 수율(Y) 평균 차이. 표본이 너무
    작은 쪽이 있으면(MIN_MNAR_SAMPLE 미만) 신뢰할 수 없으므로 None."""
    if FINAL_YIELD_COLUMN not in eval_df.columns or feature not in eval_df.columns:
        return None
    measured = pd.to_numeric(eval_df[feature], errors="coerce").notna()
    final_y = pd.to_numeric(eval_df[FINAL_YIELD_COLUMN], errors="coerce")
    measured_y = final_y[measured].dropna()
    unmeasured_y = final_y[~measured].dropna()
    if len(measured_y) < MIN_MNAR_SAMPLE or len(unmeasured_y) < MIN_MNAR_SAMPLE:
        return None
    return float(measured_y.mean() - unmeasured_y.mean())


@dataclass
class FmeaFactor:
    target: str
    feature: str
    kind: str  # "R" | "D"
    step: int
    eps2: float
    relation_shape: str
    factor_value: float | None
    range_lo: float | None
    range_hi: float | None
    measurement_rate: float
    deviation_rate_pct: float  # O의 근거 -- 권장 구간 밖 wafer 비율 (계측된 wafer 기준)
    detection_method: str
    detection_kind: str
    # 지시서 KA-1: 이 둘은 타깃 컬럼(Y1~Y5, "불량률") 기준이다 -- 계산은
    # 그대로 두고 이름만 정정했다("예상 수율"이라 부르던 게 실제로는
    # 타깃 불량률이었다). 실익 필터는 이 defect_rate_deviation_pct를
    # 그대로 쓴다(방향 불변, KA-1 "하지 말 것").
    expected_defect_rate_pct: float | None
    defect_rate_deviation_pct: float | None
    # KA-1: 진짜 수율(최종 Y 컬럼) 기준 -- "이 인자를 관리하면 수율이
    # 어떻게 되는가"에 대한 답. 위 두 필드와 다른 질문이라 합치지 않는다.
    expected_yield_pct: float | None
    # WE-2: 이 행이 선정된 근거(타깃 내 파레토 기여율, %) -- 표에도 열로
    # 노출한다.
    contribution_pct: float
    # WE-3: "계측률과 검출률을 나란히" -- 최악 10% wafer(해당 타깃 손실
    # 상위 10%, train 기준)에서의 계측률. 전체 계측률보다 훨씬 높으면
    # "계측이 사후 확인에 가깝다"는 뜻이다(WL의 MNAR 계측 편향 차트와
    # 같은 지표). 표본 부족이면 None.
    worst_decile_measurement_rate_pct: float | None
    mnar_gap_pp: float | None


def _score_factor(
    eval_df: pd.DataFrame, factor: ParetoFactor, contribution_pct: float, total_wafers: int, *, dataset_id: str
) -> FmeaFactor | None:
    # Config는 잠재 원인으로 등재하지 않는다 -- 수치형 권장구간이 없고
    # 표본 600여 건에서 FDR 통과 0건이었다(README/챗봇 컨텍스트 근거).
    if factor.kind == "Config":
        return None
    if factor.feature not in eval_df.columns or factor.target not in eval_df.columns:
        return None

    x = pd.to_numeric(eval_df[factor.feature], errors="coerce")
    y = pd.to_numeric(eval_df[factor.target], errors="coerce")
    valid = x.notna() & y.notna()
    n_observed = int(valid.sum())
    if n_observed == 0 or total_wafers <= 0:
        return None
    x_valid, y_valid = x[valid], y[valid]

    measurement_rate = (n_observed / total_wafers) * 100.0
    factor_value = float(x_valid.median())

    # "권장 구간"은 SPC 관리한계(IQR*1.5, 알람용으로 일부러 넓게 잡은
    # 값)가 아니라 recommendations.py가 산출하는, 관리한계 안쪽으로
    # clamp된 더 좁은 구간이다 -- README의 권장 구간 표(예: Step28_R1
    # 54.7~61.5)와 alarm_bands.py "구간별 평균 수율" 카드가 이미 이
    # 값을 "권장 구간"으로 쓰고 있다. SPC 관리한계를 그대로 쓰면
    # (드물게만 이탈하도록 일부러 넓힌 값이라) 이탈률이 항상 한 자리
    # 수로 나와 O(발생도)가 사실상 무의미해진다.
    control_range = compute_control_range(eval_df, factor)
    recommendation = compute_factor_recommendation(eval_df, factor, control_range, dataset_id=dataset_id)
    if recommendation is None:
        return None
    range_lo, range_hi = recommendation.recommended_lo, recommendation.recommended_hi

    in_mask = (x_valid >= range_lo) & (x_valid <= range_hi)
    out_mask = ~in_mask

    out_count = int(out_mask.sum())
    deviation_rate_pct = (out_count / n_observed) * 100.0

    # KA-1: 이 둘은 "타깃 컬럼"(Y1~Y5) 기준 -- 진짜 수율이 아니라 그
    # 타깃의 불량률이다(scatter.py 축 라벨 "{target} 불량률 (%)" 참고).
    expected_defect_rate_pct = float(y_valid[in_mask].mean()) if in_mask.any() else None
    outside_defect_rate_pct = float(y_valid[out_mask].mean()) if out_mask.any() else None
    # 권장 구간은 평균 불량률이 낮은 쪽으로 선정된다(recommendations.py)
    # -- 따라서 "실익"은 구간 밖(out)이 구간 안(in)보다 얼마나 더
    # 나쁜지로 잰다. 반대로(in - out) 재면 실제로 유의한 인자일수록 항상
    # 음수가 나온다(구간 안이 항상 더 낮으므로). 이 값이 표(WE)의 정렬
    # 기준이다(내림차순, WE-4) -- 방향·기준은 바꾸지 않는다(지시서 KA-1
    # "하지 말 것").
    defect_rate_deviation_pct = (
        outside_defect_rate_pct - expected_defect_rate_pct
        if expected_defect_rate_pct is not None and outside_defect_rate_pct is not None
        else None
    )

    # KA-1: 진짜 수율(최종 Y 컬럼) 기준 -- "이 인자를 권장 구간 안으로
    # 관리하면 최종 수율이 얼마가 되는가"에 대한 답. 위 불량률 값과는
    # 다른 질문이라 별도 열로 낸다(합치지 않는다).
    expected_yield_pct: float | None = None
    if FINAL_YIELD_COLUMN in eval_df.columns:
        final_y = pd.to_numeric(eval_df[FINAL_YIELD_COLUMN], errors="coerce")
        in_range_final_y = final_y.loc[x_valid.index[in_mask]].dropna()
        if len(in_range_final_y) > 0:
            expected_yield_pct = float(in_range_final_y.mean())

    return FmeaFactor(
        target=factor.target,
        feature=factor.feature,
        kind=factor.kind,
        step=factor.step,
        eps2=factor.eps2,
        relation_shape=factor.relation_shape,
        factor_value=factor_value,
        range_lo=range_lo,
        range_hi=range_hi,
        measurement_rate=measurement_rate,
        deviation_rate_pct=deviation_rate_pct,
        detection_method=DETECTION_METHOD_LABELS.get(factor.kind, factor.kind),
        detection_kind=factor.kind,
        expected_defect_rate_pct=expected_defect_rate_pct,
        defect_rate_deviation_pct=defect_rate_deviation_pct,
        expected_yield_pct=expected_yield_pct,
        contribution_pct=contribution_pct,
        worst_decile_measurement_rate_pct=worst_decile_measurement_rate(eval_df, factor.feature, factor.target),
        mnar_gap_pp=_mnar_gap_pp(eval_df, factor.feature),
    )


@dataclass
class NoQualifyingFactor:
    """WE-2/YG: 임계(10%) 이상 인자가 없는 타깃 -- 행을 만드는 대신 사유로 남긴다."""

    target: str
    max_contribution_pct: float


@dataclass
class FmeaTable:
    items: list[FmeaFactor]
    measurement_shortage_wafers: int
    correlation_shortage_wafers: int
    total_wafers: int
    no_qualifying_factor: list[NoQualifyingFactor]


def build_fmea_table(
    eval_df: pd.DataFrame,
    rows_by_target: dict[str, list[dict]],
    targets: list[str],
    *,
    dataset_id: str,
    min_contribution_pct: float = MIN_CONTRIBUTION_PCT,
) -> FmeaTable:
    """작업 지시서 WE-2: 타깃별로 파레토 기여율(contribution_pct)이
    `min_contribution_pct` 이상인 인자를 전부 남긴다 -- 개수 상한이 없다
    (지금 데이터에서는 타깃당 1개씩 5행이지만, 2위가 임계를 넘으면
    자동으로 6행·7행이 된다). 임계 이상이 하나도 없는 타깃은 행 대신
    `no_qualifying_factor`에 사유(최대 기여율)로 남는다. 정렬은
    defect_rate_deviation_pct(불량률 편차) 내림차순(WE-4).
    """
    total_wafers = len(eval_df)

    kept: list[FmeaFactor] = []
    no_qualifying: list[NoQualifyingFactor] = []
    for target in targets:
        rows = rows_by_target.get(target) or []  # 이미 eps2/contribution_pct 내림차순
        qualifying_rows = [r for r in rows if r["contribution_pct"] >= min_contribution_pct]
        if not qualifying_rows:
            max_pct = max((r["contribution_pct"] for r in rows), default=0.0)
            no_qualifying.append(NoQualifyingFactor(target=target, max_contribution_pct=max_pct))
            continue
        for row in qualifying_rows:
            factor = _row_to_factor(eval_df, target, row)
            scored = _score_factor(eval_df, factor, row["contribution_pct"], total_wafers, dataset_id=dataset_id)
            if scored is not None:
                kept.append(scored)

    kept.sort(key=lambda f: f.defect_rate_deviation_pct if f.defect_rate_deviation_pct is not None else float("-inf"), reverse=True)

    qualifying_features = [f.feature for f in kept]
    all_candidate_features = [row["feature"] for rows in rows_by_target.values() for row in rows]
    excluded_features = [f for f in dict.fromkeys(all_candidate_features) if f not in qualifying_features]
    measured_qualifying = _measured_any_mask(eval_df, qualifying_features)
    measured_excluded = _measured_any_mask(eval_df, excluded_features)
    correlation_shortage_mask = (~measured_qualifying) & measured_excluded
    measurement_shortage_mask = (~measured_qualifying) & (~measured_excluded)

    return FmeaTable(
        items=kept,
        measurement_shortage_wafers=int(measurement_shortage_mask.sum()),
        correlation_shortage_wafers=int(correlation_shortage_mask.sum()),
        total_wafers=total_wafers,
        no_qualifying_factor=no_qualifying,
    )
