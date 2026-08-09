"""FMEA(고장모드영향분석) 분석표 산출 -- 모니터링 홈의 '유의 인자' 표와
실행 과제/실험 확인 대상/확인 필요 대상 3개 레일을 대체한다.

타깃별 상위 인자 풀을 모아 중복을 제거하고, S(심각도)·O(발생도)·D(검출도)·
RPN을 산출한 뒤 실익(불량률 편차, defect_rate_deviation_pct) 기준으로 걸러
상위 7개만 남긴다. "구간 내/외 평균 Y"를 구하려면 원본 데이터가 필요하므로
이 계산은 전부 여기(백엔드)에서 끝내고, 프런트는 표시만 한다 (지시서 IA-5).

지시서 KA-1: 타깃 컬럼(Y1~Y5)은 "불량률"이지 수율이 아니다 -- 이 모듈이
산출하는 expected_defect_rate_pct/defect_rate_deviation_pct는 그 불량률
기준이고, 실익 필터도 이 값을 쓴다(방향·기준 불변). 진짜 수율(최종 Y
컬럼) 기준 값은 별도로 expected_yield_pct에 담는다 -- 둘을 합치지 않는다.

Config(범주형) 인자는 후보에서 제외한다 -- 수치형 권장구간 개념이 없고,
README/챗봇 컨텍스트 문서가 이미 밝힌 대로 약 600건 검정에서 FDR 통과 0건이라
"검증하고 배제한" 상태다(잠재 원인으로 등재하지 않는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.control_range import compute_control_range
from src.analysis.measurement_expansion import _measured_any_mask
from src.analysis.recommendations import compute_factor_recommendation
from src.analysis.screening.schema import FINAL_YIELD_COLUMN
from src.analysis.screening.selector import ParetoFactor, _row_to_factor

TOP_N_PER_TARGET = 5  # 타깃별로 이 안에서만 후보를 뽑는다 (지시서 IA-1)
TOP_N_FINAL = 7
# 실익 필터 하한 (지시서 IA-4) -- 불량률 편차(defect_rate_deviation_pct)
# 기준이다. 진짜 수율(Y) 기준이 아니다(지시서 KA-1: 이름이 값과 일치해야
# 한다).
MIN_DEFECT_RATE_DEVIATION_PP = 0.3
MIN_MNAR_SAMPLE = 30  # 이보다 표본이 적은 쪽은 MNAR 갭을 신뢰할 수 없어 None 처리

DETECTION_METHOD_LABELS: dict[str, str] = {
    "R": "In-line 샘플 계측",
    "D": "Defect 검사",
}


def _clamp_ceil(value: float, lo: int = 1, hi: int = 10) -> int:
    if not np.isfinite(value):
        return lo
    return int(min(hi, max(lo, math.ceil(value))))


def _target_loss_shares(eval_df: pd.DataFrame, targets: list[str]) -> dict[str, float]:
    """S(심각도)의 근거. Y1~Y5는 이미 "불량률"(%) 값이므로(scatter.py의
    축 라벨 "{target} 불량률 (%)" 참고) 평균값 자체가 그 타깃의 손실
    크기를 대표한다 -- 손실_기여율(target) = 평균(Y_target) / sum(평균(Y_i)).
    """
    means: dict[str, float] = {}
    for target in targets:
        if target not in eval_df.columns:
            means[target] = 0.0
            continue
        series = pd.to_numeric(eval_df[target], errors="coerce").dropna()
        means[target] = float(series.mean()) if len(series) else 0.0
    total = sum(means.values())
    if total <= 0:
        return {target: 0.0 for target in targets}
    return {target: (means[target] / total * 100.0) for target in targets}


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
    severity_score: int
    occurrence_score: int
    detection_score: int
    rpn: int
    mnar_gap_pp: float | None


def _score_factor(
    eval_df: pd.DataFrame, factor: ParetoFactor, loss_share_pct: float, total_wafers: int, *, dataset_id: str
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
    # 나쁜지로 잰다. 반대로(in - out) 재면 실제로 유의한 인자일수록
    # 항상 음수가 나와(구간 안이 항상 더 낮으므로) 실익 필터가 유의미한
    # 인자를 전부 걸러내 버린다 -- 실제 train.CSV로 검증(16개 후보 전부
    # 음수로 배제). 실익 필터(build_fmea_table의 MIN_DEFECT_RATE_DEVIATION_PP)
    # 는 이 값(불량률 기준)을 그대로 쓴다 -- 방향도 기준도 바꾸지 않는다
    # (지시서 KA-1 "하지 말 것").
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

    severity = _clamp_ceil(loss_share_pct / 5.0)
    occurrence = _clamp_ceil(deviation_rate_pct / 10.0)
    detection = _clamp_ceil((100.0 - measurement_rate) / 10.0)
    rpn = severity * occurrence * detection

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
        severity_score=severity,
        occurrence_score=occurrence,
        detection_score=detection,
        rpn=rpn,
        mnar_gap_pp=_mnar_gap_pp(eval_df, factor.feature),
    )


@dataclass
class FmeaTable:
    items: list[FmeaFactor]
    excluded_count: int
    excluded_negative_count: int
    measurement_shortage_wafers: int
    correlation_shortage_wafers: int
    total_wafers: int


def build_fmea_table(
    eval_df: pd.DataFrame,
    rows_by_target: dict[str, list[dict]],
    targets: list[str],
    *,
    dataset_id: str,
    top_n_per_target: int = TOP_N_PER_TARGET,
    top_n_final: int = TOP_N_FINAL,
) -> FmeaTable:
    """전 타깃 인자를 모아 RPN 상위 top_n_final개를 뽑는다 (지시서 IA-1~IA-4).

    1. 타깃별 상위 top_n_per_target개(이미 eps2 내림차순인 rows_by_target
       기준)를 후보로 모은다.
    2. 인자명 기준 중복 제거 -- 같은 인자가 여러 타깃 상위에 있으면 eps2가
       가장 큰 타깃만 남긴다.
    3. 불량률 편차 실익 필터(defect_rate_deviation_pct < 0.3%p 또는 음수
       제외) 적용 -- 진짜 수율이 아니라 불량률 기준이다(지시서 KA-1).
    4. RPN 내림차순 정렬 후 top_n_final개.
    """
    total_wafers = len(eval_df)
    loss_shares = _target_loss_shares(eval_df, targets)

    candidates: dict[str, FmeaFactor] = {}
    for target in targets:
        rows = (rows_by_target.get(target) or [])[:top_n_per_target]
        for row in rows:
            factor = _row_to_factor(eval_df, target, row)
            scored = _score_factor(
                eval_df, factor, loss_shares.get(target, 0.0), total_wafers, dataset_id=dataset_id
            )
            if scored is None:
                continue
            existing = candidates.get(scored.feature)
            if existing is None or scored.eps2 > existing.eps2:
                candidates[scored.feature] = scored

    excluded_count = 0
    excluded_negative_count = 0
    kept: list[FmeaFactor] = []
    for scored in candidates.values():
        deviation = scored.defect_rate_deviation_pct
        if deviation is None or deviation < MIN_DEFECT_RATE_DEVIATION_PP:
            excluded_count += 1
            if deviation is not None and deviation < 0:
                excluded_negative_count += 1
            continue
        kept.append(scored)

    kept.sort(key=lambda f: f.rpn, reverse=True)
    top = kept[:top_n_final]

    qualifying_features = [f.feature for f in top]
    excluded_features = [f for f in candidates if f not in qualifying_features]
    measured_qualifying = _measured_any_mask(eval_df, qualifying_features)
    measured_excluded = _measured_any_mask(eval_df, excluded_features)
    correlation_shortage_mask = (~measured_qualifying) & measured_excluded
    measurement_shortage_mask = (~measured_qualifying) & (~measured_excluded)

    return FmeaTable(
        items=top,
        excluded_count=excluded_count,
        excluded_negative_count=excluded_negative_count,
        measurement_shortage_wafers=int(measurement_shortage_mask.sum()),
        correlation_shortage_wafers=int(correlation_shortage_mask.sum()),
        total_wafers=total_wafers,
    )
