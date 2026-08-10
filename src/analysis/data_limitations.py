"""데이터 한계 진단 공용 계산 -- WE(FMEA 표의 "검출률" 열)과 WL(모니터링
홈의 계측 편향/분산 분해 블록)이 같은 지표를 공유한다.

계측 여부가 무작위가 아니라는 사실(MNAR)을 "이 인자가 계측된 wafer는
전체보다 결과가 다르다"가 아니라 "결과가 나쁜 wafer일수록 더 자주
계측됐다"로 직접 보여준다 -- 전체 계측률과 최악 10% wafer(해당 타깃
손실 상위 10%)에서의 계측률을 나란히 비교한다. 계측이 무작위였다면 두
값이 같아야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WORST_DECILE_FRACTION = 0.10
MIN_SAMPLE_FOR_RATE = 30  # 이보다 표본이 적으면 최악 10% 계측률을 신뢰할 수 없다


def worst_decile_measurement_rate(df: pd.DataFrame, feature: str, target: str) -> float | None:
    """`target`(불량률, 값이 클수록 나쁨) 기준 최악 10% wafer 부분집합에서
    `feature`의 계측률(%). 표본 부족이거나 컬럼이 없으면 None."""
    if feature not in df.columns or target not in df.columns:
        return None
    y = pd.to_numeric(df[target], errors="coerce")
    valid_idx = y.dropna().index
    if len(valid_idx) < MIN_SAMPLE_FOR_RATE:
        return None
    n_worst = max(1, int(round(len(valid_idx) * WORST_DECILE_FRACTION)))
    worst_idx = y.loc[valid_idx].sort_values(ascending=False).index[:n_worst]
    worst_measured = pd.to_numeric(df.loc[worst_idx, feature], errors="coerce").notna()
    return float(worst_measured.mean() * 100.0)


def overall_measurement_rate(df: pd.DataFrame, feature: str) -> float | None:
    if feature not in df.columns or len(df) == 0:
        return None
    return float(pd.to_numeric(df[feature], errors="coerce").notna().mean() * 100.0)


@dataclass(frozen=True)
class MnarRateRow:
    target: str
    feature: str
    overall_rate_pct: float
    worst_decile_rate_pct: float
    ratio: float  # worst_decile_rate_pct / overall_rate_pct


def build_mnar_rate_report(df: pd.DataFrame, factors: list[tuple[str, str]]) -> list[MnarRateRow]:
    """WL-1: (target, feature) 쌍마다 전체 계측률과 최악 10% 계측률·배수를
    계산해 배수 내림차순으로 반환한다. 최악 10%를 정하려면 Y가 필요하다 --
    호출부가 이미 Y1~Y5가 채워진 프레임(FMEA가 쓰는 것과 같은 eval
    프레임)을 넘겨야 한다."""
    rows: list[MnarRateRow] = []
    for target, feature in factors:
        overall = overall_measurement_rate(df, feature)
        worst = worst_decile_measurement_rate(df, feature, target)
        if overall is None or worst is None or overall <= 0:
            continue
        rows.append(
            MnarRateRow(
                target=target,
                feature=feature,
                overall_rate_pct=overall,
                worst_decile_rate_pct=worst,
                ratio=worst / overall,
            )
        )
    rows.sort(key=lambda r: r.ratio, reverse=True)
    return rows


@dataclass(frozen=True)
class VarianceDecomposition:
    """WL-2: 랏 단위로 관리해도 못 잡는 변동의 크기 -- ICC(1,1) 하나만
    보여주면 "0.007이 뭘 뜻하는지" 알 수 없어, 랏 개수 대비 무효과
    기댓값(1/랏당wafer수)을 함께 낸다."""

    lot_count: int
    wafers_per_lot: float  # 평균(랏마다 크기가 다를 수 있어 평균으로 표기)
    between_lot_pct: float  # var(랏평균) / var(Y) * 100
    within_lot_pct: float  # 100 - between_lot_pct
    no_effect_expected_pct: float  # 100 / wafers_per_lot
    icc: float  # ICC(1,1)


def compute_variance_decomposition(df: pd.DataFrame, *, lot_column: str = "Lot_ID", target_column: str = "Y") -> VarianceDecomposition | None:
    """WL-2: `between_lot_pct`는 의도적으로 편향 보정을 하지 않은 단순
    비율 var(랏평균)/var(Y)다 -- ANOVA로 표본 노이즈를 보정해 버리면(예:
    Shrout & Fleiss의 (MS_between-MS_within)/n0 보정) 랏 효과가 전혀 없는
    귀무가설 하에서도 이 값 자체가 그 보정으로 0 근처가 되어 버려,
    "무효과 기대값(1/랏당wafer수)과 비교했더니 비슷하더라"라는 이 차트의
    핵심 논증(WL-2: "무효과 기대값이 없으면 차트가 거짓말을 한다")을 할
    수 없다 -- 비교 대상 두 값이 같은 정의를 써야 그 비교가 의미 있다.
    실측 검증: train.CSV 기준 이 단순 비율이 4.49%로 무효과 기대값(1/25=
    4.0%)과 거의 같다(관측 코멘트의 근거).

    ICC(1,1)은 별도로, 표준 정의(Shrout & Fleiss 1979 One-way random
    effects) 그대로 보정해 계산한다 -- "랏 효과의 신뢰구간 추정치"라는
    본래 용도에는 편향 보정이 맞다. 두 지표(단순 비율 vs ICC)가 서로
    다른 값인 것은 의도된 것이다.
    """
    if lot_column not in df.columns or target_column not in df.columns:
        return None
    frame = df[[lot_column, target_column]].dropna()
    if frame.empty:
        return None
    y = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.assign(**{target_column: y}).dropna()
    groups = frame.groupby(lot_column)[target_column]
    lot_sizes = groups.size()
    k = len(lot_sizes)  # 랏 개수
    n = len(frame)  # 전체 wafer 수
    if k < 2 or n <= k:
        return None

    grand_mean = frame[target_column].mean()
    ss_between = float(sum(lot_sizes[lot] * (mean - grand_mean) ** 2 for lot, mean in groups.mean().items()))
    ss_within = float(sum(((group - group.mean()) ** 2).sum() for _lot, group in groups))
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # 불균형 설계의 평균 랏 크기(n0) -- Shrout & Fleiss 표준 공식. ICC 전용.
    sum_n = float(lot_sizes.sum())
    sum_n2 = float((lot_sizes**2).sum())
    n0 = (sum_n - sum_n2 / sum_n) / df_between if df_between > 0 else lot_sizes.mean()
    n0 = max(n0, 1e-9)

    var_between_corrected = max(0.0, (ms_between - ms_within) / n0)
    var_within_corrected = max(ms_within, 0.0)
    icc = (
        var_between_corrected / (var_between_corrected + var_within_corrected)
        if (var_between_corrected + var_within_corrected) > 0
        else 0.0
    )

    # 위 문단 설명대로, 차트에 그리는 between_lot_pct는 편향 보정 없는
    # 단순 var(랏평균)/var(Y)다.
    total_var = float(frame[target_column].var(ddof=1))
    lot_mean_var = float(groups.mean().var(ddof=1))
    between_pct = (lot_mean_var / total_var * 100.0) if total_var > 0 else 0.0

    wafers_per_lot = float(lot_sizes.mean())
    no_effect_expected_pct = (100.0 / wafers_per_lot) if wafers_per_lot > 0 else 0.0

    return VarianceDecomposition(
        lot_count=k,
        wafers_per_lot=wafers_per_lot,
        between_lot_pct=between_pct,
        within_lot_pct=100.0 - between_pct,
        no_effect_expected_pct=no_effect_expected_pct,
        icc=float(icc),
    )
