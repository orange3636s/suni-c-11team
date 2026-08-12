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


MIN_OVERALL_RATE_FOR_MNAR_PCT = 1.0  # 이보다 계측률이 낮은 인자는 표본 부족으로 배수가 불안정해 제외
MNAR_TOP_N = 10


def build_mnar_rate_report(
    df: pd.DataFrame,
    factors: list[tuple[str, str]],
    *,
    min_overall_rate_pct: float | None = None,
    top_n: int | None = None,
) -> list[MnarRateRow]:
    """(target, feature) 쌍마다 전체 계측률과 최악 10% 계측률·배수를
    계산해 배수 내림차순으로 반환한다. 최악 10%를 정하려면 Y가 필요하다 --
    호출부가 train.CSV처럼 실제 Y가 채워진 프레임을 넘겨야 한다(예측 Y로
    채운 프레임을 넘기면 배수가 부풀려진다 -- 예측이 나쁜 wafer일수록
    핵심 인자가 계측된 wafer이기 때문).

    `min_overall_rate_pct`를 주면 그 미만인 인자는 표본이 적어 배수가
    불안정하므로 제외한다. `top_n`을 주면 배수 상위 N개만 남긴다."""
    rows: list[MnarRateRow] = []
    for target, feature in factors:
        overall = overall_measurement_rate(df, feature)
        worst = worst_decile_measurement_rate(df, feature, target)
        if overall is None or worst is None or overall <= 0:
            continue
        if min_overall_rate_pct is not None and overall < min_overall_rate_pct:
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
    if top_n is not None:
        rows = rows[:top_n]
    return rows


@dataclass(frozen=True)
class VarianceDecomposition:
    """랏 단위로 관리해도 못 잡는 변동의 크기 -- ICC(1,1) 하나만
    보여주면 "0.007이 뭘 뜻하는지" 알 수 없어, 랏 개수 대비 무효과
    기댓값(1/랏당wafer수)을 함께 낸다."""

    lot_count: int
    wafers_per_lot: float  # 평균(랏마다 크기가 다를 수 있어 평균으로 표기)
    between_lot_pct: float  # var(랏평균) / var(Y) * 100
    within_lot_pct: float  # 100 - between_lot_pct
    no_effect_expected_pct: float  # 100 / wafers_per_lot
    icc: float  # ICC(1,1)


def compute_variance_decomposition(df: pd.DataFrame, *, lot_column: str = "Lot_ID", target_column: str = "Y") -> VarianceDecomposition | None:
    """`between_lot_pct`는 의도적으로 편향 보정을 하지 않은 단순
    비율 var(랏평균)/var(Y)다 -- ANOVA로 표본 노이즈를 보정해 버리면(예:
    Shrout & Fleiss의 (MS_between-MS_within)/n0 보정) 랏 효과가 전혀 없는
    귀무가설 하에서도 이 값 자체가 그 보정으로 0 근처가 되어 버려,
    "무효과 기대값(1/랏당wafer수)과 비교했더니 비슷하더라"라는 이 차트의
    핵심 논증("무효과 기대값이 없으면 차트가 거짓말을 한다")을 할
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


MIN_SAMPLE_FOR_MODE_SHARE = 30  # 랏 간/랏 내 분해의 MIN_SAMPLE_FOR_RATE와 같은 하한


@dataclass(frozen=True)
class ModeVarianceShareRow:
    """모니터링 홈 블록③ 「분산 분해」 하단 -- 랏 간/랏 내로 쪼갠 변동을
    불량모드(Y1~Y5)로 한 번 더 쪼갠다."""

    target: str
    mean_loss_pp: float
    mean_share_pct: float
    variance_share_pct: float


def compute_mode_variance_share(
    df: pd.DataFrame, targets: tuple[str, ...] = ("Y1", "Y2", "Y3", "Y4", "Y5")
) -> list[ModeVarianceShareRow] | None:
    """불량모드별 "변동 기여"(수율 wafer간 편차를 만드는 몫).

    총 손실 L = Y1+...+Y5, Y = 100 - L 이므로 var(Y) = var(L)이다.
    공분산의 선형성에 의해 var(L) = sum_i cov(Y_i, L)이 정확히 성립하므로

        variance_share_pct(i) = cov(Y_i, L) / var(L) * 100

    로 정의하면 합이 반드시 100%가 된다. 단순히 var(Y_i)의 비율을 쓰면
    모드 간 상관 성분이 누락돼 합이 100%가 되지 않는다 -- 그래서 이
    정의를 쓴다. ddof=1로 랏 간/랏 내 분해와 표기를 통일한다.
    """
    if any(t not in df.columns for t in targets):
        return None
    frame = df[list(targets)].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna()
    if len(frame) < MIN_SAMPLE_FOR_MODE_SHARE:
        return None

    total_loss = frame.sum(axis=1)
    var_total = float(total_loss.var(ddof=1))
    if var_total <= 0:
        return None

    rows: list[ModeVarianceShareRow] = []
    mean_total_loss = float(total_loss.mean())
    for t in targets:
        mean_loss = float(frame[t].mean())
        cov = float(np.cov(frame[t], total_loss, ddof=1)[0, 1])
        rows.append(
            ModeVarianceShareRow(
                target=t,
                mean_loss_pp=mean_loss,
                mean_share_pct=(mean_loss / mean_total_loss * 100.0) if mean_total_loss > 0 else 0.0,
                variance_share_pct=cov / var_total * 100.0,
            )
        )
    rows.sort(key=lambda r: r.variance_share_pct, reverse=True)
    return rows


MIN_WAFERS_FOR_COVERAGE = 1


@dataclass(frozen=True)
class CoreFactorCoverageRow:
    """모니터링 홈 블록⑤ 「핵심 인자 커버리지」 -- wafer 한 장이 핵심 인자
    몇 개로 판정됐는가. 지금 분석 중인 배치(eval, 보통 test_remove_y.CSV)
    기준이다 -- train과 달리 여기는 "이 배치의 계측 상태"를 묻는 질문이라
    train을 쓰면 안 된다."""

    measured_count: int
    wafer_count: int
    pct: float


def compute_core_factor_coverage(df: pd.DataFrame, core_features: list[str]) -> list[CoreFactorCoverageRow] | None:
    """`core_features`(핵심 인자, 보통 FMEA 표의 (타깃, 인자) 쌍에서 뽑은
    인자 목록) 중 몇 개가 계측됐는지를 wafer마다 세어 0개~전체개수까지
    분포로 반환한다. 핵심 인자가 하나도 없거나 데이터가 없으면 None."""
    features = [f for f in core_features if f in df.columns]
    total = len(df)
    if not features or total < MIN_WAFERS_FOR_COVERAGE:
        return None
    counts = df[features].notna().sum(axis=1)
    max_count = len(features)
    rows: list[CoreFactorCoverageRow] = []
    for k in range(max_count + 1):
        wafer_count = int((counts == k).sum())
        rows.append(CoreFactorCoverageRow(measured_count=k, wafer_count=wafer_count, pct=wafer_count / total * 100.0))
    return rows


MIN_SAMPLE_FOR_COOCCURRENCE = 30


def compute_defect_cooccurrence_matrix(
    df: pd.DataFrame,
    targets: tuple[str, ...] = ("Y1", "Y2", "Y3", "Y4", "Y5"),
    *,
    fraction: float = WORST_DECILE_FRACTION,
) -> list[list[float | None]] | None:
    """모니터링 홈 블록⑥ 「불량 원인 독립성」 -- 두 불량 원인이 동시에
    상위 `fraction`(기본 10%)에 드는 wafer 비율을 원인쌍마다 계산한다.
    두 원인이 독립이면 이 비율의 기댓값은 `fraction * fraction * 100`
    (기본 1.00%)이다. train.CSV 기준(원인 값 자체가 필요하므로 eval로는
    계산할 수 없다). 대각선은 None. 표본 부족이면 전체를 None으로
    반환한다."""
    if any(t not in df.columns for t in targets):
        return None
    frame = df[list(targets)].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(frame)
    if n < MIN_SAMPLE_FOR_COOCCURRENCE:
        return None
    n_worst = max(1, round(n * fraction))

    worst_masks: dict[str, pd.Series] = {}
    for t in targets:
        worst_idx = frame[t].sort_values(ascending=False).index[:n_worst]
        mask = pd.Series(False, index=frame.index)
        mask.loc[worst_idx] = True
        worst_masks[t] = mask

    matrix: list[list[float | None]] = []
    for a in targets:
        row: list[float | None] = []
        for b in targets:
            if a == b:
                row.append(None)
                continue
            both = int((worst_masks[a] & worst_masks[b]).sum())
            row.append(both / n * 100.0)
        matrix.append(row)
    return matrix
