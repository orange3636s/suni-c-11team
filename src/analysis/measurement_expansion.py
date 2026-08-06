"""'계측 확대 권고' 카드 (spec 문구 전수 검토 PART B) -- 5개 1위(primary)
인자의 계측률을 +10%p씩 늘리면 얼마나 더 많은 wafer를 판정할 수 있고, 그중
조치 대상(권장구간 밖)이 얼마나 되며, 그 결과 기대 수율이 얼마나 개선되는지
추정한다. 또한 현재 FDR을 통과하지 못한 인자 중 표본이 2배가 되면 통과할
것으로 추정되는 것을 계산하고, 3개의 부가 효과 카드를 구성한다.

"원인 분석 실행" 1회당 한 번만 계산되어 분석 결과에 포함된다 (spec §B-7:
"카드를 열 때마다 재계산하지 마라"). 시뮬레이션은 `random_state`를 고정해
새로고침해도 값이 바뀌지 않는다 (spec §B-2).

부트스트랩 변동계수는 권장구간 산출 시 이미 계산된 `MethodWindow.width_sd`를
재사용한다 (spec §B-7: "이미 계산된 값을 재사용한다"). 여기서 새로 부트스트랩을
돌리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.alarm_bands import WholeWaferBands
from src.analysis.recommendations import FactorRecommendation
from src.analysis.screening.selector import ParetoFactor, benjamini_hochberg

N_SIMULATION_TRIALS = 30
EXTRA_MEASUREMENT_RATE = 0.10
DOUBLING_FACTOR = 2.0
MIN_ACTION_BLOCKED_SHARE = 0.10  # spec §B-6: below this, collapse to one line
MIN_DISCOVERY_EPS2 = 0.02
MAX_DISCOVERY_CARDS = 3

RATE_LOW_THRESHOLD = 0.08
CV_HIGH_THRESHOLD = 0.18
CV_MODERATE_THRESHOLD = 0.12


def _measured_bool_columns(eval_df: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    """`features`별 계측 여부를 순수 numpy 불리언 배열(위치 기반)로 미리
    한 번씩만 계산해 캐시한다 -- pandas `.loc`/`.reindex`는 라벨 정렬 비용이
    붙어 트라이얼마다 반복 호출하면 느리다 (실측: train.CSV처럼 eval이
    10,000행이면 트라이얼 30회 x 인자 여러 개에서 수십 초까지 걸렸다).
    """
    return {
        f: pd.to_numeric(eval_df[f], errors="coerce").notna().to_numpy()
        for f in features
        if f in eval_df.columns
    }


def _measured_any_mask(eval_df: pd.DataFrame, features: list[str]) -> pd.Series:
    """wafer별로 `features` 중 하나라도 계측되어 있는지 -- '조치 불가'(전부
    미계측) 여부의 반대. 시뮬레이션의 기준선(`original_any`)이자, 개별 인자
    시뮬레이션에서 "원래도 판정 가능했던 wafer"를 걸러내는 데 재사용한다.
    """
    columns = _measured_bool_columns(eval_df, features)
    if not columns:
        return pd.Series(False, index=eval_df.index)
    any_measured = np.zeros(len(eval_df), dtype=bool)
    for values in columns.values():
        any_measured |= values
    return pd.Series(any_measured, index=eval_df.index)


def _simulate_additional_judged(
    eval_df: pd.DataFrame,
    features: list[str],
    original_any: pd.Series,
    *,
    n_trials: int = N_SIMULATION_TRIALS,
    extra_rate: float = EXTRA_MEASUREMENT_RATE,
) -> int:
    """spec §B-2 시뮬레이션 방식 그대로: `features` 각각의 미계측 wafer 중
    무작위로 전체의 10%p씩 추가 계측한다고 가정, n_trials회 반복 평균(반올림).
    trial 인덱스를 `random_state`로 고정해 같은 입력이면 항상 같은 값이 나온다.
    위치 기반 numpy 배열로만 계산해(라벨 인덱싱 없이) 큰 데이터셋에서도
    빠르게 끝난다.
    """
    columns = _measured_bool_columns(eval_df, features)
    if not columns:
        return 0
    n_total = len(eval_df)
    pick_n = int(n_total * extra_rate)
    if pick_n <= 0 or n_total == 0:
        return 0

    original_any_arr = original_any.to_numpy()
    base_arrays = list(columns.values())

    gains: list[int] = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        trial_any = np.zeros(n_total, dtype=bool)
        for base in base_arrays:
            updated = base.copy()
            unmeasured_pos = np.flatnonzero(~base)
            if len(unmeasured_pos) > 0:
                n_pick = min(pick_n, len(unmeasured_pos))
                picked = rng.choice(unmeasured_pos, size=n_pick, replace=False)
                updated[picked] = True
            trial_any |= updated
        gains.append(int(np.sum(trial_any & ~original_any_arr)))
    return int(round(float(np.mean(gains))))


def _simulate_single_factor_judged(
    eval_df: pd.DataFrame,
    feature: str,
    original_any: pd.Series,
    *,
    n_trials: int = N_SIMULATION_TRIALS,
    extra_rate: float = EXTRA_MEASUREMENT_RATE,
) -> int:
    """B-3 인자별 우선순위 표의 '추가 판정' -- 이 인자 하나만 +10%p 계측을
    늘렸을 때 새로 '판정 가능'(5개 인자 중 하나라도 계측)해지는 wafer 수.
    이 인자만 놓고 보면 뽑히는 wafer 전부가 그 인자 기준으로는 항상
    '새로 계측됨'이므로, 원래(5개 전체 기준) 전부 미계측이던 wafer만 걸러
    세지 않으면 모든 인자가 항상 같은 pick_n으로 나와 서로 구분되지 않는다.
    """
    if feature not in eval_df.columns:
        return 0
    n_total = len(eval_df)
    pick_n = int(n_total * extra_rate)
    if pick_n <= 0 or n_total == 0:
        return 0

    measured = pd.to_numeric(eval_df[feature], errors="coerce").notna().to_numpy()
    was_blocked = ~original_any.to_numpy()
    unmeasured_pos = np.flatnonzero(~measured)
    if len(unmeasured_pos) == 0:
        return 0

    gains: list[int] = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        n_pick = min(pick_n, len(unmeasured_pos))
        picked = rng.choice(unmeasured_pos, size=n_pick, replace=False)
        gains.append(int(np.sum(was_blocked[picked])))
    return int(round(float(np.mean(gains))))


@dataclass
class FactorPriority:
    feature: str
    target: str
    measurement_rate: float  # 0-100, train.CSV 기준
    recommendation: str  # "+10%p" | "+15%p" | "유지"
    # 통계 용어(변동계수/ε²/부트스트랩) 노출 금지 (spec §B-3) -- 미리 정해둔
    # 문장 중 하나를 그대로 쓴다.
    reason: str
    additional_judged: int
    yield_contribution_pp: float | None


def _bootstrap_cv(rec: FactorRecommendation) -> float:
    window = rec.methods.spc if rec.methods.adopted == "spc" else rec.methods.ml
    if window is None:
        return 0.0
    width = window.hi - window.lo
    if width <= 0:
        return 0.0
    return window.width_sd / width


def _recommend(rate: float, cv: float) -> tuple[str, str]:
    if rate < RATE_LOW_THRESHOLD:
        return "+10%p", "계측률이 가장 낮아 판정 공백이 큽니다"
    if cv > CV_HIGH_THRESHOLD:
        return "+15%p", "추정이 흔들려 권장구간 신뢰도가 낮습니다"
    if cv > CV_MODERATE_THRESHOLD:
        return "+10%p", "추정 안정화로 권장구간이 정밀해집니다"
    return "유지", "현재 추정이 충분히 안정적입니다"


def build_factor_priorities(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    primary_factors: dict[str, ParetoFactor],
    factor_summaries: dict[str, FactorRecommendation],
    original_any: pd.Series,
    *,
    action_target_ratio: float,
    yield_gap_pp: float | None,
    total_wafers: int,
) -> list[FactorPriority]:
    priorities: list[FactorPriority] = []
    for target, factor in primary_factors.items():
        summary = factor_summaries.get(target)
        if summary is None or factor.feature not in train_df.columns:
            continue
        rate = float(pd.to_numeric(train_df[factor.feature], errors="coerce").notna().mean() * 100.0)
        cv = _bootstrap_cv(summary)
        recommendation, reason = _recommend(rate / 100.0, cv)
        additional_judged = _simulate_single_factor_judged(eval_df, factor.feature, original_any)
        action_target = additional_judged * action_target_ratio
        yield_contribution = (
            (action_target * yield_gap_pp) / total_wafers if yield_gap_pp is not None and total_wafers > 0 else None
        )
        priorities.append(
            FactorPriority(
                feature=factor.feature,
                target=target,
                measurement_rate=rate,
                recommendation=recommendation,
                reason=reason,
                additional_judged=additional_judged,
                yield_contribution_pp=yield_contribution,
            )
        )
    priorities.sort(key=lambda p: p.measurement_rate)
    return priorities


@dataclass
class ProjectedFactorDiscovery:
    feature: str
    target: str
    kind: str


def project_new_factor_discoveries(
    rows_by_target: dict[str, list[dict]],
    *,
    fdr_alpha: float = 0.05,
    min_eps2: float = MIN_DISCOVERY_EPS2,
    doubling_factor: float = DOUBLING_FACTOR,
    max_cards: int = MAX_DISCOVERY_CARDS,
) -> list[ProjectedFactorDiscovery]:
    """spec §B-4 카드①: 현재 FDR을 통과하지 못한 인자 중, 표본이
    `doubling_factor`배가 되면 통과할 것으로 추정되는 것. 이미 유의한
    인자는 '새로 발견'이 아니므로 제외한다. 인자 목록은 전부 계산 결과이며
    하드코딩하지 않는다.

    `rows_by_target`은 호출자가 이미 계산해 둔 전체 인자 풀 스코어링
    결과(타깃 -> score_all_factors류 row 목록)를 그대로 받는다 -- Pareto
    화면이 같은 타깃에 대해 이미 돌린 것과 동일한 계산이라 여기서 다시
    돌리면 (타깃당 R+D+Config 88개 스코어링을) 원인 분석 실행 한 번에
    3중으로 반복하게 되어 원인 분석 실행이 눈에 띄게 느려진다.
    """
    candidates: list[tuple[str, str, str, float, float]] = []  # feature, target, kind, eps2, p_projected
    for target, rows in rows_by_target.items():
        family: list[tuple[dict, float]] = []
        for row in rows:
            if row.get("significant"):
                continue
            eps2 = row["eps2"]
            k = row.get("k_groups") or 0
            n = row["n_observed"]
            if k < 2 or not (0.0 < eps2 < 1.0):
                continue
            n_doubled = n * doubling_factor
            df1, df2 = k - 1, n_doubled - k
            if df1 <= 0 or df2 <= 0:
                continue
            f_projected = (eps2 / (1 - eps2)) * (df2 / df1) + 1
            p_projected = float(stats.f.sf(f_projected, df1, df2))
            family.append((row, p_projected))
        if not family:
            continue
        q_values = benjamini_hochberg([p for _, p in family])
        for (row, _), q in zip(family, q_values):
            if q < fdr_alpha and row["eps2"] >= min_eps2:
                candidates.append((row["feature"], target, row["kind"], row["eps2"], q))

    candidates.sort(key=lambda c: c[4])  # 가장 확실한(q 낮은) 순
    return [
        ProjectedFactorDiscovery(feature=feature, target=target, kind=kind)
        for feature, target, kind, _eps2, _q in candidates[:max_cards]
    ]


@dataclass
class MeasurementExpansionSummary:
    action_blocked_wafers: int
    total_wafers: int
    additional_judged: int
    action_target: int
    expected_yield_gain_pp: float | None
    show_full_card: bool  # False: spec §B-6 collapsed one-line message
    priorities: list[FactorPriority]
    new_factor_discoveries: list[ProjectedFactorDiscovery]


def compute_measurement_expansion(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    rows_by_target: dict[str, list[dict]],
    primary_factors: dict[str, ParetoFactor],
    factor_summaries: dict[str, FactorRecommendation],
    bands: WholeWaferBands,
    judgment_features: list[str],
    *,
    total_wafers: int,
) -> MeasurementExpansionSummary:
    action_blocked = bands.unmeasured.count
    show_full_card = total_wafers > 0 and (action_blocked / total_wafers) >= MIN_ACTION_BLOCKED_SHARE

    measured = bands.alarm.count + bands.out_of_recommended.count + bands.in_recommended.count
    action_target_ratio = bands.out_of_recommended.count / measured if measured > 0 else 0.0
    yield_gap_pp = (
        bands.in_recommended.mean_yield - bands.out_of_recommended.mean_yield
        if bands.in_recommended.mean_yield is not None and bands.out_of_recommended.mean_yield is not None
        else None
    )

    if not show_full_card:
        # spec §B-6: 계측률이 이미 충분하면 시뮬레이션 자체를 생략한다 --
        # 어차피 한 줄 메시지만 표시되므로 비용을 들일 이유가 없다.
        return MeasurementExpansionSummary(
            action_blocked_wafers=action_blocked,
            total_wafers=total_wafers,
            additional_judged=0,
            action_target=0,
            expected_yield_gain_pp=None,
            show_full_card=False,
            priorities=[],
            new_factor_discoveries=[],
        )

    original_any = _measured_any_mask(eval_df, judgment_features)
    additional_judged = _simulate_additional_judged(eval_df, judgment_features, original_any)
    action_target = int(round(additional_judged * action_target_ratio))
    expected_yield_gain_pp = (
        (action_target * yield_gap_pp) / total_wafers if yield_gap_pp is not None and total_wafers > 0 else None
    )

    priorities = build_factor_priorities(
        train_df,
        eval_df,
        primary_factors,
        factor_summaries,
        original_any,
        action_target_ratio=action_target_ratio,
        yield_gap_pp=yield_gap_pp,
        total_wafers=total_wafers,
    )
    discoveries = project_new_factor_discoveries(rows_by_target)

    return MeasurementExpansionSummary(
        action_blocked_wafers=action_blocked,
        total_wafers=total_wafers,
        additional_judged=additional_judged,
        action_target=action_target,
        expected_yield_gain_pp=expected_yield_gain_pp,
        show_full_card=True,
        priorities=priorities,
        new_factor_discoveries=discoveries,
    )
