"""수율 예측 화면(VA~VD) -- 순위는 맞지만 값은 못 맞추는 모델(R² 0.12,
상위 20장 적중 95%)이라 이 화면은 "순위 도구"다. 정렬은 y(=100 − Σ
Y1~Y5, hydrated_df가 이미 실측 우선으로 채운 값) 오름차순 하나뿐이다.

VA-1/VA-3: 타깃별 핵심 인자는 파레토 차트가 이미 쓰는 기여율
(selector.py의 contribution_pct)을 그대로 재사용하고, 웨이퍼·타깃마다
계측된 가장 높은 순위(최대 5위까지)의 인자로 폴백한다.

VC-1: 신뢰도 = (기여율 CORE_FACTOR_CONTRIBUTION_MIN 이상 인자가 계측된
타깃 수) / 5. 실측이 있는 타깃은 계측으로 센다(예측이 아니므로 근거가
확실하다). VA-2 실측상 이 임계 이상은 대체로 타깃당 1위 인자뿐이라
사실상 "1위 계측 수"와 같지만, 판정은 기여율로 한다 -- 다른
데이터셋에서 2위가 임계를 넘으면 자동으로 반영되어야 하기 때문이다.
YG: 이전에는 20%였다("하지 말 것: 20% 임계를 낮추지 마라"는 주석이
있었다) -- 작업 지시서가 명시적으로 10%로 낮추라고 요구해 그 결정을
대체했다. 폐기 배경은 docs/decisions.md 참고.

VD-1: 같은 임계로 권장사항이 두 갈래로 갈린다 -- 이상 계측된 타깃은
구간 조정 제안(SPC/ML 권장 구간 재사용 + 2차 곡선 감소량), 미만인
타깃은 계측 추가 제안(1위 인자명).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock

import pandas as pd

from src.analysis.control_range import ControlRange, compute_control_range
from src.analysis.curve_fit import CurveFit, evaluate_curve, fit_defect_rate_curve
from src.analysis.recommendations import FactorRecommendation, compute_factor_recommendation
from src.analysis.reliability_score import FAIL_RATE_TARGETS, cell_color
from src.analysis.screening.schema import Schema, parse_schema
from src.analysis.screening.selector import ParetoFactor, select_top_factors
from src.analysis.thresholds import CORE_FACTOR_CONTRIBUTION_MIN

ID_COLUMN = "Lot_Wafer_ID"
LOT_COLUMN = "Lot_ID"

# VA-3: 5위까지만 본다 -- 그 아래는 기여율 1% 미만이라 의미가 없다.
MAX_FALLBACK_RANK = 5
# VA-2/VC-1/VD-1이 공유하는 단일 임계 -- src/analysis/thresholds.py가
# 유일한 소스다(YG/ZF-1). reliability_score.SHADE_MEDIUM_MIN은 값이
# 우연히 같았을 뿐인 별개 개념(셀 농도 구간)이라 여기에 맞춰 바꾸지
# 않는다 -- 그 상수를 건드리지 마라.
CONTRIBUTION_THRESHOLD = CORE_FACTOR_CONTRIBUTION_MIN
# VD-3: 실익 없는 조치를 권하지 않는다.
MIN_MEANINGFUL_DECREASE_PCT = 0.05
# VD-2: 여러 타깃이 조정 가능해도 최대 2개까지만 나열한다.
MAX_ADJUSTMENT_ITEMS = 2

@dataclass(frozen=True)
class CoreFactorCell:
    """VA-3/VA-4: 이 (웨이퍼, 타깃)에 대해 실제로 쓰인(폴백 포함) 인자.
    `rank_used`가 1보다 크면 폴백이 일어났다는 뜻이고, `contribution_pct`는
    그 폴백된 인자 자신의 기여율이다(1위보다 낮게 표시돼 근거 강도가
    바로 드러난다)."""

    feature: str | None
    contribution_pct: float | None
    rank_used: int | None  # 1..5, 전부 미계측이면 None
    factor_value: float | None


@dataclass(frozen=True)
class ReliabilityInfo:
    """VC-1/VC-2: n/5 신뢰도와, 툴팁이 쓰는 계측/미계측 타깃 상세."""

    count: int
    measured: tuple[tuple[str, str], ...]  # (target, feature-or-"실측")
    unmeasured: tuple[tuple[str, str], ...]  # (target, 1위 feature)


@dataclass(frozen=True)
class Recommendation:
    """VD-2: 조립된 권장사항 문장(줄바꿈으로 구분된 여러 줄)과, 어느
    타깃이 어느 갈래로 분류됐는지(테스트/발송 메시지가 재사용)."""

    text: str
    adjustable_targets: tuple[str, ...]
    measurement_gap_targets: tuple[str, ...]


@dataclass(frozen=True)
class YieldCandidate:
    lot_wafer_id: str
    lot_id: str | None
    y: float
    y_components: dict[str, float]
    cells: dict[str, dict[str, object]]  # VB-3: Y1~Y5 색상(기존 cell_color 재사용)
    core_factors: dict[str, CoreFactorCell]  # VB-1: Y1핵심인자..Y5핵심인자
    reliability: ReliabilityInfo
    recommendation: Recommendation


@dataclass(frozen=True)
class FallbackSummary:
    """VA-3: "58%가 전부 미계측이다" 같은 화면에 드러낼 통계."""

    rank_counts: dict[int, int]
    none_count: int
    total_combinations: int


@dataclass(frozen=True)
class YieldPredictionTable:
    candidates: list[YieldCandidate]  # 신뢰도>=1, y 오름차순(기본 정렬) -- VB-2
    unmeasured_wafer_ids: list[str]  # 신뢰도==0 -- VB-2/VE-1: 별도 블록
    total_wafers: int
    fallback_summary: FallbackSummary
    primary_factors: dict[str, ParetoFactor | None] = field(default_factory=dict)  # 타깃별 1위(참고/발송용)


def _rank5_factors(train_df: pd.DataFrame, schema: Schema) -> dict[str, list[ParetoFactor]]:
    return {target: select_top_factors(train_df, schema, target, limit=MAX_FALLBACK_RANK) for target in FAIL_RATE_TARGETS}


# YF/ZD (성능): 라이브 프로파일 결과 `_rank5_factors`(파레토 스코어링 5회,
# 매번 eps2/quantile-bin/pearsonr 반복)가 요청마다 ~3.5초를 먹었다 --
# train_df/schema에만 의존하고 eval 데이터셋과는 무관한데도 캐시가 전혀
# 없어 같은 train 데이터셋으로 반복 조회해도 매번 다시 계산됐다(측정:
# 콜드 6.4s, 웜(하이드레이션·compare_methods 캐시만 적용) 3.5s -- 그 3.5s가
# 전부 이 함수였다). dataset_id는 재업로드로 내용이 바뀌어도 그대로일 수
# 있어(YE가 test.CSV를 그대로 갈아 끼운 것처럼) 버전 문자열을 키에 함께
# 넣는다 -- src/analysis/target_hydration.py, window_methods.py의
# (dataset_id, dataset_version)/문자열 키 관례를 그대로 따른다.
_FACTOR_CACHE_MAXSIZE = 4
_factor_cache: "OrderedDict[tuple[str, str], dict[str, list[ParetoFactor]]]" = OrderedDict()
_factor_cache_lock = RLock()


def _rank5_factors_cached(
    train_df: pd.DataFrame, schema: Schema, *, train_dataset_id: str | None, train_dataset_version: str | None
) -> dict[str, list[ParetoFactor]]:
    if train_dataset_id is None or train_dataset_version is None:
        return _rank5_factors(train_df, schema)
    key = (train_dataset_id, train_dataset_version)
    with _factor_cache_lock:
        cached = _factor_cache.get(key)
        if cached is not None:
            _factor_cache.move_to_end(key)
            return cached
    result = _rank5_factors(train_df, schema)
    with _factor_cache_lock:
        _factor_cache[key] = result
        _factor_cache.move_to_end(key)
        while len(_factor_cache) > _FACTOR_CACHE_MAXSIZE:
            _factor_cache.popitem(last=False)
    return result


def build_yield_prediction_table(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    hydrated_df: pd.DataFrame,
    *,
    dataset_id: str = "eval",
    train_dataset_id: str | None = None,
    train_dataset_version: str | None = None,
) -> YieldPredictionTable:
    """`hydrated_df`는 `target_hydration.hydrate_targets`가 반환한, 실측
    우선으로 Y1~Y5(및 Y)를 채운 프레임이다 -- 순위/표시값은 여기서
    읽는다. `eval_df`는 실측/예측 판정(신뢰도·폴백·색상)에 쓰는 원본
    (하이드레이션 전) 프레임이다.

    `train_dataset_id`/`train_dataset_version`은 선택값이다 -- 둘 다
    있으면 파레토 랭킹(가장 비싼 단계, 위 주석 참고)을 요청 간에
    캐시한다. 호출부가 못 넘기면(예: 단위 테스트) 그냥 매번 계산한다."""
    schema = parse_schema(train_df)
    ranked_factors = _rank5_factors_cached(
        train_df, schema, train_dataset_id=train_dataset_id, train_dataset_version=train_dataset_version
    )
    primary_factors: dict[str, ParetoFactor | None] = {
        target: (factors[0] if factors else None) for target, factors in ranked_factors.items()
    }

    control_range_cache: dict[tuple[str, str], ControlRange] = {}
    curve_cache: dict[tuple[str, str], CurveFit] = {}
    recommendation_cache: dict[tuple[str, str], FactorRecommendation | None] = {}

    def _control_range(factor: ParetoFactor) -> ControlRange:
        key = (factor.feature, factor.target)
        if key not in control_range_cache:
            control_range_cache[key] = compute_control_range(train_df, factor)
        return control_range_cache[key]

    def _curve(feature: str, target: str) -> CurveFit:
        key = (feature, target)
        if key not in curve_cache:
            x = pd.to_numeric(train_df[feature], errors="coerce")
            y = pd.to_numeric(train_df[target], errors="coerce")
            valid = x.notna() & y.notna()
            curve_cache[key] = fit_defect_rate_curve(x[valid].to_numpy(), y[valid].to_numpy())
        return curve_cache[key]

    def _recommendation(factor: ParetoFactor) -> FactorRecommendation | None:
        key = (factor.feature, factor.target)
        if key not in recommendation_cache:
            recommendation_cache[key] = compute_factor_recommendation(
                train_df, factor, _control_range(factor), dataset_id=dataset_id
            )
        return recommendation_cache[key]

    rank_counts: dict[int, int] = {rank: 0 for rank in range(1, MAX_FALLBACK_RANK + 1)}
    none_count = 0
    candidates: list[YieldCandidate] = []
    unmeasured_ids: list[str] = []

    for idx in hydrated_df.index:
        row = hydrated_df.loc[idx]
        eval_row = eval_df.loc[idx] if idx in eval_df.index else None

        y_components = {t: float(row[t]) for t in FAIL_RATE_TARGETS if t in hydrated_df.columns}

        core_factors: dict[str, CoreFactorCell] = {}
        target_measured: dict[str, bool] = {}
        for target in FAIL_RATE_TARGETS:
            is_target_measured = eval_row is not None and target in eval_df.columns and pd.notna(eval_row[target])
            target_measured[target] = bool(is_target_measured)

            chosen: CoreFactorCell | None = None
            for rank, factor in enumerate(ranked_factors[target], start=1):
                if eval_row is None or factor.feature not in eval_df.columns:
                    continue
                value = eval_row[factor.feature]
                if pd.notna(value):
                    chosen = CoreFactorCell(
                        feature=factor.feature,
                        contribution_pct=factor.contribution_pct,
                        rank_used=rank,
                        factor_value=float(value),
                    )
                    rank_counts[rank] += 1
                    break
            if chosen is None:
                chosen = CoreFactorCell(feature=None, contribution_pct=None, rank_used=None, factor_value=None)
                none_count += 1
            core_factors[target] = chosen

        # VC-1.
        reliability_count = 0
        measured_detail: list[tuple[str, str]] = []
        unmeasured_detail: list[tuple[str, str]] = []
        for target in FAIL_RATE_TARGETS:
            core = core_factors[target]
            if target_measured[target]:
                reliability_count += 1
                measured_detail.append((target, core.feature or "실측"))
                continue
            if core.contribution_pct is not None and core.contribution_pct >= CONTRIBUTION_THRESHOLD:
                reliability_count += 1
                measured_detail.append((target, core.feature or "-"))
            else:
                top1 = primary_factors.get(target)
                unmeasured_detail.append((target, top1.feature if top1 else "-"))

        lot_wafer_id = str(row[ID_COLUMN]) if ID_COLUMN in hydrated_df.columns else str(idx)

        if reliability_count == 0:
            # VB-2/VE-1: 미계측 웨이퍼는 판정 목록에 넣지 않고 별도 블록으로.
            unmeasured_ids.append(lot_wafer_id)
            continue

        # VB-3: Y1~Y5 색상 -- 기존 cell_color 재사용(1위 인자 기준 방향/농도).
        cells: dict[str, dict[str, object]] = {}
        for target in FAIL_RATE_TARGETS:
            target_factor = primary_factors.get(target)
            target_feature = target_factor.feature if target_factor else None
            target_value = (
                float(eval_row[target_feature])
                if eval_row is not None
                and target_feature
                and target_feature in eval_df.columns
                and pd.notna(eval_row[target_feature])
                else None
            )
            cells[target] = cell_color(target, target_value, target_measured[target], primary_factors)

        # VD/YG: 권장사항 -- 기여율 10% 이상 인자가 계측된 타깃만 구간 조정 후보.
        adjustable: list[tuple[str, str, float, float, float, float]] = []
        for target in FAIL_RATE_TARGETS:
            core = core_factors[target]
            if core.rank_used is None or core.contribution_pct is None or core.contribution_pct < CONTRIBUTION_THRESHOLD:
                continue
            factor = ranked_factors[target][core.rank_used - 1]
            rec = _recommendation(factor)
            if rec is None:
                continue
            winner = rec.methods.ml if rec.methods.adopted == "ml" else rec.methods.spc
            if winner is None:
                continue
            fit = _curve(factor.feature, target)
            current_value = core.factor_value
            assert current_value is not None
            decrease = max(0.0, evaluate_curve(fit, current_value) - evaluate_curve(fit, winner.center))
            adjustable.append((target, factor.feature, current_value, rec.recommended_lo, rec.recommended_hi, decrease))

        adjustable.sort(key=lambda item: item[5], reverse=True)
        meaningful = [item for item in adjustable if item[5] >= MIN_MEANINGFUL_DECREASE_PCT]
        top_adjustments = meaningful[:MAX_ADJUSTMENT_ITEMS]

        # YC-3: 화살표 축약형 한 줄 -- "Step16_R1 73.2 → 55.6~63.4 · Y2 −2.1%p".
        # 감소량 큰 순으로 이미 정렬돼 있다(위 adjustable.sort).
        lines: list[str] = []
        for target, feature, current_value, lo, hi, decrease in top_adjustments:
            lines.append(f"{feature} {current_value:.1f} → {lo:.1f}~{hi:.1f} · {target} −{decrease:.1f}%p")
        if adjustable and not meaningful:
            # VD-3: 실익 0.05%p 미만 -- 실익 없는 조치를 권하지 않는다.
            lines.append("개선 여지가 거의 없습니다.")

        # VD-4: 폴백으로 하위 인자를 쓰고 있어도 1위가 미계측이면 제안 대상.
        # 이미 실측인 타깃은 "예측이 부정확하다"는 문구가 성립하지 않으므로
        # 제외한다. top1이 없는(그 타깃에 후보 인자 자체가 없는) 경우도
        # "핵심 인자가 아예 없다"는 뜻이므로 YC-5 대상에 포함한다(이전에는
        # 조용히 건너뛰어 행 전체가 "—"만 남았다).
        deficient_targets: list[tuple[str, str | None]] = []
        for target in FAIL_RATE_TARGETS:
            if target_measured[target]:
                continue
            core = core_factors[target]
            eligible = core.contribution_pct is not None and core.contribution_pct >= CONTRIBUTION_THRESHOLD
            if eligible:
                continue
            top1 = primary_factors.get(target)
            deficient_targets.append((target, top1.feature if top1 is not None else None))

        # YC-2/YC-5: 타깃별로 자기 인자를 괄호로 묶어 한 줄로 -- 인자가
        # 아예 없는 타깃은 괄호 없이 타깃명만(계측할 대상 자체가 없다는
        # 뜻이라 인자를 지어내지 않는다).
        if deficient_targets:
            pairs = " · ".join(f"{t}({f})" if f else t for t, f in deficient_targets)
            prefix = "미계측" if lines else "핵심 인자 미계측"
            lines.append(f"{prefix}: {pairs} 계측 추가")

        recommendation = Recommendation(
            text="\n".join(lines),
            adjustable_targets=tuple(item[0] for item in top_adjustments),
            measurement_gap_targets=tuple(t for t, _ in deficient_targets),
        )

        candidates.append(
            YieldCandidate(
                lot_wafer_id=lot_wafer_id,
                lot_id=str(row[LOT_COLUMN]) if LOT_COLUMN in hydrated_df.columns and pd.notna(row[LOT_COLUMN]) else None,
                y=float(row["Y"]),
                y_components=y_components,
                cells=cells,
                core_factors=core_factors,
                reliability=ReliabilityInfo(count=reliability_count, measured=tuple(measured_detail), unmeasured=tuple(unmeasured_detail)),
                recommendation=recommendation,
            )
        )

    candidates.sort(key=lambda c: c.y)  # VB-2 기본: Y 낮은 순

    fallback_summary = FallbackSummary(
        rank_counts=rank_counts,
        none_count=none_count,
        total_combinations=sum(rank_counts.values()) + none_count,
    )

    return YieldPredictionTable(
        candidates=candidates,
        unmeasured_wafer_ids=unmeasured_ids,
        total_wafers=len(hydrated_df),
        fallback_summary=fallback_summary,
        primary_factors=primary_factors,
    )
