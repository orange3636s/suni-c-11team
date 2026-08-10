"""Full factor x target correlation heatmap ("why aren't the other factors
picked" view, complementing the per-factor scatter's "why this factor").

Metric modes:
  - spearman: rank correlation, pairwise deletion, no imputation. Catches
    monotonic non-linear relationships Pearson misses.
  - eps2: reuses `score_all_factors` (the same bias-corrected effect size +
    BH-FDR family used by the Pareto selector) so this view never disagrees
    with what actually got selected in the screening tab.

Config columns are excluded entirely -- correlation isn't defined for an
unordered categorical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.analysis.screening.effect_size import eps2_categorical
from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import DEFAULT_FDR_ALPHA, benjamini_hochberg, confidence_tier, score_all_factors
from src.config_parser import config_hierarchy_series

MIN_CELL_N = 30
SPEARMAN_SCALE = (-0.5, 0.5)
EPS2_SCALE = (0.0, 0.7)
# 범주형(Config) 히트맵 전용 고정 스케일 (지시서 E: "자동 정규화 금지" --
# 실측 기준 Config의 ε²는 최대 0.006, 중앙값 0.0003이라 관측 최대값에
# 맞추면 신호 없는 셀이 새빨갛게 렌더된다).
EPS2_CATEGORICAL_SCALE = (0.0, 0.05)

Metric = Literal["spearman", "eps2"]
ConfigLevel = Literal["model", "eq", "chamber"]

# Config 계층은 src/config_parser.py의 YAML 기반 공통 파서를 사용한다.
# 원본 Config 범주는 그대로 두고 히트맵 집계용 Series만 파생한다.
@dataclass
class HeatmapData:
    features: list[str]
    targets: list[str]
    values: list[list[float | None]] = field(default_factory=list)
    n: list[list[int]] = field(default_factory=list)
    q: list[list[float | None]] = field(default_factory=list)
    significant: list[list[bool]] = field(default_factory=list)
    tier: list[list[str | None]] = field(default_factory=list)
    scale: dict[str, float] = field(default_factory=dict)
    excluded_configs: int = 0


def _pairwise_n_and_rho(df: pd.DataFrame, feature: str, target: str) -> tuple[int, float | None]:
    frame = df[[feature, target]].dropna()
    n = len(frame)
    if n < MIN_CELL_N:
        return n, None
    rho = frame[feature].corr(frame[target], method="spearman")
    return n, (float(rho) if pd.notna(rho) else None)


def build_heatmap(
    df: pd.DataFrame,
    schema: Schema,
    metric: Metric = "spearman",
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> HeatmapData:
    """Full R+D x Y1~Y5 correlation heatmap. Config is always excluded --
    rho isn't defined for an unordered categorical, and eps2-only would
    make Config's scale incomparable to R/D's rho scale in the same grid.
    """
    features = [*schema.r_cols, *schema.d_cols]
    targets = schema.target_cols

    scored_by_target = {
        target: {row["feature"]: row for row in score_all_factors(df, schema, target, fdr_alpha=fdr_alpha)}
        for target in targets
    }

    values: list[list[float | None]] = []
    n_grid: list[list[int]] = []
    q_grid: list[list[float | None]] = []
    sig_grid: list[list[bool]] = []
    tier_grid: list[list[str | None]] = []
    rho_for_sort: list[list[float | None]] = []

    for feature in features:
        value_row: list[float | None] = []
        n_row: list[int] = []
        q_row: list[float | None] = []
        sig_row: list[bool] = []
        tier_row: list[str | None] = []
        rho_row: list[float | None] = []
        for target in targets:
            n, rho = _pairwise_n_and_rho(df, feature, target)
            scored = scored_by_target.get(target, {}).get(feature)
            if n < MIN_CELL_N:
                value = None
            elif metric == "eps2":
                value = scored["eps2"] if scored else None
            else:
                value = rho
            value_row.append(value)
            n_row.append(n)
            q_row.append(scored["q_value"] if scored else None)
            sig_row.append(bool(scored["significant"]) if scored else False)
            tier_row.append(confidence_tier(scored["eps2"], scored["p_value"]) if (scored and n >= MIN_CELL_N) else None)
            rho_row.append(rho if rho is not None else (scored["eps2"] if scored else None))
        values.append(value_row)
        n_grid.append(n_row)
        q_grid.append(q_row)
        sig_grid.append(sig_row)
        tier_grid.append(tier_row)
        rho_for_sort.append(rho_row)

    order = sorted(
        range(len(features)),
        key=lambda i: max((abs(v) for v in rho_for_sort[i] if v is not None), default=0.0),
        reverse=True,
    )
    scale_min, scale_max = EPS2_SCALE if metric == "eps2" else SPEARMAN_SCALE

    return HeatmapData(
        features=[features[i] for i in order],
        targets=targets,
        values=[values[i] for i in order],
        n=[n_grid[i] for i in order],
        q=[q_grid[i] for i in order],
        significant=[sig_grid[i] for i in order],
        tier=[tier_grid[i] for i in order],
        scale={"min": scale_min, "max": scale_max},
        excluded_configs=len(schema.config_cols),
    )


def _config_level_series(config_col: pd.Series, level: ConfigLevel) -> pd.Series:
    """Return one server-canonical Config hierarchy level."""
    parser_level = "equipment" if level == "eq" else level
    return config_hierarchy_series(str(config_col.name), config_col, parser_level)


def build_categorical_heatmap(
    df: pd.DataFrame,
    schema: Schema,
    level: ConfigLevel = "eq",
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> HeatmapData:
    """Config x Y1~Y5 히트맵 -- R/D 히트맵과 별도 경로다 (spec: "합치지
    말고 토글로 분리한다"). 지표는 ε² 고정(ρ는 순서 없는 범주형에 정의되지
    않는다). FDR 가족은 이 레벨 하나에서 나오는 스텝x타깃 셀 전체 --
    레벨을 바꾸면(Model/EQ/Chamber) 완전히 새 가족으로 다시 검정한다.

    색칠 여부(q<0.05)는 여기서 결정하지 않는다 -- `significant`/`q`
    그리드를 그대로 내려주고, "FDR 통과 셀만 칠한다"는 프론트가 렌더
    시점에 적용한다(숫자형 히트맵과 응답 형태를 공유하기 위함).
    """
    features = schema.config_cols
    targets = schema.target_cols

    level_series_by_feature = {feature: _config_level_series(df[feature], level) for feature in features if feature in df.columns}

    results: dict[tuple[str, str], object] = {}
    p_values: list[float] = []
    keys_with_p: list[tuple[str, str]] = []
    for feature in features:
        level_series = level_series_by_feature.get(feature)
        for target in targets:
            result = eps2_categorical(level_series, df[target]) if level_series is not None and target in df.columns else None
            results[(feature, target)] = result
            if result is not None:
                p_values.append(result.p_value)
                keys_with_p.append((feature, target))

    q_by_key = dict(zip(keys_with_p, benjamini_hochberg(p_values))) if p_values else {}

    values: list[list[float | None]] = []
    n_grid: list[list[int]] = []
    q_grid: list[list[float | None]] = []
    sig_grid: list[list[bool]] = []
    tier_grid: list[list[str | None]] = []

    for feature in features:
        value_row: list[float | None] = []
        n_row: list[int] = []
        q_row: list[float | None] = []
        sig_row: list[bool] = []
        tier_row: list[str | None] = []
        for target in targets:
            result = results[(feature, target)]
            q = q_by_key.get((feature, target))
            value_row.append(result.eps2 if result else None)
            n_row.append(result.n_observed if result else 0)
            q_row.append(q)
            sig_row.append(bool(q is not None and q < fdr_alpha))
            tier_row.append(confidence_tier(result.eps2, result.p_value) if result else None)
        values.append(value_row)
        n_grid.append(n_row)
        q_grid.append(q_row)
        sig_grid.append(sig_row)
        tier_grid.append(tier_row)

    order = sorted(
        range(len(features)),
        key=lambda i: max((v for v in values[i] if v is not None), default=0.0),
        reverse=True,
    )

    return HeatmapData(
        features=[features[i] for i in order],
        targets=targets,
        values=[values[i] for i in order],
        n=[n_grid[i] for i in order],
        q=[q_grid[i] for i in order],
        significant=[sig_grid[i] for i in order],
        tier=[tier_grid[i] for i in order],
        scale={"min": EPS2_CATEGORICAL_SCALE[0], "max": EPS2_CATEGORICAL_SCALE[1]},
        excluded_configs=len(schema.r_cols) + len(schema.d_cols),
    )
