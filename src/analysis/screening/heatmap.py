"""Full factor x target correlation heatmap ("why aren't the other factors
picked" view, complementing the per-factor scatter's "why this factor").

TC-4: every cell reports BOTH statistics now instead of toggling between
them --
  - `values` (cell intensity/displayed number): eps2, the same bias-corrected
    effect size + BH-FDR family `score_all_factors` uses, so this view never
    disagrees with what actually got selected in the screening tab. Unlike a
    rank correlation, eps2 catches non-monotonic (U-shaped) relationships
    too, so it never *undersells* a factor the way rho alone can.
  - `rho` (cell color direction only): signed Spearman rho -- eps2 has no
    sign, so without this a U-shaped factor's "does raising x help or hurt"
    question would be unanswerable from the heatmap alone.
For a gate-excluded cell (n>=MIN_CELL_N but below the R/D-specific
significance gate in `score_all_factors`), eps2 is still computed directly
here (bypassing that gate, same as rho already did) so the cell still shows
a real number instead of going blank -- only the `gate_excluded`/`tier`
flags communicate the gate outcome.

NG-1: Config x Y1~Y5 히트맵(구 `build_categorical_heatmap`)은 제거했다 --
600건 검정에서 FDR 통과 0건이라 전량 중립색이었고, Config별 트리맵 탭이
같은 정보를 더 잘 보여준다. `eps2_categorical` 자체는 트리맵이 계속 쓰므로
(src/analysis/screening/selector.py, src/analysis/llm_stats.py,
api/routes/monitoring.py) 건드리지 않았다 -- 지운 것은 이 히트맵 전용
경로뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analysis.screening.effect_size import eps2_numeric
from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import (
    DEFAULT_FDR_ALPHA,
    effective_confidence_tier,
    score_all_factors,
)

MIN_CELL_N = 30
EPS2_SCALE = (0.0, 0.7)
# TC-4: eps2가 이 이상인데 |rho|가 이 미만이면 "순위상관으로는 약해
# 보이지만 설명력은 높은" U자형 관계로 판정한다 (프런트 툴팁용 임계 --
# 백엔드는 원값만 내려주고 판정 자체는 프런트에서 한다).
U_SHAPE_EPS2_MIN = 0.05
U_SHAPE_RHO_MAX = 0.15


# Config 계층은 src/config_parser.py의 YAML 기반 공통 파서를 사용한다.
# 원본 Config 범주는 그대로 두고 히트맵 집계용 Series만 파생한다.
@dataclass
class HeatmapData:
    features: list[str]
    targets: list[str]
    values: list[list[float | None]] = field(default_factory=list)
    rho: list[list[float | None]] = field(default_factory=list)
    n: list[list[int]] = field(default_factory=list)
    q: list[list[float | None]] = field(default_factory=list)
    significant: list[list[bool]] = field(default_factory=list)
    tier: list[list[str | None]] = field(default_factory=list)
    # QA-3: 상관계수는 그려지는데(n>=MIN_CELL_N) 유의 인자 목록에서는 종류별
    # 표본 게이트 미달로 빠지는 셀 -- 히트맵과 유의 인자 판정 정합용.
    gate_excluded: list[list[bool]] = field(default_factory=list)
    scale: dict[str, float] = field(default_factory=dict)
    excluded_configs: int = 0


def _pairwise_n_rho_eps2(df: pd.DataFrame, feature: str, target: str) -> tuple[int, float | None, float | None]:
    frame = df[[feature, target]].dropna()
    n = len(frame)
    if n < MIN_CELL_N:
        return n, None, None
    rho = frame[feature].corr(frame[target], method="spearman")
    rho_value = float(rho) if pd.notna(rho) else None
    # TC-4: 종류별 게이트(R>=100/D>=40)와 무관하게, rho와 마찬가지로
    # hard_min_n(30)만 넘으면 항상 계산한다 -- 게이트 미달 셀도 진짜 ε²를
    # 보여주고, 그 셀이 "유의 인자"인지 여부는 별도의 gate_excluded/tier
    # 플래그로만 표시한다.
    eps2_result = eps2_numeric(frame[feature], frame[target], min_n=MIN_CELL_N)
    eps2_value = eps2_result.eps2 if eps2_result is not None else None
    return n, rho_value, eps2_value


def build_heatmap(
    df: pd.DataFrame,
    schema: Schema,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> HeatmapData:
    """Full R+D x Y1~Y5 correlation heatmap. Config is always excluded --
    correlation isn't defined for an unordered categorical, and Config's
    eps2 scale is incomparable to R/D's (see EPS2_CATEGORICAL_SCALE).
    """
    features = [*schema.r_cols, *schema.d_cols]
    targets = schema.target_cols

    scored_by_target = {
        target: {row["feature"]: row for row in score_all_factors(df, schema, target, fdr_alpha=fdr_alpha)}
        for target in targets
    }

    values: list[list[float | None]] = []
    rho_grid: list[list[float | None]] = []
    n_grid: list[list[int]] = []
    q_grid: list[list[float | None]] = []
    sig_grid: list[list[bool]] = []
    tier_grid: list[list[str | None]] = []
    gate_grid: list[list[bool]] = []

    for feature in features:
        value_row: list[float | None] = []
        rho_row: list[float | None] = []
        n_row: list[int] = []
        q_row: list[float | None] = []
        sig_row: list[bool] = []
        tier_row: list[str | None] = []
        gate_row: list[bool] = []
        for target in targets:
            n, rho, eps2_value = _pairwise_n_rho_eps2(df, feature, target)
            scored = scored_by_target.get(target, {}).get(feature)
            value_row.append(eps2_value)
            rho_row.append(rho)
            n_row.append(n)
            q_row.append(scored["q_value"] if scored else None)
            sig_row.append(bool(scored["significant"]) if scored else False)
            tier_row.append(
                effective_confidence_tier(scored["eps2"], scored["p_value"], under_sampled=scored.get("under_sampled", False))
                if (scored and n >= MIN_CELL_N)
                else None
            )
            # QA-3: 상관계수가 그려지는데(n>=MIN_CELL_N) 유의 인자 풀에는
            # 없는(scored=None) 셀 -- 종류별 표본 게이트 미달 등으로 두
            # 화면이 어긋나는 지점.
            gate_row.append(bool(n >= MIN_CELL_N and scored is None))
        values.append(value_row)
        rho_grid.append(rho_row)
        n_grid.append(n_row)
        q_grid.append(q_row)
        sig_grid.append(sig_row)
        tier_grid.append(tier_row)
        gate_grid.append(gate_row)

    # TC-4: 기본 정렬은 항상 "최대 eps2" -- rho 기준(최대/최소) 정렬은
    # 프런트가 같은 응답 안의 rho 그리드로 클라이언트에서 다시 정렬한다.
    order = sorted(
        range(len(features)),
        key=lambda i: max((v for v in values[i] if v is not None), default=0.0),
        reverse=True,
    )

    return HeatmapData(
        features=[features[i] for i in order],
        targets=targets,
        values=[values[i] for i in order],
        rho=[rho_grid[i] for i in order],
        n=[n_grid[i] for i in order],
        q=[q_grid[i] for i in order],
        significant=[sig_grid[i] for i in order],
        tier=[tier_grid[i] for i in order],
        gate_excluded=[gate_grid[i] for i in order],
        scale={"min": EPS2_SCALE[0], "max": EPS2_SCALE[1]},
        excluded_configs=len(schema.config_cols),
    )
