"""로트 단위 층화 표본 (작업지시 "1만~10만 행 분석이 끊기지 않게 하기" T2).

인자 순위(ε²)만 필요한 분석 단계(스크리닝/히트맵/Config 트리맵/권장구간)는
행을 전부 통계 검정에 넣을 이유가 없다 -- 추정 정밀도는 √n으로만
좋아지는데 비용은 n에 비례한다. train.CSV 10,000행을 로트 단위로 표본
추출해 전량 결과와 비교한 실측(시드 5개 x 타깃 5개 = 25회)에서 5,000행
(50%) 표본은 1위 인자를 25/25(100%) 보존했다. `ANALYSIS_SAMPLE_MAX_ROWS`
(20,000행)는 그 지점에서 4배 여유를 둔 값이다.

로트가 아니라 행을 무작위로 뽑지 않는 이유: 로트 내부 상관 구조를
깨뜨리지 않기 위해서다(이 프로젝트 실측: 랏 내 변동이 전체 분산의
97%를 차지한다 -- 로트를 통째로 빼도 나머지 로트가 그 변동을 그대로
대표하지만, 한 로트 안에서 행만 솎아내면 그 로트 고유의 변동 패턴이
왜곡된다).

적용 대상이 아닌 것 (절대 표본을 쓰면 안 된다):
  - `build_yield_prediction_table` -- 웨이퍼별 순위가 산출물 그 자체다.
  - `hydrate_targets` -- 모델 추론은 O(n)이라 가볍다.
  - `compute_variance_decomposition`/MNAR 진단 -- 단순 집계라 가볍다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

ANALYSIS_SAMPLE_MAX_ROWS = 20_000
# 산점도 점 데이터는 그 이상 찍어도 화면에서 겹쳐 안 보인다 -- 별도의
# 더 작은 상한.
SCATTER_POINT_MAX_ROWS = 5_000
DEFAULT_LOT_COLUMN = "Lot_ID"


@dataclass(frozen=True)
class SampleInfo:
    """스냅샷/응답에 실어 프런트가 "N행 중 M행 표본으로 분석했습니다"
    고지를 띄우는 데 쓴다 -- 숫자를 조용히 표본으로 바꾸고 말 안 하는
    것이 이 기능에서 가장 하면 안 되는 일이다."""

    is_sampled: bool
    original_rows: int
    sampled_rows: int
    lot_count: int | None
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "is_sampled": self.is_sampled,
            "original_rows": self.original_rows,
            "sampled_rows": self.sampled_rows,
            "lot_count": self.lot_count,
            "seed": self.seed,
        }


_NOT_SAMPLED = SampleInfo(is_sampled=False, original_rows=0, sampled_rows=0, lot_count=None, seed=0)


def _stable_seed(dataset_version: str) -> int:
    """`hash(str)`은 PYTHONHASHSEED가 프로세스마다 랜덤이라 재시작 후
    같은 데이터셋도 다른 시드를 준다 -- 그러면 "같은 데이터셋 = 같은
    표본"이 깨진다. 대신 안정적인 해시를 쓴다."""
    digest = hashlib.sha256(dataset_version.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def stratified_sample(
    df: pd.DataFrame,
    *,
    lot_column: str = DEFAULT_LOT_COLUMN,
    max_rows: int = ANALYSIS_SAMPLE_MAX_ROWS,
    dataset_version: str = "",
    seed: int | None = None,
) -> tuple[pd.DataFrame, SampleInfo]:
    """행 수가 `max_rows` 이하면 원본을 그대로 반환한다(`is_sampled=False`).
    초과하면 로트 단위로 추출한다. 시드는 `dataset_version`의 안정적인
    해시로 고정된다 -- 같은 데이터셋을 몇 번을 다시 분석해도 같은
    표본이 나온다(`seed`를 명시하면 그 값이 우선한다 -- 테스트/재현용).
    로트 컬럼이 없거나 전부 결측이면 단순 무작위 표본으로 폴백한다.
    """
    n = len(df)
    if n <= max_rows:
        return df, SampleInfo(False, n, n, None, 0)

    resolved_seed = seed if seed is not None else _stable_seed(dataset_version)
    rng = np.random.default_rng(resolved_seed)

    if lot_column not in df.columns or df[lot_column].isna().all():
        sampled = df.sample(n=max_rows, random_state=resolved_seed)
        return sampled, SampleInfo(True, n, len(sampled), None, resolved_seed)

    lot_row_counts = df.groupby(lot_column, observed=True).size()
    lots_shuffled = rng.permutation(lot_row_counts.index.to_numpy())

    chosen: list = []
    running = 0
    for lot in lots_shuffled:
        size = int(lot_row_counts.loc[lot])
        if chosen and running + size > max_rows:
            break
        chosen.append(lot)
        running += size
    if not chosen:
        # 로트 하나가 이미 max_rows보다 큰 극단적인 경우 -- 그래도 로트를
        # 쪼개지 않고 통째로 포함한다.
        chosen.append(lots_shuffled[0])

    sampled = df[df[lot_column].isin(chosen)]
    return sampled, SampleInfo(True, n, len(sampled), len(chosen), resolved_seed)
