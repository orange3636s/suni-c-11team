"""Shared feature-frame construction for the screening Pareto pipeline.

학습(`src/ml/pipeline.py`)과 추론(`src/analysis/target_hydration.py`)은
반드시 이 모듈 하나를 함께 거쳐야 한다. 각자 피처 프레임을 만들면 Config
인자 처리가 갈라진다 -- 한쪽이 `category` dtype을 유지하는데 다른 쪽이
종류 구분 없이 숫자로 강제 변환하면 Config 컬럼이 전량 NaN이 되고,
`category`로 학습된 LightGBM에 float32를 넘기면서 "승인 모델의 Y3 예측에
실패했습니다"로 터진다.

빌드 규칙:
  - Config: `category` dtype 유지, `_miss`(int8)만 추가, `_dev`는 만들지 않는다.
  - R/D: 숫자 변환(`_numeric`, +/-inf도 결측 취급) + float32, `_miss`(int8),
    u_shape면 `_dev`(|value - optimal_center|, float32) 추가.

Config 카테고리(§2-2/§2-3): 평가 데이터에 학습 시 없던 범주나 결측이
섞여도, `categories`(학습 시점 범주 목록)로 명시적으로 제한하면
LightGBM에 넘기기 전에 미등록 값이 NaN으로 정리된다 -- LightGBM이
그 자체로도 새 범주를 견뎌내지만(코드가 밀리는 것뿐, 값 기준 재매핑),
명시적으로 제한하는 쪽이 항상 기존 `build_features` 경로와 예측이
`np.allclose`로 완전히 일치함이 검증됐다. `categories`가 없으면(학습
시점 그대로, 또는 메타데이터가 없는 구형 모델의 폴백 이전 단계)
컬럼이 스스로 가진 범주를 그대로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


def numeric_series(series: pd.Series) -> pd.Series:
    """`pd.to_numeric` + inf -> NaN. 실측 CSV에 실제 inf 값이 나올 일은
    없지만(측정값 문자열 파싱 결과), coerce 과정에서 나올 수 있는 극단값을
    결측과 동일하게 다뤄 두 경로(학습/추론)가 항상 같은 결과를 내게 한다."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class FactorFeatureSpec:
    """학습은 `ParetoFactor`에서, 추론은 저장된 모델 메타데이터
    (`target_metrics[target]`)에서 각각 이 스펙을 만들어 `build_feature_frame`에
    넘긴다 -- 두 경로가 서로 다른 원본 타입을 몰라도 되게 하는 얇은 경계."""

    feature: str
    kind: str  # "R" | "D" | "Config"
    relation_shape: str | None = None
    optimal_center: float | None = None
    # Config 전용. None이면 컬럼이 스스로 가진 범주를 그대로 쓴다(학습 시점
    # 기본 동작). 값이 있으면 그 범주로 제한한다(추론 시점, §2-2/§2-3).
    categories: Sequence[Any] | None = None


def build_feature_frame(df: pd.DataFrame, specs: Sequence[FactorFeatureSpec]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.kind == "Config":
            columns.update(_config_columns(df, spec))
            continue
        columns.update(_numeric_columns(df, spec))
    return pd.DataFrame(columns, index=df.index)


def _config_columns(df: pd.DataFrame, spec: FactorFeatureSpec) -> dict[str, pd.Series]:
    if spec.feature in df.columns:
        raw = df[spec.feature].astype("category")
        if spec.categories is not None:
            # T2/§2-3: 학습 범주로 제한 -- 미등록 값은 결측으로 정리된다.
            raw = raw.cat.set_categories(list(spec.categories))
    else:
        # T2/§2-2: 평가 데이터에 이 Config 컬럼이 아예 없어도 KeyError로
        # 죽이지 않는다 -- 학습 범주로 구성된 전량 결측 category 컬럼을
        # 대신 만든다(`_miss`는 전부 1이 된다).
        raw = pd.Series(
            pd.Categorical([None] * len(df), categories=list(spec.categories or [])),
            index=df.index,
        )
    return {
        spec.feature: raw,
        f"{spec.feature}_miss": raw.isna().astype("int8"),
    }


def trained_categories_from_model(model: Any) -> list[str] | None:
    """§2-2/§2-3: LightGBM은 학습된 범주 목록을 `booster_.pandas_categorical`에
    보관한다 -- 스크리닝 파이프라인은 타깃당 인자가 하나뿐이라 categorical
    컬럼도 최대 하나뿐이므로 항상 인덱스 0을 쓴다. 학습 직후(메타데이터
    저장용, `src.ml.pipeline._trained_categories` 호출부)와 추론 시점의
    구형-모델 폴백(`src.analysis.target_hydration`) 둘 다 이 함수 하나를
    쓴다 -- 소스가 갈리면 또 같은 버그가 난다."""
    booster = getattr(model, "booster_", None)
    pandas_categorical = getattr(booster, "pandas_categorical", None) if booster is not None else None
    if not pandas_categorical:
        return None
    first = pandas_categorical[0]
    if first is None:
        return None
    return [str(value) for value in first]


def _numeric_columns(df: pd.DataFrame, spec: FactorFeatureSpec) -> dict[str, pd.Series]:
    raw = numeric_series(df[spec.feature]) if spec.feature in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
    result = {
        spec.feature: raw.astype("float32"),
        f"{spec.feature}_miss": raw.isna().astype("int8"),
    }
    if spec.relation_shape == "u_shape" and spec.optimal_center is not None:
        result[f"{spec.feature}_dev"] = (raw - spec.optimal_center).abs().astype("float32")
    return result
