"""전처리 방식 A/B/C 실시간 비교 (설정 패널 신설 §E) -- 데이터셋마다 1위가
달라지므로(예: train.CSV는 B, mentorship_final은 C) 고정 벤치마크 표를 쓸 수
없다. 데이터셋을 바꿀 때마다 실측한다.

실측 결과 가벼운 설정으로도 충분하다 (§E-2): Y 하나 + LOT 70/30 홀드아웃
1회 + max_iter=150이면 3회 학습으로 4초 안에 끝나면서도, 5-fold 실측과
같은 순위(`A < B ≈ C`)를 유지한다.

실제 파이프라인은 이 비교 결과와 무관하게 항상 B(NaN 보존)를 쓴다
(§E-5-1) -- 1위 전략을 데이터셋마다 바꾸면 권장구간·알람·모델 성능 등
다른 모든 수치가 연쇄적으로 달라져 재현성이 무너지기 때문이다. 이 모듈은
표시용 비교 결과만 계산하고, 실제 채택 여부는 손대지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

TARGET_COLUMN = "Y"
MAX_ITER = 150
HOLDOUT_TEST_SIZE = 0.3
RANDOM_STATE = 0
ADOPTED_MODE = "B"  # spec §E-5-1: 1위와 무관하게 항상 B를 채택 배지에 붙인다
CLIP_LOWER_QUANTILE = 0.01
CLIP_UPPER_QUANTILE = 0.99
B_EQUALS_C_DECIMALS = 4  # spec §E-5: 소수점 4자리까지 동일하면 "차이 없음"

MODE_LABELS = {
    "A": "A. 중앙값 대체 + 1~99% 클리핑",
    "B": "B. NaN 보존",
    "C": "C. NaN 보존 + 결측 마스크",
}


def _feature_matrix(df: pd.DataFrame, features: list[str], mode: str) -> pd.DataFrame:
    raw = pd.DataFrame(
        {f: pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan for f in features},
        index=df.index,
    )
    if mode == "A":
        out = raw.copy()
        for col in out.columns:
            lo = out[col].quantile(CLIP_LOWER_QUANTILE)
            hi = out[col].quantile(CLIP_UPPER_QUANTILE)
            out[col] = out[col].clip(lo, hi)
            out[col] = out[col].fillna(out[col].median())
        return out
    if mode == "B":
        return raw
    if mode == "C":
        out = raw.copy()
        for col in features:
            out[f"{col}__miss"] = raw[col].isna().astype(float)
        return out
    raise ValueError(f"알 수 없는 전처리 방식입니다: {mode}")


def _holdout_split(df: pd.DataFrame, group_col: str) -> tuple[np.ndarray, np.ndarray]:
    if group_col in df.columns and df[group_col].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_TEST_SIZE, random_state=RANDOM_STATE)
        return next(splitter.split(df, groups=df[group_col]))
    n = len(df)
    cut = max(1, int(n * (1 - HOLDOUT_TEST_SIZE)))
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


@dataclass
class ModeResult:
    mode: str
    label: str
    r2: float
    adopted: bool


@dataclass
class PreprocessingComparison:
    dataset_id: str
    dataset_label: str
    results: list[ModeResult] = field(default_factory=list)
    winner: str = ADOPTED_MODE
    b_equals_c: bool = True
    holdout_note: str = ""
    winner_note: str | None = None


def compute_preprocessing_comparison(
    df: pd.DataFrame,
    features: list[str],
    *,
    dataset_id: str,
    dataset_label: str,
    group_col: str = "Lot_ID",
    target_col: str = TARGET_COLUMN,
) -> PreprocessingComparison | None:
    if target_col not in df.columns or not features:
        return None
    valid = df[pd.to_numeric(df[target_col], errors="coerce").notna()]
    if len(valid) < 10:
        return None
    y = pd.to_numeric(valid[target_col], errors="coerce")

    train_idx, test_idx = _holdout_split(valid, group_col)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None

    results: list[ModeResult] = []
    scores: dict[str, float] = {}
    for mode in ("A", "B", "C"):
        matrix = _feature_matrix(valid, features, mode)
        model = LGBMRegressor(n_estimators=MAX_ITER, random_state=RANDOM_STATE, verbose=-1)
        model.fit(matrix.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(matrix.iloc[test_idx])
        r2 = float(r2_score(y.iloc[test_idx], pred))
        scores[mode] = r2
        results.append(ModeResult(mode=mode, label=MODE_LABELS[mode], r2=r2, adopted=(mode == ADOPTED_MODE)))

    winner = max(scores, key=lambda mode: scores[mode])
    b_equals_c = round(scores["B"], B_EQUALS_C_DECIMALS) == round(scores["C"], B_EQUALS_C_DECIMALS)
    winner_note = (
        f"이 데이터셋에서는 {winner}가 근소하게 우수하나, 일관성을 위해 {ADOPTED_MODE}를 채택했습니다."
        if winner != ADOPTED_MODE
        else None
    )

    return PreprocessingComparison(
        dataset_id=dataset_id,
        dataset_label=dataset_label,
        results=results,
        winner=winner,
        b_equals_c=b_equals_c,
        holdout_note=f"{dataset_label} 기준 · LOT 70/30 홀드아웃 1회",
        winner_note=winner_note,
    )
