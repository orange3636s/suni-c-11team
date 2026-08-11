"""R+D 인자 기반 GBDT 특징 준비 + 예측 구간 conformal 캘리브레이션.

옛 알람 판정(등급 심각/위험/주의, AUC 신뢰도 게이트, 부트스트랩 앙상블
예측)은 전부 폐기됐다 -- 알림은 이제 수율 예측 갱신 파이프라인
(`src/notifications/yield_update_dispatch.py`)이 전담하며, 그 판정은
등급이 아니라 `src/analysis/yield_prediction.py`의 순위 기반 표다.

이 모듈에 남은 것은 두 그룹뿐이다:
  - `feature_columns`/`step_of`/`prepare_feature_matrix`: 전체 R+D 인자를
    GBDT 특징 행렬로 준비하는 공용 유틸(§A-1의 설계 결정 -- 선정 인자가
    아니라 전체 R+D 인자를 쓰고, 결측 마스크 컬럼을 만들지 않는다. LGBM은
    NaN을 네이티브로 처리한다). `preprocessing_compare.py`가 특징 목록을
    재사용한다.
  - `compute_holdout_predictions`: 랏 단위 GroupKFold out-of-fold 잔차로
    conformal margin(q)을 내는 함수 -- `src/analysis/report.py`의 SUNI
    챗봇 컨텍스트(`/api/analysis/context`)가 "예측 구간 폭이 약
    ±{q:.1f}%p" 캐비어트 문장에 여전히 쓴다.

`random_state`는 GroupKFold의 fold 번호로 고정한다 -- 같은 입력이면
새로고침해도 항상 같은 결과가 나온다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

FINAL_YIELD_COLUMN = "Y"
GBDT_MAX_ITER = 200

# 예측 구간 conformal 캘리브레이션 (spec "예측 구간 캘리브레이션 + 미분류
# 사유 분리" §BA) -- 목표 포함률. q(홀드아웃 |잔차| 분위수)는 이 값 기준
# 분위수를 쓴다. 서버 상수로만 조절한다 -- UI는 만들지 않는다(§BA-3).
CONFORMAL_TARGET_COVERAGE = 0.90


def feature_columns(schema) -> list[str]:
    """spec §A-1: "전체 R+D 인자를 쓴다. 선정 인자가 아니라 다변량이 핵심이다."
    Config은 범주형이라 제외 -- 회귀 트리 특징으로 원-핫 없이 넣을 수 없다.
    """
    return [*schema.r_cols, *schema.d_cols]


_STEP_RE = re.compile(r"^Step(\d+)_")


def step_of(feature: str) -> int | None:
    """인자명에서 스텝 번호를 뽑는다("Step12_R3" -> 12). `Step(\\d+)_`
    패턴에 안 맞는 인자(있다면, 예: Config 계열)는 None."""
    m = _STEP_RE.match(feature)
    return int(m.group(1)) if m else None


def prepare_feature_matrix(
    df: pd.DataFrame, features: list[str], *, max_step: int | None = None
) -> pd.DataFrame:
    """`features` 중 `df`에 없는 컬럼은 전부 NaN으로 채운다 -- train과 eval의
    컬럼 구성이 다른 데이터셋 조합(예: 스키마가 다른 업로드 데이터셋을
    train으로, 내장 test를 eval로)에서도 LGBMRegressor가 그대로 동작하게
    한다. 컬럼이 아예 없다고 예측을 거부할 이유가 없다: NaN이 늘어날
    뿐이고 그 자체가 이미 네이티브로 처리된다.

    `max_step`이 주어지면 그보다 뒤 스텝의 인자는(df에 값이 있어도) 전부
    NaN으로 가린다. 현재 이 모듈의 유일한 호출부(`compute_holdout_predictions`)는
    `max_step`을 쓰지 않지만, 과거 스텝별 마스킹 기능과의 호환을 위해
    파라미터는 남겨 둔다.
    """
    out = pd.DataFrame(index=df.index)
    for f in features:
        if max_step is not None and (step := step_of(f)) is not None and step > max_step:
            out[f] = np.nan
            continue
        out[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan
    return out


@dataclass
class HoldoutPredictions:
    actual_y: np.ndarray
    pred_point: np.ndarray
    residual_std: float
    # conformal margin q -- |실제 - OOF 예측|의 `coverage` 분위수(spec
    # §BA-1). `pred_mean ± conformal_q`가 이론적으로 정규성을 가정하는
    # `residual_std` 기반 ±1.645σ 근사보다 분포 가정 없이 목표 포함률을
    # 보장한다(conformal prediction).
    conformal_q: float
    coverage: float
    n_holdout: int  # q를 낸 out-of-fold 표본 수 -- 표본 부족 경고에 쓴다
    lot_id: np.ndarray


def compute_holdout_predictions(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = FINAL_YIELD_COLUMN,
    group_col: str = "Lot_ID",
    n_splits: int = 5,
    coverage: float = CONFORMAL_TARGET_COVERAGE,
) -> HoldoutPredictions | None:
    """예측 구간 conformal 캘리브레이션 (spec §BA-1) -- train을 LOT 기준
    GroupKFold로 잘라 매 wafer가 자신이 속하지 않은 fold의 모델로만
    예측되게 한(out-of-fold) 점추정치를 모으고, `|실제 - OOF 예측|`의
    `coverage` 분위수(기본 90%)를 conformal margin `q`로 낸다.

    랏 단위로 나누는 것이 핵심이다 -- wafer 단위로 나누면 같은 랏이
    학습·검증 양쪽에 섞여 잔차가 과소평가된다(캘리브레이션이 다시
    무너진다, spec §BA-1 "랏 단위 분할 필수").

    Lot당 표본이 `n_splits`보다 적으면 None (표본 부족).
    """
    valid = train_df[pd.to_numeric(train_df[target_col], errors="coerce").notna()]
    if group_col not in valid.columns or valid[group_col].nunique() < n_splits:
        return None

    y = pd.to_numeric(valid[target_col], errors="coerce")
    x = prepare_feature_matrix(valid, features)
    groups = valid[group_col]

    gkf = GroupKFold(n_splits=n_splits)
    fold_args = list(enumerate(gkf.split(valid, groups=groups)))

    def _fold_predict(fold: int, tr_idx: np.ndarray, ev_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = LGBMRegressor(n_estimators=GBDT_MAX_ITER, random_state=fold, verbose=-1)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        return ev_idx, model.predict(x.iloc[ev_idx])

    results: list[tuple[np.ndarray, np.ndarray]] = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_fold_predict)(fold, tr_idx, ev_idx) for fold, (tr_idx, ev_idx) in fold_args
    )
    oof_pred = np.full(len(valid), np.nan)
    for ev_idx, pred in results:
        oof_pred[ev_idx] = pred

    y_arr = y.to_numpy()
    lot_arr = groups.astype(str).to_numpy()
    covered = ~np.isnan(oof_pred)
    if not covered.any():
        return None
    residuals = y_arr[covered] - oof_pred[covered]
    residual_std = float(np.std(residuals))
    conformal_q = float(np.percentile(np.abs(residuals), coverage * 100.0))
    return HoldoutPredictions(
        actual_y=y_arr[covered],
        pred_point=oof_pred[covered],
        residual_std=residual_std,
        conformal_q=conformal_q,
        coverage=coverage,
        n_holdout=int(covered.sum()),
        lot_id=lot_arr[covered],
    )
