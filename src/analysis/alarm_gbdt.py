"""GBDT 예측 수율 기반 알람 판정 (spec 알람 판정 GBDT 전환 §A) -- 관리한계
이탈량 대신, 부트스트랩 앙상블로 예측한 최종 수율(Y)의 신뢰구간 상한이
낮은 wafer만 알람으로 낸다.

핵심 설계 결정 (spec에 명시된 그대로):
  - 선정 인자가 아니라 **전체 R+D 인자**를 특징으로 쓴다 (§A-1)
  - 결측 마스크 컬럼을 만들지 않는다 -- HistGradientBoostingRegressor가
    NaN을 네이티브로 처리한다 (§A-1)
  - 등급 임계는 표준편차가 아니라 **Y 분위수**다 -- Y 분포가 정규가 아닌
    데이터셋(예: mentorship_final, 평균 10.90·표준편차 17.88)에서 σ 기준은
    음수 임계가 나와 무너진다 (§A-2)
  - 알람 판정에는 예측 평균이 아니라 **신뢰구간 상한(pred_hi)**을 쓴다 --
    예측이 흔들리는 wafer는 상한이 높아 자동으로 제외된다 (§A-2)
  - 알람 개수를 고정하지 않는다. 신뢰할 수 없는 데이터셋(예: killing_event)
    에서는 알람이 0~1건만 나오는 것이 설계 의도다

`random_state`는 부트스트랩 회차 번호(b)로 고정한다 -- 같은 입력이면
새로고침해도 항상 같은 결과가 나온다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

FINAL_YIELD_COLUMN = "Y"
N_BOOTSTRAP = 30
GBDT_MAX_ITER = 200

# §A-2: 등급 임계는 Y 분위수. §A-4/§E: AUC 평가용 "불량" 라벨도 같은
# 5% 분위수를 재사용한다 (spec §0 배경 표의 "하위 5% = 52장"과 동일 정의).
GRADE_QUANTILES = {"심각": 0.05, "위험": 0.10, "주의": 0.15}
IMPROVEMENT_QUANTILE = 0.20
BAD_LABEL_QUANTILE = 0.05

ALARM_SHARE_WARNING_THRESHOLD = 0.10  # spec §A-2: 평가 대상의 10% 초과 시 경고


def feature_columns(schema) -> list[str]:
    """spec §A-1: "전체 R+D 인자를 쓴다. 선정 인자가 아니라 다변량이 핵심이다."
    Config은 범주형이라 제외 -- 회귀 트리 특징으로 원-핫 없이 넣을 수 없다.
    """
    return [*schema.r_cols, *schema.d_cols]


def prepare_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """`features` 중 `df`에 없는 컬럼은 전부 NaN으로 채운다 -- train과 eval의
    컬럼 구성이 다른 데이터셋 조합(예: mentorship_dataset_final을 train으로,
    test를 eval로)에서도 HistGradientBoostingRegressor가 그대로 동작하게
    한다. 컬럼이 아예 없다고 예측을 거부할 이유가 없다: NaN이 늘어날 뿐이고
    그 자체가 이미 네이티브로 처리된다.
    """
    out = pd.DataFrame(index=df.index)
    for f in features:
        out[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan
    return out


@dataclass
class BootstrapPrediction:
    lot_wafer_id: list[str]
    pred_mean: np.ndarray
    pred_lo: np.ndarray  # 90% 구간 하한 (5th percentile)
    pred_hi: np.ndarray  # 90% 구간 상한 (95th percentile)


def _fit_one_bootstrap(
    train_x: pd.DataFrame, train_y: pd.Series, eval_x: pd.DataFrame, b: int
) -> np.ndarray:
    boot_idx = train_x.sample(len(train_x), replace=True, random_state=b).index
    model = HistGradientBoostingRegressor(max_iter=GBDT_MAX_ITER, random_state=b)
    model.fit(train_x.loc[boot_idx], train_y.loc[boot_idx])
    return model.predict(eval_x)


def fit_bootstrap_ensemble(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = FINAL_YIELD_COLUMN,
    n_boot: int = N_BOOTSTRAP,
    id_column: str = "Lot_Wafer_ID",
) -> BootstrapPrediction:
    train_valid = train_df[train_df[target_col].notna()]
    train_x = prepare_feature_matrix(train_valid, features)
    train_y = pd.to_numeric(train_valid[target_col], errors="coerce")
    keep = train_y.notna()
    train_x, train_y = train_x[keep], train_y[keep]

    eval_x = prepare_feature_matrix(eval_df, features)

    # 30회 부트스트랩은 서로 완전히 독립이라 병렬화해도 결정성이 깨지지
    # 않는다 (각 회차의 random_state=b는 그대로 고정) -- 직렬로 돌리면
    # 인자 58개·1만 행 기준 태스크당 2~5초씩 걸려 전체 2분을 넘겨
    # 요청 하나로 감당하기 어렵다.
    preds: list[np.ndarray] = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_fit_one_bootstrap)(train_x, train_y, eval_x, b) for b in range(n_boot)
    )

    p = np.array(preds)
    lot_wafer_id = (
        eval_df[id_column].astype(str).tolist() if id_column in eval_df.columns else [str(i) for i in eval_df.index]
    )
    return BootstrapPrediction(
        lot_wafer_id=lot_wafer_id,
        pred_mean=p.mean(axis=0),
        pred_lo=np.percentile(p, 5, axis=0),
        pred_hi=np.percentile(p, 95, axis=0),
    )


@dataclass
class GradeThresholds:
    severe: float  # 심각: Y 하위 5% 분위수
    danger: float  # 위험: Y 하위 10% 분위수
    caution: float  # 주의: Y 하위 15% 분위수
    improve: float  # 개선 권고: Y 하위 20% 분위수


def compute_grade_thresholds(train_df: pd.DataFrame, target_col: str = FINAL_YIELD_COLUMN) -> GradeThresholds:
    y = pd.to_numeric(train_df[target_col], errors="coerce").dropna()
    return GradeThresholds(
        severe=float(y.quantile(GRADE_QUANTILES["심각"])),
        danger=float(y.quantile(GRADE_QUANTILES["위험"])),
        caution=float(y.quantile(GRADE_QUANTILES["주의"])),
        improve=float(y.quantile(IMPROVEMENT_QUANTILE)),
    )


def grade_of(pred_hi: float, pred_mean: float, thresholds: GradeThresholds) -> str | None:
    """§A-2: 알람은 pred_hi(상한) 기준, 개선 권고는 pred_mean 기준(알람 제외)."""
    if pred_hi <= thresholds.severe:
        return "심각"
    if pred_hi <= thresholds.danger:
        return "위험"
    if pred_hi <= thresholds.caution:
        return "주의"
    if pred_mean <= thresholds.improve:
        return "개선 권고"
    return None


@dataclass
class WaferAlarmScore:
    lot_wafer_id: str
    lot_id: str | None
    grade: str  # "심각" | "위험" | "주의" | "개선 권고"
    risk_percentile: float  # 0-100, 낮을수록 위험 (하위 X%)


def score_alarms(
    eval_df: pd.DataFrame,
    prediction: BootstrapPrediction,
    thresholds: GradeThresholds,
    *,
    id_column: str = "Lot_Wafer_ID",
    lot_column: str = "Lot_ID",
) -> list[WaferAlarmScore]:
    """전체 wafer 중 등급이 매겨진 것만 반환한다 (개수 고정 없음, spec §A-2)."""
    n = len(prediction.pred_mean)
    order = np.argsort(prediction.pred_mean)  # 오름차순: 가장 낮은 예측이 0번째
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n)
    percentile = rank / max(n - 1, 1) * 100.0

    lot_ids = (
        eval_df[lot_column].astype(str).where(eval_df[lot_column].notna(), None).tolist()
        if lot_column in eval_df.columns
        else [None] * n
    )

    results: list[WaferAlarmScore] = []
    for i in range(n):
        grade = grade_of(float(prediction.pred_hi[i]), float(prediction.pred_mean[i]), thresholds)
        if grade is None:
            continue
        results.append(
            WaferAlarmScore(
                lot_wafer_id=prediction.lot_wafer_id[i],
                lot_id=lot_ids[i],
                grade=grade,
                risk_percentile=float(percentile[i]),
            )
        )
    return results


def cross_validate_auc(
    df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = FINAL_YIELD_COLUMN,
    group_col: str = "Lot_ID",
    n_splits: int = 5,
    bad_quantile: float = BAD_LABEL_QUANTILE,
) -> list[float] | None:
    """spec §A-4: 단일 분할이 아니라 GroupKFold 5-fold로 반복 평가한 AUC
    목록을 반환한다 (분위수 산출은 호출자가 한다: 평균은 `np.mean`, 하한은
    `np.percentile(aucs, 5)`).

    "불량"(bad) 라벨은 전체 데이터셋의 Y 하위 5% 분위수로 정의한다 (spec §0
    배경 표의 "하위 5%"와 동일 기준) -- fold마다 다시 정의하면 라벨 자체가
    흔들려 AUC를 fold 간에 비교할 수 없다. 이 라벨 정의는 모델 학습/예측을
    전혀 쓰지 않으므로 홀드아웃 누출이 아니다.

    Lot당 표본이 `n_splits`보다 적어 GroupKFold를 구성할 수 없으면 None을
    반환한다 (표본 부족 -- 호출자가 §A-4 "표본 부족" 안내로 처리한다).
    """
    valid = df[pd.to_numeric(df[target_col], errors="coerce").notna()]
    if group_col not in valid.columns or valid[group_col].nunique() < n_splits:
        return None

    y_all = pd.to_numeric(valid[target_col], errors="coerce")
    bad_threshold = float(y_all.quantile(bad_quantile))
    x_all = prepare_feature_matrix(valid, features)
    groups = valid[group_col]

    gkf = GroupKFold(n_splits=n_splits)
    fold_args = list(enumerate(gkf.split(valid, groups=groups)))

    def _fold_auc(fold: int, tr_idx: np.ndarray, ev_idx: np.ndarray) -> float | None:
        tr_x, tr_y = x_all.iloc[tr_idx], y_all.iloc[tr_idx]
        ev_x, ev_y = x_all.iloc[ev_idx], y_all.iloc[ev_idx]
        bad = (ev_y <= bad_threshold).to_numpy()
        if bad.sum() == 0 or bad.sum() == len(bad):
            return None  # AUC undefined when a fold has only one class
        model = HistGradientBoostingRegressor(max_iter=GBDT_MAX_ITER, random_state=fold)
        model.fit(tr_x, tr_y)
        pred = model.predict(ev_x)
        return float(roc_auc_score(bad, -pred))

    results: list[float | None] = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_fold_auc)(fold, tr_idx, ev_idx) for fold, (tr_idx, ev_idx) in fold_args
    )
    aucs = [a for a in results if a is not None]
    return aucs if aucs else None
