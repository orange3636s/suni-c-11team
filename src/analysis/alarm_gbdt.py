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

# 사전 알람 로그 전면 개편 (spec §B-1) -- 등급 임계는 더 이상 Y 분위수가
# 아니라 "목표 수율 ± 민감도로 조절한 σ 배수" 기준이다 (classify_wafer).
# AUC 평가용 "불량" 라벨은 여전히 Y 하위 5% 분위수를 쓴다 (spec §0 배경
# 표의 "하위 5% = 52장"과 동일 정의 -- 이건 신뢰도 게이트 전용이라
# 사용자가 조절하는 목표 수율과 별개다).
BAD_LABEL_QUANTILE = 0.05

ALARM_SHARE_WARNING_THRESHOLD = 0.10  # spec §A-2: 평가 대상의 10% 초과 시 경고

# 사용자가 아직 목표 수율/민감도를 설정하지 않은 화면(예: 원인 분석 탭의
# 알람 삼각형 마커, 알림 발송 스케줄러)이 쓰는 기본값 -- 사전 알람 로그
# 화면 자체의 기본값과 동일하다 (spec §A-1/§A-2: 목표 85.0, 민감도 균형
# 0.5).
DEFAULT_TARGET_YIELD = 85.0
DEFAULT_SENSITIVITY = 0.5

# 알람 신뢰도 게이트 (spec 알람 신뢰도 게이트 §A-2) -- 교차 데이터셋 홀드아웃
# AUC 하한이 이 값 미만이면 알람을 아예 내지 않는다. 통계적으로 도출된
# 값이 아니라 실측 AUC 분포의 빈 구간(0.55~0.70 사이에 값이 없음) 가운데를
# 잡은 경험값이다 -- 화면에도 이 사실을 명시한다 (§D-2).
AUC_GATE = 0.65


def feature_columns(schema) -> list[str]:
    """spec §A-1: "전체 R+D 인자를 쓴다. 선정 인자가 아니라 다변량이 핵심이다."
    Config은 범주형이라 제외 -- 회귀 트리 특징으로 원-핫 없이 넣을 수 없다.
    """
    return [*schema.r_cols, *schema.d_cols]


def prepare_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """`features` 중 `df`에 없는 컬럼은 전부 NaN으로 채운다 -- train과 eval의
    컬럼 구성이 다른 데이터셋 조합(예: 스키마가 다른 업로드 데이터셋을
    train으로, 내장 test를 eval로)에서도 HistGradientBoostingRegressor가
    그대로 동작하게 한다. 컬럼이 아예 없다고 예측을 거부할 이유가 없다:
    NaN이 늘어날 뿐이고 그 자체가 이미 네이티브로 처리된다.
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


def classify_offset(sensitivity: float) -> float:
    """사전 알람 로그 전면 개편 (spec §A-3/§B-1) -- 민감도 s(0~1)를 σ
    배수 오프셋으로 매핑한다. s=0(오경보 최소)일수록 보수적(+0.6σ),
    s=1(미탐 최소)일수록 민감(-0.2σ)해진다. 이 하나의 함수를 서버(알림
    발송 기본값)와 클라이언트(실시간 재계산)가 동일하게 구현해야 두 쪽의
    판정이 어긋나지 않는다.
    """
    return 0.6 - sensitivity * 0.8


def classify_wafer(
    pred_hi: float,
    pred_lo: float,
    *,
    target: float,
    sensitivity: float,
    sigma: float,
    gate_passed: bool = True,
) -> str | None:
    """사전 알람 로그 전면 개편 (spec §B-1) -- 목표 수율 기준 5분류.
    심각/위험/주의는 신뢰도 게이트를 통과했을 때만 나온다(게이트 미달이면
    정상/판별불가만 계산, spec §B-4: "정상·판별불가는 그대로 계산해
    표시한다"). 다섯 분류는 서로 겹치지 않는다 -- 심각 -> 위험 -> 주의 ->
    정상 순으로 먼저 맞는 조건 하나만 반환하고, 전부 해당 없으면
    None(판별불가: 구간이 목표를 가로지름)이다.
    """
    off = classify_offset(sensitivity)
    if gate_passed:
        if pred_hi <= target - (off + 0.4) * sigma:
            return "심각"
        if pred_hi <= target - (off + 0.2) * sigma:
            return "위험"
        if pred_hi <= target - off * sigma:
            return "주의"
    if pred_lo >= target:
        return "정상"
    return None


@dataclass
class WaferClassification:
    lot_wafer_id: str
    lot_id: str | None
    grade: str | None  # "심각" | "위험" | "주의" | "정상" | None(판별불가)
    pred_mean: float
    pred_lo: float
    pred_hi: float
    risk_percentile: float  # 0-100, 낮을수록 위험 (pred_mean 기준 순위)
    measured: bool  # False면 grade는 항상 None (판별불가·미계측)


def score_wafers(
    eval_df: pd.DataFrame,
    prediction: BootstrapPrediction,
    *,
    target: float,
    sensitivity: float,
    sigma: float,
    gate_passed: bool = True,
    measured_ids: set[str] | None = None,
    lot_column: str = "Lot_ID",
) -> list[WaferClassification]:
    """전체 eval wafer의 5분류 결과 (spec §B-1: "합이 평가 wafer 수와
    정확히 일치해야 한다" -- 개수를 거르지 않고 전부 반환한다. 알람만
    걸러 쓰려는 호출자는 결과에서 grade가 "심각"/"위험"/"주의"인 것만
    추리면 된다). `measured_ids`가 주어지면 그 밖의 wafer는
    measured=False로 표시되고 grade는 항상 None이다 -- 선정 인자가
    계측되지 않은 예측은 신뢰할 수 없어 등급을 매기지 않는다 (기존
    unmeasured_id_set 제외 로직과 동일한 기준, spec §B-2 "미계측").
    """
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

    results: list[WaferClassification] = []
    for i in range(n):
        wafer_id = prediction.lot_wafer_id[i]
        measured = measured_ids is None or wafer_id in measured_ids
        grade = (
            classify_wafer(
                float(prediction.pred_hi[i]), float(prediction.pred_lo[i]),
                target=target, sensitivity=sensitivity, sigma=sigma, gate_passed=gate_passed,
            )
            if measured
            else None
        )
        results.append(
            WaferClassification(
                lot_wafer_id=wafer_id,
                lot_id=lot_ids[i],
                grade=grade,
                pred_mean=float(prediction.pred_mean[i]),
                pred_lo=float(prediction.pred_lo[i]),
                pred_hi=float(prediction.pred_hi[i]),
                risk_percentile=float(percentile[i]),
                measured=measured,
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


def cross_validate_transfer_auc(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = FINAL_YIELD_COLUMN,
    group_col: str = "Lot_ID",
    n_splits: int = 5,
    bad_quantile: float = BAD_LABEL_QUANTILE,
) -> list[float] | None:
    """알람 신뢰도 게이트 §A-1/§A-2 -- `cross_validate_auc`는 train 자기 자신에
    대한 등급이라 분포가 다른 eval로 옮기면 그대로 무너지는 상황을 감지하지
    못한다 (train만 보고 계산하므로 eval이 무엇이든 같은 값이 나온다). 이
    함수는 train을 LOT 기준 5-fold GroupKFold로 나눠 각 fold 모델을
    **eval_df에 대해** 평가한다 -- "이 train으로 학습한 모델이 이 특정
    eval 분포에서도 통한다"는 실제 전이 성능을 잰다.

    "불량" 라벨은 train의 하위 5% 분위수 **값**을 eval의 실측 Y에 그대로
    적용한다 (spec의 알람 임계 자체가 train 분위수를 eval에 적용하는
    방식과 같은 기준을 쓴다).

    Lot당 표본이 `n_splits`보다 적으면(train) 또는 eval에 유효 라벨이
    하나도 없으면(bad가 전부 같은 클래스) None -- 호출자가 "표본 부족"으로
    처리한다.
    """
    train_valid = train_df[pd.to_numeric(train_df[target_col], errors="coerce").notna()]
    if group_col not in train_valid.columns or train_valid[group_col].nunique() < n_splits:
        return None

    y_train = pd.to_numeric(train_valid[target_col], errors="coerce")
    bad_threshold = float(y_train.quantile(bad_quantile))
    x_train = prepare_feature_matrix(train_valid, features)
    groups = train_valid[group_col]

    eval_valid = eval_df[pd.to_numeric(eval_df[target_col], errors="coerce").notna()]
    if eval_valid.empty:
        return None
    y_eval = pd.to_numeric(eval_valid[target_col], errors="coerce")
    bad_eval = (y_eval <= bad_threshold).to_numpy()
    if bad_eval.sum() == 0 or bad_eval.sum() == len(bad_eval):
        return None  # AUC undefined with a single class
    x_eval = prepare_feature_matrix(eval_valid, features)

    gkf = GroupKFold(n_splits=n_splits)
    fold_args = list(enumerate(gkf.split(train_valid, groups=groups)))

    def _fold_auc(fold: int, tr_idx: np.ndarray) -> float:
        model = HistGradientBoostingRegressor(max_iter=GBDT_MAX_ITER, random_state=fold)
        model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(x_eval)
        return float(roc_auc_score(bad_eval, -pred))

    results: list[float] = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_fold_auc)(fold, tr_idx) for fold, (tr_idx, _ev_idx) in fold_args
    )
    return results if results else None


@dataclass
class HoldoutPredictions:
    actual_y: np.ndarray
    pred_point: np.ndarray
    residual_std: float


def compute_holdout_predictions(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = FINAL_YIELD_COLUMN,
    group_col: str = "Lot_ID",
    n_splits: int = 5,
) -> HoldoutPredictions | None:
    """사전 알람 로그 전면 개편 (spec §A-4) -- "정밀도·재현율은 학습 데이터
    홀드아웃 기준 추정치"의 근거 데이터. 평가 데이터(eval)의 실제 Y는
    모르므로 정밀도/재현율을 잴 수 없다 -- 대신 train을 LOT 기준
    GroupKFold로 잘라 매 wafer가 자신이 속하지 않은 fold의 모델로만
    예측되게 한(out-of-fold) 점추정치를 모은다. 전체 잔차의 표준편차
    (`residual_std`)로 90% 구간(±1.645σ)을 근사해 pred_lo/pred_hi를 만들 수
    있게 한다 -- 실제 알람 판정에 쓰는 30회 부트스트랩 앙상블만큼
    정교하지는 않지만("추정치"라 명시하는 이유), fold마다 앙상블을 다시
    도는 것보다 5배 이상 가볍다.

    Lot당 표본이 `n_splits`보다 적으면 None (표본 부족 -- 호출자가 안내로
    처리한다).
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
        model = HistGradientBoostingRegressor(max_iter=GBDT_MAX_ITER, random_state=fold)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        return ev_idx, model.predict(x.iloc[ev_idx])

    results: list[tuple[np.ndarray, np.ndarray]] = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_fold_predict)(fold, tr_idx, ev_idx) for fold, (tr_idx, ev_idx) in fold_args
    )
    oof_pred = np.full(len(valid), np.nan)
    for ev_idx, pred in results:
        oof_pred[ev_idx] = pred

    y_arr = y.to_numpy()
    covered = ~np.isnan(oof_pred)
    if not covered.any():
        return None
    residual_std = float(np.std(y_arr[covered] - oof_pred[covered]))
    return HoldoutPredictions(actual_y=y_arr[covered], pred_point=oof_pred[covered], residual_std=residual_std)
