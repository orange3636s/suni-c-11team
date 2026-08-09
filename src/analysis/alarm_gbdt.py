"""GBDT 예측 수율 기반 알람 판정 (spec 알람 판정 GBDT 전환 §A, 이후 "민감도
슬라이더를 실제 트레이드오프로" §CA로 판정 기준을 교체) -- 관리한계
이탈량 대신, 부트스트랩 앙상블로 예측한 최종 수율(Y)의 점추정이 목표보다
낮은 wafer를 알람으로 낸다.

핵심 설계 결정 (spec에 명시된 그대로):
  - 선정 인자가 아니라 **전체 R+D 인자**를 특징으로 쓴다 (§A-1)
  - 결측 마스크 컬럼을 만들지 않는다 -- HistGradientBoostingRegressor가
    NaN을 네이티브로 처리한다 (§A-1)
  - 알람 판정에는 **점추정(pred_mean)**을 쓴다 (§CA-1) -- 신뢰구간 상한
    (pred_hi) 기준이던 이전 방식은 conformal 캘리브레이션으로 구간이
    ±5.5%p까지 넓어지면서 민감도를 끝까지 올려도 오탐이 전혀 나지 않는
    문제(오탐↔미탐 트레이드오프가 아니라 미탐만 줄어드는 슬라이더)를
    낳았다. 구간 자체는 화면 표시(EvidenceBand)에서 그대로 쓴다 --
    판정과 표시는 분리된 개념이다.
  - 등급 임계는 σ(웨이퍼 수율 산포) 배수가 아니라 **%p 절대값**이다
    (§CA-1) -- σ는 "예측 불확실성"이 아니라 "웨이퍼 수율 산포"라 개념이
    어긋나 있었다.
  - `정상`만은 여전히 구간 하한(pred_lo)을 쓴다 -- "정상"이라고 단정하려면
    보수적이어야 하고, 점추정으로 정상을 선언하면 실제 미달 wafer를
    놓친다 (§CA-3).
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

# AUC 평가용 "불량" 라벨은 여전히 Y 하위 5% 분위수를 쓴다 (spec §0 배경
# 표의 "하위 5% = 52장"과 동일 정의 -- 이건 신뢰도 게이트 전용이라
# 사용자가 조절하는 목표 수율과 별개다).
BAD_LABEL_QUANTILE = 0.05

ALARM_SHARE_WARNING_THRESHOLD = 0.10  # spec §A-2: 평가 대상의 10% 초과 시 경고

# 사용자가 아직 목표 수율/민감도를 설정하지 않은 화면(예: 원인 분석 탭의
# 알람 삼각형 마커, 알림 발송 스케줄러)이 쓰는 기본값 -- 사전 알람 로그
# 화면 자체의 기본값과 동일하다. AA-4: 민감도 기본값은 "오경보 최소"
# 프리셋(frontend/app/alerts/page.tsx SENSITIVITY_PRESETS.low_fp)과
# 반드시 같은 값을 유지한다 -- 프런트 AnalysisStateProvider의
# DEFAULT_SENSITIVITY도 함께 바꿔야 첫 로딩과 서버 판정 기준이 어긋나지
# 않는다.
DEFAULT_TARGET_YIELD = 85.0
DEFAULT_SENSITIVITY = 0.2

# 알람 신뢰도 게이트 (spec 알람 신뢰도 게이트 §A-2) -- 교차 데이터셋 홀드아웃
# AUC 하한이 이 값 미만이면 알람을 아예 내지 않는다. 통계적으로 도출된
# 값이 아니라 실측 AUC 분포의 빈 구간(0.55~0.70 사이에 값이 없음) 가운데를
# 잡은 경험값이다 -- 화면에도 이 사실을 명시한다 (§D-2).
AUC_GATE = 0.65

# 예측 구간 conformal 캘리브레이션 (spec "예측 구간 캘리브레이션 + 미분류
# 사유 분리" §BA) -- 목표 포함률. q(홀드아웃 |잔차| 분위수)는 이 값 기준
# 분위수를 쓴다. 서버 상수로만 조절한다 -- UI는 만들지 않는다(§BA-3).
CONFORMAL_TARGET_COVERAGE = 0.90


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
    pred_mean: np.ndarray  # 30회 부트스트랩 앙상블 평균 -- 점추정 안정화 전용, 구간 산출에는 안 쓴다
    pred_lo: np.ndarray  # pred_mean - conformal_q (conformal_q가 없으면 부트스트랩 5th percentile로 대체)
    pred_hi: np.ndarray  # pred_mean + conformal_q (conformal_q가 없으면 부트스트랩 95th percentile로 대체)
    # 홀드아웃 |잔차| 분위수(spec §BA-1/§BA-2) -- None이면 랏 수 부족으로
    # conformal을 낼 수 없어 구간이 부트스트랩 분위수로 대체됐다는 뜻이다
    # (§BA-5: 둘을 섞어 쓰지 않는다 -- 한 예측 안에서는 항상 둘 중 하나만).
    conformal_q: float | None
    coverage_target: float  # q를 낼 때 목표한 포함률 (기본 0.90, §BA-3)
    # eval_df에 실측 Y가 있을 때만 채워진다 -- 실제 Y가 [pred_lo, pred_hi]에
    # 들어간 비율(§BA-4 검증). 실측 Y가 없으면(일반적인 평가 상황) None.
    coverage_actual: float | None
    # 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4) -- 랏 단위
    # GroupKFold out-of-fold (실제 Y, 예측값) 쌍의 층화 샘플(최대
    # HOLDOUT_OOF_SAMPLE_SIZE개). 클라이언트가 슬라이더를 움직일 때마다
    # 서버를 다시 부르지 않고 이 쌍으로 정밀도·재현율을 즉시 재계산한다.
    # None이면 랏 수 부족으로 홀드아웃 자체를 못 낸 것(conformal_q와 같은
    # 조건).
    holdout_oof_actual: np.ndarray | None
    holdout_oof_pred: np.ndarray | None
    # 집계 수준(SUMMARY 등 eval 전체 평균) conformal 여유 (spec GA) -- 웨이퍼
    # 단위 conformal_q를 평균에 그대로 적용하면 평균의 불확실성을 개별값
    # 수준으로 과대평가한다(랏 내 상관 때문에 q/sqrt(n)도 과소평가라 쓰지
    # 않는다, GA-1). 랏 블록 부트스트랩으로 별도 산출한다. None이면
    # conformal_q와 같은 이유(랏 수 부족)로 낼 수 없었다는 뜻.
    conformal_q_agg: float | None
    pred_agg_mean: float  # eval 전체 pred_mean 평균 (SUMMARY 예측 수율 점추정)
    pred_agg_lo: float | None  # pred_agg_mean - conformal_q_agg
    pred_agg_hi: float | None  # pred_agg_mean + conformal_q_agg


HOLDOUT_OOF_SAMPLE_SIZE = 1000
HOLDOUT_OOF_SAMPLE_STRATA = 20
HOLDOUT_OOF_SAMPLE_SEED = 0


def _stratified_holdout_sample(
    actual_y: np.ndarray,
    pred_point: np.ndarray,
    *,
    n: int = HOLDOUT_OOF_SAMPLE_SIZE,
    n_strata: int = HOLDOUT_OOF_SAMPLE_STRATA,
    seed: int = HOLDOUT_OOF_SAMPLE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4) -- 홀드아웃 OOF
    쌍을 클라이언트로 그대로 내려보내면(train 1만 행 기준) payload가
    커지므로 `n`개로 층화 샘플링한다. 실제 Y 기준 `n_strata`개 구간으로
    나눠 구간마다 고르게 뽑는다 -- 무작위 샘플링만 하면 Y 분포가 두꺼운
    가운데 구간에 표본이 몰려, 정밀도·재현율 추정에 정작 중요한 미달
    영역(꼬리)의 표본이 부족해진다. `seed`를 고정해 같은 입력이면 항상
    같은 표본이 나온다(재현성).
    """
    total = len(actual_y)
    if total <= n:
        return actual_y, pred_point

    order = np.argsort(actual_y)
    strata = np.array_split(order, n_strata)
    per_stratum = max(1, n // n_strata)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stratum in strata:
        if len(stratum) <= per_stratum:
            picked.extend(stratum.tolist())
        else:
            picked.extend(rng.choice(stratum, size=per_stratum, replace=False).tolist())
    idx = np.array(sorted(picked))[:n]
    return actual_y[idx], pred_point[idx]


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
    group_col: str = "Lot_ID",
    coverage_target: float = CONFORMAL_TARGET_COVERAGE,
) -> BootstrapPrediction:
    """예측 구간 conformal 캘리브레이션 (spec §BA) -- 부트스트랩 앙상블은
    점추정(`pred_mean`)의 안정성에만 쓰고, 구간(`pred_lo`/`pred_hi`)은
    train을 랏 단위 GroupKFold로 잘라 낸 out-of-fold |잔차|의
    `coverage_target` 분위수(conformal margin `q`)로 낸다. 부트스트랩
    5/95 분위수(모델이 어느 표본을 뽑았는지에 대한 불확실성만 반영)는
    잔차 자체의 불확실성(그 모델도 얼마나 틀리는지)을 재지 못해 실측
    포함률이 목표(90%)의 1/4 수준(18~25%)으로 나왔다 -- 이 함수가 그
    캘리브레이션 붕괴를 고친다.
    """
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
    pred_mean = p.mean(axis=0)
    lot_wafer_id = (
        eval_df[id_column].astype(str).tolist() if id_column in eval_df.columns else [str(i) for i in eval_df.index]
    )

    holdout = compute_holdout_predictions(
        train_df, features, target_col=target_col, group_col=group_col, coverage=coverage_target
    )
    if holdout is not None:
        q = holdout.conformal_q
        pred_lo = pred_mean - q
        pred_hi = pred_mean + q
        holdout_oof_actual, holdout_oof_pred = _stratified_holdout_sample(holdout.actual_y, holdout.pred_point)
    else:
        holdout_oof_actual = holdout_oof_pred = None
        # 랏 수 부족(§BA-1: GroupKFold(5) 구성 불가)으로 conformal q를 낼
        # 수 없을 때만 부트스트랩 분위수로 대체한다(표본 부족 폴백 --
        # AUC 게이트 등 이 코드베이스의 다른 "표본 부족" 처리와 같은
        # 원칙). q=None이라 conformal이 적용되지 않았다는 사실이 그대로
        # 응답에 남는다.
        q = None
        pred_lo = np.percentile(p, 5, axis=0)
        pred_hi = np.percentile(p, 95, axis=0)

    actual_y = pd.to_numeric(eval_df[target_col], errors="coerce") if target_col in eval_df.columns else None
    coverage_actual = None
    if actual_y is not None:
        valid = actual_y.notna().to_numpy()
        if valid.any():
            covered = (actual_y.to_numpy()[valid] >= pred_lo[valid]) & (actual_y.to_numpy()[valid] <= pred_hi[valid])
            coverage_actual = float(covered.mean())

    # 집계 수준 conformal 여유 (spec GA-1) -- eval 전체를 랏 몇 개로
    # 나눠 평균 내는지(n_lots_eval)에 맞춰 랏 블록 부트스트랩으로 별도
    # 산출한다. 웨이퍼 q를 그대로 쓰거나 q/sqrt(n)으로 나누지 않는다.
    pred_agg_mean = float(pred_mean.mean())
    q_agg: float | None = None
    if holdout is not None:
        n_lots_eval = (
            int(eval_df[group_col].nunique()) if group_col in eval_df.columns else len(eval_df)
        )
        q_agg = compute_aggregate_conformal_q(holdout, n_lots_eval, coverage=coverage_target)
    pred_agg_lo = pred_agg_mean - q_agg if q_agg is not None else None
    pred_agg_hi = pred_agg_mean + q_agg if q_agg is not None else None

    return BootstrapPrediction(
        lot_wafer_id=lot_wafer_id,
        pred_mean=pred_mean,
        pred_lo=pred_lo,
        pred_hi=pred_hi,
        conformal_q=q,
        coverage_target=coverage_target,
        coverage_actual=coverage_actual,
        holdout_oof_actual=holdout_oof_actual,
        holdout_oof_pred=holdout_oof_pred,
        conformal_q_agg=q_agg,
        pred_agg_mean=pred_agg_mean,
        pred_agg_lo=pred_agg_lo,
        pred_agg_hi=pred_agg_hi,
    )


# 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-1) -- 민감도 0(오경보
# 최소)일 때 margin이 최대(가장 보수적), 1(미탐 최소)일 때 margin이 0(가장
# 민감)이 되도록 선형 매핑한다. %p 절대값이다 -- 이전의 σ(웨이퍼 수율
# 산포) 배수 방식은 "예측 불확실성"과 무관한 값을 쓰고 있었다.
MARGIN_MAX_PP = 4.0
# 등급(심각/위험/주의) 간 간격 -- %p. 검증 결과(§CA-5) 이 간격이 등급 간
# 실제 위험도 차이를 만들지 못하는 정황(정밀도 역전)이 있었으나, 이번
# 작업 범위는 슬라이더 트레이드오프化뿐이라 값은 그대로 두고 다음 단계
# 조사 과제로 남긴다.
GRADE_STEP_PP = 0.8


def classify_margin(sensitivity: float) -> float:
    """민감도 슬라이더를 실제 트레이드오프로 (spec §CA-1) -- 민감도
    s(0~1)를 목표 수율 대비 여유(margin, %p)로 매핑한다. s=0(오경보
    최소)일수록 여유가 크고(가장 보수적, MARGIN_MAX_PP), s=1(미탐
    최소)일수록 여유가 0(가장 민감)이다. 이 하나의 함수를 서버(알림 발송
    기본값)와 클라이언트(실시간 재계산)가 동일하게 구현해야 두 쪽의
    판정이 어긋나지 않는다.
    """
    return (1.0 - sensitivity) * MARGIN_MAX_PP


def classify_wafer(
    pred_mean: float,
    pred_lo: float,
    *,
    target: float,
    sensitivity: float,
    gate_passed: bool = True,
) -> str | None:
    """민감도 슬라이더를 실제 트레이드오프로 (spec §CA-1/§CA-3) -- 목표
    수율 기준 5분류. 심각/위험/주의는 **점추정(pred_mean)** 기준이다 --
    conformal 구간 상한(pred_hi)을 기준으로 삼으면 구간이 넓어(±5.5%p
    안팎) 민감도를 끝까지 올려도 오탐이 전혀 나지 않아 슬라이더가
    무의미해진다(미탐만 줄고 오탐↔미탐 트레이드오프가 없다). `정상`만은
    여전히 구간 하한(pred_lo) 기준이다 -- "정상"이라고 단정하려면
    보수적이어야 한다.

    심각/위험/주의는 신뢰도 게이트를 통과했을 때만 나온다(게이트 미달이면
    정상/판별불가만 계산, spec §B-4). 다섯 분류는 서로 겹치지 않는다 --
    심각 -> 위험 -> 주의 -> 정상 순으로 먼저 맞는 조건 하나만 반환하고,
    전부 해당 없으면 None(판별불가)이다.
    """
    margin = classify_margin(sensitivity)
    if gate_passed:
        if pred_mean <= target - margin - 2 * GRADE_STEP_PP:
            return "심각"
        if pred_mean <= target - margin - GRADE_STEP_PP:
            return "위험"
        if pred_mean <= target - margin:
            return "주의"
    if pred_lo >= target:
        return "정상"
    return None


# spec §BC-2: 계측 없이(measured=False) 등급이 매겨진 wafer는 어느 선정
# 인자도 근거로 들 수 없다 -- 정상 사유(build_alarm_reason) 대신 이
# 문구를 쓴다. 사유가 없는 알람은 자동 발송 대상에서도 제외한다
# (compute_alarm_notification_items/refresh.py).
NO_REASON_UNMEASURED = "사유 제시 불가 — 선정 인자 미계측"


@dataclass
class WaferClassification:
    lot_wafer_id: str
    lot_id: str | None
    grade: str | None  # "심각" | "위험" | "주의" | "정상" | None(판별불가)
    pred_mean: float
    pred_lo: float
    pred_hi: float
    risk_percentile: float  # 0-100, 낮을수록 위험 (pred_mean 기준 순위)
    # spec §BC-1: 더 이상 판정 게이트가 아니다 -- False여도 grade는 구간
    # 기준으로 정상 계산된다. 사유 표시 전용(§BB "계측 부족" 판별불가
    # 사유 / §BC-2 "사유 제시 불가" 배지·자동 발송 제외).
    measured: bool


def score_wafers(
    eval_df: pd.DataFrame,
    prediction: BootstrapPrediction,
    *,
    target: float,
    sensitivity: float,
    gate_passed: bool = True,
    measured_ids: set[str] | None = None,
    lot_column: str = "Lot_ID",
) -> list[WaferClassification]:
    """전체 eval wafer의 5분류 결과 (spec §B-1: "합이 평가 wafer 수와
    정확히 일치해야 한다" -- 개수를 거르지 않고 전부 반환한다. 알람만
    걸러 쓰려는 호출자는 결과에서 grade가 "심각"/"위험"/"주의"인 것만
    추리면 된다).

    예측 구간 캘리브레이션 + 미분류 사유 분리 (spec §BC-1) -- `measured_ids`는
    더 이상 판정 게이트가 아니다. 계측 개수가 예측 품질을 예고하지
    못했고(실측 MAE: 미계측군 2.694 / 1개만 2.923 -- 1개만 계측된 쪽이
    오히려 더 부정확했다), conformal 구간이 이미 그 불확실성을 폭으로
    반영한다. `measured_ids`가 주어지면 `measured` 필드에만 반영되고
    (§BC-1 "판정 결과가 아니라 사유 표시에만 쓴다", §BC-2 "사유 제시
    불가" 표기), grade는 measured 여부와 무관하게 항상 `classify_wafer`로
    계산한다.
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
        grade = classify_wafer(
            float(prediction.pred_mean[i]), float(prediction.pred_lo[i]),
            target=target, sensitivity=sensitivity, gate_passed=gate_passed,
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
    # conformal margin q -- |실제 - OOF 예측|의 `coverage` 분위수(spec
    # §BA-1). `pred_mean ± conformal_q`가 이론적으로 정규성을 가정하는
    # `residual_std` 기반 ±1.645σ 근사보다 분포 가정 없이 목표 포함률을
    # 보장한다(conformal prediction).
    conformal_q: float
    coverage: float
    n_holdout: int  # q를 낸 out-of-fold 표본 수 -- 표본 부족 경고에 쓴다
    # 집계 수준 여유(spec GA-1) 산출용 -- actual_y/pred_point와 같은 순서로
    # 정렬된 랏 ID. 랏 블록 부트스트랩(compute_aggregate_conformal_q)이
    # 잔차를 랏 단위로 묶는 데 쓴다.
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

    Lot당 표본이 `n_splits`보다 적으면 None (표본 부족 -- 호출자가 부트스트랩
    분위수로 폴백한다, `fit_bootstrap_ensemble` 참고).
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


# 집계 수준 conformal 여유 (spec GA-1) -- 랏 블록 부트스트랩 회차 수. 2,000회는
# 90분위 절대값 추정의 몬테카를로 오차를 실무적으로 무시할 수준까지 줄이면서도
# (표준오차가 대략 sqrt(0.9*0.1/2000)≈0.0067) 요청 하나로 감당 가능한 비용이다.
AGGREGATE_BOOTSTRAP_ROUNDS = 2000
AGGREGATE_BOOTSTRAP_SEED = 0


def compute_aggregate_conformal_q(
    holdout: HoldoutPredictions,
    n_lots: int,
    *,
    coverage: float = CONFORMAL_TARGET_COVERAGE,
    n_rounds: int = AGGREGATE_BOOTSTRAP_ROUNDS,
    seed: int = AGGREGATE_BOOTSTRAP_SEED,
) -> float | None:
    """집계(SUMMARY 등 eval 전체 평균) 수준 conformal 여유 (spec GA-1).

    웨이퍼 단위 conformal_q(홀드아웃 |잔차|의 분위수)를 그대로 평균에
    적용하면 평균의 불확실성을 개별 웨이퍼 수준으로 과대평가한다.
    `conformal_q / sqrt(n)`으로 나누지도 않는다 -- 같은 랏 안에서 잔차가
    상관돼 있어 sqrt(n) 이득이 그대로 나오지 않고, 단순 나눗셈은 그
    상관을 무시해 이번엔 반대로 과소평가한다.

    대신 홀드아웃 OOF 잔차를 랏 단위로 묶고, `n_lots`개 랏을 복원추출로
    리샘플링하는 랏 블록 부트스트랩을 `n_rounds`회 반복한다. 회차마다
    뽑힌 랏들에 속한 모든 웨이퍼 잔차의 평균을 모아 분포를 만들고, 그
    분포의 절대값 `coverage` 분위수를 여유로 쓴다 -- 랏 내부 상관을
    리샘플링 단위(랏)로 자연스럽게 반영한다.

    `n_lots`은 실제로 평균 내는 대상(eval 전체)의 랏 수와 맞춰야 한다 --
    다른 집계 크기(예: 랏 단위 25장 평균)에는 그 크기에 맞는 `n_lots`로
    별도 호출해야 하며, 이 함수가 반환한 값을 다른 크기에 재사용하면
    안 된다.
    """
    if n_lots <= 0:
        return None
    unique_lots, inverse = np.unique(holdout.lot_id, return_inverse=True)
    if len(unique_lots) == 0:
        return None
    residuals = holdout.actual_y - holdout.pred_point
    residuals_by_lot = [residuals[inverse == i] for i in range(len(unique_lots))]

    rng = np.random.default_rng(seed)
    round_means = np.empty(n_rounds)
    n_unique_lots = len(unique_lots)
    for i in range(n_rounds):
        drawn = rng.integers(0, n_unique_lots, size=n_lots)
        pooled = np.concatenate([residuals_by_lot[lot_idx] for lot_idx in drawn])
        round_means[i] = pooled.mean()
    return float(np.percentile(np.abs(round_means), coverage * 100.0))
