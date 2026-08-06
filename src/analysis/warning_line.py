"""수율 경고선 (spec 알람 판정 GBDT 전환 §C) -- 관리한계(IQR 1.5배)를
부분 의존도(partial dependence) 기반 경고선으로 교체한다.

정통 SPC의 관리한계는 "공정이 시간에 따라 변했는가"를 시계열로 판정하는
도구다. 이 앱의 인자 데이터는 횡단면(cross-sectional)이라 그 전제 자체가
맞지 않는다 (§C-1). 대신 "이 값이 넘으면 예측 수율이 얼마나 떨어지는가"를
부분 의존도 곡선에서 직접 읽어 경고선으로 쓴다.

부분 의존도 곡선 자체는 절대 높이를 신뢰할 수 없으므로(예측 오차가 Y
표준편차의 72~80%, alarm_gbdt.py 참고) 화면에 그리지 않는다 -- 경고선
"위치"만 여기서 계산해 쓴다. 경고선 범례에 표시하는 수율 차이는 예측이
아니라 **관측값**으로 별도 계산한다 (§C-4-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.analysis.alarm_gbdt import GBDT_MAX_ITER, FINAL_YIELD_COLUMN, prepare_feature_matrix

PDP_POINTS = 40
PDP_LOW_QUANTILE = 0.02
PDP_HIGH_QUANTILE = 0.98
THRESHOLD_STD_MULTIPLIER = 0.35
MIN_OBSERVED_GAP_SAMPLE = 30
REFERENCE_MODEL_RANDOM_STATE = 0


def fit_reference_model(
    train_df: pd.DataFrame, features: list[str], *, target_col: str = FINAL_YIELD_COLUMN
) -> HistGradientBoostingRegressor:
    """경고선 계산 전용 단일 모델 -- §A-1의 30개 부트스트랩 앙상블과는 별개
    목적(예측 신뢰구간이 아니라 부분 의존도 곡선의 '모양'만 필요)이라
    한 번만 학습해 모든 인자의 경고선 계산에 재사용한다.
    """
    valid = train_df[pd.to_numeric(train_df[target_col], errors="coerce").notna()]
    x = prepare_feature_matrix(valid, features)
    y = pd.to_numeric(valid[target_col], errors="coerce")
    model = HistGradientBoostingRegressor(max_iter=GBDT_MAX_ITER, random_state=REFERENCE_MODEL_RANDOM_STATE)
    model.fit(x, y)
    return model


@dataclass
class WarningLine:
    lower: float | None
    upper: float | None
    pdp_range: float  # max(pdp) - min(pdp), JSON 보고서에만 기록 (화면 비노출)


def compute_warning_line(
    model: HistGradientBoostingRegressor,
    train_df: pd.DataFrame,
    feature: str,
    features: list[str],
    *,
    target_col: str = FINAL_YIELD_COLUMN,
    n_points: int = PDP_POINTS,
) -> WarningLine | None:
    """spec §C-2. 다른 인자는 중앙값으로 고정하고 `feature`만 관측 2~98%
    분위수 구간에서 변화시켜 예측(부분 의존도 곡선)한 뒤, 전체 예측
    평균에서 0.35 x 예측 표준편차를 뺀 값을 임계로 삼는다.

    spec이 준 예시 코드는 한쪽 방향(`below.min()`)만 보여주지만 본문은
    "하한과 상한이 각각 독립적으로 나올 수 있다"고 명시한다 -- 곡선의
    전역 최댓값(가장 좋은 예측 지점)을 기준으로 왼쪽/오른쪽을 나눠 각각
    독립적으로 임계를 넘는 경계를 찾는다. 단조 곡선이면 최댓값이 한쪽
    끝에 있어 자연히 한쪽 경고선만 나온다 (Step1_D1 실측: 상한 11.8만
    존재, 하한 없음 -- 아래 골든 테스트 참고).
    """
    if feature not in train_df.columns:
        return None
    x_feature = pd.to_numeric(train_df[feature], errors="coerce").dropna()
    if len(x_feature) < 10 or x_feature.nunique() < 2:
        return None

    lo_q, hi_q = x_feature.quantile([PDP_LOW_QUANTILE, PDP_HIGH_QUANTILE])
    if not np.isfinite(lo_q) or not np.isfinite(hi_q) or hi_q <= lo_q:
        return None
    xs = np.linspace(float(lo_q), float(hi_q), n_points)

    valid = train_df[pd.to_numeric(train_df[target_col], errors="coerce").notna()]
    med = prepare_feature_matrix(valid, features).median()
    grid = pd.DataFrame([med.to_dict()] * len(xs))
    grid[feature] = xs
    pdp = model.predict(grid[features])

    all_pred = model.predict(prepare_feature_matrix(valid, features))
    threshold = float(pdp.mean() - THRESHOLD_STD_MULTIPLIER * all_pred.std())
    pdp_range = float(pdp.max() - pdp.min())

    peak_idx = int(np.argmax(pdp))
    left_xs, left_pdp = xs[: peak_idx + 1], pdp[: peak_idx + 1]
    right_xs, right_pdp = xs[peak_idx:], pdp[peak_idx:]

    lower: float | None = None
    below_left = left_xs[left_pdp <= threshold]
    if len(below_left):
        lower = float(below_left.max())

    upper: float | None = None
    below_right = right_xs[right_pdp <= threshold]
    if len(below_right):
        upper = float(below_right.min())

    return WarningLine(lower=lower, upper=upper, pdp_range=pdp_range)


def observed_yield_gap(
    train_df: pd.DataFrame,
    feature: str,
    warning_line: WarningLine,
    *,
    target_col: str = FINAL_YIELD_COLUMN,
    min_sample: int = MIN_OBSERVED_GAP_SAMPLE,
) -> dict[str, float | None]:
    """spec §C-4-1: 범례에 쓸 수율 차이는 예측이 아니라 관측값으로 계산한다.
    상한/하한 각각 독립적으로 계산하며, 경고선 밖 표본이 `min_sample`
    미만이면 그 방향의 값은 None(생략)이다.
    """
    x = pd.to_numeric(train_df[feature], errors="coerce")
    y = pd.to_numeric(train_df[target_col], errors="coerce")
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]

    def _gap(mask: pd.Series) -> float | None:
        if mask.sum() < min_sample or (~mask).sum() == 0:
            return None
        return float(y[mask].mean() - y[~mask].mean())

    upper_gap = _gap(x >= warning_line.upper) if warning_line.upper is not None else None
    lower_gap = _gap(x <= warning_line.lower) if warning_line.lower is not None else None
    return {"upper_gap": upper_gap, "lower_gap": lower_gap}


DEFAULT_REASON_MORE_COUNT_LABEL = "외 {n}건"
NO_EXCEEDANCE_REASON = "개별 인자는 정상 범위이나 조합이 위험 패턴에 해당"


def compute_all_warning_lines(
    model: "HistGradientBoostingRegressor",  # noqa: F821 -- see TYPE_CHECKING import below
    train_df: pd.DataFrame,
    features: list[str],
) -> dict[str, WarningLine]:
    """전체 R+D 인자에 대해 경고선을 한 번에 계산해 캐시할 수 있는 형태로
    반환한다 -- §A-3 알람 사유는 "선정 인자" 몇 개가 아니라 알람 판정
    자체가 쓰는 전체 인자 풀을 대상으로 초과 여부를 확인해야 하므로
    (GBDT 알람은 특정 타깃 하나가 아니라 Y 전체를 보고 판정한다), 인자마다
    반복 호출하지 않고 여기서 한 번에 모아 계산한다.
    """
    out: dict[str, WarningLine] = {}
    for feature in features:
        wl = compute_warning_line(model, train_df, feature, features)
        if wl is not None and (wl.lower is not None or wl.upper is not None):
            out[feature] = wl
    return out


@dataclass
class ExceedanceInfo:
    feature: str
    value: float
    line_value: float
    direction: str  # "초과" | "미만"
    overage_ratio: float  # (value-line)/line 절대값 -- 초과량 비율 순위용


def _exceedances_for_row(row: pd.Series, warning_lines: dict[str, WarningLine]) -> list[ExceedanceInfo]:
    out: list[ExceedanceInfo] = []
    for feature, wl in warning_lines.items():
        if feature not in row.index:
            continue
        value = row[feature]
        if pd.isna(value):
            continue
        value = float(value)
        if wl.upper is not None and value > wl.upper:
            ratio = abs(value - wl.upper) / abs(wl.upper) if wl.upper != 0 else float("inf")
            out.append(ExceedanceInfo(feature, value, wl.upper, "초과", ratio))
        elif wl.lower is not None and value < wl.lower:
            ratio = abs(wl.lower - value) / abs(wl.lower) if wl.lower != 0 else float("inf")
            out.append(ExceedanceInfo(feature, value, wl.lower, "미만", ratio))
    return out


def build_alarm_reason(row: pd.Series, warning_lines: dict[str, WarningLine]) -> str:
    """spec §A-3 "사유 표시": 경고선을 초과한 인자가 있으면 그 값을,
    없으면(다변량 조합 때문에 알람이 된 것이므로) 고정 문구를 반환한다.
    "경고선 초과 없음"은 오류가 아니라 정상 동작이다.
    """
    exceedances = _exceedances_for_row(row, warning_lines)
    if not exceedances:
        return NO_EXCEEDANCE_REASON
    exceedances.sort(key=lambda e: e.overage_ratio, reverse=True)
    top = exceedances[0]
    text = f"{top.feature} = {top.value:.1f} (경고선 {top.line_value:.1f} {top.direction})"
    if len(exceedances) > 1:
        text += f" {DEFAULT_REASON_MORE_COUNT_LABEL.format(n=len(exceedances) - 1)}"
    return text
