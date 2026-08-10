"""Tests for src/analysis/alarm_gbdt.py -- GBDT 부트스트랩 앙상블 알람 판정
(spec: 알람 판정 GBDT 전환 §A). Synthetic frames only so these always run
in CI regardless of whether data/raw/*.CSV is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.alarm_gbdt import (
    ALARM_SHARE_MAX_PCT,
    ALARM_SHARE_MIN_PCT,
    AUC_GATE,
    CONFORMAL_TARGET_COVERAGE,
    GRADE_STEP_PP,
    MARGIN_MAX_PP,
    STEP_GRID,
    classify_margin,
    classify_wafer,
    classify_wafer_by_rank,
    compute_aggregate_conformal_q,
    compute_holdout_predictions,
    compute_step_auc_profile,
    cross_validate_auc,
    cross_validate_transfer_auc,
    fit_bootstrap_ensemble,
    fit_bootstrap_models,
    gate_for_step,
    predict_with_bootstrap_models,
    prepare_feature_matrix,
    score_wafers,
    step_of,
)


def _synthetic_df(n: int = 400, n_lots: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r1 = rng.normal(50, 10, n)
    r2 = rng.normal(20, 5, n)
    noise = rng.normal(0, 2, n)
    y = 90 - 0.3 * np.abs(r1 - 50) + 0.1 * r2 + noise
    lot_ids = [f"LOT{i % n_lots:03d}" for i in range(n)]
    return pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"W{i}" for i in range(n)],
            "Lot_ID": lot_ids,
            "Step1_R1": r1,
            "Step2_R1": r2,
            "Y": y,
        }
    )


def test_prepare_feature_matrix_fills_missing_columns_with_nan():
    df = pd.DataFrame({"Step1_R1": [1.0, 2.0, np.nan]})
    x = prepare_feature_matrix(df, ["Step1_R1", "Step9_R1"])
    assert x["Step9_R1"].isna().all()
    assert x["Step1_R1"].tolist() == [1.0, 2.0, None] or x["Step1_R1"].isna().iloc[2]


def test_step_of_parses_step_prefix_and_ignores_others():
    """지시서 작업 2: "Step12_R3" -> 12, 패턴에 안 맞는 인자는 None."""
    assert step_of("Step12_R3") == 12
    assert step_of("Step1_D1") == 1
    assert step_of("Config1") is None


def test_prepare_feature_matrix_masks_features_past_max_step():
    """지시서 작업 2 핵심: max_step보다 뒤 스텝의 인자는 값이 있어도 전부
    NaN으로 가려야 한다. 스텝 패턴이 없는 인자는 절대 가려지지 않는다."""
    df = pd.DataFrame({"Step1_R1": [1.0], "Step9_R1": [2.0], "Config1": [3.0]})
    x = prepare_feature_matrix(df, ["Step1_R1", "Step9_R1", "Config1"], max_step=5)
    assert x["Step1_R1"].iloc[0] == 1.0
    assert x["Step9_R1"].isna().all()
    assert x["Config1"].iloc[0] == 3.0


def test_prepare_feature_matrix_max_step_none_is_unmasked():
    df = pd.DataFrame({"Step1_R1": [1.0], "Step30_R1": [2.0]})
    x = prepare_feature_matrix(df, ["Step1_R1", "Step30_R1"], max_step=None)
    assert x["Step30_R1"].iloc[0] == 2.0


def test_fit_bootstrap_ensemble_is_deterministic():
    df = _synthetic_df()
    train_df, eval_df = df.iloc[:300], df.iloc[300:]
    features = ["Step1_R1", "Step2_R1"]
    first = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    second = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    np.testing.assert_allclose(first.pred_mean, second.pred_mean)
    np.testing.assert_allclose(first.pred_hi, second.pred_hi)
    assert (first.pred_lo <= first.pred_mean).all()
    assert (first.pred_mean <= first.pred_hi).all()


def test_fit_bootstrap_ensemble_uses_conformal_margin_for_interval():
    """spec §BA-2 핵심: 구간은 부트스트랩 5/95 분위수가 아니라 홀드아웃
    conformal margin(q)으로 낸다 -- pred_hi - pred_mean == q, pred_mean -
    pred_lo == q가 전 wafer에 걸쳐 성립해야 한다(부트스트랩 분위수는
    wafer마다 폭이 다르다)."""
    df = _synthetic_df(n=300, n_lots=40, seed=0)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    holdout = compute_holdout_predictions(train_df, features)
    assert holdout is not None
    assert pred.conformal_q == pytest.approx(holdout.conformal_q)
    np.testing.assert_allclose(pred.pred_hi - pred.pred_mean, pred.conformal_q)
    np.testing.assert_allclose(pred.pred_mean - pred.pred_lo, pred.conformal_q)
    assert pred.coverage_target == pytest.approx(CONFORMAL_TARGET_COVERAGE)


def test_fit_bootstrap_ensemble_reports_actual_coverage_when_eval_has_known_y():
    """spec §BA-4 핵심: eval에 실측 Y가 있으면(예: 내장 test.CSV) 실제
    포함률을 계산해 coverage_actual에 채운다."""
    df = _synthetic_df(n=300, n_lots=40, seed=0)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    assert pred.coverage_actual is not None
    assert 0.0 <= pred.coverage_actual <= 1.0
    expected = float(
        (
            (eval_df["Y"].to_numpy() >= pred.pred_lo) & (eval_df["Y"].to_numpy() <= pred.pred_hi)
        ).mean()
    )
    assert pred.coverage_actual == pytest.approx(expected)


def test_fit_bootstrap_ensemble_exposes_holdout_oof_sample_for_precision_recall():
    """spec §CA-4 핵심: 클라이언트가 슬라이더를 움직일 때마다 정밀도·재현율을
    다시 추정할 수 있도록 홀드아웃 (실제, 예측) 쌍을 함께 낸다. 표본이
    HOLDOUT_OOF_SAMPLE_SIZE보다 적으면 전량, 많으면 그 이하로 층화
    샘플링된다."""
    from src.analysis.alarm_gbdt import HOLDOUT_OOF_SAMPLE_SIZE

    df = _synthetic_df(n=300, n_lots=40, seed=0)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    assert pred.holdout_oof_actual is not None
    assert pred.holdout_oof_pred is not None
    assert len(pred.holdout_oof_actual) == len(pred.holdout_oof_pred)
    assert len(pred.holdout_oof_actual) == min(len(train_df), HOLDOUT_OOF_SAMPLE_SIZE)


def test_fit_bootstrap_ensemble_holdout_oof_sample_is_deterministic():
    df = _synthetic_df(n=300, n_lots=40, seed=0)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    first = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    second = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    np.testing.assert_allclose(first.holdout_oof_actual, second.holdout_oof_actual)
    np.testing.assert_allclose(first.holdout_oof_pred, second.holdout_oof_pred)


def test_classify_margin_maps_sensitivity_to_pp_margin():
    """spec §CA-1: s=0(오경보 최소)이 가장 보수적(margin=MARGIN_MAX_PP),
    s=1(미탐 최소)이 가장 민감(margin=0)해야 한다. %p 절대값이지 σ 배수가
    아니다."""
    assert classify_margin(0.0) == pytest.approx(MARGIN_MAX_PP)
    assert classify_margin(1.0) == pytest.approx(0.0)
    assert classify_margin(0.5) == pytest.approx(MARGIN_MAX_PP / 2)


def test_classify_wafer_uses_pred_mean_not_pred_hi():
    """spec §CA-1 핵심: 알람 판정은 점추정(pred_mean) 기준이다 -- 이전
    방식(신뢰구간 상한 pred_hi)은 conformal 캘리브레이션 이후 구간이
    넓어져 민감도를 끝까지 올려도 오탐이 전혀 나지 않는 문제를 낳았다.
    구간 하한(pred_lo)이 아무리 낮아도(=구간이 넓어도) pred_mean이
    컷보다 낮으면 알람이 나와야 한다."""
    # target=85, sensitivity=0.5 -> margin=2.0, 컷은 심각 81.4/위험
    # 82.2/주의 83.0(GRADE_STEP_PP=0.8 간격 기준).
    grade = classify_wafer(pred_mean=90.0, pred_lo=70.0, target=85.0, sensitivity=0.5)
    assert grade is None  # 점추정 자체가 목표보다 높음 -- 판별불가
    assert classify_wafer(pred_mean=81.0, pred_lo=70.0, target=85.0, sensitivity=0.5) == "심각"


def test_classify_wafer_five_classes_do_not_overlap():
    """spec §CA-1/§CA-3 핵심: 심각 -> 위험 -> 주의 -> 정상 -> 판별불가는
    서로 배타적이다."""
    target, sensitivity = 85.0, 0.5
    margin = classify_margin(sensitivity)
    severe_edge = target - margin - 2 * GRADE_STEP_PP
    danger_edge = target - margin - GRADE_STEP_PP
    caution_edge = target - margin
    assert classify_wafer(pred_mean=severe_edge, pred_lo=severe_edge - 1, target=target, sensitivity=sensitivity) == "심각"
    assert classify_wafer(pred_mean=danger_edge, pred_lo=danger_edge - 1, target=target, sensitivity=sensitivity) == "위험"
    assert classify_wafer(pred_mean=caution_edge, pred_lo=caution_edge - 1, target=target, sensitivity=sensitivity) == "주의"
    assert classify_wafer(pred_mean=target + 5, pred_lo=target + 1, target=target, sensitivity=sensitivity) == "정상"
    assert classify_wafer(pred_mean=target + 5, pred_lo=target - 1, target=target, sensitivity=sensitivity) is None


def test_classify_wafer_normal_still_uses_pred_lo():
    """spec §CA-3 핵심: `정상`만은 판정 기준이 점추정으로 바뀐 뒤에도
    여전히 구간 하한(pred_lo) 기준이다 -- 점추정으로 정상을 선언하면
    실제 미달 wafer를 놓친다."""
    target, sensitivity = 85.0, 0.5
    # pred_mean은 목표를 넘지만(정상처럼 보임) pred_lo가 목표 밑이면
    # 아직 "정상"이라고 단정할 수 없다 -- 미분류여야 한다.
    grade = classify_wafer(pred_mean=90.0, pred_lo=80.0, target=target, sensitivity=sensitivity)
    assert grade is None
    grade_normal = classify_wafer(pred_mean=90.0, pred_lo=86.0, target=target, sensitivity=sensitivity)
    assert grade_normal == "정상"


def test_classify_wafer_gate_failure_suppresses_alarm_tiers_only():
    """spec §B-4 핵심: 신뢰도 게이트 미달이면 심각/위험/주의는 안 나오지만
    정상/판별불가는 그대로 계산된다."""
    target, sensitivity = 85.0, 0.5
    # 심각 조건을 만족하는 wafer라도 gate_passed=False면 알람이 아니다.
    grade = classify_wafer(pred_mean=50.0, pred_lo=40.0, target=target, sensitivity=sensitivity, gate_passed=False)
    assert grade is None
    grade_normal = classify_wafer(pred_mean=90.0, pred_lo=86.0, target=target, sensitivity=sensitivity, gate_passed=False)
    assert grade_normal == "정상"


def test_higher_sensitivity_never_produces_fewer_alarms():
    """spec §CA-4 핵심: 민감도를 올리면 margin이 줄어(느슨해져) 알람이
    단조 비감소해야 한다 -- 이게 "실제 트레이드오프"의 최소 조건이다."""
    df = _synthetic_df(n=300, n_lots=40, seed=7)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    counts = []
    for sensitivity in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        scored = score_wafers(eval_df, pred, target=85.0, sensitivity=sensitivity)
        counts.append(sum(1 for s in scored if s.grade in ("심각", "위험", "주의")))
    assert counts == sorted(counts)


def test_score_wafers_does_not_fix_a_count_and_sums_to_all_wafers():
    """spec §B-1 핵심: 알람 개수를 고정하지 않는다. 다섯 분류(None 포함)의
    합은 언제나 평가 wafer 수와 같다."""
    df = _synthetic_df(n=200, seed=1)
    train_df, eval_df = df.iloc[:150], df.iloc[150:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    scored = score_wafers(eval_df, pred, target=85.0, sensitivity=0.5)
    assert len(scored) == len(eval_df)
    assert all(s.grade in ("심각", "위험", "주의", "정상", None) for s in scored)
    assert all(0.0 <= s.risk_percentile <= 100.0 for s in scored)


def test_score_wafers_measured_ids_only_affects_the_measured_field():
    """spec §BC-1 핵심: measured_ids는 더 이상 판정 게이트가 아니다 --
    measured=False로 표시된 wafer도 구간 기준으로 정상 등급이 매겨져야
    한다(measured_ids=set()로 아무도 계측되지 않았다고 해도, 등급은
    measured_ids=None일 때와 완전히 같아야 한다)."""
    df = _synthetic_df(n=100, seed=2)
    train_df, eval_df = df.iloc[:60], df.iloc[60:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    scored_all_unmeasured = score_wafers(eval_df, pred, target=200.0, sensitivity=1.0, measured_ids=set())
    scored_ungated = score_wafers(eval_df, pred, target=200.0, sensitivity=1.0, measured_ids=None)
    assert all(not s.measured for s in scored_all_unmeasured)
    assert all(s.measured for s in scored_ungated)
    assert [s.grade for s in scored_all_unmeasured] == [s.grade for s in scored_ungated]
    # target=200(전 구간을 벗어난 목표)이므로 최소 한 wafer는 등급이
    # 매겨져야 게이트가 실제로 풀렸다는 걸 증명한다 -- 전부 None이면
    # (여전히 게이트가 걸려 있어도) 우연히 통과한 assert가 된다.
    assert any(s.grade is not None for s in scored_all_unmeasured)


def test_classify_wafer_by_rank_gate_failure_returns_none():
    all_preds = np.linspace(70, 100, 100)
    assert classify_wafer_by_rank(70.0, all_preds, sensitivity=0.5, gate_passed=False) is None


def test_classify_wafer_by_rank_lowest_wafer_is_always_severe_at_max_sensitivity():
    """지시서 작업 1 핵심: 순위 최하위(가장 낮은 예측)는 민감도가 낮아도
    (표본이 충분히 크면 하위 0.3% 안에 든다) 심각으로 잡혀야 한다. 표본이
    1,000개라 최하위의 백분위는 1/1000=0.1%로, sensitivity=0.0(base=1%)의
    심각 컷(0.3%)보다도 낮다."""
    all_preds = np.linspace(70, 100, 1000)
    lowest = all_preds[0]
    assert classify_wafer_by_rank(lowest, all_preds, sensitivity=0.0) == "심각"
    assert classify_wafer_by_rank(lowest, all_preds, sensitivity=1.0) == "심각"


def test_classify_wafer_by_rank_highest_wafer_is_never_an_alarm():
    all_preds = np.linspace(70, 100, 100)
    highest = all_preds[-1]
    assert classify_wafer_by_rank(highest, all_preds, sensitivity=1.0) is None


def test_classify_wafer_by_rank_three_tiers_split_the_share_30_60_100():
    """지시서 작업 1: sensitivity=1.0 -> base=ALARM_SHARE_MAX_PCT(10%).
    하위 3%(30% of 10)는 심각, 3~6%는 위험, 6~10%는 주의여야 한다."""
    all_preds = np.linspace(0, 999, 1000)  # percentile(i) == i/999*100 근사
    base = ALARM_SHARE_MAX_PCT
    severe_edge = np.percentile(all_preds, base * 0.3)
    danger_edge = np.percentile(all_preds, base * 0.6)
    caution_edge = np.percentile(all_preds, base)
    beyond_edge = np.percentile(all_preds, base * 1.5)
    assert classify_wafer_by_rank(severe_edge, all_preds, sensitivity=1.0) == "심각"
    assert classify_wafer_by_rank(danger_edge, all_preds, sensitivity=1.0) == "위험"
    assert classify_wafer_by_rank(caution_edge, all_preds, sensitivity=1.0) == "주의"
    assert classify_wafer_by_rank(beyond_edge, all_preds, sensitivity=1.0) is None


def test_classify_wafer_by_rank_sensitivity_widens_alarm_share():
    """s=0 -> 하위 1%만, s=1 -> 하위 10%까지 알람 후보가 된다."""
    all_preds = np.linspace(0, 999, 1000)
    just_above_1pct = np.percentile(all_preds, ALARM_SHARE_MIN_PCT + 0.5)
    assert classify_wafer_by_rank(just_above_1pct, all_preds, sensitivity=0.0) is None
    assert classify_wafer_by_rank(just_above_1pct, all_preds, sensitivity=1.0) is not None


def test_classify_wafer_by_rank_cap_at_caution_downgrades():
    all_preds = np.linspace(0, 999, 1000)
    lowest = all_preds[0]
    assert classify_wafer_by_rank(lowest, all_preds, sensitivity=1.0, cap_at_caution=True) == "주의"


def test_score_wafers_default_mode_is_rank_based():
    """지시서 작업 1 핵심: 개발자 수정지시서 §0 배경 -- 부트스트랩 앙상블
    예측은 실제 편차 대비 33~56%로 압축된다. 인자와 거의 무관하게 목표
    근처로 좁게 몰린 Y(압축 상황을 직접 흉내)를 만들면, 절대 %p
    임계값(classify_wafer)은 그 좁은 스프레드가 margin(3.2%p)보다
    작아 알람을 사실상 못 내는 반면, 순위 방식(기본값)은 스프레드
    크기와 무관하게 sensitivity가 정한 비율만큼은 항상 알람을 낸다."""
    rng = np.random.default_rng(20)
    n, n_lots = 400, 40
    r1 = rng.normal(50, 10, n)
    lot_ids = [f"LOT{i % n_lots:03d}" for i in range(n)]
    y = 88.0 + rng.normal(0, 0.3, n)  # r1과 거의 무관 + 목표 근처로 좁게 압축
    df = pd.DataFrame(
        {"Lot_Wafer_ID": [f"W{i}" for i in range(n)], "Lot_ID": lot_ids, "Step1_R1": r1, "Y": y}
    )
    train_df, eval_df = df.iloc[:300], df.iloc[300:]
    pred = fit_bootstrap_ensemble(train_df, eval_df, ["Step1_R1"], n_boot=5)
    target = float(np.median(pred.pred_mean))
    absolute_scored = score_wafers(eval_df, pred, target=target, sensitivity=0.2, mode="absolute")
    rank_scored = score_wafers(eval_df, pred, target=target, sensitivity=0.2, mode="rank")
    absolute_alarms = sum(1 for s in absolute_scored if s.grade in ("심각", "위험", "주의"))
    rank_alarms = sum(1 for s in rank_scored if s.grade in ("심각", "위험", "주의"))
    assert rank_alarms > absolute_alarms


def test_score_wafers_rank_mode_normal_still_uses_absolute_target():
    """지시서 작업 1 제약: 순위 방식이라도 `정상`은 여전히 pred_lo >=
    target(절대 기준)으로 판정해야 한다. 순위 기반 알람(심각/위험/주의)은
    target과 무관하게 항상 하위 몇 %에 매겨지므로(설계상 의도 -- 배치
    안 상대 순위일 뿐 목표 자체를 보지 않는다) 알람으로 잡히지 않은
    wafer만 놓고 검증한다."""
    df = _synthetic_df(n=300, n_lots=40, seed=12)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    low_target = float(pred.pred_lo.min()) - 1.0  # 전원이 확실히 목표를 넘는 목표
    scored = score_wafers(eval_df, pred, target=low_target, sensitivity=0.2, mode="rank")
    non_alarm = [s for s in scored if s.grade not in ("심각", "위험", "주의")]
    assert non_alarm  # 순위 하위 몇 %를 제외한 나머지가 존재해야 의미 있는 검증이다
    assert all(s.grade == "정상" for s in non_alarm)


def test_score_wafers_rank_mode_alarm_count_falls_within_sensitivity_bounds():
    """개발자 수정지시서 검증 기준: 민감도를 0~1로 올릴 때 알람 수가
    단조 증가해야 하고, 대략 ALARM_SHARE_MIN_PCT~MAX_PCT 비율 안에
    들어야 한다."""
    df = _synthetic_df(n=2000, n_lots=100, seed=13)
    train_df, eval_df = df.iloc[:1500], df.iloc[500:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    target = float(pred.pred_mean.max()) + 0.01  # 절대 기준으로는 전원 미달(압축 상황 흉내)
    for sensitivity in (0.0, 0.2, 1.0):
        scored = score_wafers(eval_df, pred, target=target, sensitivity=sensitivity, mode="rank")
        alarm_pct = 100.0 * sum(1 for s in scored if s.grade in ("심각", "위험", "주의")) / len(scored)
        base = ALARM_SHARE_MIN_PCT + sensitivity * (ALARM_SHARE_MAX_PCT - ALARM_SHARE_MIN_PCT)
        assert alarm_pct == pytest.approx(base, abs=1.0)


def test_compute_holdout_predictions_covers_full_train_out_of_fold():
    df = _synthetic_df(n=300, n_lots=60, seed=3)
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    assert len(holdout.actual_y) == len(holdout.pred_point) == len(df)
    assert holdout.residual_std >= 0
    assert holdout.n_holdout == len(df)


def test_compute_holdout_predictions_returns_none_when_too_few_lots():
    df = _synthetic_df(n=50, n_lots=2, seed=2)
    features = ["Step1_R1", "Step2_R1"]
    assert compute_holdout_predictions(df, features, n_splits=5) is None


def test_compute_holdout_predictions_conformal_q_is_nonnegative_quantile_of_abs_residual():
    """spec §BA-1 핵심: q는 |실제 - OOF 예측|의 목표 포함률 분위수다 --
    ±1.645*residual_std 근사(정규성 가정)가 아니라 분포 가정 없는
    분위수여야 한다."""
    df = _synthetic_df(n=300, n_lots=60, seed=3)
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    assert holdout.coverage == pytest.approx(CONFORMAL_TARGET_COVERAGE)
    residuals = np.abs(holdout.actual_y - holdout.pred_point)
    assert holdout.conformal_q == pytest.approx(float(np.percentile(residuals, CONFORMAL_TARGET_COVERAGE * 100.0)))
    assert holdout.conformal_q >= 0


def test_compute_holdout_predictions_higher_coverage_yields_wider_margin():
    """spec §BA-3 핵심: 목표 포함률을 올리면(보수적으로) q가 커져야
    한다(더 넓은 구간 -> 알람 감소 방향)."""
    df = _synthetic_df(n=300, n_lots=60, seed=3)
    features = ["Step1_R1", "Step2_R1"]
    q_90 = compute_holdout_predictions(df, features, n_splits=5, coverage=0.90).conformal_q
    q_95 = compute_holdout_predictions(df, features, n_splits=5, coverage=0.95).conformal_q
    assert q_95 >= q_90


def test_cross_validate_auc_returns_none_when_too_few_lots():
    df = _synthetic_df(n=50, n_lots=2, seed=2)  # fewer distinct lots than n_splits
    result = cross_validate_auc(df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert result is None


def test_cross_validate_auc_ranks_better_than_chance_on_clear_signal():
    df = _synthetic_df(n=600, n_lots=60, seed=3)
    aucs = cross_validate_auc(df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert aucs is not None
    assert len(aucs) <= 5
    assert all(0.0 <= a <= 1.0 for a in aucs)
    # Step1_R1 is a strong, clean signal for Y by construction -- expect
    # meaningfully-better-than-random ranking on average.
    assert float(np.mean(aucs)) > 0.6


def test_cross_validate_transfer_auc_stays_high_when_eval_shares_train_distribution():
    """알람 신뢰도 게이트 §A-1 -- train과 같은 생성 과정을 따르는 eval이면
    (분포가 같은 정상 조합) 전이 AUC도 self-CV와 비슷하게 높아야 한다."""
    df = _synthetic_df(n=600, n_lots=60, seed=3)
    train_df, eval_df = df.iloc[:400], df.iloc[400:]
    aucs = cross_validate_transfer_auc(train_df, eval_df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert aucs is not None
    assert len(aucs) == 5
    assert float(np.percentile(aucs, 5)) > 0.6


def test_cross_validate_transfer_auc_collapses_when_eval_distribution_shifts():
    """알람 신뢰도 게이트 §A-1 핵심: eval의 Y가 train의 인자와 무관한
    분포로 바뀌면(전형적인 "문제" 데이터셋) 전이 AUC가 무작위 수준으로
    떨어져야 한다 -- 이게 바로 게이트가 잡아야 하는 상황이다."""
    train_df = _synthetic_df(n=600, n_lots=60, seed=3)
    rng = np.random.default_rng(99)
    n = 300
    shifted_eval = pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"SW{i}" for i in range(n)],
            "Lot_ID": [f"SLOT{i % 30:03d}" for i in range(n)],
            "Step1_R1": rng.normal(50, 10, n),
            "Step2_R1": rng.normal(20, 5, n),
            # Y is pure noise, unrelated to the features -- no model can
            # transfer discrimination onto this distribution.
            "Y": rng.normal(60, 15, n),
        }
    )
    aucs = cross_validate_transfer_auc(train_df, shifted_eval, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert aucs is not None
    auc_lo = float(np.percentile(aucs, 5))
    assert auc_lo < 0.65  # falls under the AUC_GATE threshold


def test_cross_validate_transfer_auc_returns_none_when_too_few_lots():
    train_df = _synthetic_df(n=50, n_lots=2, seed=2)
    eval_df = _synthetic_df(n=50, n_lots=10, seed=4)
    result = cross_validate_transfer_auc(train_df, eval_df, ["Step1_R1", "Step2_R1"], n_splits=5)
    assert result is None


def test_compute_aggregate_conformal_q_is_much_narrower_than_wafer_q():
    """spec GA-1 핵심: 웨이퍼 conformal_q를 1,000장 평균에 그대로 쓰면
    30배 가까이 과대평가한다. 랏 블록 부트스트랩으로 낸 집계 여유
    (conformal_q_agg)는 웨이퍼 여유보다 훨씬 좁아야 한다."""
    df = _synthetic_df(n=800, n_lots=80, seed=5)
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    q_agg = compute_aggregate_conformal_q(holdout, n_lots=40, n_rounds=500)
    assert q_agg is not None
    assert q_agg > 0
    assert q_agg < holdout.conformal_q / 2


def test_compute_aggregate_conformal_q_is_not_naive_sqrt_n_division():
    """spec GA-1 핵심: q/sqrt(n) 같은 단순 나눗셈을 쓰지 않는다 -- 잔차에
    랏 내 상관이 있으면(같은 랏 wafer가 같은 방향으로 함께 틀림) 그
    상관을 무시하는 naive 나눗셈보다 여유가 넓어야 한다(과소평가
    방지). `_synthetic_df`의 잔차는 랏 간 독립이라 이 테스트는 랏
    단위 공통 오프셋 노이즈를 직접 주입해 상관을 만든다."""
    n, n_lots = 800, 80
    rng = np.random.default_rng(5)
    r1 = rng.normal(50, 10, n)
    r2 = rng.normal(20, 5, n)
    lot_ids = [f"LOT{i % n_lots:03d}" for i in range(n)]
    lot_offset = {lot: rng.normal(0, 3) for lot in set(lot_ids)}  # 랏 단위 공통 편향
    offsets = np.array([lot_offset[lot] for lot in lot_ids])
    noise = rng.normal(0, 1, n)
    y = 90 - 0.3 * np.abs(r1 - 50) + 0.1 * r2 + offsets + noise
    df = pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"W{i}" for i in range(n)],
            "Lot_ID": lot_ids,
            "Step1_R1": r1,
            "Step2_R1": r2,
            "Y": y,
        }
    )
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    n_wafers_per_round = len(holdout.actual_y) // 2  # 대략 절반 랏을 뽑을 때 웨이퍼 수 근사
    naive = holdout.conformal_q / np.sqrt(max(n_wafers_per_round, 1))
    q_agg = compute_aggregate_conformal_q(holdout, n_lots=40, n_rounds=500)
    assert q_agg is not None
    assert q_agg > naive


def test_compute_aggregate_conformal_q_is_deterministic():
    df = _synthetic_df(n=600, n_lots=60, seed=6)
    features = ["Step1_R1", "Step2_R1"]
    holdout = compute_holdout_predictions(df, features, n_splits=5)
    assert holdout is not None
    first = compute_aggregate_conformal_q(holdout, n_lots=30, n_rounds=300)
    second = compute_aggregate_conformal_q(holdout, n_lots=30, n_rounds=300)
    assert first == pytest.approx(second)


def test_fit_bootstrap_ensemble_exposes_aggregate_interval():
    """spec GA-1/GA-2 핵심: fit_bootstrap_ensemble이 웨이퍼 구간(pred_lo/hi,
    conformal_q)과 별도로 집계 구간(pred_agg_lo/hi, conformal_q_agg)을
    함께 낸다 -- 집계 여유가 웨이퍼 여유보다 좁아야 한다."""
    df = _synthetic_df(n=600, n_lots=60, seed=7)
    train_df, eval_df = df.iloc[:400], df.iloc[400:]
    features = ["Step1_R1", "Step2_R1"]
    pred = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    assert pred.conformal_q_agg is not None
    assert pred.conformal_q_agg < pred.conformal_q
    assert pred.pred_agg_lo == pytest.approx(pred.pred_agg_mean - pred.conformal_q_agg)
    assert pred.pred_agg_hi == pytest.approx(pred.pred_agg_mean + pred.conformal_q_agg)
    assert pred.pred_agg_mean == pytest.approx(float(pred.pred_mean.mean()))


def test_fit_bootstrap_models_predict_matches_fit_bootstrap_ensemble():
    """지시서 작업 2: fit_bootstrap_models + predict_with_bootstrap_models로
    나눠도(캐싱을 위한 분리) max_step=None이면 기존 fit_bootstrap_ensemble
    한 번 호출과 완전히 같은 결과가 나와야 한다(회귀 테스트)."""
    df = _synthetic_df(n=300, n_lots=40, seed=0)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    combined = fit_bootstrap_ensemble(train_df, eval_df, features, n_boot=5)
    bundle = fit_bootstrap_models(train_df, features, n_boot=5)
    split = predict_with_bootstrap_models(bundle, eval_df)
    np.testing.assert_allclose(combined.pred_mean, split.pred_mean)
    np.testing.assert_allclose(combined.pred_lo, split.pred_lo)
    np.testing.assert_allclose(combined.pred_hi, split.pred_hi)
    assert combined.conformal_q == pytest.approx(split.conformal_q)


def test_predict_with_bootstrap_models_reuses_models_across_max_step():
    """지시서 작업 2 핵심: max_step을 바꿔도 같은 bundle(이미 적합된
    모델)을 재사용할 뿐 재학습하지 않는다 -- 서로 다른 max_step으로 두
    번 예측해도 bundle.models 객체는 그대로다."""
    df = _synthetic_df(n=300, n_lots=40, seed=0)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    bundle = fit_bootstrap_models(train_df, features, n_boot=5)
    models_before = list(bundle.models)
    pred_full = predict_with_bootstrap_models(bundle, eval_df, max_step=None)
    pred_masked = predict_with_bootstrap_models(bundle, eval_df, max_step=1)
    assert bundle.models == models_before  # 같은 객체 그대로 -- 재학습 없음
    assert not np.allclose(pred_full.pred_mean, pred_masked.pred_mean)


def test_classify_wafer_cap_at_caution_downgrades_severe_and_danger_only():
    """지시서 작업 3 핵심: cap_at_caution=True면 심각/위험도 주의로
    낮아진다 -- 주의/정상/판별불가는 영향받지 않는다."""
    target, sensitivity = 85.0, 0.5
    margin = classify_margin(sensitivity)
    severe_edge = target - margin - 2 * GRADE_STEP_PP
    danger_edge = target - margin - GRADE_STEP_PP
    caution_edge = target - margin
    assert classify_wafer(
        pred_mean=severe_edge, pred_lo=severe_edge - 1, target=target, sensitivity=sensitivity, cap_at_caution=True
    ) == "주의"
    assert classify_wafer(
        pred_mean=danger_edge, pred_lo=danger_edge - 1, target=target, sensitivity=sensitivity, cap_at_caution=True
    ) == "주의"
    assert classify_wafer(
        pred_mean=caution_edge, pred_lo=caution_edge - 1, target=target, sensitivity=sensitivity, cap_at_caution=True
    ) == "주의"
    assert classify_wafer(
        pred_mean=target + 5, pred_lo=target + 1, target=target, sensitivity=sensitivity, cap_at_caution=True
    ) == "정상"


def test_compute_step_auc_profile_returns_none_when_too_few_lots():
    df = _synthetic_df(n=50, n_lots=2, seed=2)
    assert compute_step_auc_profile(df, ["Step1_R1", "Step2_R1"], group_col="Lot_ID") is None


def test_compute_step_auc_profile_covers_full_step_grid_on_clear_signal():
    """지시서 작업 3: 신호가 뚜렷하고 스텝 정보가 늘어날수록(Step2가 뒤
    스텝이라고 가정) AUC 격자를 전부 낼 수 있어야 한다."""
    df = _synthetic_df(n=600, n_lots=60, seed=3)
    profile = compute_step_auc_profile(df, ["Step1_R1", "Step2_R1"], group_col="Lot_ID")
    assert profile is not None
    assert set(profile) <= set(STEP_GRID)
    assert all(0.0 <= v <= 1.0 for v in profile.values())


def test_compute_step_auc_profile_max_step_1_masks_step2_feature():
    """max_step=1이면 Step2_R1은 전부 NaN으로 가려져야 하므로, 그 격자점의
    AUC는 Step1_R1만으로 낸 결과와 같아야 한다."""
    df = _synthetic_df(n=600, n_lots=60, seed=3)
    profile_masked = compute_step_auc_profile(df, ["Step1_R1", "Step2_R1"], group_col="Lot_ID", step_grid=[1])
    profile_direct = compute_step_auc_profile(df, ["Step1_R1"], group_col="Lot_ID", step_grid=[1])
    assert profile_masked is not None and profile_direct is not None
    assert profile_masked[1] == pytest.approx(profile_direct[1])


def test_gate_for_step_returns_none_when_profile_is_none():
    assert gate_for_step(None, 10) == (None, None)


def test_gate_for_step_snaps_to_nearest_grid_point_no_interpolation():
    """지시서 작업 3 핵심: 격자에 없는 max_step은 가장 가까운 격자점을
    쓴다 -- 보간하지 않는다."""
    profile = {1: 0.5, 5: 0.6, 10: 0.7, 30: 0.9}
    assert gate_for_step(profile, 12)[1] == pytest.approx(0.7)  # |12-10|=2 < |12-30|=18
    assert gate_for_step(profile, 8)[1] == pytest.approx(0.7)  # |8-10|=2 < |8-5|=3
    assert gate_for_step(profile, 3)[1] == pytest.approx(0.5)  # |3-1|=2 < |3-5|=2 -> ties keep first match


def test_gate_for_step_none_max_step_uses_largest_grid_point():
    profile = {1: 0.5, 5: 0.6, 30: 0.9}
    passed, auc = gate_for_step(profile, None)
    assert auc == pytest.approx(0.9)
    assert passed is True


def test_gate_for_step_passed_flag_matches_auc_gate_threshold():
    profile = {10: AUC_GATE - 0.01, 20: AUC_GATE + 0.01}
    assert gate_for_step(profile, 10) == (False, pytest.approx(AUC_GATE - 0.01))
    assert gate_for_step(profile, 20) == (True, pytest.approx(AUC_GATE + 0.01))


def test_fit_bootstrap_models_step_profile_is_opt_in():
    """지시서 작업 3: compute_step_profile은 기본 False다 -- 격자 7개 x
    5-fold GBDT 적합이 추가로 붙는 무거운 연산이라, 이를 쓰지 않는
    호출부(테스트 등)까지 매번 그 비용을 치르게 하지 않는다."""
    df = _synthetic_df(n=300, n_lots=40, seed=0)
    bundle_default = fit_bootstrap_models(df.iloc[:240], ["Step1_R1", "Step2_R1"], n_boot=3)
    assert bundle_default.step_auc_profile is None
    bundle_opt_in = fit_bootstrap_models(
        df.iloc[:240], ["Step1_R1", "Step2_R1"], n_boot=3, compute_step_profile=True
    )
    assert bundle_opt_in.step_auc_profile is not None
    assert set(bundle_opt_in.step_auc_profile) <= set(STEP_GRID)


def test_predict_with_bootstrap_models_max_step_masks_only_later_steps():
    """max_step이 첫 스텝(Step1)까지만 허용하면 Step2 인자는 전부 NaN으로
    가려진 채 예측된다 -- Step2 인자만으로 eval을 예측한 것과 같아야
    한다(둘 다 같은 마스킹을 거치므로)."""
    df = _synthetic_df(n=300, n_lots=40, seed=0)
    train_df, eval_df = df.iloc[:240], df.iloc[240:]
    features = ["Step1_R1", "Step2_R1"]
    bundle = fit_bootstrap_models(train_df, features, n_boot=5)
    pred_step1 = predict_with_bootstrap_models(bundle, eval_df, max_step=1)
    eval_masked = eval_df.copy()
    eval_masked["Step2_R1"] = np.nan
    pred_manual = predict_with_bootstrap_models(bundle, eval_masked, max_step=None)
    np.testing.assert_allclose(pred_step1.pred_mean, pred_manual.pred_mean)
