"""작업지시 "Config 인자 모델의 타깃 하이드레이션 실패 수정" 회귀 테스트.

근본 원인: 학습(`src/ml/pipeline.py build_features`)은 Config 인자를
`category` dtype으로 유지했지만, 추론(`src/analysis/target_hydration.py
_screening_features`)은 종류 구분 없이 숫자로 강제 변환해 Config
컬럼이 전량 NaN이 됐다 -- `category`로 학습된 LightGBM에 float32를
넘기면서 "categorical_feature do not match" 예외가 났다("승인 모델의
Y3 예측에 실패했습니다").

이 파일은 실제 LGBMRegressor를 소형 합성 데이터(240행)로 학습시켜
정확히 그 경로(학습 시점 dtype -> 저장된 메타데이터 -> 추론 시점
재구성)를 왕복시킨다 -- mock으로 우회하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.analysis import target_hydration
from src.analysis.screening.schema import parse_schema
from src.ml import pipeline as ml_pipeline
from src.ml.feature_builder import trained_categories_from_model

CONFIG_FEATURE = "Step1_Config"
CATEGORIES = ("EQA", "EQB", "EQC")
_MEAN_BY_CATEGORY = {"EQA": 2.0, "EQB": 5.0, "EQC": 8.0}


def _synthetic_config_frame(n: int = 240, *, categories: tuple[str, ...] = CATEGORIES, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = rng.choice(np.asarray(categories), size=n)
    y1 = np.asarray([_MEAN_BY_CATEGORY[c] for c in cats]) + rng.normal(0, 0.05, size=n)
    return pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"L{i // 10:03d}W{i % 10:02d}" for i in range(n)],
            "Lot_ID": [f"L{i // 10:03d}" for i in range(n)],
            CONFIG_FEATURE: pd.Series(cats, dtype="object"),
            "Y1": y1,
        }
    )


def _train_config_model(train_df: pd.DataFrame) -> tuple[LGBMRegressor, object]:
    """실제 학습 경로(`fit_target_pipeline` -> `build_features`)를 그대로
    태운다 -- 합성 데이터는 R/D 인자가 하나도 없으므로 Config 인자가
    Y1의 대표 인자로 뽑힌다."""
    schema = parse_schema(train_df)
    result = ml_pipeline.fit_target_pipeline(train_df, schema, "Y1")
    assert not result.no_factor_available
    assert result.factors[0].kind == "Config"
    assert result.factors[0].feature == CONFIG_FEATURE
    return result.model, result.factors[0]


class _AlwaysMatchModel:
    """Y2~Y5용 자리채우기 -- 이 테스트는 Y1(Config 대표 인자)만 살펴본다."""

    feature_names_in_ = np.asarray([])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features))


def _fake_loaded(model: LGBMRegressor, factor, *, include_categories: bool) -> object:
    details = {
        "feature": factor.feature,
        "kind": factor.kind,
        "relation_shape": factor.relation_shape,
        "optimal_center": factor.optimal_center,
    }
    if include_categories:
        details["categories"] = trained_categories_from_model(model)

    class _Bundle:
        def model_for_target(self, target: str):
            return model if target == "Y1" else _AlwaysMatchModel()

    @dataclass
    class _Loaded:
        model_id: str = "config-model-1"

        def __post_init__(self) -> None:
            self.model = _Bundle()
            self.metadata = {
                "bundle_type": "screening_pareto_pipeline",
                "pipeline_version": "pipeline-v1",
                "available_targets": list(target_hydration.FAIL_RATE_TARGETS),
                "target_metrics": {t: details for t in target_hydration.FAIL_RATE_TARGETS},
            }

    return _Loaded()


def _patch_active_model(monkeypatch: pytest.MonkeyPatch, loaded: object) -> None:
    monkeypatch.setattr(
        target_hydration,
        "_active_model",
        lambda _store, _model_dir: (loaded, {"active_model_id": loaded.model_id, "pipeline_version": "pipeline-v1"}),
    )
    target_hydration.invalidate_target_hydration_cache()


# -- 1. Config 대표 인자 모델 + 타깃 전량 결측 평가 데이터 -----------------


def test_config_primary_factor_model_hydrates_all_missing_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    train_df = _synthetic_config_frame()
    model, factor = _train_config_model(train_df)
    loaded = _fake_loaded(model, factor, include_categories=True)
    _patch_active_model(monkeypatch, loaded)

    eval_df = pd.DataFrame(
        {
            CONFIG_FEATURE: ["EQA", "EQB", "EQC", "EQA"],
            **{target: [np.nan] * 4 for target in target_hydration.ALL_TARGETS},
        }
    )

    result = target_hydration.hydrate_targets(
        eval_df, dataset_id="cfg-all-missing", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert result.dataframe[list(target_hydration.ALL_TARGETS)].notna().all().all()
    # EQA/EQB/EQC 순서대로 낮은->높은 예측이 나와야 한다(학습 시점 평균과 정합).
    predicted = result.dataframe["Y1"].to_numpy()
    assert predicted[0] < predicted[1] < predicted[2]


# -- 2. Config 값 혼합: 정상 + 미등록 범주 + 결측 --------------------------


def test_config_mixed_known_unknown_and_missing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    train_df = _synthetic_config_frame()
    model, factor = _train_config_model(train_df)
    loaded = _fake_loaded(model, factor, include_categories=True)
    _patch_active_model(monkeypatch, loaded)

    eval_df = pd.DataFrame(
        {
            CONFIG_FEATURE: ["EQA", "EQZ-NEVER-SEEN", None, "EQC"],
            **{target: [np.nan] * 4 for target in target_hydration.ALL_TARGETS},
        }
    )

    result = target_hydration.hydrate_targets(
        eval_df, dataset_id="cfg-mixed", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert result.dataframe[list(target_hydration.ALL_TARGETS)].notna().all().all()
    assert np.isfinite(result.dataframe["Y1"].to_numpy()).all()


# -- 3. Config 컬럼 자체가 없는 평가 데이터 --------------------------------


def test_config_column_entirely_missing_from_eval_data(monkeypatch: pytest.MonkeyPatch) -> None:
    train_df = _synthetic_config_frame()
    model, factor = _train_config_model(train_df)
    loaded = _fake_loaded(model, factor, include_categories=True)
    _patch_active_model(monkeypatch, loaded)

    eval_df = pd.DataFrame({target: [np.nan] * 3 for target in target_hydration.ALL_TARGETS})

    result = target_hydration.hydrate_targets(
        eval_df, dataset_id="cfg-column-missing", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert result.dataframe[list(target_hydration.ALL_TARGETS)].notna().all().all()
    assert np.isfinite(result.dataframe["Y1"].to_numpy()).all()


# -- 4. `categories` 메타데이터가 없는 구형 모델 폴백 ----------------------


def test_legacy_model_without_categories_metadata_falls_back_to_booster(monkeypatch: pytest.MonkeyPatch) -> None:
    train_df = _synthetic_config_frame()
    model, factor = _train_config_model(train_df)
    # 구형 모델 흉내 -- target_metrics에 "categories" 키 자체가 없다.
    loaded = _fake_loaded(model, factor, include_categories=False)
    assert "categories" not in loaded.metadata["target_metrics"]["Y1"]
    _patch_active_model(monkeypatch, loaded)

    eval_df = pd.DataFrame(
        {
            CONFIG_FEATURE: ["EQA", "EQB", "EQC"],
            **{target: [np.nan] * 3 for target in target_hydration.ALL_TARGETS},
        }
    )

    result = target_hydration.hydrate_targets(
        eval_df, dataset_id="cfg-legacy", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert result.dataframe[list(target_hydration.ALL_TARGETS)].notna().all().all()


# -- 5. R/D 대표 인자 모델의 예측이 수정 전과 동일 -------------------------


class _DeterministicRModel:
    """수정 전 `_screening_features`(=_finite_numeric -> float32 -> _miss/_dev)와
    수정 후(공용 build_feature_frame)가 R/D 경로에서 만드는 피처가 정말
    같은지 -- 값에 민감한 결정론적 모델로 확인한다."""

    feature_names_in_ = np.asarray(["Step1_R1", "Step1_R1_miss"])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        raw = pd.to_numeric(features["Step1_R1"], errors="coerce").fillna(0.0).to_numpy()
        miss = features["Step1_R1_miss"].to_numpy()
        return raw * 2.0 + miss * 1000.0


def test_rd_primary_factor_predictions_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _DeterministicRModel()
    details = {"feature": "Step1_R1", "kind": "R", "relation_shape": "monotonic_increasing", "optimal_center": None}

    class _Bundle:
        def model_for_target(self, target: str):
            return model if target == "Y1" else _AlwaysMatchModel()

    @dataclass
    class _Loaded:
        model_id: str = "rd-model-1"

        def __post_init__(self) -> None:
            self.model = _Bundle()
            self.metadata = {
                "bundle_type": "screening_pareto_pipeline",
                "pipeline_version": "pipeline-v1",
                "available_targets": list(target_hydration.FAIL_RATE_TARGETS),
                "target_metrics": {t: details for t in target_hydration.FAIL_RATE_TARGETS},
            }

    _patch_active_model(monkeypatch, _Loaded())

    eval_df = pd.DataFrame(
        {
            "Step1_R1": [1.0, np.nan, 3.5, np.inf],
            **{target: [np.nan] * 4 for target in target_hydration.ALL_TARGETS},
        }
    )

    result = target_hydration.hydrate_targets(
        eval_df, dataset_id="rd-unchanged", dataset_version="v1", store=object(), model_dir="unused"
    )

    # 수정 전 로직을 손으로 재현한 기대값: _finite_numeric(inf도 결측 취급) ->
    # float32 -> raw*2 + miss*1000.
    expected_raw = np.array([1.0, np.nan, 3.5, np.nan], dtype=np.float32)
    expected_miss = np.isnan(expected_raw).astype(np.int8)
    expected = np.where(np.isnan(expected_raw), 0.0, expected_raw) * 2.0 + expected_miss * 1000.0
    # `_predict_targets`가 예측을 [0, 100]으로 clip한다(기존 동작, 이
    # 테스트가 바꾸는 대상이 아니다).
    expected = np.clip(expected, 0.0, 100.0)

    assert np.allclose(result.dataframe["Y1"].to_numpy(), expected)


# -- 6. Config feature coverage가 0%가 아님 --------------------------------


def test_config_feature_coverage_is_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    train_df = _synthetic_config_frame()
    model, factor = _train_config_model(train_df)
    loaded = _fake_loaded(model, factor, include_categories=True)
    _patch_active_model(monkeypatch, loaded)

    eval_df = pd.DataFrame(
        {
            CONFIG_FEATURE: ["EQA", "EQB", "EQC", "EQA"],
            **{target: [np.nan] * 4 for target in target_hydration.ALL_TARGETS},
        }
    )

    result = target_hydration.hydrate_targets(
        eval_df, dataset_id="cfg-coverage", dataset_version="v1", store=object(), model_dir="unused"
    )

    coverage = result.provenance.feature_coverage["by_target"]["Y1"]["measured_cell_coverage"]
    assert coverage == pytest.approx(1.0)
