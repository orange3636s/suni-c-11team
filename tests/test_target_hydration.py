from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis import target_hydration
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import score_all_factors


class _FakeTargetModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.feature_names_in_ = np.asarray(["Step1_R1", "Step1_R1_miss"])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        time.sleep(0.005)
        values = pd.to_numeric(features["Step1_R1"], errors="coerce").fillna(0).to_numpy()
        return np.full(len(features), self.value, dtype=float) + values * 0.001


class _FakeBundle:
    def __init__(self) -> None:
        self.models = {target: _FakeTargetModel(index + 1.0) for index, target in enumerate(target_hydration.FAIL_RATE_TARGETS)}

    def model_for_target(self, target: str) -> _FakeTargetModel:
        return self.models[target]


@dataclass
class _FakeLoaded:
    model_id: str = "approved-model-1"

    def __post_init__(self) -> None:
        self.model = _FakeBundle()
        self.metadata = {
            "bundle_type": "screening_pareto_pipeline",
            "pipeline_version": "pipeline-v1",
            "available_targets": list(target_hydration.FAIL_RATE_TARGETS),
            "target_metrics": {
                target: {"feature": "Step1_R1", "relation_shape": "monotonic_increasing"}
                for target in target_hydration.FAIL_RATE_TARGETS
            },
        }


def _patch_model(monkeypatch) -> None:
    _patch_model_values(monkeypatch, [1.0, 2.0, 3.0, 4.0, 5.0])


def _patch_model_values(monkeypatch, values: list[float]) -> None:
    loaded = _FakeLoaded()
    loaded.model.models = {
        target: _FakeTargetModel(value)
        for target, value in zip(target_hydration.FAIL_RATE_TARGETS, values, strict=True)
    }
    monkeypatch.setattr(
        target_hydration,
        "_active_model",
        lambda _store, _model_dir: (loaded, {"active_model_id": loaded.model_id, "pipeline_version": "pipeline-v1"}),
    )
    target_hydration.invalidate_target_hydration_cache()


def _all_missing_1000x97() -> pd.DataFrame:
    rows = 1_000
    data: dict[str, object] = {
        "Lot_Wafer_ID": [f"L{index // 25:03d}_W{index:04d}" for index in range(rows)],
        "Lot_ID": [f"L{index // 25:03d}" for index in range(rows)],
        "Wafer_Slot": [(index % 25) + 1 for index in range(rows)],
    }
    # 3 IDs + 88 process columns + 6 target columns = 97 columns.
    for index in range(1, 89):
        data[f"Step{index}_R1"] = np.linspace(0.0, 1.0, rows)
    for target in target_hydration.ALL_TARGETS:
        data[target] = np.nan
    frame = pd.DataFrame(data)
    assert frame.shape == (1_000, 97)
    return frame


def test_all_missing_1000x97_is_hydrated_and_warm_cache_is_faster(monkeypatch) -> None:
    _patch_model(monkeypatch)
    raw = _all_missing_1000x97()
    before = raw.copy(deep=True)

    cold_started = time.perf_counter()
    cold = target_hydration.hydrate_targets(
        raw, dataset_id="all-missing", dataset_version="v1", store=object(), model_dir="unused"
    )
    cold_seconds = time.perf_counter() - cold_started

    warm_started = time.perf_counter()
    warm = target_hydration.hydrate_targets(
        raw, dataset_id="all-missing", dataset_version="v1", store=object(), model_dir="unused"
    )
    warm_seconds = time.perf_counter() - warm_started

    pdt.assert_frame_equal(raw, before)
    assert cold.dataframe[list(target_hydration.ALL_TARGETS)].notna().all().all()
    assert np.allclose(
        cold.dataframe["Y"],
        100.0 - cold.dataframe[list(target_hydration.FAIL_RATE_TARGETS)].sum(axis=1),
    )
    assert cold.provenance.predicted_target_cells == 5_000
    assert cold.provenance.predicted_rows == 1_000
    assert cold.provenance.cache_hit is False
    assert warm.provenance.cache_hit is True
    assert warm_seconds < cold_seconds
    assert cold_seconds < 2.0
    assert warm_seconds < 2.0
    schema = parse_schema(cold.dataframe)
    assert schema.target_cols == list(target_hydration.FAIL_RATE_TARGETS)
    assert len(score_all_factors(cold.dataframe, schema, "Y1")) == 88


def test_complete_measured_targets_are_unchanged_without_loading_a_model(monkeypatch) -> None:
    raw = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["A", "B"],
            "Step1_R1": [1.0, 2.0],
            "Y": [85.0, 80.0],
            "Y1": [1.0, 2.0],
            "Y2": [2.0, 3.0],
            "Y3": [3.0, 4.0],
            "Y4": [4.0, 5.0],
            "Y5": [5.0, 6.0],
        }
    )
    monkeypatch.setattr(
        target_hydration,
        "_active_model",
        lambda *_args: (_ for _ in ()).throw(AssertionError("complete targets must not load a model")),
    )
    target_hydration.invalidate_target_hydration_cache()

    result = target_hydration.hydrate_targets(
        raw, dataset_id="complete", dataset_version="v1", store=object(), model_dir="unused"
    )

    pdt.assert_frame_equal(result.dataframe, raw)
    assert result.provenance.predicted_target_cells == 0
    assert result.provenance.model_id is None


def test_missing_y_only_is_derived_without_loading_a_model(monkeypatch) -> None:
    raw = pd.DataFrame(
        {
            "Step1_R1": [1.0],
            "Y": [np.nan],
            "Y1": [1.0],
            "Y2": [2.0],
            "Y3": [3.0],
            "Y4": [4.0],
            "Y5": [5.0],
        }
    )
    monkeypatch.setattr(
        target_hydration,
        "_active_model",
        lambda *_args: (_ for _ in ()).throw(AssertionError("derived Y must not load a model")),
    )
    target_hydration.invalidate_target_hydration_cache()

    result = target_hydration.hydrate_targets(
        raw, dataset_id="derive-y", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert result.dataframe.loc[0, "Y"] == 85.0
    assert result.provenance.derived_y_rows == 1


def test_zero_is_measured_but_strings_and_infinity_are_hydrated(monkeypatch) -> None:
    _patch_model(monkeypatch)
    raw = pd.DataFrame(
        {
            "Step1_R1": [1.0],
            "Y": ["invalid"],
            "Y1": [0.0],
            "Y2": ["not-a-number"],
            "Y3": [np.inf],
            "Y4": [-np.inf],
            "Y5": [5.0],
        }
    )

    result = target_hydration.hydrate_targets(
        raw, dataset_id="invalid-values", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert result.dataframe.loc[0, "Y1"] == 0.0
    assert result.dataframe.loc[0, "Y5"] == 5.0
    assert result.dataframe.loc[0, ["Y2", "Y3", "Y4"]].notna().all()
    assert result.provenance.measured_target_cells == 2
    assert result.provenance.predicted_target_cells == 3


def test_prediction_bounds_and_sum_adjust_only_predicted_components(monkeypatch) -> None:
    _patch_model_values(monkeypatch, [-20.0, 150.0, 40.0, 30.0, 20.0])
    raw = pd.DataFrame(
        {
            "Step1_R1": [0.0],
            "Y": [np.nan],
            "Y1": [70.0],
            "Y2": [np.nan],
            "Y3": [np.nan],
            "Y4": [np.nan],
            "Y5": [np.nan],
        }
    )

    result = target_hydration.hydrate_targets(
        raw, dataset_id="bounded", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert result.dataframe.loc[0, "Y1"] == 70.0
    predicted = result.dataframe.loc[0, ["Y2", "Y3", "Y4", "Y5"]].astype(float)
    assert predicted.between(0.0, 100.0).all()
    assert np.isclose(predicted.sum(), 30.0)
    assert np.isclose(result.dataframe.loc[0, "Y"], 0.0)
    assert result.provenance.warning_counts["predicted_components_rescaled"] == 1


def test_missing_model_returns_actionable_error(monkeypatch) -> None:
    target_hydration.invalidate_target_hydration_cache()
    monkeypatch.setattr(
        target_hydration,
        "_active_model",
        lambda *_args: (_ for _ in ()).throw(target_hydration.TargetHydrationError("승인 모델이 없습니다.")),
    )

    with np.testing.assert_raises_regex(target_hydration.TargetHydrationError, "승인 모델"):
        target_hydration.hydrate_targets(
            pd.DataFrame({"Step1_R1": [1.0]}),
            dataset_id="no-model",
            dataset_version="v1",
            store=object(),
            model_dir="unused",
        )


def test_missing_model_feature_and_extra_step_are_safe_and_reported(monkeypatch) -> None:
    _patch_model(monkeypatch)
    raw = pd.DataFrame({"Lot_Wafer_ID": ["A"], "Step41_R3": [42.0]})

    result = target_hydration.hydrate_targets(
        raw, dataset_id="schema-drift", dataset_version="v1", store=object(), model_dir="unused"
    )

    assert "Step41_R3" in result.dataframe.columns
    assert result.dataframe.loc[0, "Step41_R3"] == 42.0
    assert result.provenance.feature_coverage["missing_features"] == ["Step1_R1"]
    assert result.dataframe[list(target_hydration.ALL_TARGETS)].notna().all().all()


def test_partial_targets_preserve_measured_and_scale_only_predictions(monkeypatch) -> None:
    _patch_model(monkeypatch)
    raw = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["A", "B"],
            "Step1_R1": [1.0, 2.0],
            "Y": [12.0, np.nan],
            "Y1": [90.0, 1.0],
            "Y2": [20.0, np.nan],
            "Y3": [np.nan, 3.0],
            "Y4": [np.nan, 4.0],
            "Y5": [np.nan, 5.0],
        }
    )
    result = target_hydration.hydrate_targets(
        raw, dataset_id="partial", dataset_version="v1", store=object(), model_dir="unused"
    )

    # Invalid observed values are warned about, never silently rewritten.
    assert result.dataframe.loc[0, "Y"] == 12.0
    assert result.dataframe.loc[0, "Y1"] == 90.0
    assert result.dataframe.loc[0, "Y2"] == 20.0
    assert result.provenance.warning_counts["observed_fail_rate_sum_over_100"] == 1
    # On the second row, only Y2 was predicted and Y is derived afterwards.
    assert result.dataframe.loc[1, "Y1"] == 1.0
    assert result.dataframe.loc[1, "Y3"] == 3.0
    assert result.dataframe.loc[1, "Y"] == 100.0 - result.dataframe.loc[1, list(target_hydration.FAIL_RATE_TARGETS)].sum()
    assert result.provenance.mixed_rows == 2


def test_cache_invalidation_is_scoped_by_dataset(monkeypatch) -> None:
    _patch_model(monkeypatch)
    raw = _all_missing_1000x97().iloc[:3]
    for dataset_id in ("one", "two"):
        target_hydration.hydrate_targets(
            raw, dataset_id=dataset_id, dataset_version="v1", store=object(), model_dir="unused"
        )
    assert target_hydration.invalidate_target_hydration_cache("one") == 1
    assert target_hydration.target_hydration_cache_info()["size"] == 1


def test_upload_target_states_are_distinct() -> None:
    base = pd.DataFrame({"Step2_R1": [1.0, 2.0]})
    assert target_hydration.inspect_target_status(base).state == "missing_columns"
    all_missing = base.assign(Y=np.nan, Y1=np.nan, Y2=np.nan, Y3=np.nan, Y4=np.nan, Y5=np.nan)
    assert target_hydration.inspect_target_status(all_missing).state == "all_missing"
    partial = all_missing.copy()
    partial.loc[0, "Y1"] = 1.0
    assert target_hydration.inspect_target_status(partial).state == "partial"
    complete = pd.DataFrame({target: [1.0, 2.0] for target in target_hydration.ALL_TARGETS})
    assert target_hydration.inspect_target_status(complete).state == "complete"
