from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from functools import lru_cache, wraps
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Response, status

from api.routes.datasets import get_dataset_registry
from api.schemas.analysis import (
    AnalysisContextResponse,
    AnalysisReportResponse,
    CategoricalScatterResponse,
    ControlRangeListResponse,
    HeatmapResponse,
    ModelPerformanceResponse,
    ParetoRankingResponse,
    PreprocessingComparisonResponse,
    ScreeningScatterResponse,
    YieldPredictionResponse,
)
from api.settings import APP_VERSION, settings
from src.analysis import alarm_gbdt, preprocessing_compare
from src.analysis.reliability_score import FAIL_RATE_TARGETS
from src.analysis.control_range import (
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.report import build_analysis_report, build_chat_context
from src.analysis.rounding import round_floats
from src.analysis.scatter import build_categorical_data, build_scatter_data
from src.analysis.screening.fmea import build_fmea_table
from src.analysis.screening.heatmap import HeatmapData, build_heatmap
from src.analysis.screening.schema import parse_schema
from src.analysis.target_hydration import (
    TARGET_HYDRATION_VERSION,
    HydratedTargets,
    TargetHydrationError,
    hydrate_targets,
)
from src.analysis.screening.selector import (
    DEFAULT_MIN_N_CATEGORICAL,
    DEFAULT_MIN_N_D,
    DEFAULT_MIN_N_R,
    PARETO_TOP_N,
    ParetoFactor,
    effective_confidence_tier,
)
from src.analysis.screening.selector import _ranked_rows_with_contribution as _ranked_rows
from src.analysis.screening.selector import _row_to_factor
from src.ml.inference import get_latest_model_metadata
from src.runtime.datasets import DatasetNotFoundError
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])

# The report always evaluates alarms/eval_result against the bundled "test"
# set -- the root-cause tab (the report button's only caller) has a single
# dataset selector (train only), the same convention /api/alarms already
# defaults to.
REPORT_EVAL_DATASET_ID = "test"


def _dataframe_or_404(dataset_id: str):
    registry = get_dataset_registry()
    try:
        return registry.get_dataframe(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.") from exc


def _hydrated_targets_or_409(dataset_id: str) -> HydratedTargets:
    registry = get_dataset_registry()
    try:
        dataframe = registry.get_dataframe(dataset_id)
        dataset_version = registry.content_version(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.") from exc
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    try:
        return hydrate_targets(
            dataframe,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            store=store,
            model_dir=settings.model_dir,
        )
    except TargetHydrationError as exc:
        # RA-B4: 이 choke point(모든 분석/모니터링/알림 라우트가 승인
        # 모델을 필요로 할 때 거치는 유일한 지점)에서 409를 올리기 직전에
        # 복구를 트리거한다 -- 레지스트리 포인터는 있는데 파일이 없는
        # 경우(RA 근본 원인) 재학습이 자동으로 시작된다. 지연 import는
        # 순환참조 회피용이다: `api.main`이 이 모듈(`analysis_router`)을
        # 기동 시 임포트하므로, 모듈 최상단에서 되돌아 임포트하면 순환이
        # 생긴다 -- 이 함수가 처음 호출되는 시점에는 이미 `api.main`
        # 로딩이 끝나 있으므로 지연 import는 안전하다(`_train_bootstrap_
        # champion`이 이미 같은 패턴을 쓰고 있다).
        from api.main import ensure_usable_champion

        recovering = not ensure_usable_champion(store)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "recovering": recovering},
        ) from exc


SCHEMA_CACHE_DATASETS = 2


@lru_cache(maxsize=SCHEMA_CACHE_DATASETS)
def _cached_schema(dataset_id: str, dataset_version: str) -> Any:
    """UB-2 (perf): `parse_schema` re-walks every column of the dataframe
    (regex match x ~90 columns) on every call -- individually cheap, but
    `get_screening_scatter` alone calls it fresh 50 times in one 5-target x
    10-factor analysis run (target_hydration.py's own cache fixed the much
    bigger cost -- model load/predict -- but left this uncached).

    `dataset_version` MUST stay in the cache key even though the function
    body never reads it -- this project has hit the "cache key missing the
    version" bug three times before (uploading a replacement dataset under
    the same id must produce a fresh schema, not a stale one). Keying by
    version alone is enough: content changes bump `dataset_version`, which
    is itself a fresh key, so old entries just age out via `maxsize`
    LRU eviction -- no explicit invalidation call is needed for this cache
    (see the UB-3 invalidation-audit note in the perf commit message).
    """
    del dataset_version  # part of the cache key only, see docstring above
    df = get_dataset_registry().get_dataframe(dataset_id)
    return parse_schema(df)


def _single_flight(fn):
    """B-4: `lru_cache`만으로는 같은 키가 아직 캐시되지 않은 상태에서
    동시에 두 스레드가 들어오면(예: 일일 발송 스케줄러 잡과 사용자의
    `/alarms` 요청이 겹치는 경우) 둘 다 캐시 미스를 보고 같은 무거운
    GBDT를 이중으로 적합시킨다. 키별 락으로 감싸 두 번째 호출은 첫 번째가
    끝날 때까지 기다렸다가 이미 채워진 캐시를 그대로 받아가게 한다 --
    서로 다른 키는 잠그지 않으므로 별개 데이터셋 요청끼리는 막지 않는다.
    데코레이터를 적용받는 함수는 반드시 위치 인자만 받아야 한다(캐시
    키가 그 튜플 그대로다).
    """
    locks: dict[tuple, threading.Lock] = {}
    locks_guard = threading.Lock()

    @wraps(fn)
    def wrapper(*args):
        with locks_guard:
            lock = locks.setdefault(args, threading.Lock())
        with lock:
            return fn(*args)

    return wrapper


def _find_cached_factor(
    dataset: str,
    df: pd.DataFrame,
    target: str,
    feature: str,
    provenance: Any | None = None,
) -> ParetoFactor | None:
    """B-2: `find_factor`가 직접 부르는 `_ranked_rows_with_contribution`
    (88인자 ANOVA+FDR 전수 스코어링)을 다시 돌리지 않고, Pareto/heatmap이
    이미 채워 뒀을 `_cached_ranked_rows(dataset, target)`를 재사용한다.
    같은 (dataset, target) 기본 파라미터(fdr_alpha=0.05, min_n=100/20)로
    스코어링하므로 결과는 find_factor와 동일하다 -- 다만 분석 실행이
    5타깃×10인자 산점도를 동시에 요청하면 이 캐시 덕분에 실제 스코어링은
    타깃당 한 번만 일어난다."""
    rows = _ranked_rows_for_provenance(dataset, target, provenance) if provenance is not None else _cached_ranked_rows(dataset, target)
    row = next((r for r in rows if r["feature"] == feature), None)
    return _row_to_factor(df, target, row) if row is not None else None


@router.get("/screening/scatter", response_model=ScreeningScatterResponse)
def get_screening_scatter(dataset: str, target: str, feature: str) -> dict[str, Any]:
    # UA-1 (perf measurement): per-phase timing for a scatter request -- a
    # single "분석 실행" fires 5 targets x 10 factors of these, so a slow
    # phase here is a slow phase x50. See UB's commit message for the
    # before/after numbers this instrumentation produced.
    t_start = time.perf_counter()
    hydrated = _hydrated_targets_or_409(dataset)
    df = hydrated.dataframe
    dataset_version = hydrated.provenance.dataset_version
    t_hydrate = time.perf_counter()

    schema = _cached_schema(dataset, dataset_version)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃 결과가 없습니다.")
    # Resolves any of the 88 factors regardless of Pareto rank -- a heatmap
    # cell click can open a scatter for a factor outside the top 5.
    factor = _find_cached_factor(dataset, df, target, feature, hydrated.provenance)
    if factor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}' 인자를 찾을 수 없습니다.")
    if factor.kind == "Config":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{feature}'은(는) Config(범주형) 인자입니다. /api/screening/scatter/categorical을 사용하세요.",
        )
    t_factor = time.perf_counter()

    data = build_scatter_data(df, df, factor, dataset_id=dataset)
    t_build = time.perf_counter()
    logger.debug(
        "scatter %s/%s hydrate=%.3f factor=%.3f build=%.3f total=%.3f",
        target, feature,
        t_hydrate - t_start, t_factor - t_hydrate, t_build - t_factor,
        t_build - t_start,
    )
    # Only the bulky per-point/per-bin arrays are rounded -- they're what
    # actually drives payload size (108KB for 1,470 points); scalar stats
    # (p_value/q_value/eps2) keep full precision since a very small
    # p-value (e.g. 7.7e-66) rounded to 4 decimals would collapse to a
    # meaningless 0.0 in the "p<0.001" exponential display.
    return {
        "points": round_floats(data.points),
        "reference_lines": round_floats(data.reference_lines),
        "normal_range": round_floats(data.normal_range),
        "bins": round_floats(data.bins),
        "optimal_center": data.optimal_center,
        "optimal_center_dropped_reason": data.optimal_center_dropped_reason,
        "eps2": data.eps2,
        "spearman_r": data.spearman_r,
        "p_value": data.p_value,
        "q_value": data.q_value,
        "significant": data.significant,
        "confidence_tier": data.confidence_tier,
        "under_sampled": data.under_sampled,
        "relation_shape": data.relation_shape,
        "n": data.n,
        "axis": data.axis,
        "methods": round_floats(data.methods),
        "target_provenance": hydrated.provenance.as_dict(),
    }


@router.get("/screening/scatter/categorical", response_model=CategoricalScatterResponse)
def get_screening_scatter_categorical(dataset: str, target: str, feature: str) -> dict[str, Any]:
    """Per-category box-plot data for a Config factor. Config never gets a
    numeric normal-range (a category has no "range"), so this is a
    separate response shape from the numeric scatter endpoint above
    rather than an overloaded variant of it.
    """
    hydrated = _hydrated_targets_or_409(dataset)
    df = hydrated.dataframe
    schema = _cached_schema(dataset, hydrated.provenance.dataset_version)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃을 찾을 수 없습니다.")
    if feature not in schema.config_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}'은(는) Config 인자가 아닙니다.")

    factor = _find_cached_factor(dataset, df, target, feature, hydrated.provenance)
    if factor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}' 인자를 찾을 수 없습니다.")

    data = build_categorical_data(df, factor)
    return {
        "groups": round_floats([vars(group) for group in data.groups]),
        "eps2": data.eps2,
        "p_value": data.p_value,
        "q_value": data.q_value,
        "significant": data.significant,
        "confidence_tier": data.confidence_tier,
        "n": data.n,
        "axis": data.axis,
        "target_provenance": hydrated.provenance.as_dict(),
    }


HEATMAP_CACHE_DATASETS = 2  # keep the most recent 2 datasets' worth, evict older ones

@lru_cache(maxsize=HEATMAP_CACHE_DATASETS)
def _cached_heatmap(
    dataset_id: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    hydration_version: str,
) -> HeatmapData:
    # 데이터셋 내용은 dataset_id가 존재하는 한 불변이므로(업로드는 매번 새
    # uuid, 번들 파일은 정적) 이 캐시는 최근 2개 데이터셋만 LRU로 유지해
    # 무한정 커지지 않는다. NG-1: categorical 보기를 제거해 kind 분기와
    # 그만큼의 캐시 슬롯도 함께 없앴다.
    del model_id, model_version, hydration_version
    df = _hydrated_targets_or_409(dataset_id).dataframe
    schema = _cached_schema(dataset_id, dataset_version)
    return build_heatmap(df, schema)


@router.get("/screening/heatmap", response_model=HeatmapResponse)
def get_screening_heatmap(dataset: str = "train") -> dict[str, Any]:
    """The correlation heatmap used identically by both the training tab
    and the root-cause tab: R+D x Y1~Y5, always both ε² and rho (TC-4).
    NG-1: the Config x Y1~Y5 categorical view was removed -- 600 tests
    found 0 FDR-significant Config factors, and the per-Config treemap tab
    already covers that ground better.
    """
    t0 = time.perf_counter()
    hydrated = _hydrated_targets_or_409(dataset)
    provenance = hydrated.provenance
    hits_before = _cached_heatmap.cache_info().hits
    heatmap = _cached_heatmap(
        dataset,
        provenance.dataset_version,
        provenance.model_id or "measured-only",
        provenance.model_version or "none",
        TARGET_HYDRATION_VERSION,
    )
    cached = _cached_heatmap.cache_info().hits > hits_before
    logger.info(
        "screening_heatmap %.1fms (cached=%s, dataset=%s)",
        (time.perf_counter() - t0) * 1000, cached, dataset,
    )
    return {
        "dataset_id": dataset,
        "metric": "eps2",
        "kind": "numeric",
        "features": heatmap.features,
        "targets": heatmap.targets,
        "values": heatmap.values,
        "rho": heatmap.rho,
        "n": heatmap.n,
        "q": heatmap.q,
        "significant": heatmap.significant,
        "tier": heatmap.tier,
        "gate_excluded": heatmap.gate_excluded,
        "scale": {"min": heatmap.scale["min"], "max": heatmap.scale["max"]},
        "excluded_configs": heatmap.excluded_configs,
        "target_provenance": provenance.as_dict(),
    }


PARETO_CACHE_DATASETS = 2  # keep the most recent 2 datasets' worth, evict older ones
PARETO_TARGETS = 5  # Y1..Y5

@lru_cache(maxsize=PARETO_CACHE_DATASETS * PARETO_TARGETS)
def _cached_ranked_rows_versioned(
    dataset_id: str,
    target: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    hydration_version: str,
) -> tuple[dict, ...]:
    # Cached per (dataset_id, target), capped at the 2 most-recent
    # datasets (LRU-evicted): the training tab and the root-cause tab both
    # request the same (dataset, target) pair and must see byte-identical
    # results -- this cache is exactly what guarantees that, not just a
    # performance nicety. Dataset content is immutable once a dataset_id
    # exists (see the heatmap cache's docstring for why that's safe).
    del dataset_version, model_id, model_version, hydration_version
    df = _hydrated_targets_or_409(dataset_id).dataframe
    schema = parse_schema(df)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃을 찾을 수 없습니다.")
    return tuple(_ranked_rows(df, schema, target, 0.05, DEFAULT_MIN_N_R, DEFAULT_MIN_N_D, DEFAULT_MIN_N_CATEGORICAL))


def _ranked_rows_for_provenance(dataset_id: str, target: str, provenance: Any) -> tuple[dict, ...]:
    return _cached_ranked_rows_versioned(
        dataset_id,
        target,
        provenance.dataset_version,
        provenance.model_id or "measured-only",
        provenance.model_version or "none",
        TARGET_HYDRATION_VERSION,
    )


def _cached_ranked_rows(dataset_id: str, target: str) -> tuple[dict, ...]:
    hydrated = _hydrated_targets_or_409(dataset_id)
    return _ranked_rows_for_provenance(dataset_id, target, hydrated.provenance)


def _pareto_payload(dataset_id: str, target: str, top_n: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    hydrated = _hydrated_targets_or_409(dataset_id)
    hits_before = _cached_ranked_rows_versioned.cache_info().hits
    ranked = list(_ranked_rows_for_provenance(dataset_id, target, hydrated.provenance))
    cached = _cached_ranked_rows_versioned.cache_info().hits > hits_before
    logger.info(
        "screening_pareto %.1fms (cached=%s, dataset=%s, target=%s)",
        (time.perf_counter() - t0) * 1000, cached, dataset_id, target,
    )
    top = ranked[:top_n]
    items = [
        {
            "feature": row["feature"],
            "kind": row["kind"],
            "step": row["step"],
            "eps2": row["eps2"],
            "p_value": row["p_value"],
            "q_value": row["q_value"],
            "significant": row["significant"],
            "confidence_tier": effective_confidence_tier(row["eps2"], row["p_value"], under_sampled=row.get("under_sampled", False)),
            "n_observed": row["n_observed"],
            "contribution_pct": row["contribution_pct"],
            "cumulative_pct": row["cumulative_pct"],
            # QA-2: 배제 대신 "표본 부족" 배지로 표시 -- 하한(30) 이상이지만
            # 종류별 정상 판정 임계 미만인 경우 True.
            "under_sampled": row.get("under_sampled", False),
        }
        for row in top
    ]
    n80 = next((index + 1 for index, row in enumerate(ranked) if row["cumulative_pct"] >= 80.0), None)
    # 차트 표시 규칙(spec §B)의 0개-타깃 안내 문구가 쓰는 전체 풀 집계치 --
    # 화면에 노출되는 top-5만으로는 "58건 중 FDR 통과 0건"을 계산할 수 없어
    # 여기서 전체 ranked 풀을 기준으로 함께 내려보낸다.
    fdr_pass_count = sum(1 for row in ranked if row["significant"])
    effect_size_pass_count = sum(
        1 for row in ranked if effective_confidence_tier(row["eps2"], row["p_value"], under_sampled=row.get("under_sampled", False)) != "reference"
    )
    max_eps2 = max((row["eps2"] for row in ranked), default=None)
    return {
        "dataset_id": dataset_id,
        "target": target,
        "total_factor_count": len(ranked),
        "n80": n80,
        "fdr_pass_count": fdr_pass_count,
        "effect_size_pass_count": effect_size_pass_count,
        "max_eps2": max_eps2,
        "items": items,
        "analyzable_target_samples": int(pd.to_numeric(hydrated.dataframe[target], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum()),
        "model_available": bool(hydrated.provenance.model_id) or not hydrated.provenance.uses_predictions,
        "factor_measurement_insufficient": len(ranked) == 0,
        "target_provenance": hydrated.provenance.as_dict(),
    }


def _fmea_payload(dataset_id: str, targets: tuple[str, ...]) -> dict[str, Any]:
    """모니터링 홈 블록③(데이터 한계) -- MNAR 계측 편향 + 분산 분해.

    MA-3: 이 함수는 원래 FMEA 분석표(행별 권장구간·편차) 전체를
    반환했지만, 그 표를 그리던 FmeaTable/ActionBlock이 모니터링 홈
    재설계로 삭제되면서 행 데이터(`items`)의 유일한 소비처가 없어졌다
    (블록①은 이제 `_action_priority_payload`가 train.CSV 기준으로 따로
    낸다 -- 이 함수의 eval 기준 표와는 다른 산출물이다). 남은 소비처
    (`DataLimitationDiagnostics`)는 `mnar_rate_report`/
    `variance_decomposition`만 읽으므로 그 둘만 내려보낸다 -- payload가
    작아진다(행마다 17개 필드였던 `items`가 통째로 빠진다).
    `build_fmea_table`은 여전히 내부에서 호출한다 -- MNAR 리포트가 그
    표의 (타깃, 인자) 쌍을 그대로 재사용하기 때문이다.
    """
    hydrated = _hydrated_targets_or_409(dataset_id)
    df = hydrated.dataframe
    schema = parse_schema(df)
    usable_targets = [t for t in targets if t in schema.target_cols]
    rows_by_target = {
        t: list(_ranked_rows_for_provenance(dataset_id, t, hydrated.provenance)) for t in usable_targets
    }
    table = build_fmea_table(df, rows_by_target, usable_targets, dataset_id=dataset_id)

    from src.analysis.data_limitations import build_mnar_rate_report, compute_variance_decomposition

    mnar_report = build_mnar_rate_report(df, [(f.target, f.feature) for f in table.items])
    variance_decomposition = compute_variance_decomposition(df)

    return round_floats(
        {
            "dataset_id": dataset_id,
            "mnar_rate_report": [
                {
                    "target": r.target,
                    "feature": r.feature,
                    "overall_rate_pct": r.overall_rate_pct,
                    "worst_decile_rate_pct": r.worst_decile_rate_pct,
                    "ratio": r.ratio,
                }
                for r in mnar_report
            ],
            "variance_decomposition": (
                {
                    "lot_count": variance_decomposition.lot_count,
                    "wafers_per_lot": variance_decomposition.wafers_per_lot,
                    "between_lot_pct": variance_decomposition.between_lot_pct,
                    "within_lot_pct": variance_decomposition.within_lot_pct,
                    "no_effect_expected_pct": variance_decomposition.no_effect_expected_pct,
                    "icc": variance_decomposition.icc,
                }
                if variance_decomposition is not None
                else None
            ),
            "target_provenance": hydrated.provenance.as_dict(),
        }
    )


def _action_priority_payload(train_dataset_id: str) -> dict[str, Any]:
    """모니터링 홈 블록①(조치 우선순위)·블록②(조치 가능 범위) -- 항상
    train.CSV 기준(작업 지시서 MB-6)이라 eval 데이터셋 선택과 무관하게
    안정적이다. 랭킹(`_ranked_rows_for_provenance`)과 권장구간 계산
    (`compare_methods`의 자체 캐시)이 이미 프로세스 전역으로 캐시되어
    있어(YF/ZD 성능 작업 참고) train 데이터셋이 바뀌지 않는 한 사실상
    즉시 반환된다.
    """
    from src.analysis.action_priority import build_action_priority_table

    hydrated = _hydrated_targets_or_409(train_dataset_id)
    df = hydrated.dataframe
    schema = parse_schema(df)
    usable_targets = [t for t in FAIL_RATE_TARGETS if t in schema.target_cols]
    rows_by_target = {
        t: list(_ranked_rows_for_provenance(train_dataset_id, t, hydrated.provenance)) for t in usable_targets
    }
    table = build_action_priority_table(df, rows_by_target)

    return round_floats(
        {
            "dataset_id": train_dataset_id,
            "total_wafers": table.total_wafers,
            "estimated_additional_action_wafers": table.estimated_additional_action_wafers,
            "no_qualifying_factor": [
                {"target": n.target, "max_contribution_pct": n.max_contribution_pct} for n in table.no_qualifying_factor
            ],
            "rows": [
                {
                    "target": r.target,
                    "feature": r.feature,
                    "relation_shape": r.relation_shape,
                    "factor_value": r.factor_value,
                    "range_lo": r.range_lo,
                    "range_hi": r.range_hi,
                    "measured_count": r.measured_count,
                    "out_of_range_count": r.out_of_range_count,
                    "total_wafers": r.total_wafers,
                    "recovery_width_pp": r.recovery_width_pp,
                    "share_pct": r.share_pct,
                    "expected_recovery_pp": r.expected_recovery_pp,
                    "contribution_pct": r.contribution_pct,
                    "dimmed": r.dimmed,
                    "dim_reason": r.dim_reason,
                }
                for r in table.rows
            ],
            "target_provenance": hydrated.provenance.as_dict(),
        }
    )


@router.get("/screening/pareto", response_model=ParetoRankingResponse)
def get_screening_pareto(dataset: str = "train", target: str = "Y1", top_n: int = PARETO_TOP_N) -> dict[str, Any]:
    """The top-eps2 Pareto ranking for one target across the full
    R+D+Config pool -- the shared source for both the training tab's
    screening table and the root-cause tab's Pareto chart, which show
    different counts (5 vs 10) and so must each pass their own `top_n`
    explicitly rather than rely on this default drifting under them.
    Not gated by FDR significance: every returned row is included
    regardless of p-value, tiered by confidence_tier instead of
    filtered out. `n80` reports the rank (across the FULL pool, not
    just `top_n`) at which cumulative contribution first reaches 80%,
    so the caller can render "80%에 도달하지 못했습니다 -- N개 더 필요"
    without a second request.
    """
    return _pareto_payload(dataset, target, top_n)


def _alarm_factors(
    train_df,
    schema,
    train_dataset_id: str,
    provenance: Any | None = None,
) -> tuple[list[ParetoFactor], list[str]]:
    """Per-target alarm-eligible factors: every BH-FDR-significant factor
    (see select_fdr_significant_factors's docstring -- deliberately kept
    unchanged so the golden 19-alarm-wafer count doesn't move). Screen
    display no longer gates on significance, but alarm generation still
    does.

    B-3: reuses `_cached_ranked_rows(train_dataset_id, target)` instead of
    `select_fdr_significant_factors` (which reruns the full 88-factor
    ANOVA+FDR scoring every call with identical default parameters) --
    same rows, same significance flags, just not recomputed from scratch.
    """
    factors: list[ParetoFactor] = []
    no_alarm_factor: list[str] = []
    for target in schema.target_cols:
        rows = (
            _ranked_rows_for_provenance(train_dataset_id, target, provenance)
            if provenance is not None
            else _cached_ranked_rows(train_dataset_id, target)
        )
        target_factors = [_row_to_factor(train_df, target, row) for row in rows if row["significant"]]
        if not target_factors:
            no_alarm_factor.append(target)
            continue
        factors.extend(target_factors)
    return factors, no_alarm_factor


def _control_range_dict(control_range) -> dict[str, Any]:
    data = vars(control_range).copy()
    data["reference_lines"] = [vars(line) for line in control_range.reference_lines]
    return round_floats(data)


@router.get("/control-ranges", response_model=ControlRangeListResponse)
def get_control_ranges(dataset: str = "train") -> dict[str, Any]:
    hydrated = _hydrated_targets_or_409(dataset)
    train_df = hydrated.dataframe
    schema = parse_schema(train_df)
    factors, no_significant = _alarm_factors(train_df, schema, dataset, hydrated.provenance)
    items = [_control_range_dict(compute_control_range(train_df, factor)) for factor in factors]
    return {
        "train_dataset_id": dataset,
        "items": items,
        "no_significant_factor_targets": no_significant,
    }


@router.get("/alarms/history")
def get_alarm_snapshot_history(limit: int = 20) -> dict[str, Any]:
    """Immutable alert snapshots; later promotions never rewrite old rows."""
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    return {"items": store.list_alert_snapshots(limit)}


@router.get("/alerts/ranking", response_model=YieldPredictionResponse)
def get_alerts_ranking(train: str = "train", eval: str = "test") -> dict[str, Any]:
    """VA~VD: 수율 예측 순위 목록 -- y(=100 − Σ Y1~Y5, RC-3 실측 우선
    규칙으로 채운 뒤 재계산) 오름차순 전체(신뢰도==0 웨이퍼는 제외)를
    내려보낸다. 상위 10/전체 보기·검색·정렬 9종은 프런트가 이 목록
    위에서 수행한다(VB그룹) -- `top_n`으로 서버가 미리 자르면 검색이
    상위 10 밖의 웨이퍼를 찾지 못한다(VB-4: "검색 중에는 상위 10 제한을
    해제한다").
    """
    from src.analysis.yield_prediction import build_yield_prediction_table

    train_df = _dataframe_or_404(train)
    eval_view = _hydrated_targets_or_409(eval)
    eval_df = _dataframe_or_404(eval)

    table = build_yield_prediction_table(
        train_df,
        eval_df,
        eval_view.dataframe,
        dataset_id=eval,
        train_dataset_id=train,
        train_dataset_version=get_dataset_registry().content_version(train),
    )
    return {
        "train_dataset_id": train,
        "eval_dataset_id": eval,
        "total_wafers": table.total_wafers,
        "candidates": [
            {
                "lot_wafer_id": c.lot_wafer_id,
                "lot_id": c.lot_id,
                "y": c.y,
                "y_components": c.y_components,
                "cells": c.cells,
                "core_factors": {
                    target: {
                        "feature": cell.feature,
                        "contribution_pct": cell.contribution_pct,
                        "rank_used": cell.rank_used,
                        "factor_value": cell.factor_value,
                    }
                    for target, cell in c.core_factors.items()
                },
                "reliability": {
                    "count": c.reliability.count,
                    "measured": [{"target": t, "feature": f} for t, f in c.reliability.measured],
                    "unmeasured": [{"target": t, "feature": f} for t, f in c.reliability.unmeasured],
                },
                "recommendation": {
                    "text": c.recommendation.text,
                    "adjustable_targets": list(c.recommendation.adjustable_targets),
                    "measurement_gap_targets": list(c.recommendation.measurement_gap_targets),
                },
            }
            for c in table.candidates
        ],
        "unmeasured_wafer_ids": table.unmeasured_wafer_ids,
        "unmeasured_count": len(table.unmeasured_wafer_ids),
        "fallback_summary": {
            "rank_counts": {str(rank): count for rank, count in table.fallback_summary.rank_counts.items()},
            "none_count": table.fallback_summary.none_count,
            "total_combinations": table.fallback_summary.total_combinations,
        },
        "target_provenance": eval_view.provenance.as_dict(),
    }


def _build_report_payload(dataset: str) -> dict[str, Any]:
    """The one function backing both /api/analysis/report (JSON download)
    and /api/analysis/context (SUNI chatbot context) -- same dict, so the
    chatbot never narrates a number the download button wouldn't also show.
    """
    registry = get_dataset_registry()
    train_view = _hydrated_targets_or_409(dataset)
    eval_view = _hydrated_targets_or_409(REPORT_EVAL_DATASET_ID)
    train_meta = registry.get_summary(dataset) or {}
    eval_meta = registry.get_summary(REPORT_EVAL_DATASET_ID) or {}
    report = build_analysis_report(
        train_view.dataframe,
        eval_view.dataframe,
        train_dataset_id=dataset,
        eval_dataset_id=REPORT_EVAL_DATASET_ID,
        train_meta=train_meta,
        eval_meta=eval_meta,
        app_version=APP_VERSION,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    report.setdefault("meta", {})["target_provenance"] = {
        "train": train_view.provenance.as_dict(),
        "eval": eval_view.provenance.as_dict(),
    }
    return report


@router.get("/analysis/report", response_model=AnalysisReportResponse)
def get_analysis_report(dataset: str = "train", *, response: Response) -> dict[str, Any]:
    """Full JSON analysis report -- always denominated by the full
    R+D+Config pool, matching the screen now that the R/D/Config split
    view has been removed entirely (see build_analysis_report's
    docstring for why the factor list and the alarm list use different,
    deliberately non-interchangeable factor sets).
    """
    response.headers["Content-Disposition"] = f'attachment; filename="analysis_report_{dataset}.json"'
    return _build_report_payload(dataset)


PREPROCESSING_COMPARISON_CACHE_DATASETS = 8


@lru_cache(maxsize=PREPROCESSING_COMPARISON_CACHE_DATASETS)
def _cached_preprocessing_comparison(dataset_id: str) -> dict[str, Any] | None:
    """설정 패널 신설 §E: 데이터셋마다 실측한 전처리 A/B/C 비교. 데이터셋당
    한 번만 계산해 캐시한다(§E-6: "탭을 열 때마다 재계산하지 마라") -- LOT
    70/30 홀드아웃 1회 × 3방식이라 4초 안팎이지만, 그 값을 매 탭 전환마다
    다시 치르지 않는다.
    """
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    features = alarm_gbdt.feature_columns(schema)
    registry = get_dataset_registry()
    summary = registry.get_summary(dataset_id)
    dataset_label = summary["original_filename"] if summary else dataset_id

    comparison = preprocessing_compare.compute_preprocessing_comparison(
        df, features, dataset_id=dataset_id, dataset_label=dataset_label
    )
    if comparison is None:
        return None
    return {
        "dataset_id": comparison.dataset_id,
        "dataset_label": comparison.dataset_label,
        "results": [
            {"mode": r.mode, "label": r.label, "r2": r.r2, "adopted": r.adopted} for r in comparison.results
        ],
        "winner": comparison.winner,
        "b_equals_c": comparison.b_equals_c,
        "holdout_note": comparison.holdout_note,
        "winner_note": comparison.winner_note,
    }


@router.get("/training/preprocessing-comparison", response_model=PreprocessingComparisonResponse)
def get_preprocessing_comparison(dataset: str) -> dict[str, Any]:
    result = _cached_preprocessing_comparison(dataset)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="전처리 비교를 계산하기에 데이터가 부족합니다.",
        )
    return round_floats(result)


@router.get("/analysis/context", response_model=AnalysisContextResponse)
def get_analysis_context(dataset: str = "train", *, response: Response) -> dict[str, Any]:
    """The SUNI chatbot's grounding context -- same underlying report as
    /api/analysis/report, but `alarms` is reshaped into `{summary,
    records}` (see build_chat_context's docstring) so the chatbot can
    answer a question about one specific wafer's alarm, not just the
    aggregate counts. No Content-Disposition, and explicitly non-cached
    since the frontend calls this right before /api/chat.
    """
    response.headers["Cache-Control"] = "no-store"
    return build_chat_context(_build_report_payload(dataset))


@router.get("/models/performance", response_model=ModelPerformanceResponse)
def get_model_performance(dataset: str | None = None) -> dict[str, Any]:
    del dataset  # Performance reflects whatever was last trained, not a live recompute.
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    metadata = get_latest_model_metadata(store)
    if metadata is None:
        return {"model_id": None, "trained_at": None, "source_filename": None, "targets": [], "final_yield": None}

    target_metrics = metadata.get("target_metrics") or {}
    targets = [
        {
            "target": target,
            "no_factor_available": bool(detail.get("no_factor_available")),
            "feature": detail.get("feature"),
            "kind": detail.get("kind"),
            "eps2": detail.get("eps2"),
            "contribution_pct": detail.get("contribution_pct"),
            "relation_shape": detail.get("relation_shape"),
            "optimal_center": detail.get("optimal_center"),
            "p_value": detail.get("p_value"),
            "confidence_tier": detail.get("confidence_tier"),
            "r2": detail.get("r2"),
            "rmse": detail.get("rmse"),
            "mae": detail.get("mae"),
            "n": detail.get("n"),
        }
        for target, detail in target_metrics.items()
    ]
    final_test_metrics = (metadata.get("final_y_metrics") or {}).get("test") or {}
    final_yield = (
        {
            "target": "Y",
            "no_factor_available": False,
            "feature": None,
            "kind": None,
            "eps2": None,
            "contribution_pct": None,
            "relation_shape": None,
            "optimal_center": None,
            "p_value": None,
            "confidence_tier": None,
            "r2": final_test_metrics.get("r2"),
            "rmse": final_test_metrics.get("rmse"),
            "mae": final_test_metrics.get("mae"),
            "n": final_test_metrics.get("n"),
        }
        if final_test_metrics
        else None
    )
    feature_columns = metadata.get("feature_columns") or []
    return {
        "model_id": metadata.get("model_id"),
        "trained_at": metadata.get("created_at"),
        "source_filename": metadata.get("source_filename"),
        "targets": targets,
        "final_yield": final_yield,
        "row_count": metadata.get("row_count"),
        "feature_count": len(feature_columns) or None,
    }
