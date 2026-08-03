"""Content-hash cache for Pareto analyses.

Mirrors the spirit of src/runtime's file-backed artifact convention
(deterministic key -> JSON result + a "latest" pointer) without touching the
existing SQLite-backed RuntimeStore, which is owned by the GBDT pipeline.
Same file + same params => no recompute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.analysis.pareto.plots import build_pareto_chart
from src.analysis.pareto.schema import Schema, parse_schema
from src.analysis.pareto.selector import (
    DEFAULT_CUTOFF,
    DEFAULT_FDR_ALPHA,
    TargetParetoResult,
    select_pareto_factors_all_targets,
)

CACHE_ROOT = Path("data/runtime/analyses/pareto")
LATEST_POINTER = CACHE_ROOT / "latest.json"


def _cache_dir(cache_key: str) -> Path:
    return CACHE_ROOT / cache_key


def compute_cache_key(content: bytes, targets: list[str] | None, cutoff: float, fdr_alpha: float) -> str:
    digest = hashlib.sha256()
    digest.update(content)
    digest.update(json.dumps({"targets": targets, "cutoff": cutoff, "fdr_alpha": fdr_alpha}, sort_keys=True).encode())
    return digest.hexdigest()[:32]


def _result_to_dict(result: TargetParetoResult) -> dict:
    return {
        "target": result.target,
        "factors": [asdict(f) for f in result.factors],
        "reference_only": [asdict(f) for f in result.reference_only],
        "excluded_count": result.excluded_count,
        "no_significant_factor": result.no_significant_factor,
        "pareto_chart": build_pareto_chart(result) if result.factors else None,
    }


def analyze_and_cache(
    content: bytes,
    targets: list[str] | None = None,
    cutoff: float = DEFAULT_CUTOFF,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> dict:
    cache_key = compute_cache_key(content, targets, cutoff, fdr_alpha)
    directory = _cache_dir(cache_key)
    results_path = directory / "results.json"

    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        _write_latest_pointer(cache_key)
        return payload

    import io

    df = pd.read_csv(io.BytesIO(content))
    schema = parse_schema(df)
    resolved_targets = targets or schema.target_cols
    results = select_pareto_factors_all_targets(
        df, schema, targets=resolved_targets, cutoff=cutoff, fdr_alpha=fdr_alpha
    )

    payload = {
        "cache_key": cache_key,
        "schema_warnings": [f"파싱하지 못한 컬럼: {c}" for c in schema.unmapped],
        "targets": [_result_to_dict(r) for r in results.values()],
    }

    directory.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_parquet(directory / "source.parquet", index=False)
    (directory / "schema.json").write_text(
        json.dumps(_schema_to_dict(schema), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_latest_pointer(cache_key)
    return payload


def _schema_to_dict(schema: Schema) -> dict:
    return {
        "r_cols": schema.r_cols,
        "d_cols": schema.d_cols,
        "config_cols": schema.config_cols,
        "target_cols": schema.target_cols,
        "id_cols": schema.id_cols,
        "max_step": schema.max_step,
        "steps_present": schema.steps_present,
        "unmapped": schema.unmapped,
    }


def _write_latest_pointer(cache_key: str) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.write_text(json.dumps({"cache_key": cache_key}), encoding="utf-8")


def load_latest_cache_key() -> str | None:
    if not LATEST_POINTER.exists():
        return None
    return json.loads(LATEST_POINTER.read_text(encoding="utf-8")).get("cache_key")


def load_cached_results(cache_key: str) -> dict | None:
    results_path = _cache_dir(cache_key) / "results.json"
    if not results_path.exists():
        return None
    return json.loads(results_path.read_text(encoding="utf-8"))


def load_cached_dataframe(cache_key: str) -> pd.DataFrame | None:
    parquet_path = _cache_dir(cache_key) / "source.parquet"
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def factor_dict_to_object(factor_dict: dict):
    from src.analysis.pareto.selector import ParetoFactor

    return ParetoFactor(**factor_dict)
