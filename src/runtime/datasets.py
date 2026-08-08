"""Dataset registry: the 2 bundled CSVs (train/test) plus user uploads.

Bundled datasets are read straight from data/bundled/ and are never
deletable. Uploads are validated immediately (blocking errors reject the
upload outright; warnings are informational and still let the file
through) and persisted as a raw CSV file plus a `datasets` row in
RuntimeStore. Deleting an upload removes both.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.analysis.screening.schema import Schema, parse_schema
from src.dataset_normalization import normalize_dataset
from src.runtime.app_state import invalidate_state_for_dataset
from src.runtime.store import RuntimeStore
from src.upload_limits import max_row_count, max_upload_size_bytes, max_upload_size_mb

logger = logging.getLogger(__name__)

LOW_ROW_COUNT_THRESHOLD = 1000
UNSTABLE_ROW_COUNT_THRESHOLD = 2000
MIN_FACTOR_OBSERVATIONS = 100
ID_COLUMN = "Lot_Wafer_ID"
LOT_COLUMN = "Lot_ID"

# Display order is explicit and fixed (spec §1) -- never sorted by name or
# size. train.CSV stays the default selection.
# 지시서 CB: mentorship_dataset_final/v7_killing_event는 구버전 스키마라
# 삭제했다 -- final은 Y = 100 - (Y1+...+Y5) 항등식이 깨져 있고(잔차 최대
# 203), v7_killing_event는 Config 대신 Eq 컬럼을 쓰는 189컬럼 이전
# 형식이라 현재 파이프라인의 스키마 가정과 맞지 않는다.
BUNDLED_DATASET_FILES = {
    "train": "train.CSV",
    "test": "test.CSV",
}


class DatasetValidationError(Exception):
    def __init__(self, blocking_errors: list[str]) -> None:
        super().__init__("; ".join(blocking_errors))
        self.blocking_errors = blocking_errors


class DatasetNotFoundError(Exception):
    pass


class BundledDatasetDeleteError(Exception):
    pass


@dataclass
class DatasetValidation:
    blocking_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)
    low_observation_factors: list[str] = field(default_factory=list)
    schema: Schema | None = None

    @property
    def is_valid(self) -> bool:
        return not self.blocking_errors


def _lot_summary(df: pd.DataFrame) -> tuple[str | None, str | None, int | None]:
    if LOT_COLUMN not in df.columns:
        return None, None, None
    lots = df[LOT_COLUMN].dropna().astype(str)
    if lots.empty:
        return None, None, None
    return str(lots.min()), str(lots.max()), int(lots.nunique())


def parse_uploaded_csv(content: bytes) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(content) > max_upload_size_bytes():
        actual_mb = len(content) / (1024 * 1024)
        raise DatasetValidationError(
            [f"파일이 너무 큽니다 (최대 {max_upload_size_mb()}MB). 현재 {actual_mb:.1f}MB"]
        )
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise DatasetValidationError([f"CSV 파싱에 실패했습니다: {exc}"]) from exc
    if df.shape[1] == 0:
        raise DatasetValidationError(["컬럼이 없는 CSV입니다."])
    return normalize_dataset(df)


def validate_dataset(
    df: pd.DataFrame,
    *,
    baseline_schema: Schema | None = None,
    normalization_report: dict[str, object] | None = None,
) -> DatasetValidation:
    blocking_errors: list[str] = []
    warnings: list[str] = []

    schema = parse_schema(df)
    if not (schema.r_cols or schema.d_cols or schema.config_cols):
        blocking_errors.append(
            "Step{n}_R{m} / Step{n}_D{m} / Step{n}_Config 패턴에 맞는 컬럼이 하나도 없습니다."
        )
    if not schema.target_cols:
        blocking_errors.append("이 파일에는 타깃 열이 없어 원인 분석에 사용할 수 없습니다.")

    if blocking_errors:
        return DatasetValidation(blocking_errors=blocking_errors, schema=schema)

    row_count = len(df)
    if row_count < LOW_ROW_COUNT_THRESHOLD:
        warnings.append("표본이 부족해 인자 선정 결과를 신뢰하기 어렵습니다.")
    elif row_count < UNSTABLE_ROW_COUNT_THRESHOLD:
        warnings.append("인자 선정이 불안정할 수 있습니다.")
    if row_count > max_row_count():
        warnings.append(f"행 수가 많습니다 ({row_count:,}행, 권장 상한 {max_row_count():,}행). 처리 시간과 메모리 사용량이 늘어날 수 있습니다.")

    # Lot_ID may have been derived from Lot_Wafer_ID just now (spec §2-2)
    # rather than having existed in the source file -- either way, if it's
    # missing entirely (couldn't be parsed for a single row) GroupKFold is
    # still impossible and gets the same existing warning. A *partial*
    # parse failure (most rows fine, a handful malformed) is a distinct,
    # narrower warning instead of the blanket "no LOT column" one.
    failed_count = int((normalization_report or {}).get("lot_id_parse_failed_count", 0) or 0)
    if LOT_COLUMN not in df.columns or (row_count > 0 and failed_count >= row_count):
        warnings.append(
            "LOT 열이 없어 GroupKFold를 적용할 수 없습니다. 단순 KFold로 대체하며 "
            "결과가 낙관적으로 나올 수 있습니다."
        )
    elif failed_count > 0:
        warnings.append(f"Lot_Wafer_ID에서 LOT을 파싱하지 못한 행이 {failed_count}개 있습니다. 해당 행은 LOT 미상으로 처리됩니다.")

    if baseline_schema is not None:
        diffs: list[str] = []
        if len(schema.r_cols) != len(baseline_schema.r_cols):
            diffs.append(f"R 컬럼 {len(schema.r_cols)}개 (기준 {len(baseline_schema.r_cols)}개)")
        if len(schema.d_cols) != len(baseline_schema.d_cols):
            diffs.append(f"D 컬럼 {len(schema.d_cols)}개 (기준 {len(baseline_schema.d_cols)}개)")
        if schema.max_step != baseline_schema.max_step:
            diffs.append(f"step 최대 {schema.max_step}개 (기준 {baseline_schema.max_step}개)")
        if diffs:
            warnings.append("내장 train.CSV와 step 구성이 다릅니다: " + ", ".join(diffs))

    low_observation_factors = [
        column
        for column in [*schema.r_cols, *schema.d_cols]
        if df[column].notna().sum() < MIN_FACTOR_OBSERVATIONS
    ]
    if low_observation_factors:
        preview = ", ".join(low_observation_factors[:10])
        suffix = " 외 %d개" % (len(low_observation_factors) - 10) if len(low_observation_factors) > 10 else ""
        warnings.append(f"관측 행이 {MIN_FACTOR_OBSERVATIONS}개 미만인 인자는 스크리닝에서 자동 제외됩니다: {preview}{suffix}")

    return DatasetValidation(
        warnings=warnings,
        unmapped_columns=schema.unmapped,
        low_observation_factors=low_observation_factors,
        schema=schema,
    )


# Both caches below are keyed at module level, not on `self`, because a
# fresh DatasetRegistry is constructed on every request (see
# get_dataset_registry() in api/routes/datasets.py) -- a per-instance dict
# cache would never survive past the request that built it. Bundled files
# are re-read only if their mtime/size changes (handles a redeployed CSV);
# uploaded datasets are immutable for the life of their uuid dataset_id
# (see analysis.py's heatmap-cache docstring for why that's safe), so no
# invalidation key is needed there.
_BUNDLED_CACHE_SIZE = len(BUNDLED_DATASET_FILES)
_UPLOADED_CACHE_SIZE = 8  # small LRU; bounds memory across a long-running process


@lru_cache(maxsize=_BUNDLED_CACHE_SIZE)
def _read_bundled_csv(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size  # part of the cache key only, invalidates on file replacement
    raw = pd.read_csv(path_str)
    normalized, _ = normalize_dataset(raw)
    return normalized


@lru_cache(maxsize=_UPLOADED_CACHE_SIZE)
def _read_uploaded_csv(path_str: str) -> pd.DataFrame:
    raw = pd.read_csv(path_str)
    normalized, _ = normalize_dataset(raw)
    return normalized


class DatasetRegistry:
    def __init__(self, store: RuntimeStore, upload_root: Path, bundled_root: Path) -> None:
        self.store = store
        self.upload_root = Path(upload_root)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.bundled_root = Path(bundled_root)

    def _bundled_path(self, dataset_id: str) -> Path:
        return self.bundled_root / BUNDLED_DATASET_FILES[dataset_id]

    def _load_bundled(self, dataset_id: str) -> pd.DataFrame:
        path = self._bundled_path(dataset_id)
        stat = path.stat()
        return _read_bundled_csv(str(path), stat.st_mtime_ns, stat.st_size)

    def bundled_schema(self, dataset_id: str = "train") -> Schema:
        return parse_schema(self._load_bundled(dataset_id))

    def _bundled_summary(self, dataset_id: str) -> dict[str, Any]:
        df = self._load_bundled(dataset_id)
        lot_min, lot_max, lot_count = _lot_summary(df)
        return {
            "dataset_id": dataset_id,
            "kind": "bundled",
            "original_filename": BUNDLED_DATASET_FILES[dataset_id],
            "uploaded_at": None,
            "row_count": len(df),
            "column_count": df.shape[1],
            "lot_min": lot_min,
            "lot_max": lot_max,
            "lot_count": lot_count,
            "warnings": [],
            "unmapped_columns": [],
            "schema_diff": {},
            "deletable": False,
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        bundled = [self._bundled_summary(dataset_id) for dataset_id in BUNDLED_DATASET_FILES]
        uploaded = [
            {**record, "kind": "uploaded", "deletable": True}
            for record in self.store.list_datasets()
        ]
        return [*bundled, *uploaded]

    def get_summary(self, dataset_id: str) -> dict[str, Any] | None:
        if dataset_id in BUNDLED_DATASET_FILES:
            return self._bundled_summary(dataset_id)
        record = self.store.get_dataset(dataset_id)
        if record is None:
            return None
        return {**record, "kind": "uploaded", "deletable": True}

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id in BUNDLED_DATASET_FILES:
            return self._load_bundled(dataset_id)
        record = self.store.get_dataset(dataset_id)
        if record is None:
            raise DatasetNotFoundError(dataset_id)
        return _read_uploaded_csv(str(self.upload_root / record["stored_path"]))

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        import gc

        try:
            try:
                df, normalization_report = parse_uploaded_csv(content)
            except DatasetValidationError as exc:
                return {
                    "success": False,
                    "dataset_id": None,
                    "blocking_errors": exc.blocking_errors,
                    "warnings": [],
                    "unmapped_columns": [],
                }

            baseline_schema = self.bundled_schema("train")
            validation = validate_dataset(df, baseline_schema=baseline_schema, normalization_report=normalization_report)
            if not validation.is_valid:
                return {
                    "success": False,
                    "dataset_id": None,
                    "blocking_errors": validation.blocking_errors,
                    "warnings": [],
                    "unmapped_columns": [],
                }

            dataset_id = uuid4().hex
            stored_name = f"{dataset_id}.csv"
            (self.upload_root / stored_name).write_bytes(content)
            lot_min, lot_max, lot_count = _lot_summary(df)
            self.store.create_dataset(
                dataset_id=dataset_id,
                original_filename=filename,
                stored_path=stored_name,
                row_count=len(df),
                column_count=df.shape[1],
                lot_min=lot_min,
                lot_max=lot_max,
                lot_count=lot_count,
                warnings=validation.warnings,
                unmapped_columns=validation.unmapped_columns,
                schema_diff={},
            )
            return {
                "success": True,
                "dataset_id": dataset_id,
                "blocking_errors": [],
                "warnings": validation.warnings,
                "unmapped_columns": validation.unmapped_columns,
                "row_count": len(df),
                "column_count": df.shape[1],
                "lot_min": lot_min,
                "lot_max": lot_max,
                "lot_count": lot_count,
            }
        finally:
            # The uploaded DataFrame is only needed for this validation pass --
            # release it explicitly rather than waiting on Python's own GC,
            # since RSS doesn't reliably shrink back to the OS otherwise.
            df = None
            gc.collect()

    def delete(self, dataset_id: str) -> None:
        if dataset_id in BUNDLED_DATASET_FILES:
            raise BundledDatasetDeleteError(dataset_id)
        record = self.store.get_dataset(dataset_id)
        if record is None:
            raise DatasetNotFoundError(dataset_id)
        path = self.upload_root / record["stored_path"]
        if path.is_file():
            path.unlink()
        self.store.delete_dataset(dataset_id)
        # A deleted dataset's saved 학습/원인 분석/사전 알람 results would
        # otherwise keep pointing a selector at data that no longer exists
        # (spec §3-5) -- best-effort, deletion itself already succeeded above.
        try:
            invalidate_state_for_dataset(self.store, dataset_id)
        except Exception:
            logger.warning("데이터셋 삭제에 따른 최근 결과 무효화 실패: %s", dataset_id, exc_info=True)
