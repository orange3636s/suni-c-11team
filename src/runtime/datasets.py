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
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from src.analysis.screening.schema import Schema, parse_schema
from src.analysis.target_hydration import (
    inspect_target_status,
    invalidate_target_hydration_cache,
)
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

# Display order is explicit and fixed -- never sorted by name or size.
# train.CSV stays the default selection. 번들 데이터셋은 Y = 100 -
# (Y1+...+Y5) 항등식이 성립하고 Config 컬럼 체계를 쓰는 것만 넣는다 --
# 현재 파이프라인의 스키마 가정이 그 두 가지에 의존한다.
BUNDLED_DATASET_FILES = {
    "train": "train.CSV",
    "test": "test_remove_y.CSV",
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

    if blocking_errors:
        return DatasetValidation(blocking_errors=blocking_errors, schema=schema)

    # 타깃(Y) 열이 없어도 차단하지 않는다 -- 업로드 연동(원인
    # 분석·수율 예측에서 새 파일을 올리는 것)은 대부분 "평가 데이터셋"
    # 용도라 Y가 없는 게 정상이다. 학습 전용 업로드(`/api/train/jobs`)는
    # 이 함수를 쓰지 않고 별도 검증(src/data_validation.py)을 거치므로
    # 여기서 풀어도 학습 경로에는 영향이 없다. 대신 경고로 남겨 "학습에는
    # 못 쓴다"를 알린다.
    target_status = inspect_target_status(df)
    if target_status.state == "missing_columns":
        warnings.append("이 파일에는 타깃(Y/Y1~Y5) 열이 없습니다 -- 평가 데이터셋으로 허용하며 승인 모델의 예측값으로 분석합니다.")
    elif target_status.state == "all_missing":
        warnings.append("실측 수율·불량률이 없어 승인 모델의 예측값으로 분석합니다.")
    elif target_status.state == "partial":
        warnings.append("실측값을 우선 사용하고 결측값만 예측값으로 보완합니다.")

    row_count = len(df)
    if row_count < LOW_ROW_COUNT_THRESHOLD:
        warnings.append("표본이 부족해 인자 선정 결과를 신뢰하기 어렵습니다.")
    elif row_count < UNSTABLE_ROW_COUNT_THRESHOLD:
        warnings.append("인자 선정이 불안정할 수 있습니다.")
    # 행 수 상한은 경고가 아니라 차단이어야 한다 -- 통과시키면 한도 근처
    # 파일이 업로드는 성공하고서 분석 단계에서 OOM으로 죽는다. 업로드
    # 시점에 명확히 거부한다.
    if row_count > max_row_count():
        blocking_errors.append(
            f"행 수가 너무 많습니다 ({row_count:,}행, 최대 {max_row_count():,}행). "
            "파일을 나눠서 업로드해 주세요."
        )
        return DatasetValidation(blocking_errors=blocking_errors, warnings=warnings, schema=schema)

    # Lot_ID may have been derived from Lot_Wafer_ID just now
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
# Bound the cache by total cached row count (LRU-evicted), never by the
# *count* of cached uploads: with a 200,000-row upload limit, 8 cached
# frames would be several GB, well past Railway's 512MB free-tier ceiling.
# Bounding by rows keeps the cache small regardless of how large any single
# upload is.
_UPLOADED_CACHE_MAX_ROWS = 200_000
_uploaded_cache_lock = threading.Lock()
_uploaded_cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()


@lru_cache(maxsize=_BUNDLED_CACHE_SIZE)
def _read_bundled_csv(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size  # part of the cache key only, invalidates on file replacement
    raw = pd.read_csv(path_str)
    normalized, _ = normalize_dataset(raw)
    return normalized


def _read_uploaded_csv(path_str: str) -> pd.DataFrame:
    with _uploaded_cache_lock:
        cached = _uploaded_cache.get(path_str)
        if cached is not None:
            _uploaded_cache.move_to_end(path_str)
            return cached
    raw = pd.read_csv(path_str)
    normalized, _ = normalize_dataset(raw)
    with _uploaded_cache_lock:
        _uploaded_cache[path_str] = normalized
        _uploaded_cache.move_to_end(path_str)
        total_rows = sum(len(df) for df in _uploaded_cache.values())
        while total_rows > _UPLOADED_CACHE_MAX_ROWS and len(_uploaded_cache) > 1:
            _, evicted = _uploaded_cache.popitem(last=False)
            total_rows -= len(evicted)
    return normalized


def _read_uploaded_csv_cache_clear() -> None:
    with _uploaded_cache_lock:
        _uploaded_cache.clear()


# 호출부가 lru_cache의 `.cache_clear()` 관례에 기대므로 같은 이름의
# 속성으로 붙여준다(위 delete()의 docstring 참고).
_read_uploaded_csv.cache_clear = _read_uploaded_csv_cache_clear


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

    def content_version(self, dataset_id: str) -> str:
        """Stable content identifier for analysis/hydration cache keys.

        Uploaded dataset IDs are immutable UUIDs. Bundled IDs are stable across
        deploys, so include file stat values to invalidate a replaced bundle.
        """
        if dataset_id in BUNDLED_DATASET_FILES:
            path = self._bundled_path(dataset_id)
            stat = path.stat()
            return f"bundled:{stat.st_mtime_ns}:{stat.st_size}"
        record = self.store.get_dataset(dataset_id)
        if record is None:
            raise DatasetNotFoundError(dataset_id)
        path = self.upload_root / record["stored_path"]
        stat = path.stat()
        return f"uploaded:{dataset_id}:{stat.st_mtime_ns}:{stat.st_size}"

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        """Bytes-in-memory path -- used by callers that already build the CSV
        in memory (SQL fetch-from-db, yield_dispatch), not by the browser
        upload route (that streams to disk via `upload_from_path` instead,
        because a 150MB browser upload must never sit fully in memory
        twice at once -- once as the raw bytes, once as the parsed
        DataFrame)."""

        def _load() -> tuple[pd.DataFrame, dict[str, object]]:
            return parse_uploaded_csv(content)

        return self._finalize_upload(
            filename,
            size_bytes=len(content),
            load=_load,
            persist=lambda dest: dest.write_bytes(content),
        )

    def upload_from_path(self, filename: str, tmp_path: Path) -> dict[str, Any]:
        """Counterpart to `upload()` for a file already streamed to
        disk in chunks by the route handler -- `tmp_path` holds the raw
        bytes, so this never holds the full upload in memory as `bytes` at
        all, only the parsed DataFrame. On success `tmp_path` is moved (not
        copied) into the registry's storage; on failure it's left in place
        for the caller to clean up."""
        size_bytes = tmp_path.stat().st_size

        def _load() -> tuple[pd.DataFrame, dict[str, object]]:
            try:
                df = pd.read_csv(tmp_path)
            except Exception as exc:
                raise DatasetValidationError([f"CSV 파싱에 실패했습니다: {exc}"]) from exc
            if df.shape[1] == 0:
                raise DatasetValidationError(["컬럼이 없는 CSV입니다."])
            return normalize_dataset(df)

        def _persist(dest: Path) -> None:
            os.replace(tmp_path, dest)

        return self._finalize_upload(filename, size_bytes=size_bytes, load=_load, persist=_persist)

    def _finalize_upload(
        self,
        filename: str,
        *,
        size_bytes: int,
        load: Callable[[], tuple[pd.DataFrame, dict[str, object]]],
        persist: Callable[[Path], None],
    ) -> dict[str, Any]:
        import gc

        if size_bytes > max_upload_size_bytes():
            actual_mb = size_bytes / (1024 * 1024)
            return {
                "success": False,
                "dataset_id": None,
                "blocking_errors": [f"파일이 너무 큽니다 (최대 {max_upload_size_mb()}MB). 현재 {actual_mb:.1f}MB"],
                "warnings": [],
                "unmapped_columns": [],
            }

        df: pd.DataFrame | None = None
        try:
            try:
                df, normalization_report = load()
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
            persist(self.upload_root / stored_name)
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
                # 프런트가 "이 파일에는 Y 계열이 있습니다" 안내를
                # 띄울지 결정하는 데 쓴다 -- 경고 문구를 문자열로 매칭하지
                # 않도록 별도 필드로 내려준다.
                "has_target_columns": bool(inspect_target_status(df).present_columns),
                "target_status": inspect_target_status(df).as_dict(),
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
        # `_read_uploaded_csv`는 파일 경로로 lru_cache된다 -- 삭제
        # 후 같은 stored_path로 새 파일이 올라오면(업로드 파일명이 겹치는
        # 경우) 캐시가 낡은 DataFrame을 계속 돌려준다. `lru_cache`는 키
        # 하나만 지우는 API가 없으므로 전체를 비운다 (업로드 캐시는
        # 최대 8개뿐이라 비용이 작다).
        _read_uploaded_csv.cache_clear()
        invalidate_target_hydration_cache(dataset_id)
        # A deleted dataset's saved 학습/원인 분석/사전 알람 results would
        # otherwise keep pointing a selector at data that no longer exists
        # -- best-effort, deletion itself already succeeded above.
        try:
            invalidate_state_for_dataset(self.store, dataset_id)
        except Exception:
            logger.warning("데이터셋 삭제에 따른 최근 결과 무효화 실패: %s", dataset_id, exc_info=True)
