"""Schema-compatibility normalization applied once, right after a CSV is
read, so every downstream consumer (screening, training, the dataset
registry) only ever has to know one column-naming convention -- not every
variant a given source dataset happens to use (spec: "스키마 호환 확장").

Two independent CSV-loading paths exist in this codebase (the dataset
registry's in-memory cache for screening/analysis, and the training job's
own re-read of the uploaded bytes) -- both call `normalize_dataset` so
neither silently sees an un-normalized frame.
"""

from __future__ import annotations

import re

import pandas as pd

ID_COLUMN = "Lot_Wafer_ID"
LOT_COLUMN = "Lot_ID"
WAFER_SLOT_COLUMN = "Wafer_Slot"

# Step{n}_Config and Step{n}_EQ are the same concept -- the step's
# equipment configuration (Model/Equipment/Chamber) -- under a different
# column name depending on the dataset. `Step{n}_EQ1`/`Step{n}_EQ2`/...
# (multiple numbered equipment channels per step, as in
# mentorship_dataset_v7_killing_event.csv) are deliberately NOT matched
# here -- those are a different, unrelated shape and stay whatever
# category they already fall into (the broader `equipment` pattern in
# config/data_schema.yaml keeps excluding them from numeric features).
_EQ_COLUMN_RE = re.compile(r"^Step(\d+)_EQ$", re.IGNORECASE)
_CONFIG_COLUMN_RE = re.compile(r"^Step(\d+)_Config$", re.IGNORECASE)
_LOT_WAFER_RE = re.compile(r"^(?P<lot>.+?)W(?P<slot>\d+)$", re.IGNORECASE)


def _normalize_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Step{n}_EQ -> Step{n}_Config. If a dataset has both for the
    same step, _Config wins and the _EQ duplicate is dropped entirely
    (spec §2-1: "같은 데이터셋에 _Config 와 _EQ 가 동시에 존재하면
    _Config 를 우선하고, 중복 step은 하나만 쓴다").
    """
    existing_config_steps = {
        match.group(1) for column in df.columns if (match := _CONFIG_COLUMN_RE.match(str(column)))
    }
    rename_map: dict[str, str] = {}
    drop_columns: list[str] = []
    for column in df.columns:
        match = _EQ_COLUMN_RE.match(str(column))
        if not match:
            continue
        step = match.group(1)
        if step in existing_config_steps:
            drop_columns.append(column)
        else:
            rename_map[column] = f"Step{step}_Config"
            existing_config_steps.add(step)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _parse_lot_wafer_id(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Derive Lot_ID/Wafer_Slot from Lot_Wafer_ID (e.g. "L001W01" ->
    Lot_ID="L001", Wafer_Slot=1) when a dataset doesn't already carry
    Lot_ID as its own column (spec §2-2) -- GroupKFold and every
    LOT-based summary need Lot_ID specifically, not just the combined
    wafer id. A dataset that already has Lot_ID is left untouched.

    Returns (normalized_df, failed_row_count) -- rows whose
    Lot_Wafer_ID doesn't match the "<lot>W<slot>" shape get Lot_ID=null
    rather than raising, so one malformed id doesn't reject the whole
    file.
    """
    if LOT_COLUMN in df.columns or ID_COLUMN not in df.columns:
        return df, 0
    extracted = df[ID_COLUMN].astype(str).str.extract(_LOT_WAFER_RE)
    df = df.copy()
    df[LOT_COLUMN] = extracted["lot"]
    df[WAFER_SLOT_COLUMN] = pd.to_numeric(extracted["slot"], errors="coerce").astype("Int64")
    failed_count = int(df[LOT_COLUMN].isna().sum())
    return df, failed_count


def normalize_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply every schema-compatibility normalization in one place.

    Returns (normalized_df, report). `report["lot_id_parsed"]` is True
    only when Lot_ID didn't already exist and had to be derived;
    `report["lot_id_parse_failed_count"]` is how many rows didn't match
    the "<lot>W<slot>" shape (0 when Lot_ID already existed natively, or
    when parsing succeeded for every row).
    """
    had_lot_id = LOT_COLUMN in df.columns
    df = _normalize_config_columns(df)
    df, failed_count = _parse_lot_wafer_id(df)
    return df, {
        "lot_id_parsed": (not had_lot_id) and LOT_COLUMN in df.columns,
        "lot_id_parse_failed_count": failed_count,
    }
