"""Schema fingerprints and model/data compatibility decisions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.config_parser import CONFIG_PARSER_VERSION


def schema_fingerprint(
    feature_columns: list[str],
    parser_version: str = CONFIG_PARSER_VERSION,
) -> str:
    payload = json.dumps(
        {"features": sorted(dict.fromkeys(feature_columns)), "parser": parser_version},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_schema_status(
    metadata: dict[str, Any],
    raw_feature_columns: list[str] | None = None,
) -> str:
    version = metadata.get("schema_version")
    fingerprint = metadata.get("schema_fingerprint")
    if not version or not fingerprint:
        if raw_feature_columns is not None:
            legacy = metadata.get("raw_feature_columns") or metadata.get("feature_columns")
            if isinstance(legacy, list) and list(legacy) == list(raw_feature_columns):
                return "compatible"
        return "legacy"
    if version != "semicon_yield_v2":
        return "incompatible"
    if raw_feature_columns is None:
        return "compatible"
    expected = schema_fingerprint(
        raw_feature_columns,
        str(metadata.get("config_parser_version", CONFIG_PARSER_VERSION)),
    )
    return "compatible" if fingerprint == expected else "incompatible"
