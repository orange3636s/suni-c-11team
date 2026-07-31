from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from api.settings import settings


DEFAULT_MODEL_DIR = settings.model_dir


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _model_slug(model_name: str) -> str:
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", model_name).lower()
    sanitized = re.sub(r"[^a-z0-9_]+", "_", snake_case).strip("_")
    return sanitized or "model"


def save_model_bundle(
    model: Any,
    *,
    target: str,
    model_name: str,
    feature_columns: list[str],
    metrics: dict[str, dict[str, float | None]],
    random_state: int,
    split_method: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    created_at: datetime | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = created_at or datetime.now().astimezone()
    timestamp_for_name = timestamp.strftime("%Y%m%d_%H%M%S")
    base_name = f"{target}_{_model_slug(model_name)}_{timestamp_for_name}"
    model_path = output_dir / f"{base_name}.joblib"
    metadata_path = output_dir / f"{base_name}.json"

    metadata = to_json_safe(
        {
            "target": target,
            "model_name": model_name,
            "feature_columns": feature_columns,
            "created_at": timestamp.isoformat(),
            "metrics": metrics,
            "random_state": random_state,
            "split_method": split_method,
            "scikit_learn_version": sklearn.__version__,
            "sklearn_version": sklearn.__version__,
        }
    )
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_path, metadata_path, metadata


def load_model(model_path: str | Path) -> Any:
    return joblib.load(Path(model_path))


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("모델 메타데이터는 JSON 객체여야 합니다.")
    return loaded
