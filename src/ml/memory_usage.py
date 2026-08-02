from __future__ import annotations

import logging
import os
import sys
from typing import Any


def _psutil_memory() -> tuple[float | None, float | None]:
    """Return current and peak RSS in MiB without requiring psutil."""
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None, None

    try:
        current = float(psutil.Process(os.getpid()).memory_info().rss) / (1024**2)
    except (OSError, RuntimeError, ValueError):
        return None, None
    return current, None


def _resource_peak_memory() -> float | None:
    try:
        import resource
    except ImportError:
        return None

    try:
        maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, RuntimeError, ValueError):
        return None
    # macOS reports bytes; Linux and the BSDs used by common containers report KiB.
    return maximum / (1024**2) if sys.platform == "darwin" else maximum / 1024


def memory_snapshot() -> dict[str, float | None]:
    current, psutil_peak = _psutil_memory()
    return {
        "rss_mb": current,
        "max_rss_mb": psutil_peak or _resource_peak_memory(),
    }


def log_memory_stage(
    logger: logging.Logger,
    stage: str,
    **context: Any,
) -> None:
    """Log process memory for diagnostics without exposing it in API payloads."""
    snapshot = memory_snapshot()
    details = " ".join(
        f"{key}={value}"
        for key, value in context.items()
        if value is not None
    )
    logger.info(
        "ML memory stage=%s rss_mb=%s max_rss_mb=%s%s",
        stage,
        f"{snapshot['rss_mb']:.1f}" if snapshot["rss_mb"] is not None else "unavailable",
        (
            f"{snapshot['max_rss_mb']:.1f}"
            if snapshot["max_rss_mb"] is not None
            else "unavailable"
        ),
        f" {details}" if details else "",
    )
