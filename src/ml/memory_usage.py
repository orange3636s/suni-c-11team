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
        info = psutil.Process(os.getpid()).memory_info()
        current = float(info.rss) / (1024**2)
        peak_bytes = getattr(info, "peak_wset", None)
        peak = float(peak_bytes) / (1024**2) if peak_bytes else None
    except (OSError, RuntimeError, ValueError):
        return None, None
    return current, peak


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


def _windows_memory() -> tuple[float | None, float | None]:
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        handle = get_current_process()
        succeeded = get_process_memory_info(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            return None, None
        scale = float(1024**2)
        return counters.WorkingSetSize / scale, counters.PeakWorkingSetSize / scale
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None


def memory_snapshot() -> dict[str, float | None]:
    current, psutil_peak = _psutil_memory()
    if current is None:
        current, windows_peak = _windows_memory()
        psutil_peak = psutil_peak or windows_peak
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
