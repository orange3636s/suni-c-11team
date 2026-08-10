"""Shared Sturges-rule bin count -- the one place every quantile-binning
caller gets its default bin count from, so a factor's effective bin count
never silently disagrees between callers again (TC-5: `effect_size.py` was
hardcoded to 8 bins, `quantile_profile.py` to 12 -- two independent
constants computing "the same kind of thing" for the same data).

Sturges' rule (k = ceil(1 + log2(n))) grows the bin count with sample size
instead of using one fixed number for every n from 30 to 10,000. Clamped to
[MIN_BINS, MAX_BINS]:
  - MIN_BINS=5: fewer bins than this and a U-shape's bottom gets smoothed
    away entirely (there's nowhere for the minimum to sit).
  - MAX_BINS=15: more bins than this and each bin's own mean becomes noise
    once per-bin sample counts drop too low (verified: 100 bins collapsed
    a recommended-window's width to ~0.3 -- meaningless precision).
"""

from __future__ import annotations

import math

MIN_BINS = 5
MAX_BINS = 15


def suggest_bin_count(n: int) -> int:
    """Sturges' rule, clamped to [MIN_BINS, MAX_BINS]. `n` is the number of
    valid (non-null) observations the caller is about to bin -- not the
    dataset's total row count."""
    if n <= 0:
        return MIN_BINS
    k = math.ceil(1 + math.log2(n))
    return max(MIN_BINS, min(MAX_BINS, k))
