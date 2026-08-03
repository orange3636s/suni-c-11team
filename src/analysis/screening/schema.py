"""Column-name schema parser for the Pareto correlation-factor module.

Parses ``Step{n}_R{m}`` / ``Step{n}_D{m}`` / ``Step{n}_Config`` columns via
regex so the analysis works regardless of how many steps or measurement
channels a given dataset has. Nothing here is hardcoded to the 30-step /
train.CSV shape used in golden tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

R_PAT = r"^Step(\d+)_R(\d+)$"
D_PAT = r"^Step(\d+)_D(\d+)$"
CONFIG_PAT = r"^Step(\d+)_Config$"

_R_RE = re.compile(R_PAT)
_D_RE = re.compile(D_PAT)
_CONFIG_RE = re.compile(CONFIG_PAT)

ID_COLUMNS = ("Lot_Wafer_ID", "Lot_ID", "Wafer_Slot")
ALL_TARGET_COLUMNS = ("Y1", "Y2", "Y3", "Y4", "Y5")
FINAL_YIELD_COLUMN = "Y"
FAIL_BIT_PATTERN = re.compile(r"^Y([6-9]|10)$")


@dataclass
class Schema:
    r_cols: list[str] = field(default_factory=list)
    d_cols: list[str] = field(default_factory=list)
    config_cols: list[str] = field(default_factory=list)
    target_cols: list[str] = field(default_factory=list)
    id_cols: list[str] = field(default_factory=list)
    max_step: int = 0
    steps_present: list[int] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)

    @property
    def factor_cols(self) -> list[str]:
        return [*self.r_cols, *self.d_cols, *self.config_cols]

    def step_of(self, feature: str) -> int | None:
        for pattern in (_R_RE, _D_RE, _CONFIG_RE):
            match = pattern.match(feature)
            if match:
                return int(match.group(1))
        return None

    def kind_of(self, feature: str) -> str | None:
        if _R_RE.match(feature):
            return "R"
        if _D_RE.match(feature):
            return "D"
        if _CONFIG_RE.match(feature):
            return "Config"
        return None


def parse_schema(df: pd.DataFrame) -> Schema:
    r_cols: list[str] = []
    d_cols: list[str] = []
    config_cols: list[str] = []
    unmapped: list[str] = []
    steps: set[int] = set()

    known_ids = [c for c in ID_COLUMNS if c in df.columns]
    known_ids_set = set(known_ids)
    known_targets = {FINAL_YIELD_COLUMN, *ALL_TARGET_COLUMNS}

    for col in df.columns:
        if col in known_ids_set or col in known_targets or FAIL_BIT_PATTERN.match(col):
            continue

        r_match = _R_RE.match(col)
        d_match = _D_RE.match(col)
        c_match = _CONFIG_RE.match(col)

        if r_match:
            r_cols.append(col)
            steps.add(int(r_match.group(1)))
        elif d_match:
            d_cols.append(col)
            steps.add(int(d_match.group(1)))
        elif c_match:
            config_cols.append(col)
            steps.add(int(c_match.group(1)))
        else:
            unmapped.append(col)

    target_cols = [t for t in ALL_TARGET_COLUMNS if t in df.columns]

    return Schema(
        r_cols=sorted(r_cols, key=_sort_key),
        d_cols=sorted(d_cols, key=_sort_key),
        config_cols=sorted(config_cols, key=_sort_key),
        target_cols=target_cols,
        id_cols=known_ids,
        max_step=max(steps) if steps else 0,
        steps_present=sorted(steps),
        unmapped=unmapped,
    )


def _sort_key(col: str) -> tuple[int, int]:
    for pattern in (_R_RE, _D_RE, _CONFIG_RE):
        match = pattern.match(col)
        if match:
            groups = match.groups()
            step = int(groups[0])
            sub = int(groups[1]) if len(groups) > 1 else 0
            return (step, sub)
    return (0, 0)
