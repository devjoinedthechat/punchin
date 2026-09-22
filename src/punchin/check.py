"""Compare a run against a recorded baseline, and fail when it got worse.

The rest of punchin finds out what a change did. This is what you put in front of a deploy: it takes
the numbers a run produced, holds them against the numbers the last good run produced, and names every
scenario that moved the wrong way. Nothing here calls a model; it works on recordings.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FORMAT = 1
Direction = Literal["higher_is_worse", "lower_is_worse", "must_stay_true"]


@dataclass(frozen=True)
class Rule:
    key: str
    direction: Direction
    slack: float = 0.0
    why: str = ""

    def broken(self, was: Any, now: Any) -> str | None:
        """What went wrong, or None. A metric absent from either run is not a regression."""
        if was is None or now is None:
            return None
        if self.direction == "must_stay_true":
            return f"{self.key}: was true, now false" if bool(was) and not bool(now) else None
        if not isinstance(was, int | float) or not isinstance(now, int | float):
            return None
        if self.direction == "higher_is_worse" and now > was + self.slack:
            return f"{self.key}: {_n(was)} -> {_n(now)}"
        if self.direction == "lower_is_worse" and now < was - self.slack:
            return f"{self.key}: {_n(was)} -> {_n(now)}"
        return None


def _n(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


RULES: tuple[Rule, ...] = (
    Rule("correct", "must_stay_true", why="the call reached the outcome the scenario expects"),
    Rule("day_ok", "must_stay_true", why="the day booked is the day the customer meant"),
    Rule("reg_survived", "must_stay_true", why="the plate the agent used is the plate she said"),
    Rule("note_ok", "must_stay_true", why="what the workshop needs to know reached the booking"),
    Rule("options_max", "higher_is_worse", why="times read out in one breath"),
    Rule("agent_repeats", "higher_is_worse", why="the agent saying the same thing again"),
    Rule("customer_stalls", "higher_is_worse", why="the customer not knowing what was asked"),
    Rule("turns", "higher_is_worse", slack=2, why="how long the call took to get there"),
    Rule("lookup_attempts", "higher_is_worse", slack=1, why="tries needed to find the car"),
)


@dataclass
class Regression:
    scenario: str
    detail: str


def baseline_from(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The numbers to hold a later run against, keyed by scenario."""
    kept = {rule.key for rule in RULES}
    return {
        "format": FORMAT,
        "scenarios": {str(row["scenario"]): {k: v for k, v in row.items() if k in kept} for row in rows},
    }


def load_baseline(path: Path) -> dict[str, Any]:
    stored: dict[str, Any] = json.loads(path.read_text())
    if stored.get("format") != FORMAT:
        raise ValueError(
            f"{path} was written by a different version of punchin (format {stored.get('format')}, "
            f"this is {FORMAT}); record it again with `punchin check --update`"
        )
    return stored


def compare(
    rows: Sequence[dict[str, Any]], baseline: dict[str, Any], rules: Sequence[Rule] = RULES
) -> list[Regression]:
    """Every way this run is worse than the baseline, scenario by scenario."""
    known = baseline.get("scenarios", {})
    found: list[Regression] = []
    seen = set()
    for row in rows:
        scenario = str(row["scenario"])
        seen.add(scenario)
        was = known.get(scenario)
        if was is None:
            continue  # a new scenario has nothing to be worse than
        found.extend(
            Regression(scenario, broke)
            for rule in rules
            if (broke := rule.broken(was.get(rule.key), row.get(rule.key))) is not None
        )
    found.extend(
        Regression(missing, "in the baseline, not in this run") for missing in sorted(known.keys() - seen)
    )
    return found


def report(found: Sequence[Regression], total: int) -> str:
    if not found:
        return f"no regressions across {total} scenarios"
    lines = [f"{len(found)} regressions across {total} scenarios"]
    lines.extend(f"  {found_one.scenario:18} {found_one.detail}" for found_one in found)
    return "\n".join(lines)
