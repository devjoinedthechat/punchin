"""Compare a run against a recorded baseline, and fail when it got worse.

The rest of punchin finds out what a change did. This is what you put in front of a deploy.

It takes more than one sample per scenario, and that is the whole design. An agent is sampled, so a
scenario that passes four times in five will fail a single-sample gate one build in five, and the
rational response to that is to re-run CI until it is green — which is the same as having no gate. A
baseline here records how often a scenario came out right, not whether it did once, and a regression is
a rate that fell, not a coin that landed differently.

A scenario that disagrees with itself inside one run is reported as flaky rather than as passing or
failing. For a voice agent that is a finding, not a nuisance: it means the outcome a customer gets
depends on the sampler.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal

FORMAT = 3
Direction = Literal["higher_is_worse", "lower_is_worse", "must_stay_true"]


@dataclass(frozen=True)
class Rule:
    key: str
    direction: Direction
    slack: float = 0.0
    why: str = ""

    def broken(self, was: float | None, now: float | None, spread: float | None = None) -> str | None:
        """What went wrong, or None. A metric absent from either side is not a regression.

        `spread` is how much this metric moved across the baseline's own repeated runs. When it is
        known it replaces the hand-set slack, because a scenario that naturally wanders by three turns
        should not be failed for wandering by three turns. A hand-set number cannot know that and this
        can, since the baseline already recorded it.
        """
        if was is None or now is None:
            return None
        if self.direction == "must_stay_true":
            # Both are rates now. Falling from "always" to "usually" is a regression too.
            return f"{self.key}: {was:.0%} -> {now:.0%} of runs" if now < was - 1e-9 else None
        allowed = self.slack if spread is None else max(spread, self.slack)
        note = "" if spread is None else f" (it varies by {_n(spread)} on its own)"
        if self.direction == "higher_is_worse" and now > was + allowed:
            return f"{self.key}: {_n(was)} -> {_n(now)}{note}"
        if self.direction == "lower_is_worse" and now < was - allowed:
            return f"{self.key}: {_n(was)} -> {_n(now)}{note}"
        return None


def _n(value: float) -> str:
    return f"{value:g}"


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
BOOLEAN = {rule.key for rule in RULES if rule.direction == "must_stay_true"}


@dataclass
class Regression:
    scenario: str
    detail: str


@dataclass
class Flaky:
    scenario: str
    passed: int
    trials: int

    def __str__(self) -> str:
        return f"{self.scenario}: came out right in {self.passed} of {self.trials} runs"


def _aggregate(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    """A rate for something that is true or false, a median for a number."""
    seen = [row[key] for row in rows if row.get(key) is not None]
    if not seen:
        return None
    if key in BOOLEAN:
        return sum(bool(value) for value in seen) / len(seen)
    numbers = [float(value) for value in seen if isinstance(value, int | float)]
    return median(numbers) if numbers else None


def _spread(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    """How far this metric moved across these runs. None for a rate, or for a single run."""
    if key in BOOLEAN:
        return None
    numbers = [float(row[key]) for row in rows if isinstance(row.get(key), int | float)]
    return max(numbers) - min(numbers) if len(numbers) > 1 else None


def by_scenario(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario"])].append(row)
    return dict(grouped)


def baseline_from(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What a later run is held against: one aggregate per metric, and how many runs it rests on."""
    return {
        "format": FORMAT,
        "scenarios": {
            scenario: {
                "trials": len(group),
                **{rule.key: value for rule in RULES if (value := _aggregate(group, rule.key)) is not None},
                # What each number did across these runs, so a later check can tell wandering from
                # regression without anybody having to guess a tolerance.
                "spread": {
                    rule.key: moved for rule in RULES if (moved := _spread(group, rule.key)) is not None
                },
            }
            for scenario, group in sorted(by_scenario(rows).items())
        },
    }


def load_baseline(path: Path) -> dict[str, Any]:
    stored: dict[str, Any] = json.loads(path.read_text())
    written = stored.get("format")
    if written != FORMAT:
        raise ValueError(
            f"{path} was written by a different version of punchin (baseline format {written}, this "
            f"is {FORMAT}); record it again with `punchin check --update`"
        )
    return stored


def flaky(rows: Sequence[dict[str, Any]], key: str = "correct") -> list[Flaky]:
    """Scenarios that disagreed with themselves inside this run."""
    found = []
    for scenario, group in sorted(by_scenario(rows).items()):
        verdicts = [bool(row.get(key)) for row in group if row.get(key) is not None]
        if len(verdicts) > 1 and 0 < sum(verdicts) < len(verdicts):
            found.append(Flaky(scenario, sum(verdicts), len(verdicts)))
    return found


def compare(
    rows: Sequence[dict[str, Any]], baseline: dict[str, Any], rules: Sequence[Rule] = RULES
) -> list[Regression]:
    """Every way this run is worse than the baseline, scenario by scenario."""
    known = baseline.get("scenarios", {})
    grouped = by_scenario(rows)
    found: list[Regression] = [
        Regression(scenario, broke)
        for scenario, group in sorted(grouped.items())
        if (was := known.get(scenario)) is not None
        for rule in rules
        if (
            broke := rule.broken(
                was.get(rule.key),
                _aggregate(group, rule.key),
                (was.get("spread") or {}).get(rule.key),
            )
        )
        is not None
    ]
    found.extend(
        Regression(missing, "in the baseline, not in this run")
        for missing in sorted(known.keys() - grouped.keys())
    )
    return found


@dataclass
class Checked:
    regressions: list[Regression]
    flaky: list[Flaky] = field(default_factory=list)
    scenarios: int = 0
    trials: int = 0

    @property
    def failed(self) -> bool:
        return bool(self.regressions)

    def text(self) -> str:
        runs = f"{self.trials} run{'s' if self.trials != 1 else ''} of {self.scenarios} scenarios"
        lines = []
        if self.regressions:
            lines.append(f"{len(self.regressions)} regressions across {runs}")
            lines.extend(f"  {found.scenario:18} {found.detail}" for found in self.regressions)
        else:
            lines.append(f"no regressions across {runs}")
        if self.flaky:
            lines.append(
                f"\n{len(self.flaky)} scenario{'s' if len(self.flaky) != 1 else ''} disagreed with "
                f"{'themselves' if len(self.flaky) != 1 else 'itself'} — the outcome a customer gets "
                f"depends on the sampler:"
            )
            lines.extend(f"  {found}" for found in self.flaky)
        elif self.trials > self.scenarios:
            lines.append("  every scenario agreed with itself across runs")
        return "\n".join(lines)


def check(rows: Sequence[dict[str, Any]], baseline: dict[str, Any]) -> Checked:
    grouped = by_scenario(rows)
    return Checked(compare(rows, baseline), flaky(rows), len(grouped), len(rows))
