"""Does a fork tell the truth?

Everything punchin claims rests on one thing: that forking a recording at turn `k` and running the
change from there says what a full live re-run would have said. Nothing in the tool proves that, and
there is a specific reason to doubt it — a simulated customer can be more helpful than the real one
was, which would make every fix look like it worked.

So measure it. A fork differs from a live run in two ways at once, and they are separated here:

    live, full re-run        the scripted customer, from the first turn      the ground truth
    fork, scripted customer  the recorded prefix, then the same script       the fork mechanism alone
    fork, pinned customer    the recorded prefix, then the simulator         what punchin actually does

`fork-scripted` against `live` measures whether replaying a prefix changes the answer. `fork-pinned`
against `fork-scripted` measures whether the simulator does. Reporting them apart is the point: one is
a bug in the tool, the other is a limit of simulation, and they need different fixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from punchin.agent import Agent
from punchin.call import Call
from punchin.customer import ScriptedCustomer
from punchin.fork import Budget, agent_turns, fork_once
from punchin.metrics import outcome
from punchin.model import Model
from punchin.record import record
from punchin.scenarios import GoalState, Scenario


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% interval that stays honest at the handful of trials this can afford."""
    if total == 0:
        return 0.0, 1.0
    rate = passed / total
    centre = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Arm:
    name: str
    scenario: Scenario
    calls: list[Call] = field(default_factory=list)

    @property
    def verdicts(self) -> list[bool]:
        return [bool(outcome(call, self.scenario)["correct"]) for call in self.calls]

    @property
    def passed(self) -> int:
        return sum(self.verdicts)

    @property
    def rate(self) -> float:
        return self.passed / len(self.calls) if self.calls else 0.0

    @property
    def cost_usd(self) -> float:
        return sum(float(c.notes.get("live_cost_usd", c.cost_usd)) for c in self.calls)

    def row(self) -> str:
        low, high = wilson(self.passed, len(self.calls))
        return (
            f"  {self.name:26} correct {self.passed}/{len(self.calls)}"
            f"  ({low:.0%}-{high:.0%})   ${self.cost_usd:.3f}"
        )


@dataclass
class Soundness:
    scenario: Scenario
    at: int
    change: str
    baseline: Call
    live: Arm
    fork_scripted: Arm
    fork_pinned: Arm

    @property
    def mechanism_gap(self) -> float:
        """How much replaying a prefix moves the answer. Anything but ~0 is a bug in the fork."""
        return self.fork_scripted.rate - self.live.rate

    @property
    def simulator_gap(self) -> float:
        """How much the pinned customer moves it. Positive means forks are optimistic."""
        return self.fork_pinned.rate - self.fork_scripted.rate

    @property
    def total_gap(self) -> float:
        return self.fork_pinned.rate - self.live.rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.id,
            "at": self.at,
            "change": self.change,
            "baseline_correct": bool(outcome(self.baseline, self.scenario)["correct"]),
            "arms": {
                arm.name: {"passed": arm.passed, "trials": len(arm.calls), "rate": arm.rate}
                for arm in (self.live, self.fork_scripted, self.fork_pinned)
            },
            "mechanism_gap": self.mechanism_gap,
            "simulator_gap": self.simulator_gap,
            "total_gap": self.total_gap,
        }

    def text(self) -> str:
        was = "correct" if outcome(self.baseline, self.scenario)["correct"] else "wrong"
        lines = [
            f"soundness of forking {self.scenario.id} at turn {self.at}",
            f"  change: {self.change}",
            f"  the recording being forked was {was}",
            "",
            self.live.row() + "   <- ground truth",
            self.fork_scripted.row(),
            self.fork_pinned.row() + "   <- what punchin does",
            "",
            f"  the fork mechanism moves the answer by {self.mechanism_gap:+.0%}",
            f"  the pinned customer moves it by       {self.simulator_gap:+.0%}",
            f"  a fork is {self.total_gap:+.0%} against a full re-run",
        ]
        lines.append("  " + self._verdict())
        return "\n".join(lines)

    def _verdict(self) -> str:
        trials = len(self.live.calls)
        if abs(self.total_gap) < 1e-9:
            return f"forks and live runs agreed on every one of {trials} trials"
        direction = "optimistic" if self.total_gap > 0 else "pessimistic"
        blame = "the simulator" if abs(self.simulator_gap) >= abs(self.mechanism_gap) else "the mechanism"
        return (
            f"forks were {direction} here, mostly {blame}. At {trials} trials this is a signal to "
            f"look at, not a number to quote."
        )


def measure(
    scenario: Scenario,
    *,
    agent_with_change: Agent,
    agent_without: Agent,
    goal: GoalState,
    model: Model,
    state_path: Path,
    at: int | None = None,
    trials: int = 3,
    change: str = "the agent under test",
    budget: Budget | None = None,
    out: Path | None = None,
) -> Soundness:
    """Record a baseline, then run the three arms and report where they disagree."""
    if not scenario.script:
        raise ValueError(
            f"{scenario.id} has no script, so there is no live re-run to compare a fork against; "
            f"soundness needs a scenario punchin can drive from the start"
        )

    baseline = record(scenario, agent_without, ScriptedCustomer(scenario), state_path, out)
    points = agent_turns(baseline)
    if at is None:
        at = points[len(points) // 2]  # the middle of the call, unless told otherwise
    if at not in points:
        raise ValueError(f"turn {at} is not an agent turn; fork at one of: {points}")

    live = Arm("live, full re-run", scenario)
    fork_scripted = Arm("fork, scripted customer", scenario)
    fork_pinned = Arm("fork, pinned customer", scenario)

    for _ in range(trials):
        if budget is not None and budget.exhausted:
            break
        live.calls.append(record(scenario, agent_with_change, ScriptedCustomer(scenario), state_path, out))
        fork_scripted.calls.append(
            fork_once(
                baseline,
                scenario,
                at,
                agent=agent_with_change,
                goal=goal,
                model=model,
                state_path=state_path,
                make_customer=lambda prefix: ScriptedCustomer(scenario).fast_forward(prefix),
            )
        )
        fork_pinned.calls.append(
            fork_once(
                baseline,
                scenario,
                at,
                agent=agent_with_change,
                goal=goal,
                model=model,
                state_path=state_path,
            )
        )
        if budget is not None:
            budget.spend(live.calls[-1].cost_usd + fork_scripted.cost_usd + fork_pinned.cost_usd)

    return Soundness(scenario, at, change, baseline, live, fork_scripted, fork_pinned)
