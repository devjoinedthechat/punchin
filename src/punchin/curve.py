"""Where a call was lost, and where it could still have been saved.

A fork answers "did this change fix this call". It does not say *when* the call became unsalvageable,
and that is usually the more useful question: a fix that only works if applied before turn 4 is telling
you the problem is upstream of the prompt.

Forking the same recording at every agent turn answers it. The same loop reads two ways:

    with a change      how late can this change still rescue the call?   (recoverability)
    without one        which turn is the outcome actually decided at?    (attribution)

The second is credit assignment over a conversation. A turn where repeated runs disagree is a turn that
decides the call; one where they all agree, wrongly, means the call was already lost before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from punchin.agent import Agent
from punchin.call import Call
from punchin.fork import Budget, BudgetSpent, ForkReport, agent_turns, fork
from punchin.model import Model
from punchin.scenarios import GoalState, Scenario


@dataclass
class Point:
    at: int
    report: ForkReport

    @property
    def passed(self) -> int:
        return self.report.fixed

    @property
    def trials(self) -> int:
        return len(self.report.attempts)

    @property
    def rate(self) -> float:
        return self.passed / self.trials if self.trials else 0.0

    @property
    def decided(self) -> bool:
        """True when every attempt from here agreed. The call is settled by this turn, either way."""
        return self.trials > 0 and self.passed in (0, self.trials)


@dataclass
class Curve:
    call: Call
    scenario: Scenario
    change: str
    points: list[Point] = field(default_factory=list)
    stopped: str | None = None

    @property
    def was_correct(self) -> bool:
        return bool(self.points[0].report.before["correct"]) if self.points else False

    @property
    def last_saved(self) -> int | None:
        """The latest turn a fork still came out right from. None if none of them did."""
        saved = [point.at for point in self.points if point.rate > 0.5]
        return max(saved) if saved else None

    @property
    def always_lost(self) -> list[int]:
        """Turns where no attempt ever came out right."""
        return [point.at for point in self.points if point.trials and point.passed == 0]

    @property
    def first_lost(self) -> int | None:
        """The earliest turn after which nothing recovers: the point of no return, if there is one."""
        for at in self.always_lost:
            if all(p.passed == 0 for p in self.points if p.at >= at and p.trials):
                return at
        return None

    @property
    def carried_by_prefix(self) -> bool:
        """True when a later fork does better than an earlier one.

        Forking late is not a harder test, it is an easier one: the recorded prefix has already made
        the decisions, and the agent only has to not undo them. A curve that rises is measuring the
        recording more than the agent.
        """
        rates = [point.rate for point in self.points if point.trials]
        return any(later > earlier + 1e-9 for earlier, later in pairwise(rates))

    @property
    def undecided(self) -> list[int]:
        """Turns where repeated runs disagreed: the ones the outcome actually turns on."""
        return [point.at for point in self.points if point.trials > 1 and not point.decided]

    @property
    def live_cost_usd(self) -> float:
        return sum(point.report.live_cost_usd for point in self.points)

    def as_dict(self) -> dict[str, Any]:
        return {
            "call": self.call.id,
            "scenario": self.scenario.id,
            "change": self.change,
            "was_correct": self.was_correct,
            "points": [{"at": p.at, "passed": p.passed, "trials": p.trials} for p in self.points],
            "last_saved": self.last_saved,
            "first_lost": self.first_lost,
            "always_lost": self.always_lost,
            "carried_by_prefix": self.carried_by_prefix,
            "undecided": self.undecided,
            "live_cost_usd": round(self.live_cost_usd, 4),
            "stopped": self.stopped,
        }

    def text(self) -> str:
        if not self.points:
            return f"{self.call.id}: nothing measured"
        header = f"{self.call.id}  was {'correct' if self.was_correct else 'wrong'}"
        lines = [header, f"  change: {self.change}", ""]
        lines.append("  fork at  " + "".join(f"{p.at:>6}" for p in self.points))
        lines.append("   correct" + "".join(f"{p.passed:>4}/{p.trials}" for p in self.points))
        lines.append("")
        lines.extend(self._reading())
        lines.append(f"  live cost ${self.live_cost_usd:.3f}; every prefix was free")
        if self.stopped:
            lines.append(f"  stopped early: {self.stopped}")
        return "\n".join(lines)

    def _reading(self) -> list[str]:
        lines = []
        if self.always_lost:
            turns = ", ".join(str(at) for at in self.always_lost)
            lines.append(f"  wrong every time from turn(s) {turns}: that is where this call goes astray")
        if self.first_lost is not None:
            lines.append(f"  and nothing after turn {self.first_lost} recovers it — the point of no return")
        if self.undecided:
            turns = ", ".join(str(at) for at in self.undecided)
            lines.append(f"  still open at turn(s) {turns}: the outcome is decided there, not before")
        if self.carried_by_prefix:
            lines.append(
                "  the curve rises, so a later fork did better than an earlier one. That is the "
                "recording carrying the call, not the agent: a late fork tests less."
            )
        if not self.always_lost and not self.undecided:
            lines.append("  every fork came out right; no single turn decided this call")
        elif self.last_saved is None:
            lines.append("  no fork at any turn came out right; this change does not address this call")
        return lines


def measure(
    call: Call,
    scenario: Scenario,
    *,
    agent: Agent,
    goal: GoalState,
    model: Model,
    state_path: Path,
    repeat: int = 3,
    change: str = "the agent unchanged",
    budget: Budget | None = None,
    wrap: Any = None,
    out: Path | None = None,
) -> Curve:
    """Fork the recording at every agent turn and collect the outcome at each."""
    curve = Curve(call, scenario, change)
    for at in agent_turns(call):
        if budget is not None and budget.exhausted:
            curve.stopped = f"budget of ${budget.limit_usd:.2f} spent after {len(curve.points)} turns"
            break
        try:
            report = fork(
                call,
                scenario,
                at,
                agent=agent,
                goal=goal,
                model=model,
                state_path=state_path,
                repeat=repeat,
                changed=change,
                budget=budget,
                out=out,
                wrap=wrap,
            )
        except BudgetSpent as spent:
            curve.stopped = str(spent)
            break
        curve.points.append(Point(at, report))
    return curve
