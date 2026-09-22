"""Fork a recorded call at a turn: the prefix comes from the recording, everything after it runs live.

The prefix is free and identical, so the only thing under test is what the change did from turn `at`
onward. The customer from `at + 1` is pinned to the goal state of the customer who was really on the
call, which is what makes the continuation that customer's and not a generic caller's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from punchin.agent import Agent, Lead
from punchin.call import Call, call_id
from punchin.dms import Dms, fresh
from punchin.metrics import feel, outcome
from punchin.model import Model
from punchin.pinned import PinnedCustomer
from punchin.record import converse, finish, now
from punchin.scenarios import GoalState, Scenario

MOVED = ("options_max", "turns", "agent_words_max", "customer_stalls")


class Budget:
    """Stops the run starting another attempt once it has spent what it was allowed."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0

    def spend(self, amount: float) -> None:
        self.spent_usd += amount

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.limit_usd


def agent_turns(call: Call) -> list[int]:
    """The turns a fork can start at: the agent is what changed, so only its turns are fork points."""
    return [t.index for t in call.turns if t.speaker == "agent"]


def replay_prefix(call: Call, at: int, dms: Dms) -> None:
    """Re-apply what the recorded agent did to the world before turn `at`, so the fork starts where it did."""
    for turn in call.turns[:at]:
        for made in turn.tool_calls:
            try:
                dms.call(made.tool, **made.arguments)
            except ToolError:
                continue  # it failed against this same state when recorded, and fails the same way now


def fork_once(
    call: Call,
    scenario: Scenario,
    at: int,
    *,
    agent: Agent,
    goal: GoalState,
    model: Model,
    state_path: Path,
) -> Call:
    """One attempt: the turns before `at` from the recording, then the agent and the pinned customer live."""
    dms = Dms(state_path)
    dms.save(fresh([scenario.vehicle]))
    replay_prefix(call, at, dms)
    lead = Lead(owner=scenario.vehicle.owner, syn_due=scenario.vehicle.syn_due)
    customer = PinnedCustomer(model, goal)
    started = now()
    forked = Call(
        id=call_id(f"{scenario.id}-at{at}", agent.name, started),
        scenario=scenario.id,
        agent=agent.name,
        customer=customer.name,
        started_at=started,
        notes={"forked_from": call.id, "forked_at": at},
    )
    forked.turns = [turn.model_copy(deep=True) for turn in call.turns[:at]]
    try:
        converse(forked, agent, customer, lead, dms)
    finally:
        finish(forked, dms)
        # What the attempt really cost: the prefix was served from the recording and paid for nothing.
        live = sum(turn.cost_usd for turn in forked.turns[at:])
        forked.notes["live_cost_usd"] = round(live + customer.cost_usd, 4)
    return forked


@dataclass
class ForkReport:
    original: Call
    scenario: Scenario
    at: int
    changed: str
    attempts: list[Call] = field(default_factory=list)
    stopped: str | None = None

    @property
    def before(self) -> dict[str, Any]:
        return {**outcome(self.original, self.scenario), **feel(self.original)}

    @property
    def after(self) -> list[dict[str, Any]]:
        return [{**outcome(a, self.scenario), **feel(a)} for a in self.attempts]

    @property
    def fixed(self) -> int:
        return sum(1 for row in self.after if row["correct"])

    @property
    def live_cost_usd(self) -> float:
        return sum(float(a.notes.get("live_cost_usd", 0.0)) for a in self.attempts)

    def text(self) -> str:
        before, after = self.before, self.after
        said = self.original.turns[self.at].spoken if self.at < len(self.original.turns) else ""
        lines = [
            f"fork of {self.original.id} at turn {self.at}",
            f"  changed: {self.changed}",
            f"  the recorded turn {self.at} said: {said[:88]}",
            f"  original   correct={before['correct']!s:5}  " + "  ".join(f"{k}={before[k]}" for k in MOVED),
        ]
        for number, (row, call) in enumerate(zip(after, self.attempts, strict=True), start=1):
            lines.append(
                f"  attempt {number}  correct={row['correct']!s:5}  "
                + "  ".join(f"{k}={row[k]}" for k in MOVED)
                + f"  live ${float(call.notes.get('live_cost_usd', 0.0)):.3f}"
            )
        if not after:
            lines.append("  no attempts ran")
            return "\n".join(lines)
        lines.append(f"  correct in {self.fixed} of {len(after)} attempts (original: {before['correct']})")
        for key in MOVED:
            moved = median(row[key] for row in after)
            if moved != before[key]:
                lines.append(f"  {key}: {before[key]} -> {moved} (median of attempts)")
        lines.append(
            f"  live cost ${self.live_cost_usd:.3f}; the {self.at} turns before the fork cost nothing"
        )
        if self.stopped:
            lines.append(f"  stopped early: {self.stopped}")
        return "\n".join(lines)


def fork(
    call: Call,
    scenario: Scenario,
    at: int,
    *,
    agent: Agent,
    goal: GoalState,
    model: Model,
    state_path: Path,
    repeat: int = 1,
    changed: str = "nothing",
    budget: Budget | None = None,
    out: Path | None = None,
) -> ForkReport:
    if at not in agent_turns(call):
        allowed = ", ".join(str(i) for i in agent_turns(call))
        raise ValueError(f"turn {at} is not an agent turn in {call.id}; fork at one of: {allowed}")
    report = ForkReport(call, scenario, at, changed)
    for _ in range(repeat):
        if budget is not None and budget.exhausted:
            report.stopped = f"budget of ${budget.limit_usd:.2f} spent after {len(report.attempts)} attempts"
            break
        attempt = fork_once(call, scenario, at, agent=agent, goal=goal, model=model, state_path=state_path)
        report.attempts.append(attempt)
        if budget is not None:
            budget.spend(float(attempt.notes.get("live_cost_usd", 0.0)))
        if out is not None:
            attempt.save(out)
    return report
