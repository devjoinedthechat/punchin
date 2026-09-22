"""Fork a recorded call at a turn: the prefix comes from the recording, everything after it runs live.

The prefix is free and identical, so the only thing under test is what the change did from turn `at`
onward. The customer from `at + 1` is pinned to the goal state of the customer who was really on the
call, which is what makes the continuation that customer's and not a generic caller's.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal

from mcp.server.mcpserver.exceptions import ToolError

from punchin.agent import Agent, Lead
from punchin.call import Call, call_id
from punchin.customer import Customer
from punchin.dms import Dms, fresh
from punchin.metrics import feel, outcome
from punchin.model import Model
from punchin.pinned import PinnedCustomer
from punchin.record import converse, finish, now
from punchin.scenarios import GoalState, Scenario

MOVED = ("options_max", "turns", "agent_words_max", "customer_stalls")


class BudgetSpent(RuntimeError):
    """The run spent what it was allowed, in the middle of an attempt."""


class Budget:
    """What a run may spend. Checked between attempts, and inside one through `charge`.

    Checking only between attempts is not enough: one attempt that loops can spend the whole limit
    several times over before anybody looks. `charge` is called per turn, so the run stops within one
    turn of the limit rather than one attempt.
    """

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        # What `charge` has already counted for the attempt in progress, so the attempt's own total is
        # not added on top of it when the attempt finishes.
        self.charged = 0.0

    def spend(self, amount: float) -> None:
        self.spent_usd += amount

    def charge(self, amount: float) -> None:
        self.charged += amount
        self.spend(amount)
        if self.exhausted:
            raise BudgetSpent(f"spent ${self.spent_usd:.2f} of the ${self.limit_usd:.2f} allowed")

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
    wrap: Callable[[Customer], Customer] | None = None,
    make_customer: Callable[[Call], Customer] | None = None,
    budget: Budget | None = None,
) -> Call:
    """One attempt: the turns before `at` from the recording, then the agent and the pinned customer live.

    `wrap` puts something between the pinned customer and the agent — a voice and a recogniser, so that
    forking a spoken call does not quietly remove the thing that broke it.
    """
    dms = Dms(state_path)
    dms.save(fresh([scenario.vehicle]))
    replay_prefix(call, at, dms)
    lead = Lead(owner=scenario.vehicle.owner, syn_due=scenario.vehicle.syn_due)
    prefix_turns = [turn.model_copy(deep=True) for turn in call.turns[:at]]
    prefix = call.model_copy(update={"turns": prefix_turns})
    inner: Customer = make_customer(prefix) if make_customer else PinnedCustomer(model, goal)
    customer: Customer = wrap(inner) if wrap else inner
    started = now()
    forked = Call(
        id=call_id(f"{scenario.id}-at{at}", agent.name, started),
        scenario=scenario.id,
        agent=agent.name,
        customer=customer.name,
        started_at=started,
        notes={"forked_from": call.id, "forked_at": at},
    )
    forked.turns = prefix_turns
    try:
        converse(forked, agent, customer, lead, dms, budget=budget)
    finally:
        finish(forked, dms)
        # What the attempt really cost: the prefix was served from the recording and paid for nothing.
        live = sum(turn.cost_usd for turn in forked.turns[at:])
        spent = float(getattr(inner, "cost_usd", 0.0))
        forked.notes["live_cost_usd"] = round(live + spent, 4)
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

    def as_dict(self) -> dict[str, Any]:
        before, after = self.before, self.after
        return {
            "original": self.original.id,
            "scenario": self.scenario.id,
            "at": self.at,
            "changed": self.changed,
            "was_correct": bool(before["correct"]),
            "fixed": self.fixed,
            "attempts": len(after),
            "live_cost_usd": round(self.live_cost_usd, 4),
            "before": {key: before[key] for key in MOVED},
            "after": [{key: row[key] for key in MOVED} for row in after],
            "stopped": self.stopped,
        }

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
    wrap: Callable[[Customer], Customer] | None = None,
    make_customer: Callable[[Call], Customer] | None = None,
) -> ForkReport:
    if at not in agent_turns(call):
        allowed = ", ".join(str(i) for i in agent_turns(call))
        raise ValueError(f"turn {at} is not an agent turn in {call.id}; fork at one of: {allowed}")
    report = ForkReport(call, scenario, at, changed)
    for _ in range(repeat):
        if budget is not None and budget.exhausted:
            report.stopped = f"budget of ${budget.limit_usd:.2f} spent after {len(report.attempts)} attempts"
            break
        try:
            attempt = fork_once(
                call,
                scenario,
                at,
                agent=agent,
                goal=goal,
                model=model,
                state_path=state_path,
                wrap=wrap,
                make_customer=make_customer,
                budget=budget,
            )
        except BudgetSpent as spent:
            # Stopped inside an attempt rather than between two. The partial call is already saved.
            report.stopped = str(spent)
            break
        report.attempts.append(attempt)
        if budget is not None:
            budget.spend(float(attempt.notes.get("live_cost_usd", 0.0)) - budget.charged)
            budget.charged = 0.0
        if out is not None:
            attempt.save(out)
    return report


def fork_point(call: Call, where: str) -> int:
    """Which agent turn to fork this call at. An index, or `first`, `half` or `last`.

    A sweep needs this because turn 6 is a different moment in every call. The names resolve against
    each recording's own agent turns, so one instruction means the same thing across an archive.
    """
    points = agent_turns(call)
    if not points:
        raise ValueError(f"{call.id} has no agent turns to fork at")
    if where == "first":
        return points[0]
    if where == "last":
        return points[-1]
    if where == "half":
        return points[len(points) // 2]
    try:
        at = int(where)
    except ValueError:
        raise ValueError(f"--at takes a turn number, or first, half or last; not {where!r}") from None
    if at not in points:
        raise ValueError(f"turn {at} is not an agent turn in {call.id}; fork at one of: {points}")
    return at


Verdict = Literal["fixed", "broke", "still wrong", "held"]


def verdict_of(report: ForkReport) -> Verdict:
    """What the change did to this call. The majority of attempts decides, because agents are sampled."""
    was = bool(report.before["correct"])
    # Counted against `after`, which is what `fixed` counts, so the two can never disagree.
    outcomes = report.after
    now = report.fixed * 2 > len(outcomes) if outcomes else was
    if was and now:
        return "held"
    if was and not now:
        return "broke"
    return "fixed" if now else "still wrong"


@dataclass
class Sweep:
    """One change, across many recordings. The number that matters is not how many it fixed."""

    change: str
    reports: list[ForkReport] = field(default_factory=list)
    stopped: str | None = None

    @property
    def verdicts(self) -> dict[Verdict, list[str]]:
        found: dict[Verdict, list[str]] = {"fixed": [], "broke": [], "still wrong": [], "held": []}
        for report in self.reports:
            found[verdict_of(report)].append(report.scenario.id)
        return found

    @property
    def live_cost_usd(self) -> float:
        return sum(report.live_cost_usd for report in self.reports)

    def as_dict(self) -> dict[str, Any]:
        return {
            "change": self.change,
            "calls": len(self.reports),
            "verdicts": self.verdicts,
            "live_cost_usd": round(self.live_cost_usd, 4),
            "stopped": self.stopped,
        }

    def text(self) -> str:
        found = self.verdicts
        lines = [f"{self.change}", f"  across {len(self.reports)} calls:"]
        for name in ("fixed", "broke", "still wrong", "held"):
            if found[name]:
                lines.append(f"    {name:12} {len(found[name]):3}  {', '.join(found[name][:6])}")
        lines.append("")
        if found["broke"]:
            lines.append(
                f"  This change breaks {len(found['broke'])} call(s) that were right before. "
                f"Fixing {len(found['fixed'])} is not the number to look at."
            )
        elif found["fixed"]:
            lines.append(f"  Fixes {len(found['fixed'])}, breaks nothing.")
        else:
            lines.append("  Changes nothing that was measured.")
        lines.append(f"  live cost ${self.live_cost_usd:.3f}; every prefix was free")
        if self.stopped:
            lines.append(f"  stopped early: {self.stopped}")
        return "\n".join(lines)


def sentences(change: str) -> list[str]:
    """A prompt change, split where a person would split it. Blank if there is nothing to split."""
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", change.strip()) if part.strip()]
    return parts if len(parts) > 1 else []


@dataclass
class Ingredient:
    """One sentence of a change, and what the change does without it."""

    dropped: str
    report: ForkReport

    @property
    def still_fixes(self) -> int:
        return self.report.fixed

    @property
    def trials(self) -> int:
        return len(self.report.attempts)


@dataclass
class Recipe:
    """Which part of a prompt change did the work.

    A fix that works tells you nothing about which of its three sentences mattered. Dropping one at a
    time and re-forking does: a sentence whose removal costs nothing was not carrying the fix, and it
    should come out before it calcifies into folklore.
    """

    change: str
    whole: ForkReport
    without: list[Ingredient] = field(default_factory=list)
    stopped: str | None = None

    @property
    def live_cost_usd(self) -> float:
        return self.whole.live_cost_usd + sum(one.report.live_cost_usd for one in self.without)

    def as_dict(self) -> dict[str, Any]:
        return {
            "change": self.change,
            "whole": {"fixed": self.whole.fixed, "of": len(self.whole.attempts)},
            "without": [
                {"sentence": one.dropped, "fixed": one.still_fixes, "of": one.trials} for one in self.without
            ],
            "live_cost_usd": round(self.live_cost_usd, 4),
            "stopped": self.stopped,
        }

    def text(self) -> str:
        lines = [
            f"what carries {self.change[:60]!r}",
            f"  the whole change    fixes {self.whole.fixed}/{len(self.whole.attempts)}",
            "",
        ]
        for one in sorted(self.without, key=lambda i: i.still_fixes):
            cost = self.whole.fixed - one.still_fixes
            verdict = "carries it" if cost > 0 else "does nothing here"
            lines.append(f"  without {one.dropped[:44]:44} {one.still_fixes}/{one.trials}  {verdict}")
        idle = [one.dropped for one in self.without if one.still_fixes >= self.whole.fixed]
        lines.append("")
        if idle:
            lines.append(f"  {len(idle)} sentence(s) could come out without changing the outcome.")
        else:
            lines.append("  every sentence is doing something.")
        lines.append(f"  live cost ${self.live_cost_usd:.3f}")
        if self.stopped:
            lines.append(f"  stopped early: {self.stopped}")
        return "\n".join(lines)
