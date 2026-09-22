"""Teacher-forced fidelity: does the pinned customer say what the real one said, turn by turn?

For every customer turn t in the recording, the simulator sees the real conversation up to the agent's
line before it and writes turn t. The real turn t is the answer key. The agent's own randomness never
enters, because the agent's lines are always the recorded ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean, median

from punchin.call import Call
from punchin.goal import facts
from punchin.model import Model
from punchin.pinned import PinnedCustomer
from punchin.scenarios import GoalState


@dataclass
class TurnScore:
    index: int
    real: str
    simulated: str
    real_facts: set[str]
    simulated_facts: set[str]

    @property
    def jaccard(self) -> float:
        union = self.real_facts | self.simulated_facts
        return len(self.real_facts & self.simulated_facts) / len(union) if union else 1.0

    @property
    def exact(self) -> bool:
        return self.real_facts == self.simulated_facts

    @property
    def length_ratio(self) -> float:
        return len(self.simulated.split()) / max(len(self.real.split()), 1)


@dataclass
class Report:
    call: str
    turns: list[TurnScore] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def mean_jaccard(self) -> float:
        return mean(t.jaccard for t in self.turns) if self.turns else 1.0

    @property
    def exact_rate(self) -> float:
        return mean(t.exact for t in self.turns) if self.turns else 1.0

    @property
    def mean_length_ratio(self) -> float:
        return mean(t.length_ratio for t in self.turns) if self.turns else 1.0

    def one_line(self) -> str:
        return (
            f"  {self.call[-46:]:46} jaccard {self.mean_jaccard:.2f}  exact {self.exact_rate:.0%}"
            f"  length x{self.mean_length_ratio:.2f}  ${self.cost_usd:.3f}"
        )

    def text(self) -> str:
        lines = [
            f"{self.call}: facts jaccard {self.mean_jaccard:.2f}, exact {self.exact_rate:.0%}, "
            f"length x{self.mean_length_ratio:.2f}, ${self.cost_usd:.3f}"
        ]
        for t in self.turns:
            mark = "=" if t.exact else "≠"
            lines.append(f"  {t.index:2} {mark} real: {t.real}")
            lines.append(f"       sim: {t.simulated}")
            if not t.exact:
                lines.append(f"       real {sorted(t.real_facts)}  sim {sorted(t.simulated_facts)}")
        return "\n".join(lines)


# What the pinned customer is told about the customer it is playing. Dropping one and re-measuring is
# how you find out whether it was worth putting there.
FIELDS = ("prefers_time", "manner", "reveals", "constraints", "extras", "mood")


def without(goal: GoalState, field: str) -> GoalState:
    """The same goal state with one thing withheld, for an ablation."""
    blank: dict[str, object] = (
        {"mood": "neutral"}
        if field == "mood"
        else {field: [] if field in ("manner", "reveals", "constraints", "extras") else None}
    )
    return goal.model_copy(update=blank)


def teacher_forced(call: Call, goal: GoalState, model: Model) -> Report:
    report = Report(call.id)
    for turn in call.turns:
        if turn.speaker != "customer":
            continue
        prefix = call.model_copy(update={"turns": call.turns[: turn.index]})
        customer = PinnedCustomer(model, goal)
        simulated = customer.line(prefix)
        report.cost_usd += customer.cost_usd
        report.turns.append(
            TurnScore(turn.index, turn.spoken, simulated, facts(goal, turn.spoken), facts(goal, simulated))
        )
    return report


def across(reports: Sequence[Report]) -> str:
    """The spread over several calls. One call is an anecdote; this is the number worth quoting."""
    if not reports:
        return "no calls measured"
    scores = sorted(report.mean_jaccard for report in reports)
    turns = sum(len(report.turns) for report in reports)
    spent = sum(report.cost_usd for report in reports)
    exact = mean(report.exact_rate for report in reports)
    return (
        f"\n{len(reports)} calls, {turns} customer turns: "
        f"jaccard mean {mean(scores):.2f}, median {median(scores):.2f}, "
        f"range {scores[0]:.2f}-{scores[-1]:.2f}; exact {exact:.0%}; ${spent:.3f}"
    )


@dataclass
class Ablation:
    call: str
    full: Report
    dropped: dict[str, Report] = field(default_factory=dict)

    def text(self) -> str:
        lines = [
            f"{self.call}: what each part of the goal state is worth",
            f"  {'everything':18} jaccard {self.full.mean_jaccard:.2f}  exact {self.full.exact_rate:.0%}",
        ]
        for name, report in sorted(self.dropped.items(), key=lambda kv: kv[1].mean_jaccard):
            cost = self.full.mean_jaccard - report.mean_jaccard
            worth = f"{cost:+.2f}" if abs(cost) >= 0.005 else "  ~0"
            lines.append(
                f"  without {name:10} jaccard {report.mean_jaccard:.2f}  exact "
                f"{report.exact_rate:.0%}   worth {worth}"
            )
        spent = self.full.cost_usd + sum(r.cost_usd for r in self.dropped.values())
        lines.append(f"  ${spent:.3f}. A field worth ~0 is one the simulator was not using.")
        return "\n".join(lines)


def ablation(call: Call, goal: GoalState, model: Model, fields: Sequence[str]) -> Ablation:
    """Measure fidelity once with everything, then once per field withheld."""
    found = Ablation(call.id, teacher_forced(call, goal, model))
    for name in fields:
        found.dropped[name] = teacher_forced(call, without(goal, name), model)
    return found
