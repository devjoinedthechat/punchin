"""Teacher-forced fidelity: does the pinned customer say what the real one said, turn by turn?

For every customer turn t in the recording, the simulator sees the real conversation up to the agent's
line before it and writes turn t. The real turn t is the answer key. The agent's own randomness never
enters, because the agent's lines are always the recorded ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean, median, stdev

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
            f"  {self.call[-42:]:42} jaccard {self.mean_jaccard:.2f}  exact {self.exact_rate:.0%}"
            f"  length x{self.mean_length_ratio:.2f}  over {len(self.turns)} turns  ${self.cost_usd:.3f}"
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


@dataclass
class Repeated:
    """The same measurement several times, because one run of it is not a number.

    The simulator is sampled, so fidelity has a spread. Measured on one call, the same goal state has
    come back anywhere from 0.60 to 0.80. Any claim that a change to the simulator moved fidelity has
    to clear that spread, and the only way to know what it is, is to run the thing again.
    """

    call: str
    runs: list[Report] = field(default_factory=list)

    @property
    def scores(self) -> list[float]:
        return [run.mean_jaccard for run in self.runs]

    @property
    def mean(self) -> float:
        return mean(self.scores) if self.scores else 0.0

    @property
    def spread(self) -> float:
        """Widest minus narrowest: the noise a claimed improvement has to beat."""
        return max(self.scores) - min(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def cost_usd(self) -> float:
        return sum(run.cost_usd for run in self.runs)

    @property
    def turns(self) -> int:
        """Customer turns each run scored. A handful of them quantises the whole measurement."""
        return len(self.runs[0].turns) if self.runs else 0

    def one_line(self, label: str = "") -> str:
        name = label or self.call[-36:]
        each = " ".join(f"{score:.2f}" for score in self.scores)
        return (
            f"  {name:36} jaccard {self.mean:.2f}  spread {self.spread:.2f}  ({each})"
            f"  over {self.turns} turns  ${self.cost_usd:.3f}"
        )


def repeated(call: Call, goal: GoalState, model: Model, times: int) -> Repeated:
    found = Repeated(call.id)
    for _ in range(max(1, times)):
        found.runs.append(teacher_forced(call, goal, model))
    return found


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
    """The spread over several calls. One call is an anecdote; this is the number worth quoting.

    Reported per turn rather than per call. Averaging the calls' averages gives a two-turn call the
    same weight as a ten-turn one, and a two-turn call's score is a coin flip on whether one keyword
    fired — it moves in steps of 0.50 and carries almost no information. Pooling the turns weights
    each call by how much it actually measured.
    """
    if not reports:
        return "no calls measured"
    scores = sorted(report.mean_jaccard for report in reports)
    every_turn = [turn.jaccard for report in reports for turn in report.turns]
    turns = len(every_turn)
    spent = sum(report.cost_usd for report in reports)
    exact = mean(turn.exact for report in reports for turn in report.turns) if turns else 0.0
    shortest = min(len(report.turns) for report in reports)
    note = (
        f"\n  The shortest call scored {shortest} turns, so its per-call number moves in steps of "
        f"{1 / shortest:.2f} and is close to a coin flip. That is why the pooled figure is the one to "
        f"read."
        if shortest <= 4
        else ""
    )
    return (
        f"\n{len(reports)} calls, {turns} customer turns"
        f"\n  per turn (pooled): jaccard {mean(every_turn):.2f}, exact {exact:.0%}"
        f"\n  per call:          mean {mean(scores):.2f}, median {median(scores):.2f}, "
        f"range {scores[0]:.2f}-{scores[-1]:.2f}"
        f"\n  ${spent:.3f}{note}"
    )


def paired(full: Repeated, dropped: Repeated) -> tuple[float, float, int]:
    """How much one field is worth, paired turn by turn: the mean difference, its standard error, n.

    Comparing the two arms' averages throws away the thing that makes them comparable. Most of the
    variance in this metric is turn difficulty — a turn where the customer says "Ja." scores differently
    from one where she corrects herself, whatever the goal state says — and that difficulty is identical
    on both sides of the comparison. Pairing by turn cancels it, so a field worth 0.10 can be seen
    through 0.13 of run-to-run spread instead of drowning in it.
    """
    differences: list[float] = []
    for with_it, without_it in zip(full.runs, dropped.runs, strict=False):
        by_index = {turn.index: turn.jaccard for turn in without_it.turns}
        differences.extend(
            turn.jaccard - by_index[turn.index] for turn in with_it.turns if turn.index in by_index
        )
    if not differences:
        return 0.0, 0.0, 0
    middle = mean(differences)
    if len(differences) < 2:
        return middle, 0.0, len(differences)
    spread = stdev(differences) / math.sqrt(len(differences))
    return middle, spread, len(differences)


@dataclass
class Ablated:
    """An ablation where every arm was measured several times, so a delta can be read against noise."""

    call: str
    full: Repeated
    dropped: dict[str, Repeated] = field(default_factory=dict)

    @property
    def noise(self) -> float:
        """The widest spread any single arm showed. A delta under this is not a finding."""
        return max([self.full.spread, *(arm.spread for arm in self.dropped.values())], default=0.0)

    def worth(self, name: str) -> tuple[float, float, int]:
        return paired(self.full, self.dropped[name])

    def text(self) -> str:
        lines = [
            f"{self.call}: what each part of the goal state is worth",
            self.full.one_line("everything"),
            "",
            f"  {'field':18} {'paired':>8} {'± s.e.':>8} {'turns':>6}   verdict",
        ]
        for name in sorted(self.dropped, key=lambda n: -self.worth(n)[0]):
            middle, error, count = self.worth(name)
            # Two standard errors either side of zero: the usual bar, stated so it can be argued with.
            if error > 0 and abs(middle) > 2 * error:
                verdict = "carries its weight" if middle > 0 else "makes it worse"
            else:
                verdict = "not distinguishable from zero"
            lines.append(f"  {name:18} {middle:+8.3f} {error:8.3f} {count:6}   {verdict}")
        spent = self.full.cost_usd + sum(a.cost_usd for a in self.dropped.values())
        lines.append(
            f"\n  Paired by turn, so turn difficulty cancels. Run-to-run spread on the whole call was "
            f"{self.noise:.2f}, which is why the means alone say nothing. ${spent:.3f}."
        )
        return "\n".join(lines)


def ablation(call: Call, goal: GoalState, model: Model, fields: Sequence[str], times: int = 1) -> Ablated:
    """Measure fidelity with everything, then once per field withheld, `times` runs each."""
    found = Ablated(call.id, repeated(call, goal, model, times))
    for name in fields:
        found.dropped[name] = repeated(call, without(goal, name), model, times)
    return found
