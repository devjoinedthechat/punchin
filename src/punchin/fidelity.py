"""Teacher-forced fidelity: does the pinned customer say what the real one said, turn by turn?

For every customer turn t in the recording, the simulator sees the real conversation up to the agent's
line before it and writes turn t. The real turn t is the answer key. The agent's own randomness never
enters, because the agent's lines are always the recorded ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

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
