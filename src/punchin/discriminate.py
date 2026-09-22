"""Can anything tell the simulated customer from the real one?

Fact overlap asks whether the simulator said the same things. That is a proxy. The question underneath
is whether a reader could tell the two apart, and *indistinguishable* is what faithful actually means.

So: show a model the conversation so far and two candidate next turns, one real and one simulated, in
random order, and ask which came from the real customer. Fifty per cent means it cannot tell. A hundred
means the simulator is obvious.

The usual objection to a model judging a model is that the scale is invented and nobody can check it.
That objection does not apply here, because **this judge can be calibrated against known answers**:

    blind   the real turn against itself      must come out at 50%, or the judge has a position bias
    floor   the real turn against a real turn
            from a different call              should come out high, or the judge is not discriminating

A run that fails either control is thrown away rather than reported. A quality judge scoring
"naturalness" out of five has no equivalent check, which is the difference.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from punchin.call import Call, Turn
from punchin.model import Model
from punchin.pinned import PinnedCustomer
from punchin.scenarios import GoalState

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "real": {"type": "string", "enum": ["A", "B"], "description": "Which one the real customer said"},
        "why": {"type": "string", "description": "In a few words, what gave it away"},
    },
    "required": ["real", "why"],
    "additionalProperties": False,
}

SYSTEM = """\
Du får et uddrag af en telefonsamtale mellem et autoværksteds agent og en kunde, og to bud på, hvad
kunden sagde derefter. Den ene er, hvad kunden faktisk sagde. Den anden er skrevet af en model, der
spiller kunden.

Vælg den, du tror kom fra det rigtige menneske. Hvis de er lige troværdige, så vælg en af dem alligevel
— gæt hellere end at lade være. Sig kort, hvad der afgjorde det.
"""

PROMPT = """\
Samtalen indtil nu:

{transcript}

A: {a}
B: {b}

Hvilken af dem sagde den rigtige kunde?
"""


@dataclass
class Guess:
    index: int
    real: str
    other: str
    shown_first: str  # "real" or "other"
    picked_real: bool
    why: str


@dataclass
class Round:
    """One arm: the measurement, or one of its two controls."""

    name: str
    expected: str
    guesses: list[Guess] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def correct(self) -> int:
        return sum(guess.picked_real for guess in self.guesses)

    @property
    def rate(self) -> float:
        return self.correct / len(self.guesses) if self.guesses else 0.0

    @property
    def first_rate(self) -> float:
        """How often it chose whichever was shown first. Far from 50% means position bias."""
        chose_first = [guess.picked_real == (guess.shown_first == "real") for guess in self.guesses]
        return mean(chose_first) if chose_first else 0.0

    def interval(self) -> tuple[float, float]:
        return wilson(self.correct, len(self.guesses))

    def line(self) -> str:
        low, high = self.interval()
        return (
            f"  {self.name:24} {self.correct:>3}/{len(self.guesses):<3} = {self.rate:.0%}"
            f"  ({low:.0%}-{high:.0%})   want {self.expected}"
        )


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    rate = passed / total
    centre = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Discrimination:
    call: str
    paired: Round
    blind: Round
    floor: Round

    @property
    def cost_usd(self) -> float:
        return self.paired.cost_usd + self.blind.cost_usd + self.floor.cost_usd

    @property
    def trustworthy(self) -> tuple[bool, str]:
        """Whether the controls held. A run that fails them says nothing about the simulator."""
        low, high = self.blind.interval()
        if not (low <= 0.5 <= high):
            return False, f"the blind control came out at {self.blind.rate:.0%}, not 50%: the judge is biased"
        if self.floor.rate < 0.6:
            return False, (
                f"the floor control came out at {self.floor.rate:.0%}: the judge cannot tell a turn "
                f"from another call apart either, so it is not discriminating at all"
            )
        return True, ""

    def as_dict(self) -> dict[str, Any]:
        held, why = self.trustworthy
        return {
            "call": self.call,
            "paired": {
                "correct": self.paired.correct,
                "of": len(self.paired.guesses),
                "rate": self.paired.rate,
            },
            "blind": {"rate": self.blind.rate},
            "floor": {"rate": self.floor.rate},
            "trustworthy": held,
            "why_not": why or None,
            "cost_usd": round(self.cost_usd, 4),
        }

    def text(self) -> str:
        lines = [f"{self.call}: can anything tell her from the simulator?", ""]
        lines += [self.paired.line(), self.blind.line(), self.floor.line(), ""]
        held, why = self.trustworthy
        if not held:
            lines.append(f"  this run says nothing: {why}")
            return "\n".join(lines)
        low, high = self.paired.interval()
        if low <= 0.5 <= high:
            lines.append("  indistinguishable at this sample size: the interval covers 50%")
        elif self.paired.rate > 0.5:
            lines.append(f"  distinguishable {self.paired.rate:.0%} of the time, and the interval misses 50%")
            told = [g.why for g in self.paired.guesses if g.picked_real][:3]
            lines.extend(f"    gave it away: {why}" for why in told)
        else:
            # Below 50% is not "worse at telling them apart". A judge that is reliably wrong is
            # carrying just as much signal as one that is reliably right, with the label inverted:
            # the simulated turn is the one that reads as human. Its reasons are the useful half.
            lines.append(
                f"  distinguishable, but inverted: the judge named the simulated turn as the real "
                f"one {1 - self.paired.rate:.0%} of the time, and the interval misses 50%. The "
                f"simulator is not failing to sound like her — it sounds more like a person than "
                f"the recording does."
            )
            told = [g.why for g in self.paired.guesses if not g.picked_real][:3]
            lines.extend(f"    picked the simulator because: {why}" for why in told)
        lines.append(f"  ${self.cost_usd:.3f}")
        return "\n".join(lines)


def pooled(runs: Sequence[Discrimination]) -> Discrimination:
    """One verdict over many calls, because a single call cannot carry one.

    A five-turn call is five judgements, and five judgements put the interval somewhere around
    12% to 77% — wide enough to cover both a faithful simulator and an obvious one. The turn is the
    unit that was measured, so the turns are what get pooled; the controls pool with them, which is
    also the only way the floor threshold means anything.
    """
    if not runs:
        raise ValueError("nothing to pool")
    together = Discrimination(
        f"{len(runs)} calls",
        Round("real vs simulated", "50%"),
        Round("blind (real vs real)", "50%"),
        Round("floor (real vs another call)", "high"),
    )
    for one in runs:
        for into, arm in (
            (together.paired, one.paired),
            (together.blind, one.blind),
            (together.floor, one.floor),
        ):
            into.guesses.extend(arm.guesses)
            into.cost_usd += arm.cost_usd
    return together


def _ask(
    model: Model, prefix: Call, *, real: str, other: str, index: int, seed: random.Random
) -> tuple[Guess, float]:
    real_first = seed.random() < 0.5
    a, b = (real, other) if real_first else (other, real)
    done = model.complete(
        SYSTEM,
        PROMPT.format(transcript=prefix.transcript(heard=False) or "(intet endnu)", a=a, b=b),
        schema=SCHEMA,
    )
    answer = done.structured if isinstance(done.structured, dict) else {}
    chose = str(answer.get("real", "A")).strip().upper()
    picked_real = (chose == "A") == real_first
    return (
        Guess(index, real, other, "real" if real_first else "other", picked_real, str(answer.get("why", ""))),
        done.cost_usd,
    )


def _customer_turns(call: Call) -> list[Turn]:
    return [turn for turn in call.turns if turn.speaker == "customer"]


def discriminate(
    call: Call,
    goal: GoalState,
    model: Model,
    *,
    elsewhere: Sequence[str] = (),
    workers: int = 4,
    seed: int = 0,
) -> Discrimination:
    """The measurement and both its controls, one pass over the call's customer turns."""
    # Which candidate is shown first, nothing more. Reproducible so a run can be repeated exactly.
    rng = random.Random(seed)  # noqa: S311
    spoken = _customer_turns(call)
    if not spoken:
        raise ValueError(f"{call.id} has no customer turns to judge")
    pool = list(elsewhere) or [turn.spoken for turn in reversed(spoken)]

    def simulated(turn: Turn) -> str:
        prefix = call.model_copy(update={"turns": call.turns[: turn.index]})
        return PinnedCustomer(model, goal).line(prefix)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as runner:
        fakes = list(runner.map(simulated, spoken))

    arms = {
        "real vs simulated": (Round("real vs simulated", "50%"), fakes),
        "blind (real vs real)": (Round("blind (real vs real)", "50%"), [t.spoken for t in spoken]),
        "floor (real vs another call)": (
            Round("floor (real vs another call)", "high"),
            [pool[i % len(pool)] for i in range(len(spoken))],
        ),
    }

    for round_, others in arms.values():

        def one(pair: tuple[Turn, str]) -> tuple[Guess, float]:
            turn, other = pair
            prefix = call.model_copy(update={"turns": call.turns[: turn.index]})
            return _ask(model, prefix, real=turn.spoken, other=other, index=turn.index, seed=rng)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as runner:
            for guess, cost in runner.map(one, zip(spoken, others, strict=True)):
                round_.guesses.append(guess)
                round_.cost_usd += cost

    return Discrimination(
        call.id,
        arms["real vs simulated"][0],
        arms["blind (real vs real)"][0],
        arms["floor (real vs another call)"][0],
    )
