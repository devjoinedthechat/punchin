"""Can anything tell the simulated customer from the real one, and is the judge worth believing?"""

import datetime as dt

import pytest

from punchin.call import Call, Turn
from punchin.discriminate import Discrimination, Guess, Round, discriminate, wilson
from punchin.model import Completion
from punchin.scenarios import BY_ID

SCENARIO = BY_ID["self-correction"]


class Judge:
    """A model that answers the discrimination however it is told to."""

    name = "judge"

    def __init__(self, always: str = "A", line: str = "Onsdag.") -> None:
        self.always, self.line, self.asked = always, line, []

    def complete(self, system: str, prompt: str, *, mcp=None, schema=None) -> Completion:
        self.asked.append(prompt)
        if schema and "real" in schema.get("properties", {}):
            return Completion("", structured={"real": self.always, "why": "a hunch"}, cost_usd=0.001)
        return Completion(self.line, cost_usd=0.002)


def _call() -> Call:
    at = dt.datetime.now(dt.UTC)
    call = Call(id="c", scenario=SCENARIO.id, agent="a", customer="k", started_at=at)
    call.turns = [
        Turn(index=0, speaker="agent", text="Hvilken dag?", started_at=at, ended_at=at),
        Turn(index=1, speaker="customer", text="Onsdag, tak.", started_at=at, ended_at=at),
        Turn(index=2, speaker="agent", text="Kl. 8?", started_at=at, ended_at=at),
        Turn(index=3, speaker="customer", text="Ja tak.", started_at=at, ended_at=at),
    ]
    return call


def test_it_judges_every_customer_turn_and_runs_both_controls() -> None:
    found = discriminate(_call(), SCENARIO.goal, Judge(), workers=1)
    assert len(found.paired.guesses) == 2
    assert len(found.blind.guesses) == 2
    assert len(found.floor.guesses) == 2
    assert found.cost_usd > 0


def test_a_judge_that_always_picks_the_first_fails_its_own_blind_control() -> None:
    """The control exists to catch exactly this, and a run that fails it says nothing."""
    found = discriminate(_call(), SCENARIO.goal, Judge(always="A"), workers=1, seed=1)
    held, why = found.trustworthy
    if not held:
        assert "biased" in why or "not discriminating" in why
        assert "says nothing" in found.text()


def test_a_judge_at_fifty_per_cent_means_the_simulator_is_indistinguishable() -> None:
    perfect = Round("real vs simulated", "50%")
    perfect.guesses = [Guess(i, "r", "s", "real", i % 2 == 0, "") for i in range(20)]
    blind = Round("blind", "50%")
    blind.guesses = [Guess(i, "r", "r", "real", i % 2 == 0, "") for i in range(20)]
    floor = Round("floor", "high")
    floor.guesses = [Guess(i, "r", "o", "real", True, "") for i in range(20)]

    found = Discrimination("c", perfect, blind, floor)
    held, why = found.trustworthy
    assert held, why
    assert "indistinguishable" in found.text()


def test_a_simulator_that_is_always_spotted_is_reported_as_such() -> None:
    obvious = Round("real vs simulated", "50%")
    obvious.guesses = [Guess(i, "r", "s", "real", True, "for pænt sprog") for i in range(20)]
    blind = Round("blind", "50%")
    blind.guesses = [Guess(i, "r", "r", "real", i % 2 == 0, "") for i in range(20)]
    floor = Round("floor", "high")
    floor.guesses = [Guess(i, "r", "o", "real", True, "") for i in range(20)]

    printed = Discrimination("c", obvious, blind, floor).text()
    assert "distinguishable 100%" in printed
    assert "gave it away: for pænt sprog" in printed


def test_a_judge_that_cannot_spot_a_turn_from_another_call_is_not_discriminating() -> None:
    fine = Round("real vs simulated", "50%")
    fine.guesses = [Guess(i, "r", "s", "real", i % 2 == 0, "") for i in range(20)]
    blind = Round("blind", "50%")
    blind.guesses = [Guess(i, "r", "r", "real", i % 2 == 0, "") for i in range(20)]
    useless = Round("floor", "high")
    useless.guesses = [Guess(i, "r", "o", "real", i % 2 == 0, "") for i in range(20)]

    held, why = Discrimination("c", fine, blind, useless).trustworthy
    assert held is False
    assert "not discriminating at all" in why


def test_the_interval_is_binomial_so_the_noise_needs_no_measuring() -> None:
    """The reason to prefer this over fact overlap: its spread is known, not estimated."""
    assert wilson(10, 20) == pytest.approx((0.30, 0.70), abs=0.02)
    assert wilson(20, 20)[0] > 0.8
    assert wilson(0, 0) == (0.0, 1.0)


def test_a_call_with_no_customer_turns_is_refused() -> None:
    bare = _call()
    bare.turns = [t for t in bare.turns if t.speaker == "agent"]
    with pytest.raises(ValueError, match="no customer turns"):
        discriminate(bare, SCENARIO.goal, Judge())


def test_the_judge_sees_what_was_said_not_what_was_heard() -> None:
    """It is deciding which line a person produced, so it gets the words, not the recogniser's guess."""
    call = _call()
    call.turns[1].heard = "Onsdag, tvak."
    judge = Judge()
    discriminate(call, SCENARIO.goal, judge, workers=1)
    assert not any("tvak" in asked for asked in judge.asked)


def test_many_calls_pool_into_one_verdict_because_a_single_call_cannot_carry_one() -> None:
    """Five turns puts the interval near 12%-77%, which covers a faithful simulator and an obvious one."""
    from punchin.discriminate import pooled

    def run(name: str, paired_right: int, of: int) -> Discrimination:
        p = Round("real vs simulated", "50%")
        p.guesses = [Guess(i, "r", "s", "real", i < paired_right, "") for i in range(of)]
        p.cost_usd = 0.1
        b = Round("blind", "50%")
        b.guesses = [Guess(i, "r", "r", "real", i % 2 == 0, "") for i in range(of)]
        f = Round("floor", "high")
        f.guesses = [Guess(i, "r", "o", "real", True, "") for i in range(of)]
        return Discrimination(name, p, b, f)

    alone = run("one call", 2, 5)
    low, high = alone.paired.interval()
    assert high - low > 0.5  # no power at all on its own

    together = pooled([run(f"call {i}", 2, 5) for i in range(10)])
    assert len(together.paired.guesses) == 50
    assert together.paired.correct == 20
    wide = together.paired.interval()
    assert wide[1] - wide[0] < 0.3  # pooling is what buys the interval
    assert together.cost_usd == pytest.approx(1.0)
    assert "10 calls" in together.text()


def test_pooling_nothing_is_refused() -> None:
    from punchin.discriminate import pooled

    with pytest.raises(ValueError, match="nothing to pool"):
        pooled([])


def _run(right: int, of: int, *, blind_right: int | None = None, floor_right: int | None = None):
    """A discrimination with a known outcome in each arm, without asking a model anything."""
    paired = Round("real vs simulated", "50%")
    paired.guesses = [Guess(i, "r", "s", "real", i < right, f"reason {i}") for i in range(of)]
    blind = Round("blind (real vs real)", "50%")
    blind.guesses = [
        Guess(i, "r", "r", "real", i < (blind_right if blind_right is not None else of // 2), "")
        for i in range(of)
    ]
    floor = Round("floor (real vs another call)", "high")
    floor.guesses = [
        Guess(i, "r", "o", "real", i < (floor_right if floor_right is not None else of), "")
        for i in range(of)
    ]
    return Discrimination("c", paired, blind, floor)


def test_a_judge_that_is_reliably_wrong_is_not_a_judge_that_cannot_tell() -> None:
    """28% is separation with the label inverted, not 28% of the way to being fooled."""
    printed = _run(13, 47).text()
    assert "inverted" in printed
    assert "72% of the time" in printed  # it named the simulator as real
    assert "distinguishable 28% of the time" not in printed


def test_an_inverted_result_quotes_the_reasons_it_preferred_the_simulator() -> None:
    """At 28% the informative half is the 34 it got wrong, not the 13 it got right."""
    printed = _run(13, 47).text()
    assert "picked the simulator because: reason 13" in printed
    assert "gave it away" not in printed
