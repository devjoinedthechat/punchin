"""Repeats, and the noise floor an ablation delta has to clear."""

import sys
from pathlib import Path

import pytest

from punchin.agent import ScriptedAgent
from punchin.customer import ScriptedCustomer
from punchin.fidelity import FIELDS, Ablated, Repeated, ablation, repeated, without
from punchin.model import ClaudeCodeModel
from punchin.record import record
from punchin.scenarios import BY_ID

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"
SCENARIO = BY_ID["self-correction"]


@pytest.fixture
def recorded(tmp_path: Path):
    return record(SCENARIO, ScriptedAgent(careful=True), ScriptedCustomer(SCENARIO), tmp_path / "s.json")


def _model() -> ClaudeCodeModel:
    return ClaudeCodeModel([sys.executable, str(FAKE)])


def test_a_repeat_runs_the_measurement_again_and_keeps_every_score(recorded) -> None:
    found = repeated(recorded, SCENARIO.goal, _model(), 3)
    assert len(found.runs) == 3
    assert len(found.scores) == 3
    assert found.mean == pytest.approx(sum(found.scores) / 3)
    assert "spread" in found.one_line()


def test_a_deterministic_simulator_has_no_spread(recorded) -> None:
    """The fake answers by rule, so its spread is zero. A sampled model's will not be."""
    assert repeated(recorded, SCENARIO.goal, _model(), 3).spread == 0.0


def test_one_run_reports_no_spread_rather_than_pretending_to_know(recorded) -> None:
    assert repeated(recorded, SCENARIO.goal, _model(), 1).spread == 0.0


def test_the_spread_of_an_arm_is_still_reported_for_context() -> None:
    """The means are not the verdict any more, but the spread is what says why they cannot be."""

    class FakeRun:
        def __init__(self, score: float) -> None:
            self.mean_jaccard = score
            self.cost_usd = 0.0
            self.turns: list[object] = []

    def arm(*scores: float) -> Repeated:
        found = Repeated("c")
        found.runs = [FakeRun(s) for s in scores]  # type: ignore[list-item]
        return found

    noisy = Ablated("c", arm(0.60, 0.80), {"mood": arm(0.70, 0.70)})
    assert noisy.noise == pytest.approx(0.20)
    printed = noisy.text()
    assert "0.20" in printed  # the spread that makes a comparison of means useless
    assert "not distinguishable from zero" in printed  # no turns, so nothing to pair


def test_the_ablation_measures_every_field_and_the_full_goal(recorded) -> None:
    found = ablation(recorded, SCENARIO.goal, _model(), ("mood", "manner"), times=2)
    assert set(found.dropped) == {"mood", "manner"}
    assert len(found.full.runs) == 2
    assert all(len(arm.runs) == 2 for arm in found.dropped.values())
    printed = found.text()
    assert "everything" in printed
    assert "paired by turn" in printed.lower()
    assert "± s.e." in printed


def test_withholding_a_field_really_withholds_it() -> None:
    goal = BY_ID["courtesy-car"].goal
    assert goal.constraints and goal.extras
    assert without(goal, "constraints").constraints == []
    assert without(goal, "extras").extras == []
    assert without(goal, "mood").mood == "neutral"
    assert without(goal, "prefers_time").prefers_time is None
    assert without(goal, "manner").manner == []
    # and changes nothing else
    assert without(goal, "mood").reg == goal.reg


def test_every_ablated_field_is_a_real_field_of_the_goal_state() -> None:
    from punchin.scenarios import GoalState

    assert set(FIELDS) <= set(GoalState.model_fields)


def test_pairing_by_turn_cancels_the_difficulty_that_swamps_the_means() -> None:
    """Two arms whose means are identical can still differ on every turn, and pairing finds it."""
    from punchin.fidelity import Repeated, paired

    class FakeTurn:
        def __init__(self, index: int, score: float) -> None:
            self.index, self.jaccard = index, score

    class FakeRun:
        def __init__(self, scores: dict[int, float]) -> None:
            self.turns = [FakeTurn(i, s) for i, s in scores.items()]
            self.mean_jaccard = sum(scores.values()) / len(scores)
            self.cost_usd = 0.0

    def arm(*runs: dict[int, float]) -> Repeated:
        found = Repeated("c")
        found.runs = [FakeRun(r) for r in runs]  # type: ignore[list-item]
        return found

    # Every turn is 0.10 better with the field, on turns of wildly different difficulty.
    full = arm({1: 0.90, 3: 0.30, 5: 0.60}, {1: 0.95, 3: 0.35, 5: 0.65})
    dropped = arm({1: 0.80, 3: 0.20, 5: 0.50}, {1: 0.85, 3: 0.25, 5: 0.55})
    middle, error, count = paired(full, dropped)
    assert middle == pytest.approx(0.10)
    assert count == 6
    assert error < 0.01  # the difficulty cancelled, so the difference is clean
    assert abs(middle) > 2 * error  # and therefore visible


def test_a_field_that_changes_nothing_is_not_distinguishable_from_zero() -> None:
    from punchin.fidelity import Ablated, Repeated

    class FakeTurn:
        def __init__(self, index: int, score: float) -> None:
            self.index, self.jaccard = index, score

    class FakeRun:
        def __init__(self, scores: dict[int, float]) -> None:
            self.turns = [FakeTurn(i, s) for i, s in scores.items()]
            self.mean_jaccard = sum(scores.values()) / len(scores)
            self.cost_usd = 0.0

    def arm(*runs: dict[int, float]) -> Repeated:
        found = Repeated("c")
        found.runs = [FakeRun(r) for r in runs]  # type: ignore[list-item]
        return found

    same = {1: 0.9, 3: 0.3, 5: 0.6}
    noisy = {1: 0.4, 3: 0.9, 5: 0.1}
    found = Ablated("c", arm(same, noisy), {"mood": arm(same, noisy)})
    middle, _, _ = found.worth("mood")
    assert middle == pytest.approx(0.0)
    assert "not distinguishable from zero" in found.text()
    assert "paired" in found.text().lower()


def test_the_corpus_number_is_pooled_by_turn_not_averaged_over_calls() -> None:
    """A two-turn call and a ten-turn call must not weigh the same: one measured five times as much."""
    from punchin.fidelity import across

    class FakeTurn:
        def __init__(self, score: float) -> None:
            self.jaccard, self.exact = score, score == 1.0

    class FakeReport:
        def __init__(self, scores: list[float]) -> None:
            self.turns = [FakeTurn(s) for s in scores]
            self.mean_jaccard = sum(scores) / len(scores)
            self.exact_rate = sum(s == 1.0 for s in scores) / len(scores)
            self.cost_usd = 0.0

    short = FakeReport([0.0, 0.0])  # a two-turn call that scored badly
    long = FakeReport([1.0] * 10)  # a ten-turn call that scored perfectly
    printed = across([short, long])  # type: ignore[list-item]

    assert "12 customer turns" in printed
    assert "jaccard 0.83" in printed  # pooled: 10 of 12 turns were right
    assert "mean 0.50" in printed  # per call: the two calls averaged, which is the misleading one
    assert "coin flip" in printed  # and it says why


def test_a_corpus_of_long_calls_gets_no_warning() -> None:
    from punchin.fidelity import across

    class FakeTurn:
        def __init__(self) -> None:
            self.jaccard, self.exact = 0.8, False

    class FakeReport:
        def __init__(self) -> None:
            self.turns = [FakeTurn() for _ in range(8)]
            self.mean_jaccard, self.exact_rate, self.cost_usd = 0.8, 0.0, 0.0

    assert "coin flip" not in across([FakeReport(), FakeReport()])  # type: ignore[list-item]
