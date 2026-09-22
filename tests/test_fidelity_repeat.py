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


def test_a_delta_under_the_noise_is_not_reported_as_a_delta() -> None:
    """The whole point: a number that cannot beat its own sampler is not a finding."""

    class FakeRun:
        def __init__(self, score: float) -> None:
            self.mean_jaccard = score
            self.cost_usd = 0.0

    def arm(*scores: float) -> Repeated:
        found = Repeated("c")
        found.runs = [FakeRun(s) for s in scores]  # type: ignore[list-item]
        return found

    noisy = Ablated("c", arm(0.60, 0.80), {"mood": arm(0.70, 0.70)})
    assert noisy.noise == pytest.approx(0.20)
    assert "under the noise" in noisy.text()  # a 0.00 delta against 0.20 of noise

    quiet = Ablated("c", arm(0.80, 0.80), {"mood": arm(0.50, 0.50)})
    assert quiet.noise == 0.0
    assert "+0.30" in quiet.text()


def test_the_ablation_measures_every_field_and_the_full_goal(recorded) -> None:
    found = ablation(recorded, SCENARIO.goal, _model(), ("mood", "manner"), times=2)
    assert set(found.dropped) == {"mood", "manner"}
    assert len(found.full.runs) == 2
    assert all(len(arm.runs) == 2 for arm in found.dropped.values())
    printed = found.text()
    assert "everything" in printed
    assert "noise floor" in printed


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
