"""Does a fork tell the truth? The three arms, exercised with the scripted pair and a fake model."""

import sys
from pathlib import Path

import pytest

from punchin.agent import ScriptedAgent
from punchin.fork import Budget
from punchin.model import ClaudeCodeModel
from punchin.scenarios import BY_ID
from punchin.soundness import measure, wilson

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"
SCENARIO = BY_ID["self-correction"]


def _measure(tmp_path: Path, *, at: int = 4, **kwargs: object):
    return measure(
        SCENARIO,
        agent_with_change=ScriptedAgent(careful=True),
        agent_without=ScriptedAgent(careful=False),
        goal=SCENARIO.goal,
        model=ClaudeCodeModel([sys.executable, str(FAKE)]),
        state_path=tmp_path / "s.json",
        at=at,
        trials=2,
        change="careful instead of careless",
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_three_arms_all_run_and_the_baseline_is_the_broken_one(tmp_path: Path) -> None:
    found = _measure(tmp_path)
    assert len(found.live.calls) == 2
    assert len(found.fork_scripted.calls) == 2
    assert len(found.fork_pinned.calls) == 2
    # the recording being forked was made by the careless agent, which books the day she took back
    assert found.live.scenario is SCENARIO
    assert found.fork_scripted.rate == 1.0  # the change fixes it, replayed prefix and all
    assert found.live.rate == 1.0  # and fixes it from the first turn too


def test_the_mechanism_and_the_simulator_are_reported_apart(tmp_path: Path) -> None:
    """One is a bug in the tool, the other a limit of simulation. They need different fixes."""
    found = _measure(tmp_path)
    assert found.mechanism_gap == found.fork_scripted.rate - found.live.rate
    assert found.simulator_gap == found.fork_pinned.rate - found.fork_scripted.rate
    assert found.total_gap == pytest.approx(found.mechanism_gap + found.simulator_gap)

    printed = found.text()
    assert "ground truth" in printed
    assert "what punchin does" in printed
    assert "the fork mechanism moves the answer by" in printed
    assert "the pinned customer moves it by" in printed


def test_a_forked_prefix_does_not_change_the_answer_here(tmp_path: Path) -> None:
    """The claim the whole tool rests on, on the one scenario a test can afford to check."""
    found = _measure(tmp_path)
    assert found.mechanism_gap == 0.0, found.text()


def test_the_scripted_customer_resumes_mid_call_rather_than_starting_again(tmp_path: Path) -> None:
    found = _measure(tmp_path)
    forked = found.fork_scripted.calls[0]
    said = [t.spoken for t in forked.turns if t.speaker == "customer"]
    assert len(said) == len(set(said)), said  # no line said twice
    assert said[0] == SCENARIO.script[0].text  # the prefix still holds the opening
    assert forked.notes["forked_at"] == 4


def test_the_report_carries_the_numbers_as_json(tmp_path: Path) -> None:
    found = _measure(tmp_path).as_dict()
    assert found["scenario"] == "self-correction"
    assert found["at"] == 4
    assert set(found["arms"]) == {"live, full re-run", "fork, scripted customer", "fork, pinned customer"}
    assert found["baseline_correct"] is False
    assert "change" in found


def test_a_budget_stops_the_trials(tmp_path: Path) -> None:
    found = _measure(tmp_path, budget=Budget(0.0001))
    assert len(found.live.calls) <= 1


def test_a_scenario_with_no_script_cannot_be_checked(tmp_path: Path) -> None:
    """An imported call has no script, so there is no live re-run to compare a fork against."""
    bare = SCENARIO.model_copy(update={"script": [], "id": "imported-1"})
    with pytest.raises(ValueError, match="no script"):
        measure(
            bare,
            agent_with_change=ScriptedAgent(careful=True),
            agent_without=ScriptedAgent(careful=False),
            goal=bare.goal,
            model=ClaudeCodeModel([sys.executable, str(FAKE)]),
            state_path=tmp_path / "s.json",
        )


def test_a_fork_point_that_is_not_an_agent_turn_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an agent turn"):
        _measure(tmp_path, at=3)


@pytest.mark.parametrize(
    ("passed", "total", "low", "high"),
    [(0, 0, 0.0, 1.0), (3, 3, 0.44, 1.0), (0, 3, 0.0, 0.56), (1, 2, 0.09, 0.91)],
)
def test_the_interval_stays_honest_at_a_handful_of_trials(
    passed: int, total: int, low: float, high: float
) -> None:
    got_low, got_high = wilson(passed, total)
    assert got_low == pytest.approx(low, abs=0.02)
    assert got_high == pytest.approx(high, abs=0.02)
