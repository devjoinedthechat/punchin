"""Does a fork tell the truth? The three arms, exercised with the scripted pair and a fake model."""

import sys
from pathlib import Path
from unittest.mock import patch

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


def test_a_sweep_says_whether_a_gap_is_the_tool_or_one_scenario(tmp_path: Path) -> None:
    """One scenario cannot tell you much. A gap on most of them is the tool; on one it is that one."""
    from punchin.soundness import summarise_soundness

    found = [_measure(tmp_path), _measure(tmp_path)]
    printed = summarise_soundness(found)
    assert "2 scenarios" in printed
    assert "the fork mechanism moved the answer by" in printed
    assert "the pinned customer moved it by" in printed
    assert "agreed on every scenario" in printed


def test_a_sweep_names_the_scenarios_that_disagreed() -> None:
    from punchin.soundness import Arm, Soundness, summarise_soundness

    def rigged(passes_live: int, passes_fork: int) -> Soundness:
        live, scripted, pinned = (Arm(n, SCENARIO) for n in ("a", "b", "c"))
        live.calls = [_fake(True)] * passes_live + [_fake(False)] * (2 - passes_live)
        scripted.calls = [_fake(True)] * passes_live + [_fake(False)] * (2 - passes_live)
        pinned.calls = [_fake(True)] * passes_fork + [_fake(False)] * (2 - passes_fork)
        return Soundness(SCENARIO, 4, "x", _fake(False), live, scripted, pinned)

    optimistic = rigged(passes_live=1, passes_fork=2)
    assert optimistic.simulator_gap > 0
    assert SCENARIO.id in summarise_soundness([optimistic])


def _fake(correct: bool):
    """A call whose booking makes `outcome` say correct or not."""
    import datetime as when

    from punchin.call import Call

    call = Call(id="x", scenario=SCENARIO.id, agent="a", customer="c", started_at=when.datetime.now(when.UTC))
    if correct:
        call.bookings = [
            {
                "reg": SCENARIO.expected.reg,
                "date": SCENARIO.expected.day.isoformat(),
                "time": "08:00",
                "note": "",
            }
        ]
    return call


def test_a_run_where_every_arm_sat_at_its_ceiling_says_so(tmp_path: Path) -> None:
    """Agreement with no chance of disagreement is not evidence, and the report admits it."""
    from punchin.soundness import summarise_soundness

    assert "ceiling" in summarise_soundness([_measure(tmp_path)])


BASELINE_AGENT = """
import json, sys
json.load(sys.stdin)
print(json.dumps({"text": "Jeg booker bare noget. [FARVEL]"}))
"""
CHANGED_AGENT = """
import json, sys
json.load(sys.stdin)
print(json.dumps({"text": "Hvilken dag passer dig? [FARVEL]"}))
"""


def test_soundness_can_be_run_on_an_agent_punchin_does_not_own(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The check that makes the tool credible has to work on your agent, not only on punchin's."""
    from punchin.cli import main

    before, after = tmp_path / "before.py", tmp_path / "after.py"
    before.write_text(BASELINE_AGENT)
    after.write_text(CHANGED_AGENT)

    code = main(
        [
            "soundness",
            "--scenario",
            "self-correction",
            "--trials",
            "1",
            "--at",
            "0",
            "--agent",
            "command",
            "--baseline-command",
            f"{sys.executable} {before}",
            "--agent-command",
            f"{sys.executable} {after}",
            "--out",
            str(tmp_path / "out"),
            "-q",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "ground truth" in printed
    assert "what punchin does" in printed
    assert str(after) in printed  # the change under test is named


def test_a_missing_flag_is_reported_before_anything_is_built(tmp_path: Path) -> None:
    """Building the model first made a missing flag report a missing `claude` binary: a true sentence
    about the wrong problem, and only on a machine without Claude Code — never the one it was written on.
    """
    from punchin.cli import main

    with patch("punchin.model.find_claude", return_value=None):
        with pytest.raises(SystemExit, match="needs both"):
            main(
                [
                    "soundness",
                    "--scenario",
                    "self-correction",
                    "--agent",
                    "command",
                    "--agent-command",
                    "echo",
                    "--out",
                    str(tmp_path),
                    "-q",
                ]
            )
        with pytest.raises(SystemExit, match="--system-suffix"):
            main(["soundness", "--scenario", "self-correction", "--out", str(tmp_path), "-q"])
