"""One change, across many recordings. Fixing four and breaking one is not an improvement."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from punchin.agent import ScriptedAgent
from punchin.call import Call
from punchin.cli import main
from punchin.customer import ScriptedCustomer
from punchin.fork import ForkReport, Sweep, fork_point, verdict_of
from punchin.record import record
from punchin.scenarios import BY_ID, SCENARIOS

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"


def _stand_in():
    """The fake Claude Code binary, which the pinned customer talks to for nothing."""
    from punchin.model import ClaudeCodeModel

    return ClaudeCodeModel([sys.executable, str(FAKE)])


@pytest.fixture
def corpus(tmp_path: Path) -> list[Call]:
    state = tmp_path / "s.json"
    return [record(s, ScriptedAgent(careful=True), ScriptedCustomer(s), state, tmp_path) for s in SCENARIOS]


def test_a_fork_point_means_the_same_thing_in_every_call(corpus: list[Call]) -> None:
    """Turn 6 is a different moment in each recording; `half` is the same instruction everywhere."""
    for call in corpus:
        points = [t.index for t in call.turns if t.speaker == "agent"]
        assert fork_point(call, "first") == points[0]
        assert fork_point(call, "last") == points[-1]
        assert fork_point(call, "half") in points
        assert fork_point(call, str(points[0])) == points[0]


def test_an_impossible_fork_point_is_refused(corpus: list[Call]) -> None:
    with pytest.raises(ValueError, match="not an agent turn"):
        fork_point(corpus[0], "3")
    with pytest.raises(ValueError, match="first, half or last"):
        fork_point(corpus[0], "middle-ish")


def _report(scenario_id: str, was: bool, now: int, attempts: int) -> ForkReport:
    """A report with a known before and after, without running anything."""
    scenario = BY_ID[scenario_id]

    class Fake(ForkReport):
        @property
        def before(self):  # type: ignore[override]
            return {"correct": was, "options_max": 0, "turns": 0, "agent_words_max": 0, "customer_stalls": 0}

        @property
        def after(self):  # type: ignore[override]
            return [
                {"correct": i < now, "options_max": 0, "turns": 0, "agent_words_max": 0, "customer_stalls": 0}
                for i in range(attempts)
            ]

    return Fake(
        Call(
            id="c",
            scenario=scenario_id,
            agent="a",
            customer="c",
            started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        ),
        scenario,
        0,
        "x",
    )


@pytest.mark.parametrize(
    ("was", "now", "attempts", "expected"),
    [
        (False, 3, 3, "fixed"),
        (False, 2, 3, "fixed"),  # a majority is enough, because agents are sampled
        (False, 1, 3, "still wrong"),
        (False, 0, 3, "still wrong"),
        (True, 3, 3, "held"),
        (True, 1, 3, "broke"),
        (True, 0, 3, "broke"),
    ],
)
def test_what_the_change_did_to_one_call(was: bool, now: int, attempts: int, expected: str) -> None:
    assert verdict_of(_report("plain-booking", was, now, attempts)) == expected


def test_breaking_one_call_outranks_fixing_four() -> None:
    swept = Sweep("a prompt change")
    swept.reports = [
        _report("plain-booking", False, 3, 3),
        _report("self-correction", False, 3, 3),
        _report("next-week", False, 3, 3),
        _report("hurried", False, 3, 3),
        _report("courtesy-car", True, 0, 3),
    ]
    found = swept.verdicts
    assert len(found["fixed"]) == 4
    assert found["broke"] == ["courtesy-car"]

    printed = swept.text()
    assert "breaks 1 call" in printed
    assert "is not the number to look at" in printed


def test_a_change_that_breaks_nothing_says_so() -> None:
    swept = Sweep("x")
    swept.reports = [_report("plain-booking", False, 3, 3), _report("hurried", True, 3, 3)]
    assert "Fixes 1, breaks nothing." in swept.text()


def test_a_change_that_does_nothing_says_that_too() -> None:
    swept = Sweep("x")
    swept.reports = [_report("plain-booking", True, 3, 3)]
    assert "Changes nothing that was measured." in swept.text()


def test_the_command_sweeps_many_calls_and_fails_when_it_breaks_one(
    corpus: list[Call], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The careless agent, forked after the customer has said the thing it gets wrong."""
    from unittest.mock import patch

    from punchin.model import ClaudeCodeModel

    paths = [str(tmp_path / f"{call.id}.json") for call in corpus[:3]]
    stand_in = ClaudeCodeModel([sys.executable, str(FAKE)])
    with patch("punchin.commands.model_for", return_value=stand_in):
        code = main(
            [
                "fork",
                *paths,
                "--at",
                "6",
                "--agent",
                "careless",
                "--goal",
                "truth",
                "--out",
                str(tmp_path / "forks"),
                "--max-usd",
                "1",
                "-q",
            ]
        )
    printed = capsys.readouterr().out
    assert "across 3 calls" in printed
    assert code == 1  # something that was right before is now wrong
    assert "breaks" in printed


def test_a_sweep_spends_nothing_it_was_not_given(
    corpus: list[Call], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A sweep over an archive must stop at its allowance, not after it."""
    from unittest.mock import patch

    from punchin.model import ClaudeCodeModel

    paths = [str(tmp_path / f"{call.id}.json") for call in corpus]
    stand_in = ClaudeCodeModel([sys.executable, str(FAKE)])
    with patch("punchin.commands.model_for", return_value=stand_in):
        main(
            [
                "fork",
                *paths,
                "--at",
                "last",
                "--agent",
                "careful",
                "--goal",
                "truth",
                "--json",
                "--out",
                str(tmp_path / "f2"),
                "--max-usd",
                "0.001",
                "-q",
            ]
        )
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["calls"] < len(corpus)
    assert "budget" in (parsed["stopped"] or "")


def test_a_sweep_reports_as_json(corpus: list[Call], tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    paths = [str(tmp_path / f"{call.id}.json") for call in corpus[:2]]
    with patch("punchin.commands.model_for", return_value=_stand_in()):
        main(
            [
                "fork",
                *paths,
                "--at",
                "last",
                "--agent",
                "careful",
                "--goal",
                "truth",
                "--json",
                "--out",
                str(tmp_path / "forks"),
                "--max-usd",
                "1",
                "-q",
            ]
        )
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["calls"] == 2
    assert set(parsed["verdicts"]) == {"fixed", "broke", "still wrong", "held"}
    assert "live_cost_usd" in parsed


def test_one_call_still_gets_the_single_call_report(
    corpus: list[Call], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    with patch("punchin.commands.model_for", return_value=_stand_in()):
        main(
            [
                "fork",
                str(tmp_path / f"{corpus[0].id}.json"),
                "--at",
                "last",
                "--agent",
                "careful",
                "--goal",
                "truth",
                "--out",
                str(tmp_path / "forks"),
                "--max-usd",
                "1",
                "-q",
            ]
        )
    printed = capsys.readouterr().out
    assert "fork of" in printed
    assert "across" not in printed


def test_a_change_splits_where_a_person_would_split_it() -> None:
    from punchin.fork import sentences

    assert sentences("Læs dagen tilbage. Gæt aldrig. Book efter et ja.") == [
        "Læs dagen tilbage.",
        "Gæt aldrig.",
        "Book efter et ja.",
    ]
    assert sentences("Tilbyd én tid ad gangen.") == []  # nothing to take out of one sentence
    assert sentences("  ") == []


def test_a_sentence_whose_removal_costs_nothing_was_not_carrying_the_fix() -> None:
    from punchin.fork import Ingredient, Recipe

    class R:
        def __init__(self, passed: int) -> None:
            self.fixed, self.live_cost_usd = passed, 0.0

        @property
        def attempts(self) -> list[None]:
            return [None] * 3

    whole = R(3)
    found = Recipe("A. B.", whole)  # type: ignore[arg-type]
    found.without = [
        Ingredient("A.", R(0)),  # type: ignore[arg-type]
        Ingredient("B.", R(3)),  # type: ignore[arg-type]
    ]
    printed = found.text()
    assert "without A." in printed and "carries it" in printed
    assert "without B." in printed and "does nothing here" in printed
    assert "1 sentence(s) could come out" in printed


def test_a_change_where_every_sentence_matters_says_so() -> None:
    from punchin.fork import Ingredient, Recipe

    class R:
        def __init__(self, passed: int) -> None:
            self.fixed, self.live_cost_usd = passed, 0.0

        @property
        def attempts(self) -> list[None]:
            return [None] * 3

    found = Recipe("A. B.", R(3))  # type: ignore[arg-type]
    found.without = [Ingredient("A.", R(1)), Ingredient("B.", R(0))]  # type: ignore[arg-type]
    assert "every sentence is doing something" in found.text()


def test_ablating_a_single_sentence_change_is_refused(corpus: list[Call], tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="more than one sentence"):
        main(
            [
                "fork",
                str(tmp_path / f"{corpus[0].id}.json"),
                "--at",
                "last",
                "--ablate-change",
                "--system-suffix",
                "Tilbyd én tid ad gangen.",
                "--out",
                str(tmp_path / "ab"),
                "-q",
            ]
        )


def test_a_change_that_only_breaks_does_not_claim_fixing_zero_is_the_wrong_number() -> None:
    """'Fixing 0 is not the number to look at' is a sentence no one should have to read."""
    sweep = Sweep(
        "system suffix 'x'",
        [_report("hurried", True, 0, 1), _report("plain-booking", True, 1, 1)],
    )
    printed = sweep.text()
    assert "breaks 1 call(s) that were right before, and fixes none." in printed
    assert "Fixing 0" not in printed


def test_a_long_list_of_calls_says_how_many_it_left_out() -> None:
    """Six names printed against a count of nine reads as a contradiction."""
    ids = [s.id for s in SCENARIOS][:9]
    held = [_report(name, True, 1, 1) for name in ids]
    printed = Sweep("system suffix 'x'", held).text()
    assert "held           9" in printed
    assert "and 3 more" in printed
