"""Grouping the calls that went wrong, so the work is ordered by how many customers a fix reaches."""

from pathlib import Path

import pytest

from punchin.agent import ScriptedAgent
from punchin.cli import main
from punchin.customer import ScriptedCustomer
from punchin.record import record
from punchin.scenarios import SCENARIOS, load_scenarios
from punchin.triage import SYMPTOMS, symptom_of, triage


def _corpus(tmp_path: Path, *, careful: bool) -> list:
    state = tmp_path / "s.json"
    return [record(s, ScriptedAgent(careful=careful), ScriptedCustomer(s), state) for s in SCENARIOS]


def test_a_corpus_that_went_right_has_nothing_to_triage(tmp_path: Path) -> None:
    found = triage(_corpus(tmp_path, careful=True), load_scenarios(None))
    assert found.clean == 10
    assert found.clusters == []
    assert "nothing to fix" in found.text()


def test_the_careless_corpus_groups_by_what_went_wrong(tmp_path: Path) -> None:
    found = triage(_corpus(tmp_path, careful=False), load_scenarios(None))
    assert found.total == 10
    assert found.clean < 10
    symptoms = [c.symptom for c in found.clusters]
    assert "wrong day booked" in symptoms
    # biggest group first, so the fix that reaches the most customers is read first
    sizes = [len(c.rows) for c in found.clusters]
    assert sizes == sorted(sizes, reverse=True)
    assert all(c.why for c in found.clusters)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"reg_heard": False}, "plate never arrived"),
        ({"booked": True, "day_ok": False, "should_book": True}, "wrong day booked"),
        ({"booked": True, "day_ok": True, "should_book": False}, "booked when it should not have"),
        ({"booked": False, "should_book": True}, "never booked"),
        ({"booked": True, "day_ok": True, "should_book": True, "note_ok": False}, "note lost"),
        ({"booked": True, "day_ok": True, "should_book": True, "agent_repeats": 5}, "agent repeated itself"),
        ({"booked": True, "day_ok": True, "should_book": True, "ended_by": "customer"}, "customer gave up"),
        ({"booked": True, "day_ok": True, "should_book": True, "options_max": 5}, "read out a menu"),
        ({"booked": True, "day_ok": True, "should_book": True}, None),
    ],
)
def test_each_symptom_is_recognised(row: dict, expected: str | None) -> None:
    assert symptom_of(row) == expected


def test_a_call_with_several_faults_is_filed_under_the_one_to_fix_first() -> None:
    """A lost plate causes the rest, so a call showing both is a plate problem, counted once."""
    both = {
        "reg_heard": False,
        "agent_repeats": 5,
        "ended_by": "customer",
        "booked": False,
        "should_book": True,
    }
    assert symptom_of(both) == "plate never arrived"
    assert next(name for name, _ in SYMPTOMS) == "plate never arrived"


def test_the_cheapest_call_in_a_group_is_the_one_to_fork() -> None:
    rows = [
        {"scenario": "a", "call": "long", "turns": 20, "reg_heard": False},
        {"scenario": "b", "call": "short", "turns": 6, "reg_heard": False},
    ]
    from punchin.triage import Cluster

    assert Cluster("s", "w", rows).example() == "short"


def test_the_command_exits_nonzero_when_there_is_something_to_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    good, bad = tmp_path / "good", tmp_path / "bad"
    assert main(["record", "--agent", "careful", "--out", str(good), "-q"]) == 0
    assert main(["record", "--agent", "careless", "--out", str(bad), "-q"]) == 0
    capsys.readouterr()

    assert main(["triage", *[str(p) for p in good.glob("2026*.json")], "-q"]) == 0
    assert "nothing to fix" in capsys.readouterr().out

    assert main(["triage", *[str(p) for p in bad.glob("2026*.json")], "-q"]) == 1
    printed = capsys.readouterr().out
    assert "wrong day booked" in printed
    assert "fork this one first" in printed


def test_json_output_is_one_object(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import json

    bad = tmp_path / "bad"
    main(["record", "--agent", "careless", "--out", str(bad), "-q"])
    capsys.readouterr()
    main(["triage", *[str(p) for p in bad.glob("2026*.json")], "--json", "-q"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["total"] == 10
    assert parsed["clusters"][0]["calls"] >= parsed["clusters"][-1]["calls"]
    assert "fork_first" in parsed["clusters"][0]
