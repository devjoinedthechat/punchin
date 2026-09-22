"""The regression gate: what counts as worse, and what does not."""

import json
from pathlib import Path

import pytest

from punchin.agent import ScriptedAgent
from punchin.check import FORMAT, Rule, baseline_from, compare, load_baseline, report
from punchin.cli import main
from punchin.customer import ScriptedCustomer
from punchin.metrics import summarize
from punchin.record import record
from punchin.scenarios import SCENARIOS


def _rows(tmp_path: Path, *, careful: bool) -> list[dict]:
    state = tmp_path / "s.json"
    return [
        summarize(record(s, ScriptedAgent(careful=careful), ScriptedCustomer(s), state), s) for s in SCENARIOS
    ]


def test_a_run_is_not_a_regression_against_itself(tmp_path: Path) -> None:
    rows = _rows(tmp_path, careful=True)
    assert compare(rows, baseline_from(rows)) == []
    assert "no regressions across 10 scenarios" in report([], 10)


def test_the_careless_agent_regresses_against_the_careful_baseline(tmp_path: Path) -> None:
    baseline = baseline_from(_rows(tmp_path, careful=True))
    found = compare(_rows(tmp_path, careful=False), baseline)
    broke = {(f.scenario, f.detail.split(":")[0]) for f in found}
    assert ("self-correction", "correct") in broke  # booked the day she took back
    assert ("next-week", "day_ok") in broke  # booked this week, not next
    assert ("courtesy-car", "note_ok") in broke  # dropped the courtesy car
    assert ("wrong-reg-first", "agent_repeats") in broke  # looped on the wrong plate
    assert len(found) > 10


def test_slack_lets_a_call_wander_a_turn_or_two_without_failing() -> None:
    rule = Rule("turns", "higher_is_worse", slack=2)
    assert rule.broken(10, 12) is None
    assert rule.broken(10, 13) == "turns: 10 -> 13"
    assert rule.broken(10, 8) is None  # shorter is not worse


def test_a_metric_missing_from_either_side_is_not_a_regression() -> None:
    """Text runs have no `reg_survived`. Comparing them to an audio baseline must not invent failures."""
    rule = Rule("reg_survived", "must_stay_true")
    assert rule.broken(None, False) is None
    assert rule.broken(True, None) is None
    assert rule.broken(True, False) == "reg_survived: was true, now false"
    assert rule.broken(False, True) is None  # getting better is allowed


def test_a_scenario_that_vanished_from_the_run_is_a_regression(tmp_path: Path) -> None:
    rows = _rows(tmp_path, careful=True)
    found = compare(rows[:-1], baseline_from(rows))
    assert [f.detail for f in found] == ["in the baseline, not in this run"]


def test_a_new_scenario_has_nothing_to_be_worse_than(tmp_path: Path) -> None:
    rows = _rows(tmp_path, careful=True)
    assert compare(rows, baseline_from(rows[:3])) == []


def test_a_baseline_from_another_version_is_refused(tmp_path: Path) -> None:
    stale = tmp_path / "baseline.json"
    stale.write_text(json.dumps({"format": FORMAT + 1, "scenarios": {}}))
    with pytest.raises(ValueError, match="different version"):
        load_baseline(stale)


def test_the_command_writes_a_baseline_then_gates_on_it(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    calls = tmp_path / "calls"
    baseline = tmp_path / "baseline.json"
    assert main(["record", "--agent", "careful", "--out", str(calls), "-q"]) == 0
    recorded = [str(p) for p in sorted(calls.glob("2026*.json"))]

    assert main(["check", *recorded, "--baseline", str(baseline), "--update", "-q"]) == 0
    assert baseline.exists()
    assert main(["check", *recorded, "--baseline", str(baseline), "-q"]) == 0

    worse = tmp_path / "worse"
    assert main(["record", "--agent", "careless", "--out", str(worse), "-q"]) == 0
    regressed = [str(p) for p in sorted(worse.glob("2026*.json"))]
    assert main(["check", *regressed, "--baseline", str(baseline), "-q"]) == 1
    assert "regressions across 10 scenarios" in capsys.readouterr().out


def test_checking_without_a_baseline_says_how_to_make_one(tmp_path: Path) -> None:
    calls = tmp_path / "calls"
    main(["record", "--agent", "careful", "--scenario", "plain-booking", "--out", str(calls), "-q"])
    recorded = [str(p) for p in calls.glob("2026*.json")]
    with pytest.raises(SystemExit, match="--update"):
        main(["check", *recorded, "--baseline", str(tmp_path / "nope.json"), "-q"])
