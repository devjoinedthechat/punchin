"""The regression gate: what counts as worse, and why one sample of a sampled agent is not enough."""

import json
from pathlib import Path

import pytest

from punchin.agent import ScriptedAgent
from punchin.check import FORMAT, Rule, baseline_from, by_scenario, check, compare, flaky, load_baseline
from punchin.cli import main
from punchin.customer import ScriptedCustomer
from punchin.metrics import summarize
from punchin.record import record
from punchin.scenarios import SCENARIOS


def _rows(tmp_path: Path, *, careful: bool, runs: int = 1) -> list[dict]:
    state = tmp_path / "s.json"
    return [
        summarize(record(s, ScriptedAgent(careful=careful), ScriptedCustomer(s), state), s)
        for _ in range(runs)
        for s in SCENARIOS
    ]


def test_a_run_is_not_a_regression_against_itself(tmp_path: Path) -> None:
    rows = _rows(tmp_path, careful=True)
    assert compare(rows, baseline_from(rows)) == []
    assert "no regressions" in check(rows, baseline_from(rows)).text()


def test_the_careless_agent_regresses_against_the_careful_baseline(tmp_path: Path) -> None:
    baseline = baseline_from(_rows(tmp_path, careful=True))
    found = compare(_rows(tmp_path, careful=False), baseline)
    broke = {(f.scenario, f.detail.split(":")[0]) for f in found}
    assert ("self-correction", "correct") in broke  # booked the day she took back
    assert ("next-week", "day_ok") in broke  # booked this week, not next
    assert ("courtesy-car", "note_ok") in broke  # dropped the courtesy car
    assert ("wrong-reg-first", "agent_repeats") in broke  # looped on the wrong plate


def test_an_outcome_is_a_rate_so_getting_worse_more_often_is_a_regression() -> None:
    """A scenario that used to pass every run and now passes most of them has still regressed."""
    always = Rule("correct", "must_stay_true")
    assert always.broken(1.0, 1.0) is None
    assert always.broken(1.0, 0.67) == "correct: 100% -> 67% of runs"
    assert always.broken(0.5, 0.5) is None
    assert always.broken(0.5, 1.0) is None  # better is allowed
    assert always.broken(0.67, 1.0) is None


def test_slack_lets_a_call_wander_a_turn_or_two_without_failing() -> None:
    rule = Rule("turns", "higher_is_worse", slack=2)
    assert rule.broken(10, 12) is None
    assert rule.broken(10, 13) == "turns: 10 -> 13"
    assert rule.broken(10, 8) is None  # shorter is not worse


def test_a_metric_missing_from_either_side_is_not_a_regression() -> None:
    """Text runs have no `reg_survived`. Comparing them to an audio baseline must not invent failures."""
    rule = Rule("reg_survived", "must_stay_true")
    assert rule.broken(None, 0.0) is None
    assert rule.broken(1.0, None) is None
    assert rule.broken(1.0, 0.0) == "reg_survived: 100% -> 0% of runs"


def test_several_runs_of_one_scenario_become_one_rate(tmp_path: Path) -> None:
    rows = _rows(tmp_path, careful=True, runs=3)
    assert len(rows) == 30
    assert len(by_scenario(rows)) == 10
    stored = baseline_from(rows)
    assert stored["scenarios"]["plain-booking"]["trials"] == 3
    assert stored["scenarios"]["plain-booking"]["correct"] == 1.0


def test_a_scenario_that_disagrees_with_itself_is_flaky_not_failing() -> None:
    """The outcome a customer gets depending on the sampler is a finding, not a nuisance."""
    rows = [
        {"scenario": "a", "correct": True},
        {"scenario": "a", "correct": False},
        {"scenario": "a", "correct": True},
        {"scenario": "b", "correct": True},
        {"scenario": "b", "correct": True},
    ]
    found = flaky(rows)
    assert [f.scenario for f in found] == ["a"]
    assert (found[0].passed, found[0].trials) == (2, 3)
    assert "2 of 3" in str(found[0])

    checked = check(rows, {"scenarios": {}})
    assert not checked.failed  # flaky is reported, not failed on, without a baseline to fall short of
    assert "disagreed with themselves" in checked.text()


def test_a_single_run_is_never_called_flaky() -> None:
    assert flaky([{"scenario": "a", "correct": False}]) == []


def test_a_scenario_that_vanished_from_the_run_is_a_regression(tmp_path: Path) -> None:
    rows = _rows(tmp_path, careful=True)
    found = compare(rows[:-1], baseline_from(rows))
    assert [f.detail for f in found] == ["in the baseline, not in this run"]


def test_a_new_scenario_has_nothing_to_be_worse_than(tmp_path: Path) -> None:
    rows = _rows(tmp_path, careful=True)
    assert compare(rows, baseline_from(rows[:3])) == []


def test_a_baseline_from_another_version_is_refused(tmp_path: Path) -> None:
    stale = tmp_path / "baseline.json"
    stale.write_text(json.dumps({"format": FORMAT - 1, "scenarios": {}}))
    with pytest.raises(ValueError, match="different version"):
        load_baseline(stale)


def test_the_command_writes_a_baseline_then_gates_on_it(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    calls = tmp_path / "calls"
    baseline = tmp_path / "baseline.json"
    assert main(["record", "--agent", "careful", "--repeat", "2", "--out", str(calls), "-q"]) == 0
    recorded = [str(p) for p in sorted(calls.glob("2026*.json"))]
    assert len(recorded) == 20  # two runs of ten, none overwriting another

    assert main(["check", *recorded, "--baseline", str(baseline), "--update", "-q"]) == 0
    assert "2 run(s) each" in capsys.readouterr().out
    assert main(["check", *recorded, "--baseline", str(baseline), "-q"]) == 0

    worse = tmp_path / "worse"
    assert main(["record", "--agent", "careless", "--out", str(worse), "-q"]) == 0
    regressed = [str(p) for p in sorted(worse.glob("2026*.json"))]
    capsys.readouterr()
    assert main(["check", *regressed, "--baseline", str(baseline), "-q"]) == 1
    assert "regressions" in capsys.readouterr().out


def test_a_one_run_baseline_says_it_is_a_coin_toss(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    calls = tmp_path / "calls"
    main(["record", "--agent", "careful", "--out", str(calls), "-q"])
    capsys.readouterr()
    main(
        [
            "check",
            *[str(p) for p in calls.glob("2026*.json")],
            "--baseline",
            str(tmp_path / "b.json"),
            "--update",
            "-q",
        ]
    )
    assert "coin toss" in capsys.readouterr().out


def test_the_result_is_available_as_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    calls = tmp_path / "calls"
    main(["record", "--agent", "careful", "--out", str(calls), "-q"])
    recorded = [str(p) for p in calls.glob("2026*.json")]
    main(["check", *recorded, "--baseline", str(tmp_path / "b.json"), "--update", "-q"])
    capsys.readouterr()
    main(["check", *recorded, "--baseline", str(tmp_path / "b.json"), "--json", "-q"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["scenarios"] == 10
    assert parsed["regressions"] == []


def test_checking_without_a_baseline_says_how_to_make_one(tmp_path: Path) -> None:
    calls = tmp_path / "calls"
    main(["record", "--agent", "careful", "--scenario", "plain-booking", "--out", str(calls), "-q"])
    recorded = [str(p) for p in calls.glob("2026*.json")]
    with pytest.raises(SystemExit, match="--update"):
        main(["check", *recorded, "--baseline", str(tmp_path / "nope.json"), "-q"])


def test_one_unlucky_run_does_not_fail_a_build_but_a_consistent_one_does() -> None:
    """Numbers are compared on their median, so a single slow call is absorbed and a trend is not."""
    was = baseline_from([{"scenario": "a", "turns": 10}])
    one_slow = [
        {"scenario": "a", "turns": 10},
        {"scenario": "a", "turns": 10},
        {"scenario": "a", "turns": 40},
    ]
    assert compare(one_slow, was) == []

    all_slow = [{"scenario": "a", "turns": 40} for _ in range(3)]
    assert [r.detail for r in compare(all_slow, was)] == ["turns: 10 -> 40"]


def test_an_outcome_that_used_to_be_certain_and_is_now_usual_has_regressed() -> None:
    was = baseline_from([{"scenario": "a", "correct": True}])
    sometimes = [
        {"scenario": "a", "correct": True},
        {"scenario": "a", "correct": False},
        {"scenario": "a", "correct": True},
    ]
    assert [r.detail for r in compare(sometimes, was)] == ["correct: 100% -> 67% of runs"]
    # and the other way round is an improvement, not a regression
    assert compare([{"scenario": "a", "correct": True}], baseline_from(sometimes)) == []
