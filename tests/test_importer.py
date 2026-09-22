"""Reading somebody else's transcript, and stating what should have happened."""

import datetime as dt
import json
from pathlib import Path

import pytest

from punchin.cli import main
from punchin.importer import ImportError_, read_call, scenario_for, turns_from
from punchin.metrics import outcome
from punchin.scenarios import SCENARIO_DIR, load_scenarios

PLAIN = """
Agent: Hej, synet udløber snart. Passer det nu?
Kunde: Ja, det er fint.
Agent: Må jeg få nummerpladen?
Kunde: Det er XY 55 123.
"""


def test_a_plain_transcript_is_a_conversation() -> None:
    call = read_call(PLAIN, scenario_id="dealer-1")
    assert [t.speaker for t in call.turns] == ["agent", "customer", "agent", "customer"]
    assert call.turns[3].spoken == "Det er XY 55 123."
    assert call.notes["imported"] is True
    assert call.scenario == "dealer-1"


@pytest.mark.parametrize("who", ["Agent", "Assistant", "Bot", "AI", "Sofie", "agent", "ASSISTANT"])
def test_the_names_a_real_archive_uses_for_each_side_are_understood(who: str) -> None:
    call = read_call(f"{who}: Hej.\nCustomer: Ja.", scenario_id="s")
    assert [t.speaker for t in call.turns] == ["agent", "customer"]


def test_a_line_that_wraps_stays_one_turn() -> None:
    call = read_call("Agent: Hej, det er Sofie\nfra værkstedet.\nKunde: Ja.", scenario_id="s")
    assert call.turns[0].spoken == "Hej, det er Sofie fra værkstedet."
    assert len(call.turns) == 2


def test_json_jsonl_and_a_wrapper_object_all_read() -> None:
    turns = [{"speaker": "agent", "text": "Hej."}, {"speaker": "kunde", "text": "Ja."}]
    assert len(turns_from(json.dumps(turns))) == 2
    assert len(turns_from(json.dumps({"messages": turns}))) == 2
    assert len(turns_from("\n".join(json.dumps(t) for t in turns))) == 2


def test_timestamps_asr_text_and_tool_calls_survive_the_crossing() -> None:
    raw = json.dumps(
        [
            {
                "role": "assistant",
                "content": "Nummerpladen?",
                "timestamp": "2026-09-22T10:00:00Z",
                "tool_calls": [{"name": "lookup_vehicle", "arguments": {"reg": "XY55123"}}],
            },
            {"role": "user", "content": "XY 55 123.", "asr": "XY 55 one two three."},
        ]
    )
    call = read_call(raw, scenario_id="s")
    assert call.turns[0].started_at.year == 2026
    assert [c.tool for c in call.turns[0].tool_calls] == ["lookup_vehicle"]
    assert call.turns[1].spoken == "XY 55 123."
    assert call.turns[1].as_heard == "XY 55 one two three."  # the agent only ever had this


@pytest.mark.parametrize(
    ("raw", "complaint"),
    [
        ("", "empty"),
        ("just some prose with no speakers", "no turns found"),
        ('{"nothing": 1}', "needs a 'turns'"),
        ('{"turns": [{"speaker": "narrator", "text": "hm"}]}', "no turns had a speaker"),
        ("{not json}\n{also not}", "not a JSON object"),
    ],
)
def test_a_file_punchin_cannot_read_says_what_it_wanted(raw: str, complaint: str) -> None:
    with pytest.raises(ImportError_, match=complaint):
        read_call(raw, scenario_id="s")


def test_the_scenario_carries_the_outcome_somebody_stated() -> None:
    call = read_call(PLAIN, scenario_id="dealer-1")
    scenario = scenario_for(call, reg="XY 55 123", day=dt.date(2026, 10, 1), booked=True, extras=["lånebil"])
    assert scenario.id == "dealer-1"
    assert scenario.pattern == "imported"
    assert scenario.vehicle.reg == "XY55123"
    assert scenario.expected.booked and scenario.expected.day == dt.date(2026, 10, 1)
    assert scenario.goal.extras == ["lånebil"]
    assert scenario.script == []  # it is forked with a pinned customer, never replayed from a script


def test_a_scenario_needs_a_registration() -> None:
    with pytest.raises(ImportError_, match="needs the registration"):
        scenario_for(read_call(PLAIN, scenario_id="s"), reg="", day=None, booked=False)


def test_scenarios_on_disk_join_the_built_in_corpus(tmp_path: Path) -> None:
    built_in = load_scenarios(None)
    assert "self-correction" in built_in and len(built_in) == 10

    call = read_call(PLAIN, scenario_id="dealer-1")
    scenario = scenario_for(call, reg="XY55123", day=None, booked=False)
    (tmp_path / "dealer-1.json").write_text(scenario.model_dump_json())
    both = load_scenarios(tmp_path)
    assert "dealer-1" in both and "self-correction" in both


def test_a_file_that_is_not_a_scenario_is_named(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text('{"id": "x"}')
    with pytest.raises(ValueError, match="is not a scenario"):
        load_scenarios(tmp_path)


def test_a_missing_scenario_directory_is_not_an_error() -> None:
    assert load_scenarios(Path("/nonexistent")) == load_scenarios(None)
    assert SCENARIO_DIR.name == "scenarios"


def test_the_command_writes_both_and_the_call_is_then_gradeable(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = tmp_path / "call.txt"
    source.write_text(PLAIN)
    calls, scenarios = tmp_path / "calls", tmp_path / "scenarios"
    code = main(
        [
            "import",
            str(source),
            "--id",
            "dealer-1",
            "--reg",
            "XY 55 123",
            "--day",
            "2026-10-01",
            "--booked",
            "--booked-day",
            "2026-10-02",
            "--out",
            str(calls),
            "--scenarios",
            str(scenarios),
            "-q",
        ]
    )
    assert code == 0
    capsys.readouterr()

    recorded = next(calls.glob("2026*.json"))
    assert (scenarios / "dealer-1.json").exists()
    assert main(["metrics", str(recorded), "--scenarios", str(scenarios), "-q"]) == 0
    printed = capsys.readouterr().out
    assert "dealer-1" in printed
    assert "False" in printed  # booked friday, she said thursday


def test_a_booking_without_a_day_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "call.txt"
    source.write_text(PLAIN)
    with pytest.raises(SystemExit, match="--booked needs --day"):
        main(
            [
                "import",
                str(source),
                "--id",
                "d",
                "--reg",
                "XY55123",
                "--booked",
                "--out",
                str(tmp_path),
                "--scenarios",
                str(tmp_path),
                "-q",
            ]
        )


def test_grading_a_call_nobody_has_stated_an_outcome_for_says_so(tmp_path: Path) -> None:
    call = read_call(PLAIN, scenario_id="never-graded")
    written = call.save(tmp_path)
    with pytest.raises(SystemExit, match="nothing to grade this call against"):
        main(["metrics", str(written), "--scenarios", str(tmp_path / "none"), "-q"])


def test_the_imported_outcome_is_graded_the_same_way_as_a_built_in_one() -> None:
    call = read_call(PLAIN, scenario_id="dealer-1")
    scenario = scenario_for(call, reg="XY55123", day=dt.date(2026, 10, 1), booked=True)
    call.bookings = [{"reg": "XY55123", "date": "2026-10-02", "time": "08:00", "note": ""}]
    assert outcome(call, scenario)["day_ok"] is False
    call.bookings = [{"reg": "XY55123", "date": "2026-10-01", "time": "08:00", "note": ""}]
    assert outcome(call, scenario)["correct"] is True
