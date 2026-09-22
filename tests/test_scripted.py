"""The scripted pair: `careful` gets every scenario right; `careless` makes each built-in mistake."""

import datetime as dt
from pathlib import Path

import pytest

from punchin.agent import ScriptedAgent
from punchin.call import Call
from punchin.customer import ScriptedCustomer
from punchin.record import record
from punchin.scenarios import BY_ID, SCENARIOS, Scenario, next_weekday


def _record(tmp_path: Path, scenario: Scenario, *, careful: bool) -> Call:
    return record(
        scenario,
        ScriptedAgent(careful=careful),
        ScriptedCustomer(scenario),
        tmp_path / "state.json",
        tmp_path,
    )


def _booked_day(call: Call) -> dt.date | None:
    return dt.date.fromisoformat(call.bookings[0]["date"]) if call.bookings else None


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_the_careful_agent_reaches_the_expected_outcome(tmp_path: Path, scenario: Scenario) -> None:
    call = _record(tmp_path, scenario, careful=True)
    assert bool(call.bookings) == scenario.expected.booked, call.transcript()
    assert _booked_day(call) == scenario.expected.day, call.transcript()
    if scenario.expected.booked:
        assert call.bookings[0]["reg"] == scenario.expected.reg
    assert call.notes["ended_by"] == "agent", call.transcript()


def test_the_careless_agent_books_the_day_the_customer_took_back(tmp_path: Path) -> None:
    scenario = BY_ID["self-correction"]
    call = _record(tmp_path, scenario, careful=False)
    assert _booked_day(call) == next_weekday("tirsdag")  # the tool call succeeded, and it is wrong
    assert scenario.expected.day == next_weekday("onsdag")


def test_the_careless_agent_looks_up_the_plate_the_customer_corrected(tmp_path: Path) -> None:
    call = _record(tmp_path, BY_ID["wrong-reg-first"], careful=False)
    first = next(c for t in call.turns for c in t.tool_calls if c.tool == "lookup_vehicle")
    assert first.arguments["reg"] == "KL99010"
    assert first.error is not None


def test_the_careless_agent_books_this_week_when_asked_for_next(tmp_path: Path) -> None:
    call = _record(tmp_path, BY_ID["next-week"], careful=False)
    assert _booked_day(call) == next_weekday("onsdag")
    assert BY_ID["next-week"].expected.day == next_weekday("onsdag", weeks_ahead=1)


def test_the_careless_agent_drops_the_courtesy_car(tmp_path: Path) -> None:
    careful = _record(tmp_path, BY_ID["courtesy-car"], careful=True)
    careless = _record(tmp_path, BY_ID["courtesy-car"], careful=False)
    assert careful.bookings[0]["note"] == "lånebil"
    assert careless.bookings[0]["note"] == ""


def test_the_careless_agent_keeps_asking_a_customer_who_said_no(tmp_path: Path) -> None:
    call = _record(tmp_path, BY_ID["already-booked"], careful=False)
    assert not call.bookings
    assert call.notes["ended_by"] == "customer"  # the customer gave up on it


def test_a_call_round_trips_through_its_file(tmp_path: Path) -> None:
    call = _record(tmp_path, BY_ID["plain-booking"], careful=True)
    again = Call.load(tmp_path / f"{call.id}.json")
    assert again == call
    assert "Kunde: CD 67 890." in again.transcript()
