import datetime as dt
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from punchin.dms import TODAY, Dms, Vehicle, fresh, normalize_reg


def _dms(tmp_path: Path) -> Dms:
    dms = Dms(tmp_path / "state.json")
    dms.save(
        fresh(
            [Vehicle(reg="AB12345", make="Škoda", model="Octavia", year=2019, owner="Mette", syn_due=TODAY)]
        )
    )
    return dms


@pytest.mark.parametrize("said", ["AB 12 345", "ab12345", "AB-12-345", "AB.12.345"])
def test_plates_are_found_however_they_were_said(tmp_path: Path, said: str) -> None:
    assert normalize_reg(said) == "AB12345"
    assert _dms(tmp_path).call("lookup_vehicle", reg=said)["model"] == "Octavia"


def test_slots_are_weekdays_only_and_booking_takes_one(tmp_path: Path) -> None:
    dms = _dms(tmp_path)
    week = dms.call("find_slots", date_from=TODAY, date_to=TODAY + dt.timedelta(days=7))
    assert {dt.date.fromisoformat(s["date"]).weekday() for s in week["slots"]} <= {0, 1, 2, 3, 4}
    first = week["slots"][0]
    booked = dms.call("book", slot_id=first["id"], reg="AB 12 345", note="lånebil")
    assert (booked["date"], booked["time"], booked["note"]) == (first["date"], first["time"], "lånebil")
    with pytest.raises(ToolError, match="already taken"):
        dms.call("book", slot_id=first["id"], reg="AB12345")


def test_every_call_is_on_the_record_including_failures(tmp_path: Path) -> None:
    dms = _dms(tmp_path)
    with pytest.raises(ToolError, match="no vehicle"):
        dms.call("lookup_vehicle", reg="ZZ 99 999")
    dms.call("lookup_vehicle", reg="AB12345")
    calls = dms.load().calls
    assert [(c["tool"], c["error"] is None) for c in calls] == [
        ("lookup_vehicle", False),
        ("lookup_vehicle", True),
    ]


def test_the_state_path_is_absolute_so_a_server_started_elsewhere_finds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert Dms(Path("relative/state.json")).path == tmp_path / "relative/state.json"
