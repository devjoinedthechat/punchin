"""The recording format, and what happens when it moves."""

import datetime as dt
import json
from pathlib import Path

import pytest

from punchin.call import FORMAT, Call


def _call() -> Call:
    return Call(id="c", scenario="plain-booking", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC))


def test_a_recording_carries_the_format_it_was_written_with(tmp_path: Path) -> None:
    path = _call().save(tmp_path)
    assert json.loads(path.read_text())["format"] == FORMAT
    assert Call.load(path).format == FORMAT


def test_a_recording_from_a_newer_punchin_is_refused_with_a_sentence(tmp_path: Path) -> None:
    path = _call().save(tmp_path)
    raw = json.loads(path.read_text())
    raw["format"] = FORMAT + 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="newer punchin"):
        Call.load(path)


def test_a_recording_from_before_the_format_field_still_loads(tmp_path: Path) -> None:
    """Recordings made before versioning have no `format` key and are readable as format 1."""
    path = _call().save(tmp_path)
    raw = json.loads(path.read_text())
    del raw["format"]
    path.write_text(json.dumps(raw))
    assert Call.load(path).format == FORMAT


def test_two_calls_recorded_in_the_same_second_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """A gate records each scenario several times; without this, `--repeat 3` quietly does one."""
    from punchin.call import call_id

    at = dt.datetime.now(dt.UTC)
    names = {call_id("plain-booking", "careful", at) for _ in range(50)}
    assert len(names) == 50

    import datetime as when

    made = [
        _call().model_copy(update={"id": call_id("s", "a", at), "started_at": when.datetime.now(when.UTC)})
        for _ in range(20)
    ]
    for one in made:
        one.save(tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 20


def test_a_call_id_still_sorts_by_when_it_happened() -> None:
    from punchin.call import call_id

    early = call_id("s", "a", dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    late = call_id("s", "a", dt.datetime(2026, 6, 1, tzinfo=dt.UTC))
    assert early < late


def test_a_recording_never_silently_lands_on_a_different_one(tmp_path: Path) -> None:
    """Entropy lowers the odds of losing a recording; only a check removes them."""
    first = _call()
    first.save(tmp_path)
    first.save(tmp_path)  # the same call again is fine

    other = first.model_copy(update={"started_at": dt.datetime(2020, 1, 1, tzinfo=dt.UTC)})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        other.save(tmp_path)
