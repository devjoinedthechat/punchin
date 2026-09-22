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
