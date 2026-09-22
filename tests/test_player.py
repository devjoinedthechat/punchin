"""The side-by-side page: it must stand alone, and it must not confuse said with heard."""

import datetime as dt
from pathlib import Path

import pytest

from punchin.call import Call, ToolCall, Turn
from punchin.player import MAX_EMBED_BYTES, build, write


def _when() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _call(call_id: str, *, forked_at: int | None = None, audio: Path | None = None) -> Call:
    call = Call(id=call_id, scenario="plain-booking", agent="a", customer="c", started_at=_when())
    call.turns = [
        Turn(
            index=0,
            speaker="agent",
            text="Hvilken dag passer dig?",
            started_at=_when(),
            ended_at=_when(),
            tool_calls=[ToolCall(tool="find_slots", arguments={"date_from": "2026-10-01"})],
        ),
        Turn(
            index=1,
            speaker="customer",
            text="Torsdag formiddag ville være godt.",
            started_at=_when(),
            ended_at=_when(),
            heard="2. derform i dag ville være godt.",
            audio=str(audio) if audio else None,
            audio_ms=1800,
        ),
    ]
    call.bookings = [{"reg": "CD67890", "date": "2026-10-01", "time": "08:00", "note": ""}]
    if forked_at is not None:
        call.notes["forked_at"] = forked_at
    return call


def test_the_page_stands_alone_and_carries_both_sides() -> None:
    page = build(_call("before"), _call("after", forked_at=1), title="Before and after")
    assert page.startswith("<!doctype html>")
    assert "__DATA__" not in page and "__TITLE__" not in page and "__SUB__" not in page
    assert "Before and after" in page
    assert "before" in page and "after" in page
    assert "prefers-color-scheme: dark" in page  # readable in either theme
    assert "served from the recording" in page  # the fork is marked


def test_what_was_said_and_what_was_heard_are_both_shown_and_kept_apart() -> None:
    page = build(_call("a"), _call("b"))
    assert "Torsdag formiddag ville være godt." in page
    assert "2. derform i dag ville være godt." in page
    assert "find_slots" in page


def test_audio_is_embedded_so_the_file_can_be_sent_to_somebody(tmp_path: Path) -> None:
    wav = tmp_path / "turn.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")  # not a real wav; the page only base64s the bytes
    page = build(_call("a", audio=wav), _call("b", audio=wav))
    assert page.count("data:audio/wav;base64,") == 2


def test_a_missing_wav_is_not_a_crash(tmp_path: Path) -> None:
    """Recordings move. A page built from a call whose audio has gone is still worth having."""
    page = build(_call("a", audio=tmp_path / "gone.wav"), _call("b"))
    assert "data:audio/wav" not in page
    assert "Torsdag formiddag" in page


def test_a_script_tag_in_a_transcript_cannot_end_the_script_early() -> None:
    """Turn text reaches the page inside a <script>. A customer quoting HTML must not break it."""
    call = _call("a")
    call.turns[1].text = "Jeg sagde </script><img src=x onerror=alert(1)> til ham."
    page = build(call, _call("b"))
    assert "</script><img" not in page
    assert "<\\/script>" in page


def test_a_page_too_large_to_open_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("punchin.player.MAX_EMBED_BYTES", 32)
    assert MAX_EMBED_BYTES > 32
    with pytest.raises(ValueError, match="over the"):
        build(_call("a"), _call("b"))


def test_writing_creates_the_directory(tmp_path: Path) -> None:
    dest = write(_call("a"), _call("b"), tmp_path / "nested" / "page.html")
    assert dest.exists()
    assert dest.read_text().startswith("<!doctype html>")
