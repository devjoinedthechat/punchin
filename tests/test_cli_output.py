"""How a recording reads. The README quotes this, so it must not drift silently."""

import datetime as dt

from punchin.call import Call, ToolCall, Turn
from punchin.commands import show


def _call() -> Call:
    at = dt.datetime.now(dt.UTC)
    call = Call(id="c-1", scenario="plain-booking", agent="agent:x", started_at=at, customer="k")
    call.turns = [
        Turn(
            index=4,
            speaker="agent",
            text="Hvilken dag passer dig?",
            started_at=at,
            ended_at=at,
            model_ms=4614,
            tool_calls=[ToolCall(tool="lookup_vehicle", arguments={"reg": "CD 67 890"})],
        ),
        Turn(
            index=5,
            speaker="customer",
            text="Torsdag formiddag ville være godt.",
            started_at=at,
            ended_at=at,
            heard="2. derform i dag ville være godt.",
        ),
        Turn(
            index=6,
            speaker="agent",
            text="Den 2. oktober kl. 8.",
            started_at=at,
            ended_at=at,
            tool_calls=[
                ToolCall(
                    tool="lookup_vehicle", arguments={"reg": "ZZ"}, error="no vehicle registered as 'ZZ'"
                )
            ],
        ),
    ]
    call.bookings = [{"reg": "CD67890", "date": "2026-10-02", "time": "08:00", "note": "lånebil"}]
    call.notes["ended_by"] = "agent"
    return call


def test_a_turn_is_its_number_its_speaker_and_then_everything_about_it() -> None:
    lines = show(_call()).splitlines()
    assert "  4  Agent  Hvilken dag passer dig?   (4614 ms)" in lines
    assert "  5  Kunde  Torsdag formiddag ville være godt." in lines


def test_what_was_heard_and_what_was_called_line_up_under_the_words() -> None:
    """Continuations sit in the same column as the text, so a turn reads as one thing."""
    lines = show(_call()).splitlines()
    said = next(line for line in lines if "Torsdag formiddag" in line)
    heard = next(line for line in lines if "heard" in line)
    called = next(line for line in lines if "calls" in line)
    text_column = said.index("Torsdag")
    assert heard.index("heard") == text_column
    assert called.index("calls") == text_column


def test_tool_arguments_read_as_words_not_as_a_python_dict() -> None:
    printed = show(_call())
    assert "calls  lookup_vehicle(reg=CD 67 890)" in printed
    assert "{'reg'" not in printed


def test_a_failed_call_says_what_failed_underneath_it() -> None:
    lines = show(_call()).splitlines()
    error = next(i for i, line in enumerate(lines) if "error  lookup_vehicle" in line)
    assert "no vehicle registered as 'ZZ'" in lines[error + 1]


def test_the_footer_says_what_happened_and_where_it_can_be_forked() -> None:
    printed = show(_call())
    assert "  booked  CD67890 2026-10-02 08:00 (lånebil)" in printed
    assert "ended by  agent" in printed
    assert "fork at  4, 6" in printed


def test_a_call_that_cost_nothing_does_not_say_so() -> None:
    assert "$" not in show(_call()).splitlines()[0]
