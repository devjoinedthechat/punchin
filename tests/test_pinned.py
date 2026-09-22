"""The pinned customer: the thing a fork replaces a real person with."""

import datetime as dt
import sys
from pathlib import Path

import pytest

from punchin.call import Call, Turn
from punchin.customer import CustomerTurn
from punchin.model import ClaudeCodeModel, Completion
from punchin.pinned import HANGUP, PinnedCustomer
from punchin.scenarios import BY_ID

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"
SCENARIO = BY_ID["self-correction"]


class Spy:
    """A model that answers with whatever it is told to, and keeps the prompt it was given."""

    name = "spy"

    def __init__(self, reply: str = "Ja.") -> None:
        self.reply = reply
        self.system = ""
        self.prompt = ""
        self.calls = 0

    def complete(self, system: str, prompt: str, *, mcp=None, schema=None) -> Completion:
        self.system, self.prompt, self.calls = system, prompt, self.calls + 1
        return Completion(self.reply, cost_usd=0.01)


def _call(*turns: tuple[str, str, str | None]) -> Call:
    at = dt.datetime.now(dt.UTC)
    call = Call(id="c", scenario="self-correction", agent="a", customer="c", started_at=at)
    for index, (speaker, text, heard) in enumerate(turns):
        call.turns.append(
            Turn(index=index, speaker=speaker, text=text, started_at=at, ended_at=at, heard=heard)  # type: ignore[arg-type]
        )
    return call


def test_the_goal_state_is_what_the_customer_is_told_and_nothing_else() -> None:
    """The simulator may only say what the real customer knew. It has to be given that, and only that."""
    spy = Spy()
    PinnedCustomer(spy, SCENARIO.goal).respond(_call(("agent", "Hvilken dag?", None)))

    assert "AB 12 345" in spy.prompt  # the plate she had
    assert "onsdag" in spy.prompt.lower()  # the day she meant
    assert "retter sig selv" in spy.prompt  # how she talks
    assert "må ikke finde på" in spy.system  # and the instruction not to invent
    assert "Hvilken dag?" in spy.prompt  # plus the conversation so far


def test_the_hangup_marker_ends_the_call_and_never_reaches_the_transcript() -> None:
    customer = PinnedCustomer(Spy(f"Tak, hej. {HANGUP}"), SCENARIO.goal)
    said = customer.respond(_call(("agent", "Hej hej.", None)))
    assert isinstance(said, CustomerTurn)
    assert said.text == "Tak, hej."
    assert HANGUP not in said.text
    assert customer.hung_up is True
    assert customer.respond(_call(("agent", "Er du der?", None))) is None


def test_an_empty_reply_is_a_customer_who_has_stopped_talking() -> None:
    assert PinnedCustomer(Spy("   "), SCENARIO.goal).respond(_call(("agent", "Hallo?", None))) is None


def test_what_the_customer_costs_is_kept() -> None:
    customer = PinnedCustomer(Spy(), SCENARIO.goal)
    for _ in range(3):
        customer.respond(_call(("agent", "Ja?", None)))
    assert customer.cost_usd == pytest.approx(0.03)


def test_line_asks_for_a_turn_without_ending_the_call() -> None:
    """Teacher-forced fidelity asks for turn after turn; it must not be hung up on halfway."""
    customer = PinnedCustomer(Spy(f"Tak, hej. {HANGUP}"), SCENARIO.goal)
    assert customer.line(_call(("agent", "Hej hej.", None))).endswith(HANGUP)
    assert customer.hung_up is False
    assert customer.line(_call(("agent", "Igen.", None)))  # still answering


def test_a_customer_who_wants_nothing_is_told_so() -> None:
    spy = Spy()
    PinnedCustomer(spy, BY_ID["already-booked"].goal).respond(_call(("agent", "Passer det?", None)))
    assert "ingen, vil ikke booke" in spy.prompt


def test_the_name_says_which_model_is_playing_her() -> None:
    assert PinnedCustomer(Spy(), SCENARIO.goal).name == "pinned:spy"
    assert PinnedCustomer(Spy(), SCENARIO.goal, name="her").name == "her"


def test_the_whole_thing_runs_against_the_stand_in_binary() -> None:
    model = ClaudeCodeModel([sys.executable, str(FAKE)])
    said = PinnedCustomer(model, SCENARIO.goal).respond(_call(("agent", "Må jeg få nummerpladen?", None)))
    assert said is not None
    assert "AB 12 345" in said.text


def test_she_is_shown_what_she_said_not_what_the_recogniser_made_of_it() -> None:
    """The recogniser sits between her mouth and the agent's ears, not between her and herself."""
    spy = Spy()
    call = _call(
        ("agent", "Nummerpladen?", None),
        ("customer", "Det er AB 12 345.", "Det er AB-12300-354."),
        ("agent", "Hvilken dag?", None),
    )
    PinnedCustomer(spy, SCENARIO.goal).respond(call)
    assert "Det er AB 12 345." in spy.prompt
    assert "AB-12300-354" not in spy.prompt  # she never said that, so she must not react to it
