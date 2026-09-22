"""Forking a recorded call: the prefix is reproduced exactly, the rest runs live."""

import sys
from pathlib import Path

import pytest

from punchin.agent import ScriptedAgent
from punchin.call import Call
from punchin.customer import CustomerTurn, ScriptedCustomer
from punchin.dms import Dms, fresh
from punchin.fork import Budget, agent_turns, fork, fork_once, replay_prefix
from punchin.model import ClaudeCodeModel
from punchin.record import record
from punchin.scenarios import BY_ID

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"
SCENARIO = BY_ID["self-correction"]


@pytest.fixture
def recorded(tmp_path: Path) -> Call:
    return record(SCENARIO, ScriptedAgent(careful=True), ScriptedCustomer(SCENARIO), tmp_path / "s.json")


def _model() -> ClaudeCodeModel:
    return ClaudeCodeModel([sys.executable, str(FAKE)])


def test_only_agent_turns_are_fork_points(recorded: Call) -> None:
    assert agent_turns(recorded) == [0, 2, 4, 6, 8]
    with pytest.raises(ValueError, match=r"not an agent turn.*fork at one of: 0, 2, 4, 6, 8"):
        fork(
            recorded,
            SCENARIO,
            3,
            agent=ScriptedAgent(careful=True),
            goal=SCENARIO.goal,
            model=_model(),
            state_path=Path("unused.json"),
        )


def test_the_prefix_puts_the_world_back_where_the_recording_left_it(recorded: Call, tmp_path: Path) -> None:
    dms = Dms(tmp_path / "replay.json")
    dms.save(fresh([SCENARIO.vehicle]))
    replay_prefix(recorded, len(recorded.turns), dms)  # the whole call, booking included
    state = dms.load()
    assert [b.reg for b in state.bookings] == ["AB12345"]
    assert state.slots[state.bookings[0].slot_id].taken

    # The boundary is exclusive: turn 6 is where the agent searched for slots, so a prefix of 6
    # stops just before it. That is the point — turn 6 is the turn under test.
    dms.save(fresh([SCENARIO.vehicle]))
    replay_prefix(recorded, 6, dms)
    early = dms.load()
    assert early.bookings == []
    assert [c["tool"] for c in early.calls] == ["lookup_vehicle"]

    dms.save(fresh([SCENARIO.vehicle]))
    replay_prefix(recorded, 8, dms)
    later = dms.load()
    assert [c["tool"] for c in later.calls] == ["lookup_vehicle", "find_slots"]
    assert later.bookings == []  # the booking itself is turn 8, still ahead


def test_a_fork_keeps_the_prefix_verbatim_and_runs_the_rest_live(recorded: Call, tmp_path: Path) -> None:
    forked = fork_once(
        recorded,
        SCENARIO,
        6,
        agent=ScriptedAgent(careful=True),
        goal=SCENARIO.goal,
        model=_model(),
        state_path=tmp_path / "f.json",
    )
    assert [t.spoken for t in forked.turns[:6]] == [t.spoken for t in recorded.turns[:6]]
    assert len(forked.turns) > 6
    assert forked.notes["forked_from"] == recorded.id
    assert forked.notes["forked_at"] == 6
    # The scripted agent has no memory of its own, so it still knows the car the prefix looked up.
    assert forked.bookings and forked.bookings[0]["reg"] == "AB12345"


def test_the_prefix_is_not_charged_for(recorded: Call, tmp_path: Path) -> None:
    """The point of a fork: the turns served from the recording cost nothing to replay."""
    forked = fork_once(
        recorded,
        SCENARIO,
        6,
        agent=ScriptedAgent(careful=True),
        goal=SCENARIO.goal,
        model=_model(),
        state_path=tmp_path / "f.json",
    )
    live = float(forked.notes["live_cost_usd"])
    assert live == pytest.approx(
        sum(t.cost_usd for t in forked.turns[6:]) + 0.002 * _customer_turns(forked, 6)
    )
    assert live < 0.01


def _customer_turns(call: Call, after: int) -> int:
    return sum(1 for t in call.turns[after:] if t.speaker == "customer")


def test_repeat_runs_several_attempts_and_the_budget_stops_them(recorded: Call, tmp_path: Path) -> None:
    report = fork(
        recorded,
        SCENARIO,
        6,
        agent=ScriptedAgent(careful=True),
        goal=SCENARIO.goal,
        model=_model(),
        state_path=tmp_path / "f.json",
        repeat=3,
        changed="nothing",
    )
    assert len(report.attempts) == 3
    assert report.fixed == 3  # the careful agent still books the right day from turn 6
    assert "correct in 3 of 3 attempts" in report.text()
    assert "the 6 turns before the fork cost nothing" in report.text()

    broke = fork(
        recorded,
        SCENARIO,
        6,
        agent=ScriptedAgent(careful=True),
        goal=SCENARIO.goal,
        model=_model(),
        state_path=tmp_path / "f.json",
        repeat=5,
        budget=Budget(0.001),
    )
    assert len(broke.attempts) == 1  # the first attempt spends past a one-tenth-of-a-cent budget
    assert broke.stopped is not None and "budget" in broke.stopped


def test_a_fork_can_put_a_recogniser_between_the_customer_and_the_agent(
    recorded: Call, tmp_path: Path
) -> None:
    """Forking a spoken call must not quietly remove the thing that broke it."""
    heard: list[str] = []

    class Deafening:
        """Stands in for a bad line: everything the customer says arrives as noise."""

        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.name = "deafening"

        def respond(self, call: Call) -> CustomerTurn | None:
            said = self.inner.respond(call)  # type: ignore[attr-defined]
            if said is None:
                return None
            heard.append(said.text)
            return CustomerTurn(said.text, heard="øh hvad")

    forked = fork_once(
        recorded,
        SCENARIO,
        6,
        agent=ScriptedAgent(careful=True),
        goal=SCENARIO.goal,
        model=_model(),
        state_path=tmp_path / "f.json",
        wrap=Deafening,
    )
    assert heard, "the wrapper never saw the customer"
    live = [t for t in forked.turns[6:] if t.speaker == "customer"]
    assert all(t.heard == "øh hvad" for t in live)
    assert all(t.spoken != "øh hvad" for t in live)  # what was said is still the answer key
    assert not forked.bookings  # nothing arrives, so nothing is booked


def test_a_fork_with_the_careless_agent_books_the_day_the_customer_took_back(
    recorded: Call, tmp_path: Path
) -> None:
    """The report has to be able to say a fork made things worse, not only better.

    Forked at 6, so the self-correction is already in the prefix and the careless agent has to read
    past it. Forked at 4 the trap is gone: the correction had not been said yet, and the customer the
    simulator plays from there is free to ask for Wednesday and nothing else. Where you fork decides
    what you are still testing.
    """
    report = fork(
        recorded,
        SCENARIO,
        6,
        agent=ScriptedAgent(careful=False),
        goal=SCENARIO.goal,
        model=_model(),
        state_path=tmp_path / "f.json",
        repeat=1,
    )
    assert report.before["correct"] is True
    assert report.fixed == 0
    assert "correct in 0 of 1 attempts (original: True)" in report.text()
