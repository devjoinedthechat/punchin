"""The contract punchin offers somebody else's agent."""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

from punchin.adapter import PROTOCOL, AgentProtocolError, CommandAgent, request_for
from punchin.agent import Lead
from punchin.call import Call, Turn
from punchin.dms import TODAY, Dms, fresh
from punchin.record import record
from punchin.scenarios import BY_ID

SCENARIO = BY_ID["self-correction"]
LEAD = Lead(owner="Mette", syn_due=TODAY + dt.timedelta(days=12))


def _dms(tmp_path: Path) -> Dms:
    dms = Dms(tmp_path / "state.json")
    dms.save(fresh([SCENARIO.vehicle]))
    return dms


def _agent(body: str, tmp_path: Path, **kwargs: object) -> CommandAgent:
    script = tmp_path / "agent.py"
    script.write_text(body)
    return CommandAgent([sys.executable, str(script)], TODAY, **kwargs)  # type: ignore[arg-type]


SAYS_HELLO = """
import json, sys
json.load(sys.stdin)
print(json.dumps({"text": "Hej."}))
"""


def test_an_agent_is_asked_for_one_line_and_answers_with_one(tmp_path: Path) -> None:
    dms = _dms(tmp_path)
    call = Call(id="c", scenario="s", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC))
    turn = _agent(SAYS_HELLO, tmp_path).respond(call, LEAD, dms)
    assert turn.text == "Hej."
    assert turn.tool_calls == []
    assert turn.model_ms is not None and turn.model_ms >= 0


def test_the_agent_is_told_what_was_heard_and_never_what_was_said(tmp_path: Path) -> None:
    """An external agent gets exactly what a real one would: the recogniser's version."""
    call = Call(id="c", scenario="s", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC))
    call.turns = [
        Turn(
            index=0,
            speaker="customer",
            text="Det er AB 12 345.",
            started_at=dt.datetime.now(dt.UTC),
            ended_at=dt.datetime.now(dt.UTC),
            heard="Det er AB-12300-354.",
        )
    ]
    asked = request_for(call, LEAD, _dms(tmp_path), TODAY)
    assert asked["protocol"] == PROTOCOL
    assert asked["conversation"] == [{"speaker": "customer", "text": "Det er AB-12300-354."}]
    assert json.dumps(asked)  # the whole request has to survive a round trip to another process
    assert asked["lead"]["owner"] == "Mette"
    assert "mcp" in asked["tools"] and "state_path" in asked["tools"]


CALLS_A_TOOL = """
import json, subprocess, sys
request = json.load(sys.stdin)
server = request["tools"]["mcp"]["mcpServers"]["dms"]
sys.path.insert(0, "src")
from punchin.dms import Dms
Dms(__import__("pathlib").Path(request["tools"]["state_path"])).call("lookup_vehicle", reg="AB 12 345")
print(json.dumps({"text": "Tak.", "cost_usd": 0.25}))
"""


def test_tool_calls_come_from_the_dealership_system_not_from_the_agents_word(tmp_path: Path) -> None:
    """The agent is never asked what it called, because it would have to be believed about it."""
    dms = _dms(tmp_path)
    call = Call(id="c", scenario="s", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC))
    turn = _agent(CALLS_A_TOOL, tmp_path).respond(call, LEAD, dms)
    assert [(c.tool, c.error) for c in turn.tool_calls] == [("lookup_vehicle", None)]
    assert turn.tool_calls[0].result["model"] == "Octavia"
    assert turn.cost_usd == 0.25


def test_only_the_turns_own_tool_calls_are_attributed_to_it(tmp_path: Path) -> None:
    dms = _dms(tmp_path)
    call = Call(id="c", scenario="s", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC))
    agent = _agent(CALLS_A_TOOL, tmp_path)
    first = agent.respond(call, LEAD, dms)
    second = agent.respond(call, LEAD, dms)
    assert len(first.tool_calls) == 1
    assert len(second.tool_calls) == 1  # not two; the earlier call belongs to the earlier turn


CHATTY = """
import json, sys
json.load(sys.stdin)
print("loading model...", flush=True)
print(json.dumps({"text": "Hej."}))
"""


def test_an_agent_that_logs_to_stdout_still_works(tmp_path: Path) -> None:
    assert (
        _agent(CHATTY, tmp_path)
        .respond(
            Call(id="c", scenario="s", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC)),
            LEAD,
            _dms(tmp_path),
        )
        .text
        == "Hej."
    )


@pytest.mark.parametrize(
    ("body", "complaint"),
    [
        ("import sys; sys.exit(3)", "exited 3"),
        ("print('not json')", "no JSON object"),
        ("import json; print(json.dumps({'reply': 'hej'}))", "without a string `text`"),
    ],
)
def test_a_broken_agent_is_named_rather_than_guessed_at(tmp_path: Path, body: str, complaint: str) -> None:
    call = Call(id="c", scenario="s", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC))
    with pytest.raises(AgentProtocolError, match=complaint):
        _agent(body, tmp_path).respond(call, LEAD, _dms(tmp_path))


def test_an_agent_that_hangs_is_given_up_on(tmp_path: Path) -> None:
    call = Call(id="c", scenario="s", agent="a", customer="c", started_at=dt.datetime.now(dt.UTC))
    agent = _agent("import time; time.sleep(30)", tmp_path, timeout_s=0.5)
    with pytest.raises(AgentProtocolError, match="did not answer"):
        agent.respond(call, LEAD, _dms(tmp_path))


def test_an_empty_command_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        CommandAgent([], TODAY)


REFERENCE = Path(__file__).parent.parent / "examples" / "rule_agent.py"


@pytest.mark.slow
def test_the_reference_agent_in_examples_satisfies_the_corpus(tmp_path: Path) -> None:
    """The example is executable documentation: if the contract drifts, this fails."""
    from punchin.customer import ScriptedCustomer
    from punchin.metrics import outcome

    agent = CommandAgent([sys.executable, str(REFERENCE)], TODAY)
    call = record(SCENARIO, agent, ScriptedCustomer(SCENARIO), tmp_path / "s.json")
    assert outcome(call, SCENARIO)["correct"], call.transcript()
