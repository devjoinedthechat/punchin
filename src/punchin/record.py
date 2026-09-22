"""Record one call: the agent opens, the customer answers, until someone hangs up."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from punchin.agent import Agent, Lead
from punchin.call import Call, Turn, call_id
from punchin.customer import Customer
from punchin.dms import Dms, fresh
from punchin.scenarios import Scenario

MAX_TURNS = 20


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def record(
    scenario: Scenario, agent: Agent, customer: Customer, state_path: Path, out: Path | None = None
) -> Call:
    dms = Dms(state_path)
    dms.save(fresh([scenario.vehicle]))
    lead = Lead(owner=scenario.vehicle.owner, syn_due=scenario.vehicle.syn_due)
    started = _now()
    call = Call(
        id=call_id(scenario.id, agent.name, started),
        scenario=scenario.id,
        agent=agent.name,
        customer=customer.name,
        started_at=started,
    )

    while len(call.turns) < MAX_TURNS:
        at = _now()
        spoken = agent.respond(call, lead, dms)
        turn = Turn(
            index=len(call.turns),
            speaker="agent",
            text=spoken.text,
            started_at=at,
            ended_at=_now(),
            tool_calls=spoken.tool_calls,
            model_ms=spoken.model_ms,
            cost_usd=spoken.cost_usd,
        )
        call.turns.append(turn)
        at = _now()
        answer = customer.respond(call)
        if answer is None:
            call.notes["ended_by"] = "customer"
            break
        call.turns.append(
            Turn(index=len(call.turns), speaker="customer", text=answer, started_at=at, ended_at=_now())
        )
        if turn.ends_call:
            call.notes["ended_by"] = "agent"
            break
    else:
        call.notes["ended_by"] = "max_turns"

    final = dms.load()
    call.bookings = [b.model_dump(mode="json") for b in final.bookings]
    call.notes["tool_calls"] = len(final.calls)
    if out is not None:
        call.save(out)
    return call
