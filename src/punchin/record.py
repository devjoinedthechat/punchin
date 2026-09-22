"""Record one call, and the turn loop that both recording and forking run."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from punchin.agent import Agent, Lead
from punchin.call import Call, Turn, call_id
from punchin.customer import Customer
from punchin.dms import Dms, fresh
from punchin.scenarios import Scenario

MAX_TURNS = 20


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def converse(
    call: Call, agent: Agent, customer: Customer, lead: Lead, dms: Dms, *, max_turns: int = MAX_TURNS
) -> Call:
    """Alternate agent and customer turns onto `call` until one of them ends it."""
    while len(call.turns) < max_turns:
        at = now()
        spoken = agent.respond(call, lead, dms)
        turn = Turn(
            index=len(call.turns),
            speaker="agent",
            text=spoken.text,
            started_at=at,
            ended_at=now(),
            tool_calls=spoken.tool_calls,
            model_ms=spoken.model_ms,
            cost_usd=spoken.cost_usd,
        )
        call.turns.append(turn)
        at = now()
        answer = customer.respond(call)
        if answer is None:
            call.notes["ended_by"] = "customer"
            break
        call.turns.append(
            Turn(index=len(call.turns), speaker="customer", text=answer, started_at=at, ended_at=now())
        )
        if turn.ends_call:
            call.notes["ended_by"] = "agent"
            break
    else:
        call.notes["ended_by"] = "max_turns"
    return call


def finish(call: Call, dms: Dms) -> Call:
    """Read the world back out of the DMS and onto the call."""
    final = dms.load()
    call.bookings = [b.model_dump(mode="json") for b in final.bookings]
    call.notes["tool_calls"] = len(final.calls)
    return call


def record(
    scenario: Scenario, agent: Agent, customer: Customer, state_path: Path, out: Path | None = None
) -> Call:
    dms = Dms(state_path)
    dms.save(fresh([scenario.vehicle]))
    lead = Lead(owner=scenario.vehicle.owner, syn_due=scenario.vehicle.syn_due)
    started = now()
    call = Call(
        id=call_id(scenario.id, agent.name, started),
        scenario=scenario.id,
        agent=agent.name,
        customer=customer.name,
        started_at=started,
    )
    try:
        converse(call, agent, customer, lead, dms)
    finally:
        # A call that died halfway is the interesting one; keep whatever was said before it broke.
        finish(call, dms)
        if out is not None:
            call.save(out)
    return call
