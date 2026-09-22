"""Record one call, and the turn loop that both recording and forking run."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Protocol

from punchin.agent import Agent, Lead
from punchin.call import Call, Turn, call_id
from punchin.customer import Customer
from punchin.dms import Dms, fresh
from punchin.scenarios import Scenario

MAX_TURNS = 20

log = logging.getLogger(__name__)


class Spender(Protocol):
    """Whatever is keeping the bill. `punchin.fork.Budget` is the one this package ships."""

    def charge(self, amount: float) -> None:
        """Add to the running total, and raise if the allowance has gone."""


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def converse(
    call: Call,
    agent: Agent,
    customer: Customer,
    lead: Lead,
    dms: Dms,
    *,
    max_turns: int = MAX_TURNS,
    budget: Spender | None = None,
) -> Call:
    """Alternate agent and customer turns onto `call` until one of them ends it.

    `budget` is charged per turn rather than per call. Checking between calls is not enough: one call
    that loops can spend a whole run's allowance before anything looks at it, and the caller then finds
    out by reading the bill.
    """
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
        if budget is not None:
            budget.charge(turn.cost_usd)
        log.info("%2d agent  %s", turn.index, _short(turn.spoken))
        for made in turn.tool_calls:
            log.info("        -> %s %s", made.tool, "ERROR" if made.error else "ok")
        at = now()
        answer = customer.respond(call)
        if answer is None:
            call.notes["ended_by"] = "customer"
            break
        call.turns.append(
            Turn(
                index=len(call.turns),
                speaker="customer",
                text=answer.text,
                started_at=at,
                ended_at=now(),
                audio=answer.audio,
                heard=answer.heard,
                audio_ms=answer.audio_ms,
            )
        )
        said = call.turns[-1]
        log.info("%2d kunde  %s", said.index, _short(said.spoken))
        if said.heard is not None and said.heard.strip() != said.spoken.strip():
            log.info("    heard  %s", _short(said.heard))
        if turn.ends_call:
            call.notes["ended_by"] = "agent"
            break
    else:
        call.notes["ended_by"] = "max_turns"
    return call


def _short(text: str, width: int = 88) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def finish(call: Call, dms: Dms) -> Call:
    """Read the world back out of the DMS and onto the call."""
    final = dms.load()
    call.bookings = [b.model_dump(mode="json") for b in final.bookings]
    call.notes["tool_calls"] = len(final.calls)
    return call


def record(
    scenario: Scenario,
    agent: Agent,
    customer: Customer,
    state_path: Path,
    out: Path | None = None,
    *,
    budget: Spender | None = None,
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
        converse(call, agent, customer, lead, dms, budget=budget)
    finally:
        # A call that died halfway is the interesting one; keep whatever was said before it broke.
        finish(call, dms)
        if out is not None:
            call.save(out)
    return call
