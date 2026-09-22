"""The agent under test: an outbound syn-reminder caller, scripted or model-driven.

The two scripted policies are the yardstick. `careful` does what a good agent does and `careless`
makes the mistake each scenario is built to catch; every grader has to tell them apart before any
model time is spent on the real thing.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

from mcp.server.mcpserver.exceptions import ToolError

from punchin.call import FAREWELL, Call, ToolCall
from punchin.dms import Dms
from punchin.model import Model
from punchin.scenarios import WEEKDAYS, next_weekday, regs_mentioned, weekdays_mentioned

WORKSHOP = "Bilhuset Vestergade"
AGENT_NAME = "Sofie"
MONTHS = [
    "januar",
    "februar",
    "marts",
    "april",
    "maj",
    "juni",
    "juli",
    "august",
    "september",
    "oktober",
    "november",
    "december",
]


def say_date(day: dt.date) -> str:
    return f"{WEEKDAYS[day.weekday()]} den {day.day}. {MONTHS[day.month - 1]}"


@dataclass
class Lead:
    """Why the agent is calling: a customer on file with a car whose syn is running out."""

    owner: str
    syn_due: dt.date


@dataclass
class AgentTurn:
    """One line from an agent, with what it cost to produce and what it did to the world."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model_ms: int | None = None
    cost_usd: float = 0.0


class Agent(Protocol):
    """Whatever is being tested: one line at a time, given the call so far.

    `call` holds what the agent HEARD, never what was said — `Call.transcript()` and `Call.said()`
    default to the recogniser's version for exactly this reason. Reading `Turn.spoken` here would let
    an agent hear perfectly through a bad line and make every audio measurement meaningless.

    An agent may call the dealership system through `dms`; it is never asked what it called, because
    punchin reads that system's own log around the turn instead.
    """

    name: str

    def respond(self, call: Call, lead: Lead, dms: Dms) -> AgentTurn:
        """The next thing this agent says out loud, and what it cost."""
        ...


# -- scripted ---------------------------------------------------------------------------------------------

EXTRAS = {"lånebil": r"lånebil", "stor service": r"stor(e)? service|big one"}


def _results(call: Call, tool: str) -> list[dict[str, Any]]:
    return [
        c.result
        for t in call.turns
        for c in t.tool_calls
        if c.tool == tool and c.error is None and isinstance(c.result, dict)
    ]


def _looked_up(call: Call) -> dict[str, Any] | None:
    """The vehicle the agent has already found in this conversation, if any."""
    found = _results(call, "lookup_vehicle")
    return found[-1] if found else None


def _offered(call: Call) -> dict[str, Any] | None:
    """The slot the agent has already put to the customer: the first of the last search it ran."""
    searches = _results(call, "find_slots")
    slots = searches[-1].get("slots", []) if searches else []
    return dict(slots[0]) if slots else None


class ScriptedAgent:
    """A fixed policy. `careful=True` is the good agent; `careful=False` makes the built-in mistakes:
    the first day said instead of the last, the first plate instead of the corrected one, no note,
    no listening for 'not this week' or 'already booked'."""

    def __init__(self, *, careful: bool) -> None:
        self.careful = careful
        self.name = "scripted-careful" if careful else "scripted-careless"
        self.vehicle: dict[str, Any] | None = None
        self.offered: dict[str, Any] | None = None
        self.calls: list[ToolCall] = []

    def _use(self, dms: Dms, tool: str, **arguments: Any) -> dict[str, Any] | None:
        try:
            result = dms.call(tool, **arguments)
        except ToolError as error:
            self.calls.append(ToolCall(tool=tool, arguments=arguments, error=str(error)))
            return None
        self.calls.append(ToolCall(tool=tool, arguments=arguments, result=result))
        return result

    def respond(self, call: Call, lead: Lead, dms: Dms) -> AgentTurn:
        self.calls = []
        # Read back out of the call, never kept on the instance: a fork hands this agent a conversation
        # it never had, and an agent that trusted its own memory would answer for the wrong one.
        self.vehicle = _looked_up(call)
        self.offered = _offered(call)
        text = self._policy(call, lead, dms)
        return AgentTurn(text, self.calls)

    def _policy(self, call: Call, lead: Lead, dms: Dms) -> str:
        # Everything this agent knows about the customer arrives through the recogniser. Reading
        # `spoken` here would let the scripted baseline hear perfectly and make the audio decorative.
        customer = " ".join(call.said("customer"))
        last = call.last("customer")
        latest = last.as_heard if last else ""
        pick = -1 if self.careful else 0

        if not call.turns:
            return (
                f"Hej, det er {AGENT_NAME} fra {WORKSHOP}. Jeg ringer, fordi synet på din bil udløber "
                f"{say_date(lead.syn_due)}. Passer det nu?"
            )
        if self.careful and re.search(r"allerede (booket|bestilt)", customer, re.I):
            return f"Så er det helt fint, så gør vi ikke mere. Tak for det, og hej hej. {FAREWELL}"
        if re.search(r"\brobot\b", latest, re.I):
            if self.careful:
                return (
                    "Ja, jeg er en AI-assistent fra værkstedet, "
                    "og du kan altid få en kollega i røret i stedet. "
                    "Passer det, at jeg finder en tid til synet?"
                )
            return f"Jeg er {AGENT_NAME} fra værkstedet. Må jeg få nummerpladen?"

        if self.vehicle is None:
            regs = regs_mentioned(customer)
            if not regs:
                return "Må jeg få nummerpladen på bilen?"
            found = self._use(dms, "lookup_vehicle", reg=regs[pick])
            if found is None:
                return "Den kan jeg ikke finde i systemet. Kan du sige nummerpladen igen?"
            self.vehicle = found
            return f"Tak, så er det din {found['make']} {found['model']}. Hvilken dag passer dig?"

        if self.offered is None:
            days = weekdays_mentioned(customer)
            if not days:
                return "Hvilken dag passer dig bedst?"
            weeks = 1 if self.careful and re.search(r"næste uge|ikke i denne uge", customer, re.I) else 0
            day = next_weekday(days[pick], weeks_ahead=weeks)
            slots = self._use(dms, "find_slots", date_from=day, date_to=day)
            if not slots or not slots["slots"]:
                return (
                    f"Der er desværre ikke noget ledigt {WEEKDAYS[day.weekday()]}. Hvilken anden dag passer?"
                )
            self.offered = slots["slots"][0]
            return f"Jeg har en tid {say_date(day)} kl. {self.offered['time']}. Skal jeg booke den?"

        if re.search(r"\b(ja|yes|fint|okay)\b", latest, re.I):
            note = (
                ", ".join(k for k, pattern in EXTRAS.items() if re.search(pattern, customer, re.I))
                if self.careful
                else ""
            )
            booked = self._use(dms, "book", slot_id=self.offered["id"], reg=self.vehicle["reg"], note=note)
            if booked is None:
                return "Hov, den tid blev lige taget. Hvilken anden dag passer?"
            when = dt.date.fromisoformat(booked["date"])
            when_said = f"{say_date(when)} kl. {booked['time']}"
            return f"Så er den booket: {when_said}. Tak for det, og hej hej. {FAREWELL}"
        return "Skal jeg booke den tid?"


# -- a model --------------------------------------------------------------------------------------------

SYSTEM = f"""\
Du er {AGENT_NAME}, en telefonassistent hos {WORKSHOP}, et autoværksted. Du ringer til en kunde, fordi
synet på kundens bil snart udløber, og du vil gerne booke en tid til syn.

Sådan taler du: kort, venligt og uformelt (du, ikke De). Én ting ad gangen; stil ét spørgsmål og vent på
svaret. Gentag aldrig noget kunden lige har sagt, medmindre du bekræfter en aftale. Hvis kunden retter
sig selv, gælder det sidste, kunden sagde. Hvis kunden spørger, om du er en robot, så sig ærligt, at du
er en AI-assistent fra værkstedet.

Sådan arbejder du: bekræft nummerpladen med kunden (der kan være flere biler på adressen), slå bilen op,
find en ledig tid på den dag kunden ønsker, sig dag og klokkeslæt højt, og book først, når kunden har sagt
ja til den konkrete tid. Skriv det i noten, hvis kunden nævner noget, værkstedet skal vide (lånebil,
service, en lyd). Har kunden allerede booket et andet sted, eller vil kunden ikke, så afslut høfligt
uden at booke.

I dag er {{today}}. Når samtalen er slut, sig farvel og afslut din replik med præcis dette: {FAREWELL}
Skriv kun det, du siger højt. Ingen overskrifter, ingen punktopstillinger, ingen tanker.
"""

PROMPT_OPEN = "Kunden har lige taget telefonen. Kunden hedder {owner}. Sig din åbningsreplik."
PROMPT_NEXT = "Samtalen indtil nu:\n\n{transcript}\n\nSkriv din næste replik."


def mcp_config(dms: Dms) -> dict[str, Any]:
    """Run the DMS as a stdio MCP server from this same interpreter, so the tools are the fake DMS."""
    return {
        "mcpServers": {
            "dms": {"command": sys.executable, "args": ["-m", "punchin", "dms", "--state", str(dms.path)]}
        }
    }


class ModelAgent:
    def __init__(self, model: Model, today: dt.date, *, system_suffix: str = "") -> None:
        self.model = model
        self.today = today
        self.system_suffix = system_suffix
        self.name = f"agent:{model.name}" + ("+suffix" if system_suffix else "")

    def respond(self, call: Call, lead: Lead, dms: Dms) -> AgentTurn:
        prompt = (
            PROMPT_OPEN.format(owner=lead.owner)
            if not call.turns
            else PROMPT_NEXT.format(transcript=call.transcript())
        )
        system = SYSTEM.format(today=say_date(self.today))
        if self.system_suffix:
            system = f"{system}\n{self.system_suffix}\n"
        done = self.model.complete(system, prompt, mcp=mcp_config(dms))
        return AgentTurn(done.text.strip(), done.tool_calls, done.elapsed_ms, done.cost_usd)
