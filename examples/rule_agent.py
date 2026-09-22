#!/usr/bin/env python3
"""A syn-reminder agent that is not part of punchin, to show what the contract asks of one.

It reads a JSON request on stdin, calls the dealership system over MCP exactly as any other client
would, and prints one JSON object on stdout. It knows nothing about punchin's internals, and punchin
knows nothing about its: the only thing they share is the shape below and the MCP server.

    punchin record --agent command --agent-command "python examples/rule_agent.py"

Request  {"protocol":1,"today":"…","lead":{…},"tools":{"mcp":{…}},"conversation":[{"speaker","text"}]}
Reply    {"text":"…"}
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

DAYS = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
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
PLATE = re.compile(r"\b[A-Za-z]{2}[ .-]?\d{2}[ .-]?\d{3}\b")
FAREWELL = "[FARVEL]"


def plates(text: str) -> list[str]:
    return [re.sub(r"[^A-Z0-9]", "", found.upper()) for found in PLATE.findall(text)]


def weekdays(text: str) -> list[str]:
    return re.findall(r"\b(mandag|tirsdag|onsdag|torsdag|fredag)\b", text.lower())


def next_weekday(name: str, after: dt.date, weeks: int = 0) -> dt.date:
    days = (DAYS.index(name) - after.weekday()) % 7 or 7
    return after + dt.timedelta(days=days + 7 * weeks)


def spoken(day: dt.date) -> str:
    return f"{DAYS[day.weekday()]} den {day.day}. {MONTHS[day.month - 1]}"


async def use(config: dict[str, Any], tool: str, **arguments: Any) -> dict[str, Any] | None:
    server = config["mcpServers"]["dms"]
    params = StdioServerParameters(command=server["command"], args=server["args"])
    async with Client(params) as client:
        result = await client.call_tool(tool, arguments)
    text = "".join(getattr(block, "text", "") for block in result.content)
    try:
        answer: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return None
    return None if "error" in answer and "reg" not in answer else answer


async def reply(request: dict[str, Any]) -> str:
    config = request["tools"]["mcp"]
    today = dt.date.fromisoformat(request["today"])
    turns = request["conversation"]
    lead = request["lead"]
    if not turns:
        due = dt.date.fromisoformat(lead["syn_due"])
        return (
            f"Hej, det er Sofie fra værkstedet. Jeg ringer, fordi synet på din bil udløber "
            f"{spoken(due)}. Passer det nu?"
        )

    said = " ".join(t["text"] for t in turns if t["speaker"] == "customer")
    latest = next((t["text"] for t in reversed(turns) if t["speaker"] == "customer"), "")
    mine = [t["text"] for t in turns if t["speaker"] == "agent"]

    if re.search(r"allerede (booket|bestilt)", said, re.I):
        return f"Så gør vi ikke mere. Tak for det, og hej hej. {FAREWELL}"
    if re.search(r"\brobot\b", latest, re.I):
        return "Ja, jeg er en AI-assistent fra værkstedet. Skal jeg finde en tid til synet?"

    found = plates(said)
    vehicle = await use(config, "lookup_vehicle", reg=found[-1]) if found else None
    if vehicle is None:
        if len(found) >= 2:
            return "Den kan jeg ikke finde. Kan du sige nummerpladen ét tegn ad gangen?"
        return "Må jeg få nummerpladen på bilen?"

    days = weekdays(said)
    if not days:
        return f"Tak, så er det din {vehicle['make']} {vehicle['model']}. Hvilken dag passer dig?"
    weeks = 1 if re.search(r"næste uge|ikke i denne uge", said, re.I) else 0
    day = next_weekday(days[-1], today, weeks)

    slots = await use(config, "find_slots", date_from=day.isoformat(), date_to=day.isoformat())
    free = (slots or {}).get("slots") or []
    if not free:
        return f"Der er desværre ikke noget ledigt {DAYS[day.weekday()]}. Hvilken anden dag passer?"

    offered = free[0]
    already_offered = any("Skal jeg booke" in line for line in mine)
    # The corpus has a customer who switches to English mid-call, so a Danish-only yes is not enough.
    if already_offered and re.search(r"\b(ja|jo|fint|okay|ok|yes|yeah|sure|book it)\b", latest, re.I):
        note = ", ".join(
            word
            for word, pattern in (("lånebil", r"lånebil"), ("stor service", r"stor(e)? service|big one"))
            if re.search(pattern, said, re.I)
        )
        booked = await use(config, "book", slot_id=offered["id"], reg=vehicle["reg"], note=note)
        if booked is None:
            return "Hov, den tid blev taget. Hvilken anden dag passer?"
        when = dt.date.fromisoformat(booked["date"])
        return f"Så er den booket: {spoken(when)} kl. {booked['time']}. Hej hej. {FAREWELL}"
    return f"Jeg har en tid {spoken(day)} kl. {offered['time']}. Skal jeg booke den?"


async def main() -> None:
    request = json.loads(sys.stdin.read())
    if request.get("protocol") != 1:
        raise SystemExit(f"this agent speaks protocol 1, not {request.get('protocol')}")
    print(json.dumps({"text": await reply(request)}, ensure_ascii=False), flush=True)


anyio.run(main)
