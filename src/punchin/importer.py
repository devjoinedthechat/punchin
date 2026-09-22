"""Turn a call that punchin did not record into one it can grade and fork.

An archive of real calls is the reason to want any of this, and it never arrives in punchin's shape.
This reads the shapes it does arrive in — a JSON array of turns, one JSON object per line, or a plain
`Agent:` / `Kunde:` transcript — and writes a recording.

It also asks for something a transcript cannot contain: **what should have happened**. A call cannot be
graded against nothing, and no amount of parsing recovers the day the customer actually meant. That is a
person's judgement, stated once, and everything downstream reads it from the scenario written here.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from punchin.call import Call, ToolCall, Turn, call_id
from punchin.dms import TODAY, Vehicle, normalize_reg
from punchin.scenarios import Expected, GoalState, Scenario

SPEAKERS = {
    "agent": "agent",
    "assistant": "agent",
    "bot": "agent",
    "ai": "agent",
    "sofie": "agent",
    "customer": "customer",
    "kunde": "customer",
    "user": "customer",
    "caller": "customer",
    "human": "customer",
    "client": "customer",
}
LINE = re.compile(r"^\s*([A-Za-zÆØÅæøå ]{2,20})\s*[:>]\s*(.*)$")
TEXT_KEYS = ("text", "content", "message", "utterance", "transcript", "value")
SPEAKER_KEYS = ("speaker", "role", "who", "from", "source", "party")
HEARD_KEYS = ("heard", "asr", "recognised", "recognized", "asr_text")
TIME_KEYS = ("started_at", "timestamp", "time", "ts", "at", "start")


class ImportError_(ValueError):
    """The file did not contain a conversation punchin can read."""


def _first(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((entry[key] for key in keys if entry.get(key) not in (None, "")), None)


def _speaker(raw: Any) -> str | None:
    if raw is None:
        return None
    return SPEAKERS.get(str(raw).strip().lower())


def _moment(raw: Any, fallback: dt.datetime) -> dt.datetime:
    if isinstance(raw, int | float):
        return dt.datetime.fromtimestamp(float(raw), dt.UTC)
    if isinstance(raw, str):
        try:
            parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    return fallback


def _tool_calls(entry: dict[str, Any]) -> list[ToolCall]:
    raw = entry.get("tool_calls") or entry.get("tools") or entry.get("function_calls") or []
    made = []
    for one in raw if isinstance(raw, list) else []:
        if not isinstance(one, dict):
            continue
        name = one.get("tool") or one.get("name") or one.get("function")
        if not name:
            continue
        arguments = one.get("arguments") or one.get("args") or one.get("input") or {}
        made.append(
            ToolCall(
                tool=str(name),
                arguments=arguments if isinstance(arguments, dict) else {"input": arguments},
                result=one.get("result") or one.get("output"),
                error=one.get("error"),
            )
        )
    return made


def turns_from(raw: str) -> list[dict[str, Any]]:
    """The conversation, as a list of plain dicts, whatever shape the file arrived in."""
    stripped = raw.strip()
    if not stripped:
        raise ImportError_("the file is empty")

    lines = [line for line in stripped.splitlines() if line.strip()]
    # JSONL before JSON: a file of objects, one per line, also starts with "{".
    if len(lines) > 1 and all(line.lstrip().startswith("{") for line in lines):
        parsed_lines = []
        for number, line in enumerate(lines, start=1):
            try:
                parsed_lines.append(json.loads(line))
            except json.JSONDecodeError as bad:
                raise ImportError_(f"line {number} is not a JSON object: {bad}") from bad
        return [entry for entry in parsed_lines if isinstance(entry, dict)]

    if stripped.startswith(("[", "{")):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as bad:
            raise ImportError_(f"looks like JSON but does not parse: {bad}") from bad
        if isinstance(parsed, dict):
            for key in ("turns", "conversation", "messages", "transcript", "events"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                raise ImportError_(
                    "a JSON object needs a 'turns', 'conversation', 'messages' or 'transcript' list"
                )
        if not isinstance(parsed, list):
            raise ImportError_("expected a list of turns")
        return [entry for entry in parsed if isinstance(entry, dict)]

    spoken: list[dict[str, Any]] = []
    for line in lines:
        found = LINE.match(line)
        if found and _speaker(found.group(1)):
            spoken.append({"speaker": _speaker(found.group(1)), "text": found.group(2).strip()})
        elif spoken:
            spoken[-1]["text"] = f"{spoken[-1]['text']} {line.strip()}".strip()
    if not spoken:
        raise ImportError_(
            "no turns found. Expected JSON, JSONL, or lines like 'Agent: ...' and 'Kunde: ...'"
        )
    return spoken


def read_call(
    raw: str, *, scenario_id: str, agent: str = "imported", started_at: dt.datetime | None = None
) -> Call:
    """A recording, from somebody else's transcript."""
    entries = turns_from(raw)
    if not entries:
        raise ImportError_("no turns found")
    began = started_at or dt.datetime.now(dt.UTC)
    call = Call(
        id=call_id(scenario_id, agent, began),
        scenario=scenario_id,
        agent=agent,
        customer="imported",
        started_at=began,
        notes={"imported": True},
    )
    unknown: set[str] = set()
    at = began
    for entry in entries:
        speaker = _speaker(_first(entry, SPEAKER_KEYS))
        if speaker is None:
            unknown.add(str(_first(entry, SPEAKER_KEYS)))
            continue
        text = _first(entry, TEXT_KEYS)
        if text is None:
            continue
        at = _moment(_first(entry, TIME_KEYS), at)
        heard = _first(entry, HEARD_KEYS)
        call.turns.append(
            Turn(
                index=len(call.turns),
                speaker=speaker,  # type: ignore[arg-type]
                text=str(text).strip(),
                started_at=at,
                ended_at=at,
                heard=str(heard).strip() if heard is not None else None,
                tool_calls=_tool_calls(entry),
            )
        )
    if not call.turns:
        named = ", ".join(sorted(name for name in unknown if name != "None")) or "none"
        raise ImportError_(f"no turns had a speaker punchin recognises (saw: {named})")
    call.notes["ended_by"] = "imported"
    return call


def scenario_for(
    call: Call,
    *,
    reg: str,
    day: dt.date | None,
    booked: bool,
    why: str = "",
    extras: list[str] | None = None,
) -> Scenario:
    """What should have happened, said once, so the call can be graded.

    The vehicle is a stand-in: it exists so a fork has something to look up and book against. Only the
    registration has to be the real one.
    """
    plate = normalize_reg(reg)
    if not plate:
        raise ImportError_("a scenario needs the registration the customer gave")
    said = " ".join(turn.spoken for turn in call.turns if turn.speaker == "customer")
    return Scenario(
        id=call.scenario,
        pattern="imported",
        vehicle=Vehicle(
            reg=plate,
            make="ukendt",
            model="ukendt",
            year=TODAY.year,
            owner="ukendt",
            syn_due=(day or TODAY) + dt.timedelta(days=12),
        ),
        goal=GoalState(
            intent="book syn" if booked else "no booking",
            reg=plate,
            wants_day=day,
            extras=extras or [],
            mood="unknown, read from a real call",
        ),
        expected=Expected(booked=booked, day=day if booked else None, reg=plate if booked else None),
        why=why or f"a real call, imported; the customer said {len(said.split())} words",
    )
