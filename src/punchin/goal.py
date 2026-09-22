"""What the customer wanted, read back out of a recorded call, and the facts a line reveals."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from punchin.call import Call
from punchin.dms import normalize_reg
from punchin.model import Model
from punchin.scenarios import WEEKDAYS, GoalState, regs_mentioned, weekdays_mentioned

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "description": "What the customer wanted from the call, in a few words"},
        "reg": {
            "type": "string",
            "description": "The plate the customer meant, letters and digits only; '' if none",
        },
        "wants_day": {
            "type": ["string", "null"],
            "description": "The day the customer wanted, YYYY-MM-DD, after any correction; null if none",
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Conditions the customer set",
        },
        "extras": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Things the workshop should know",
        },
        "formality": {"type": "string", "enum": ["informal", "formal"]},
        "prefers_time": {
            "type": ["string", "null"],
            "description": "A time of day the customer asked for, in their own Danish words; null if none",
        },
        "manner": {
            "type": "array",
            "items": {"type": "string"},
            "description": "How they talk, in Danish: 'retter sig selv om dagen', 'siger alt i én sætning'",
        },
        "mood": {"type": "string", "description": "How the customer came across, in a few words"},
        "reveals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The facts above, in the order the customer first said them: ['reg', 'day', ...]",
        },
    },
    "required": [
        "intent",
        "reg",
        "wants_day",
        "constraints",
        "extras",
        "formality",
        "mood",
        "prefers_time",
        "manner",
        "reveals",
    ],
    "additionalProperties": False,
}

SYSTEM = """\
You read the transcript of a phone call between a car workshop's agent and a customer, in Danish, and
write down the customer's goal state: what they wanted, what they knew, what they required, and how they
talked. Only what the customer said or clearly meant counts; nothing the agent said or did is evidence of
what the customer wanted. When the customer corrects themselves, the correction is what they meant.
Weekdays are resolved against today's date, given in the prompt; 'næste uge' means the week after this one.

Write down how they talked as well as what they said: if they corrected themselves, said everything in one
breath, or asked for a particular time of day, that is part of who this customer is and belongs in `manner`
and `prefers_time`. Describe the habit, never quote the line.
"""

PROMPT = "Today is {today} ({weekday}).\n\nTranscript:\n\n{transcript}\n\nWrite the customer's goal state."


def extract(call: Call, model: Model, today: dt.date) -> tuple[GoalState, float]:
    """The goal state a model reads out of the call, and what it cost."""
    prompt = PROMPT.format(
        today=today.isoformat(), weekday=WEEKDAYS[today.weekday()], transcript=call.transcript()
    )
    done = model.complete(SYSTEM, prompt, schema=SCHEMA)
    if not isinstance(done.structured, dict):
        raise TypeError(f"extraction did not return an object: {done.text[:200]!r}")
    raw = dict(done.structured)
    raw["reg"] = normalize_reg(raw.get("reg") or "")
    return GoalState.model_validate(raw), done.cost_usd


YES = re.compile(r"\b(ja|jo|yes|fint|okay|ok|gerne|perfekt)\b", re.I)
NO = re.compile(r"\b(nej|no|ikke|vent)\b", re.I)
# Only an actual parting. A bare "hej" opens a Danish call as often as it closes one, and "tak" is
# politeness anywhere in it; both fired on greetings and scored a faithful line as a miss.
BYE = re.compile(r"\bhej hej\b|\bfarvel\b|\bvi ses\b|\bha'? det\b|\bhav en god\b", re.I)
ROBOT = re.compile(r"\brobot\b", re.I)
NEXT_WEEK = re.compile(r"næste uge|ikke i denne uge", re.I)
TIME = re.compile(
    r"\b\d{1,2}[:.]\d{2}\b|\bkl\.?\s*\d{1,2}\b|\bklokken\b|\btidlig\w*|\bformiddag\b"
    r"|\beftermiddag\b|\bmorgen\w*|\bførste\b|\bsen\w*\b",
    re.I,
)


def facts(goal: GoalState, text: str) -> set[str]:
    """Which of the goal's facts a line reveals, plus the moves anyone can make: yes, no, bye, robot."""
    found: set[str] = set()
    if goal.reg and goal.reg in regs_mentioned(text):
        found.add("reg")
    if goal.wants_day is not None and WEEKDAYS[goal.wants_day.weekday()] in weekdays_mentioned(text):
        found.add("day")
    if NEXT_WEEK.search(text):
        found.add("next_week")
    if goal.prefers_time and TIME.search(text):
        found.add("time")  # a turn that is only a time preference scored as saying nothing at all
    for kind, items in (("constraint", goal.constraints), ("extra", goal.extras)):
        for item in items:
            words = [w for w in re.findall(r"\w+", item.lower()) if len(w) >= 4]
            if words and any(w in text.lower() for w in words):
                found.add(f"{kind}:{item}")
    if YES.search(text):
        found.add("yes")
    if NO.search(text):
        found.add("no")
    if BYE.search(text):
        found.add("bye")
    if ROBOT.search(text):
        found.add("robot")
    return found


def score(found: GoalState, truth: GoalState) -> dict[str, Any]:
    """How an extracted goal state compares with the scenario's: the plate, the day, and the extras."""
    want = {e.lower() for e in truth.constraints + truth.extras}
    got = {e.lower() for e in found.constraints + found.extras}
    overlap = _loose_overlap(got, want)
    return {
        "reg": found.reg == truth.reg,
        "day": found.wants_day == truth.wants_day,
        "extras_recall": overlap / len(want) if want else 1.0,
        "formality": found.formality == truth.formality,
    }


def _loose_overlap(got: set[str], want: set[str]) -> int:
    hits = 0
    for w in want:
        words = [x for x in re.findall(r"\w+", w) if len(x) >= 4]
        if any(any(x in g for x in words) for g in got):
            hits += 1
    return hits
