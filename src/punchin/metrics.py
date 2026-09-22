"""What a recorded call did, and what it felt like, as numbers.

Outcome is against the scenario's ground truth. Feel is from the transcript and timing; in text mode the
timing is the model's latency only, so it is reported as that and not as anything a caller heard.
"""

from __future__ import annotations

import datetime as dt
import re
from statistics import mean
from typing import Any

from punchin.call import Call
from punchin.customer import FILLERS
from punchin.scenarios import Scenario

TIME_SENTENCE = re.compile(r"[^.!?]*\b(?:kl\.?|klokken)\b[^.!?]*", re.I)
SPELLED = "otte|ni|ti|elleve|tolv|tretten|fjorten|femten|ét"
TIME_TOKEN = re.compile(
    rf"\b\d{{1,2}}[:.]\d{{2}}\b|\bhalv \w+|\b(?:kl\.?|klokken)\s*\d{{1,2}}\b|\b(?:{SPELLED})\b",
    re.I,
)


def options_offered(text: str) -> int:
    """How many clock times one line puts in front of the customer. Five is a menu, not a question."""
    # 'kl.' and '8.00' end no sentence; without this the splitter cuts a time in half and counts none.
    flat = re.sub(r"\bkl\.", "kl", text)
    flat = re.sub(r"(\d)\.(\d{2})\b", r"\1:\2", flat)
    return max((len(TIME_TOKEN.findall(s)) for s in TIME_SENTENCE.findall(flat)), default=0)


def outcome(call: Call, scenario: Scenario) -> dict[str, Any]:
    booked = call.bookings[0] if call.bookings else None
    day = dt.date.fromisoformat(booked["date"]) if booked else None
    want = scenario.expected
    extras = [e.lower() for e in scenario.goal.extras]
    note = (booked or {}).get("note", "").lower()
    carried = all(any(w in note for w in re.findall(r"\w{4,}", e)) for e in extras) if booked else not extras
    return {
        "booked": booked is not None,
        "should_book": want.booked,
        "day_ok": day == want.day,
        "reg_ok": (booked or {}).get("reg") == want.reg if want.booked else booked is None,
        "note_ok": carried,
        "correct": (booked is not None) == want.booked
        and day == want.day
        and (booked is None or booked["reg"] == want.reg),
    }


def feel(call: Call) -> dict[str, Any]:
    agent = [t for t in call.turns if t.speaker == "agent"]
    customer = [t for t in call.turns if t.speaker == "customer"]
    latencies = [t.model_ms for t in agent if t.model_ms]
    return {
        "turns": len(call.turns),
        "agent_words_mean": round(mean(len(t.spoken.split()) for t in agent), 1) if agent else 0.0,
        "agent_words_max": max((len(t.spoken.split()) for t in agent), default=0),
        "options_max": max((options_offered(t.spoken) for t in agent), default=0),
        "questions_per_turn_max": max((t.spoken.count("?") for t in agent), default=0),
        "customer_stalls": sum(t.spoken in FILLERS for t in customer),
        "ended_by": call.notes.get("ended_by"),
        "model_ms_mean": round(mean(latencies)) if latencies else None,
        "model_ms_max": max(latencies) if latencies else None,
        "cost_usd": round(call.cost_usd, 4),
    }


def summarize(call: Call, scenario: Scenario) -> dict[str, Any]:
    return {
        "call": call.id,
        "scenario": scenario.id,
        "agent": call.agent,
        **outcome(call, scenario),
        **feel(call),
    }
