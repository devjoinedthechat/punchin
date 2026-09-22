"""A recorded call: who said what, when, and what the agent did to the DMS in between."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Speaker = Literal["agent", "customer"]
FAREWELL = "[FARVEL]"  # the agent ends the call by ending its line with this


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None

    @field_validator("arguments", "result", mode="before")
    @classmethod
    def _as_json(cls, value: Any) -> Any:
        """Dates become strings at construction, so a call in memory equals the same call read back."""
        return json.loads(json.dumps(value, default=str))


class Turn(BaseModel):
    index: int
    speaker: Speaker
    text: str
    started_at: dt.datetime
    ended_at: dt.datetime
    tool_calls: list[ToolCall] = []
    model_ms: int | None = None  # how long the model took, when a model produced the turn
    cost_usd: float = 0.0

    @property
    def ends_call(self) -> bool:
        return self.speaker == "agent" and self.text.rstrip().endswith(FAREWELL)

    @property
    def spoken(self) -> str:
        return self.text.replace(FAREWELL, "").strip()


class Call(BaseModel):
    id: str
    scenario: str
    agent: str
    customer: str
    started_at: dt.datetime
    turns: list[Turn] = []
    bookings: list[dict[str, Any]] = Field(default_factory=list)  # what ended up in the DMS
    notes: dict[str, Any] = Field(default_factory=dict)

    def transcript(self, upto: int | None = None) -> str:
        """The conversation as text, `Agent:` and `Kunde:` lines, for a model to read."""
        lines = []
        for turn in self.turns[:upto]:
            who = "Agent" if turn.speaker == "agent" else "Kunde"
            lines.append(f"{who}: {turn.spoken}")
        return "\n".join(lines)

    def last(self, speaker: Speaker) -> Turn | None:
        return next((t for t in reversed(self.turns) if t.speaker == speaker), None)

    def said(self, speaker: Speaker) -> list[str]:
        return [t.spoken for t in self.turns if t.speaker == speaker]

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.json"
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Call:
        return cls.model_validate_json(path.read_text())


def call_id(scenario: str, agent: str, at: dt.datetime) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", agent.lower()).strip("-")
    return f"{at:%Y%m%d-%H%M%S}-{scenario}-{safe}"


def load_all(directory: Path) -> list[Call]:
    return [Call.load(p) for p in sorted(directory.glob("*.json"))]


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
