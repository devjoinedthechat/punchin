"""The customer side of a recording: the scenario's script, played line by line."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from punchin.call import Call
from punchin.scenarios import Line, Scenario

FILLERS = ["Ja?", "Hm, hvad mener du?", "Øh, ja..."]


@dataclass
class CustomerTurn:
    """What the customer said, and what survived of it if it was spoken down a line."""

    text: str
    audio: str | None = None
    heard: str | None = None
    audio_ms: int | None = None


class Customer(Protocol):
    name: str

    def respond(self, call: Call) -> CustomerTurn | None:
        """The customer's next line, or None when they hang up."""


class ScriptedCustomer:
    """Speaks the first unspoken line whose `when` matches what the agent just said.

    An ungated line is said as soon as it is reached. When nothing matches the customer stalls with a
    filler, and hangs up after three in a row; a call that ends that way is a bad call, on purpose.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.name = f"script:{scenario.id}"
        self.queue: list[Line] = list(scenario.script)
        self.stalled = 0

    def respond(self, call: Call) -> CustomerTurn | None:
        last = call.last("agent")
        heard = last.spoken if last else ""
        for line in self.queue:
            if line.when is None or re.search(line.when, heard, re.I):
                self.queue.remove(line)
                self.stalled = 0
                if line.hangup:
                    self.queue.clear()
                return CustomerTurn(line.text)
        self.stalled += 1
        if self.stalled > len(FILLERS):
            return None
        return CustomerTurn(FILLERS[self.stalled - 1])

    @property
    def done(self) -> bool:
        return not self.queue
