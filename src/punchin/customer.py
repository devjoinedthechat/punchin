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
    """The other side of the call: a script, a simulator pinned to a real customer, or either of
    those spoken aloud and heard back through a recogniser.

    Whatever it returns is the answer key. A wrapper may add what a recogniser made of it, but the
    words on `CustomerTurn.text` are what the customer really said, and graders score against those.
    """

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

    def fast_forward(self, call: Call) -> ScriptedCustomer:
        """Drop the lines this customer has already said in `call`, so a fork resumes mid-script.

        A fork hands the customer a conversation it did not have. Without this it would start again
        at the top of its script and say the opening line into the middle of a call.
        """
        already = {" ".join(turn.spoken.split()) for turn in call.turns if turn.speaker == "customer"}
        self.queue = [line for line in self.queue if " ".join(line.text.split()) not in already]
        return self

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
