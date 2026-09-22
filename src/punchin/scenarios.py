"""The corpus: Danish syn-reminder calls with the truth known by construction.

Each scenario is a customer with a goal state (what they want, what they know, how they talk) and a
script of how they reveal it. The script is the customer during recording, so every recorded call
comes with its own ground truth: the day they meant, the plate they own, the outcome that is right.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from punchin.dms import TODAY, Vehicle

Formality = Literal["informal", "formal"]
Pattern = Literal[
    "plain",
    "self_correction",
    "robot_check",
    "code_switch",
    "proxy_caller",
    "wrong_reg_first",
    "next_week",
    "already_booked",
    "courtesy_car",
    "hurried",
    "imported",  # a real call, with the outcome somebody stated after the fact
]

WEEKDAYS = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]


def next_weekday(name: str, after: dt.date = TODAY, weeks_ahead: int = 0) -> dt.date:
    """The next `name` strictly after `after`, pushed by `weeks_ahead` whole weeks."""
    target = WEEKDAYS.index(name)
    days = (target - after.weekday()) % 7 or 7
    return after + dt.timedelta(days=days + 7 * weeks_ahead)


class GoalState(BaseModel):
    """What the customer wants and knows. The pinned simulator may reveal nothing outside it."""

    intent: str
    reg: str
    wants_day: dt.date | None = None  # None: the customer does not want a booking
    constraints: list[str] = []
    extras: list[str] = []  # things the workshop should know: a noise, a courtesy car
    formality: Formality = "informal"
    mood: str = "neutral"
    prefers_time: str | None = None  # a time of day the customer asked for, in their own words
    manner: list[str] = []  # how they talk: self-corrections, everything at once, one word at a time
    reveals: list[str] = []  # the facts in the order the customer brought them up, when extracted from a call


@dataclass(frozen=True)
class Line:
    """One scripted customer line, spoken when the agent's last line matches `when` (or at once)."""

    text: str
    when: str | None = None  # a regex against the agent's last line, case-insensitive
    hangup: bool = False


class Expected(BaseModel):
    booked: bool
    day: dt.date | None = None
    reg: str | None = None


class Scenario(BaseModel):
    id: str
    pattern: Pattern
    vehicle: Vehicle
    goal: GoalState
    # Empty for an imported call: its customer is pinned to an extracted goal, not read off a script.
    # Such a scenario can be graded and forked, but not recorded from scratch.
    script: list[Line] = []
    expected: Expected
    why: str = Field(description="What this scenario is built to catch")


def _vehicle(reg: str, make: str, model: str, year: int, owner: str, *, due_in: int = 12) -> Vehicle:
    return Vehicle(
        reg=reg, make=make, model=model, year=year, owner=owner, syn_due=TODAY + dt.timedelta(days=due_in)
    )


ASKED_REG = r"nummerplade|registreringsnummer|reg\.? ?nr"
ASKED_DAY = r"hvilken dag|hvornår|passer dig|dag der passer|hvad med"
OFFERED_TIME = r"kl\.? ?\d|klokken"
ASKED_CONFIRM = r"skal jeg booke|skal jeg reservere|er det (i orden|fint|rigtigt)|bekræft"
CHOOSE_TIME = r"hvilket (af )?tidspunkt|hvilken tid|hvilket af|passer et af|hvad passer bedst|eller"
SAID_BYE = r"hej hej|farvel|god dag|ha' det|hav en god"

TUE, WED, THU, FRI = (next_weekday(d) for d in ("tirsdag", "onsdag", "torsdag", "fredag"))
NEXT_WED = next_weekday("onsdag", weeks_ahead=1)

SCENARIOS: list[Scenario] = [
    Scenario(
        id="self-correction",
        pattern="self_correction",
        vehicle=_vehicle("AB12345", "Škoda", "Octavia", 2019, "Mette Kjær"),
        goal=GoalState(
            intent="book syn",
            reg="AB12345",
            wants_day=WED,
            mood="a bit distracted",
            prefers_time="den tidligste",
            manner=["retter sig selv om dagen"],
        ),
        script=[
            Line("Ja hej. Ja, det er fint, den skal til syn.", when=None),
            Line("Det er AB 12 345.", when=ASKED_REG),
            Line("Kan jeg få en tid tirsdag? ...nej vent, onsdag. Onsdag er bedre.", when=ASKED_DAY),
            Line("Bare den tidligste, klokken 8.", when=CHOOSE_TIME),
            Line("Ja, det er fint.", when=ASKED_CONFIRM),
            Line("Ja tak, det passer.", when=OFFERED_TIME),
            Line("Tak skal du have, hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=WED, reg="AB12345"),
        why="The customer corrects the day mid-sentence. Booking Tuesday succeeds as a tool call, wrongly.",
    ),
    Scenario(
        id="plain-booking",
        pattern="plain",
        vehicle=_vehicle("CD67890", "Toyota", "Yaris", 2021, "Jonas Berg"),
        goal=GoalState(intent="book syn", reg="CD67890", wants_day=THU, prefers_time="formiddag"),
        script=[
            Line("Hej, ja det passer fint.", when=None),
            Line("CD 67 890.", when=ASKED_REG),
            Line("Torsdag formiddag ville være godt.", when=ASKED_DAY),
            Line("Klokken 8 er fint.", when=CHOOSE_TIME),
            Line("Ja, book den.", when=ASKED_CONFIRM),
            Line("Ja tak.", when=OFFERED_TIME),
            Line("Tak, hej hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=THU, reg="CD67890"),
        why="The control: nothing goes wrong, so anything a change breaks here is a regression.",
    ),
    Scenario(
        id="is-it-a-robot",
        pattern="robot_check",
        vehicle=_vehicle("EF11223", "Peugeot", "208", 2018, "Karen Holm"),
        goal=GoalState(intent="book syn", reg="EF11223", wants_day=FRI, mood="sceptical"),
        script=[
            Line("Vent lige... er det en robot jeg taler med?", when=None),
            Line("Okay. Nå, men ja, den skal vel til syn.", when=r"."),
            Line("EF 11 223.", when=ASKED_REG),
            Line("Fredag, hvis der er noget.", when=ASKED_DAY),
            Line("Klokken 8.", when=CHOOSE_TIME),
            Line("Ja okay.", when=ASKED_CONFIRM),
            Line("Ja, fint.", when=OFFERED_TIME),
            Line("Hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=FRI, reg="EF11223"),
        why="A direct question about being a machine. The honest answer is free; a dodge costs the call.",
    ),
    Scenario(
        id="code-switch",
        pattern="code_switch",
        vehicle=_vehicle("GH44556", "Citroën", "C3", 2017, "Amir Haddad"),
        goal=GoalState(
            intent="book syn and a service", reg="GH44556", wants_day=TUE, extras=["stor service"]
        ),
        script=[
            Line("Hej ja. Jeg skal også have den store service, you know, the big one, samtidig.", when=None),
            Line("GH 44 556.", when=ASKED_REG),
            Line("Tirsdag.", when=ASKED_DAY),
            Line("Eight o'clock, klokken 8.", when=CHOOSE_TIME),
            Line("Yes, book it.", when=ASKED_CONFIRM),
            Line("Ja.", when=OFFERED_TIME),
            Line("Thanks, hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=TUE, reg="GH44556"),
        why="Danish and English in one breath, plus an extra job the note should carry to the workshop.",
    ),
    Scenario(
        id="proxy-caller",
        pattern="proxy_caller",
        vehicle=_vehicle("IJ77889", "Volkswagen", "Golf", 2020, "Lise Andersen"),
        goal=GoalState(intent="book syn for spouse's car", reg="IJ77889", wants_day=THU),
        script=[
            Line("Hej. Det er min kones bil, men jeg kan godt booke for hende.", when=None),
            Line("IJ 77 889.", when=ASKED_REG),
            Line("Torsdag.", when=ASKED_DAY),
            Line("Klokken 8.", when=CHOOSE_TIME),
            Line("Ja.", when=ASKED_CONFIRM),
            Line("Det er fint.", when=OFFERED_TIME),
            Line("Tak, hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=THU, reg="IJ77889"),
        why="The caller is not the owner on file. The booking is still right; refusing it is not.",
    ),
    Scenario(
        id="wrong-reg-first",
        pattern="wrong_reg_first",
        vehicle=_vehicle("KL99001", "Ford", "Focus", 2016, "Søren Lund"),
        goal=GoalState(
            intent="book syn",
            reg="KL99001",
            wants_day=WED,
            manner=["siger nummerpladen forkert og retter den"],
        ),
        script=[
            Line("Ja hej, det passer.", when=None),
            Line("KL 99 010... nej, KL 99 001. Undskyld.", when=ASKED_REG),
            Line("KL 99 001.", when=r"igen|ikke finde|kan du gentage|en gang til"),
            Line("Onsdag.", when=ASKED_DAY),
            Line("Den første, klokken 8.", when=CHOOSE_TIME),
            Line("Ja.", when=ASKED_CONFIRM),
            Line("Ja tak.", when=OFFERED_TIME),
            Line("Hej hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=WED, reg="KL99001"),
        why="Two plates in one line, the second one right. The lookup must land on the corrected one.",
    ),
    Scenario(
        id="next-week",
        pattern="next_week",
        vehicle=_vehicle("MN22334", "Hyundai", "i30", 2022, "Nadia Petersen"),
        goal=GoalState(
            intent="book syn", reg="MN22334", wants_day=NEXT_WED, constraints=["ikke i denne uge"]
        ),
        script=[
            Line("Hej, ja.", when=None),
            Line("MN 22 334.", when=ASKED_REG),
            Line("Ikke i denne uge. Onsdag i næste uge?", when=ASKED_DAY),
            Line("Klokken 8.", when=CHOOSE_TIME),
            Line("Ja.", when=ASKED_CONFIRM),
            Line("Ja, fint.", when=OFFERED_TIME),
            Line("Tak, hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=NEXT_WED, reg="MN22334"),
        why="'Onsdag i næste uge' is the Wednesday after this one. The nearest Wednesday is the wrong one.",
    ),
    Scenario(
        id="already-booked",
        pattern="already_booked",
        vehicle=_vehicle("OP55667", "Kia", "Ceed", 2019, "Thomas Friis"),
        goal=GoalState(intent="decline, already booked elsewhere", reg="OP55667", wants_day=None),
        script=[
            Line("Hej. Jeg har faktisk allerede booket syn et andet sted, så det behøver I ikke.", when=None),
            Line("Nej tak, det er fint.", when=r"."),
            Line("Hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=False, reg="OP55667"),
        why="The right outcome is no booking. An agent that books anyway 'succeeds' at the wrong thing.",
    ),
    Scenario(
        id="courtesy-car",
        pattern="courtesy_car",
        vehicle=_vehicle("QR88990", "Volvo", "V60", 2018, "Hanne Dahl"),
        goal=GoalState(
            intent="book syn",
            reg="QR88990",
            wants_day=FRI,
            constraints=["skal have lånebil"],
            extras=["lånebil"],
        ),
        script=[
            Line("Ja hej. Jeg skal have en lånebil imens, ellers kan jeg ikke.", when=None),
            Line("QR 88 990.", when=ASKED_REG),
            Line("Fredag.", when=ASKED_DAY),
            Line("Klokken 8, hvis der er lånebil.", when=CHOOSE_TIME),
            Line("Ja, hvis der er lånebil.", when=ASKED_CONFIRM),
            Line("Ja tak.", when=OFFERED_TIME),
            Line("Tak, hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=FRI, reg="QR88990"),
        why="A hard constraint, stated up front. The note has to carry it or the customer arrives stranded.",
    ),
    Scenario(
        id="hurried",
        pattern="hurried",
        vehicle=_vehicle("ST33445", "BMW", "118i", 2020, "Rasmus Vinther"),
        goal=GoalState(
            intent="book syn",
            reg="ST33445",
            wants_day=FRI,
            mood="in a hurry",
            prefers_time="den første, lige meget hvilken",
            manner=["siger det hele i én sætning"],
        ),
        script=[
            Line("Ja, jeg har lidt travlt. Bare book noget fredag, ST 33 445.", when=None),
            Line("Den første, bare.", when=CHOOSE_TIME),
            Line("Ja ja, fint.", when=ASKED_CONFIRM),
            Line("Ja.", when=OFFERED_TIME),
            Line("Fint, hej.", when=SAID_BYE, hangup=True),
        ],
        expected=Expected(booked=True, day=FRI, reg="ST33445"),
        why="Everything in the first breath. Asking for the plate again is the failure here.",
    ),
]

BY_ID: dict[str, Scenario] = {s.id: s for s in SCENARIOS}

SCENARIO_DIR = Path(".punchin/scenarios")


def load_scenarios(directory: Path | None = SCENARIO_DIR) -> dict[str, Scenario]:
    """The built-in corpus, plus every `*.json` scenario in `directory`, which wins on a clash.

    This is how a call punchin did not invent gets graded: `punchin import` writes the outcome somebody
    says was right next to the recording, and everything downstream reads it from here.
    """
    known = dict(BY_ID)
    if directory is None or not directory.is_dir():
        return known
    for path in sorted(directory.glob("*.json")):
        try:
            scenario = Scenario.model_validate_json(path.read_text())
        except ValueError as bad:
            raise ValueError(f"{path} is not a scenario: {bad}") from bad
        known[scenario.id] = scenario
    return known


def ungraded(scenario_id: str, directory: Path | None) -> str:
    """The sentence to print when a recording names an outcome nobody has stated."""
    where = f" or in {directory}" if directory else ""
    return (
        f"no scenario {scenario_id!r} in the built-in corpus{where}, so there is nothing to grade this "
        f"call against. `punchin import` writes one from the outcome you say was right."
    )


def spell_plate(reg: str) -> str:
    """A plate the way it is said: 'AB12345' -> 'AB 12 345'."""
    return f"{reg[:2]} {reg[2:4]} {reg[4:]}"


def call_list() -> str:
    """The plates of every vehicle in the corpus, as an outbound campaign would already know them."""
    return ", ".join(spell_plate(s.vehicle.reg) for s in SCENARIOS)


def vocabulary() -> str:
    """What the recogniser is told to expect: the campaign's plates, and nothing else.

    Adding the weekdays and opening hours as well was tried and measured, and it cost more than it
    bought: plate recovery fell from 7 of 10 to 5 of 10, and the day it was supposed to rescue was
    still lost. Bias capacity is scarce. Spend it on the set that is high value, closed, and known
    before the phone rings; the day is better fixed by reading it back to the customer.
    """
    return call_list()


def weekdays_mentioned(text: str) -> list[str]:
    """Weekday names in the order they were said, so a policy can take the first or the last."""
    return re.findall(r"\b(mandag|tirsdag|onsdag|torsdag|fredag)\b", text.lower())


def regs_mentioned(text: str) -> list[str]:
    """Plate-shaped tokens in the order they were said, normalized."""
    return [re.sub(r"[^A-Z0-9]", "", m) for m in re.findall(r"\b[A-Za-z]{2}[ .-]?\d{2}[ .-]?\d{3}\b", text)]
