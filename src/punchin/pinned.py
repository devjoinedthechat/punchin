"""The pinned customer: a simulator that may say only what the original customer knew and wanted."""

from __future__ import annotations

from punchin.call import Call
from punchin.model import Model
from punchin.scenarios import WEEKDAYS, GoalState

HANGUP = "[LÆGGER PÅ]"

SYSTEM = """\
Du spiller en bestemt kunde i en telefonsamtale med et autoværksted, der ringer om syn. Du er kunden, ikke
agenten, og du skriver kun det, kunden siger højt i sin næste replik.

Det du ved og vil, står under "Kunden". Du må ikke finde på noget, der ikke står der: ingen andre biler,
dage, navne eller ønsker. Spørger agenten om noget, du ikke ved, så sig kort, at det ved du ikke. Nævn
tingene i den rækkefølge, de står i under "Rækkefølge", medmindre agenten spørger om noget tidligere.
Sig kun det, der er relevant for det, agenten lige har sagt; svar på ét spørgsmål ad gangen, kort, som
man taler i telefon. Hold tonen: {formality_da}, og {mood}.

Når agenten siger farvel, så sig farvel og afslut din replik med præcis dette: {hangup}
Ingen anførselstegn, ingen forklaringer, ingen regibemærkninger.
"""

PROMPT = """\
Kunden:
- vil: {intent}
- nummerplade: {reg}
- ønsket dag: {day}
- betingelser: {constraints}
- værkstedet skal vide: {extras}

Rækkefølge: {reveals}

Samtalen indtil nu:

{transcript}

Skriv kundens næste replik.
"""

FORMALITY_DA = {"informal": "uformel, siger du", "formal": "høflig, siger De"}


class PinnedCustomer:
    def __init__(self, model: Model, goal: GoalState, *, name: str | None = None) -> None:
        self.model = model
        self.goal = goal
        self.name = name or f"pinned:{model.name}"
        self.hung_up = False
        self.cost_usd = 0.0

    def line(self, call: Call) -> str:
        """The next line, without ending the call; what teacher-forcing asks for."""
        goal = self.goal
        day = (
            f"{WEEKDAYS[goal.wants_day.weekday()]} {goal.wants_day.isoformat()}"
            if goal.wants_day
            else "ingen, vil ikke booke"
        )
        prompt = PROMPT.format(
            intent=goal.intent,
            reg=" ".join([goal.reg[:2], goal.reg[2:4], goal.reg[4:]]) if goal.reg else "ukendt",
            day=day,
            constraints=", ".join(goal.constraints) or "ingen",
            extras=", ".join(goal.extras) or "ikke noget",
            reveals=", ".join(goal.reveals) or "som det falder naturligt",
            transcript=call.transcript() or "(agenten har ikke sagt noget endnu)",
        )
        system = SYSTEM.format(formality_da=FORMALITY_DA[goal.formality], mood=goal.mood, hangup=HANGUP)
        done = self.model.complete(system, prompt)
        self.cost_usd += done.cost_usd
        return done.text.strip()

    def respond(self, call: Call) -> str | None:
        if self.hung_up:
            return None
        said = self.line(call)
        if said.endswith(HANGUP):
            self.hung_up = True
            said = said.removesuffix(HANGUP).strip()
        return said or None
