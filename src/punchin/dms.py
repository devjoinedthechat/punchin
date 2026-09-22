"""A small dealership management system with three tools, kept in a JSON file.

The state lives in a file rather than in memory because Claude Code in print mode starts the MCP
server afresh for every agent turn; the file is what makes the second turn remember the first.
"""

import datetime as dt
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

TODAY = dt.date(2026, 9, 28)  # a Monday; every scenario is dated relative to it
OPENING_HOURS = ("08:00", "08:30", "10:00", "13:00", "14:30")

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
)


def normalize_reg(reg: str) -> str:
    """Danish plates are two letters and five digits; people say them with spaces, dots or dashes."""
    return re.sub(r"[^A-Z0-9]", "", reg.upper())


class Vehicle(BaseModel):
    reg: str
    make: str
    model: str
    year: int
    owner: str
    syn_due: dt.date


class Slot(BaseModel):
    id: str
    date: dt.date
    time: str
    taken: bool = False


class Booking(BaseModel):
    reg: str
    slot_id: str
    date: dt.date
    time: str
    note: str = ""


class State(BaseModel):
    vehicles: dict[str, Vehicle] = {}
    slots: dict[str, Slot] = {}
    bookings: list[Booking] = []
    calls: list[dict[str, Any]] = []  # every tool call, in order, for the recording

    def vehicle(self, reg: str) -> Vehicle:
        found = self.vehicles.get(normalize_reg(reg))
        if found is None:
            raise ToolError(f"no vehicle registered as {reg!r}")
        return found

    def free_slots(self, date_from: dt.date, date_to: dt.date) -> list[Slot]:
        return sorted(
            (s for s in self.slots.values() if not s.taken and date_from <= s.date <= date_to),
            key=lambda s: (s.date, s.time),
        )

    def book(self, slot_id: str, reg: str, note: str = "") -> Booking:
        vehicle = self.vehicle(reg)
        slot = self.slots.get(slot_id)
        if slot is None:
            raise ToolError(f"no slot {slot_id!r}")
        if slot.taken:
            raise ToolError(f"slot {slot_id} is already taken")
        slot.taken = True
        booking = Booking(reg=vehicle.reg, slot_id=slot.id, date=slot.date, time=slot.time, note=note)
        self.bookings.append(booking)
        return booking


def fresh(vehicles: list[Vehicle], today: dt.date = TODAY, weeks: int = 2) -> State:
    """A workshop with every weekday slot free for the next `weeks`, and the given vehicles on file."""
    slots: dict[str, Slot] = {}
    for offset in range(1, weeks * 7 + 1):
        day = today + dt.timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        for time in OPENING_HOURS:
            slot_id = f"{day:%Y%m%d}-{time.replace(':', '')}"
            slots[slot_id] = Slot(id=slot_id, date=day, time=time)
    return State(vehicles={v.reg: v for v in vehicles}, slots=slots)


class Dms:
    """The three tools, callable in-process; `serve` exposes the same three over MCP."""

    def __init__(self, path: Path) -> None:
        # Absolute, because the MCP server is started by Claude Code from a directory of its own.
        self.path = path.resolve()

    def load(self) -> State:
        return State.model_validate_json(self.path.read_text())

    def save(self, state: State) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(state.model_dump_json(indent=2))
        Path(tmp).replace(self.path)

    def _record(
        self, state: State, tool: str, arguments: dict[str, Any], result: Any, error: str | None
    ) -> None:
        state.calls.append({"tool": tool, "arguments": arguments, "result": result, "error": error})

    DATES = ("date_from", "date_to")

    def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        # A recording stores arguments as JSON, so replaying one hands back "2026-09-30" where the
        # tool wants a date. Without this, every fork whose prefix searched for slots dies comparing
        # a string to a date.
        arguments = {
            key: dt.date.fromisoformat(value) if key in self.DATES and isinstance(value, str) else value
            for key, value in arguments.items()
        }
        state = self.load()
        handler: Callable[..., dict[str, Any]] = getattr(self, f"_{tool}")
        try:
            result = handler(state, **arguments)
        except ToolError as error:
            self._record(state, tool, arguments, None, str(error))
            self.save(state)
            raise
        self._record(state, tool, arguments, result, None)
        self.save(state)
        return result

    def _lookup_vehicle(self, state: State, reg: str) -> dict[str, Any]:
        vehicle = state.vehicle(reg)
        return {**vehicle.model_dump(mode="json"), "syn_due_in_days": (vehicle.syn_due - TODAY).days}

    def _find_slots(self, state: State, date_from: dt.date, date_to: dt.date) -> dict[str, Any]:
        slots = state.free_slots(date_from, date_to)
        return {
            "slots": [s.model_dump(mode="json", exclude={"taken"}) for s in slots[:12]],
            "total": len(slots),
        }

    def _book(self, state: State, slot_id: str, reg: str, note: str = "") -> dict[str, Any]:
        return state.book(slot_id, reg, note).model_dump(mode="json")


INSTRUCTIONS = """\
Værkstedets bookingsystem. lookup_vehicle finder bilen ud fra nummerpladen, find_slots viser ledige
tider, og book reserverer en tid til bilen. Book aldrig uden at kunden har sagt ja til dagen og tiden.
"""


def build_server(dms: Dms) -> MCPServer:
    server = MCPServer(name="dms", instructions=INSTRUCTIONS)

    @server.tool(annotations=READ)
    def lookup_vehicle(
        reg: Annotated[str, Field(description="Nummerpladen, fx 'AB 12 345'")],
    ) -> dict[str, Any]:
        """Bilen og dens ejer ud fra nummerpladen, og hvor mange dage der er til synet udløber."""
        return dms.call("lookup_vehicle", reg=reg)

    @server.tool(annotations=READ)
    def find_slots(
        date_from: Annotated[dt.date, Field(description="Første dag, YYYY-MM-DD")],
        date_to: Annotated[dt.date, Field(description="Sidste dag, YYYY-MM-DD")],
    ) -> dict[str, Any]:
        """Ledige tider mellem to datoer, tidligste først, med det slot_id der skal bruges til book."""
        return dms.call("find_slots", date_from=date_from, date_to=date_to)

    @server.tool(annotations=WRITE)
    def book(
        slot_id: Annotated[str, Field(description="slot_id fra find_slots")],
        reg: Annotated[str, Field(description="Nummerpladen på bilen der skal bookes")],
        note: Annotated[
            str, Field(description="Kort note til værkstedet, fx 'lånebil' eller 'raslelyd'")
        ] = "",
    ) -> dict[str, Any]:
        """Reserverer tiden til bilen. Kun når kunden har sagt ja til dag og tid."""
        return dms.call("book", slot_id=slot_id, reg=reg, note=note)

    return server


def serve(path: Path) -> None:
    build_server(Dms(path)).run("stdio")
