"""Which calls are worth forking.

A fork costs a model run and somebody's attention, and an archive has more failures than either. This
groups the failed calls by what went wrong, so the work is ordered by how many customers a fix would
reach rather than by which call somebody happened to listen to.

The grouping is deliberately not a model: it reads the recording's own facts — what the grader said
failed, whether the plate survived the line, whether the agent repeated itself, who hung up. A cluster
you cannot explain in a sentence is not a cluster anybody can act on.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from punchin.call import Call
from punchin.metrics import summarize
from punchin.scenarios import Scenario

# In the order they are reported: the first symptom that fits is the one a call is filed under, so a
# call with several appears once, under the one worth fixing first.
SYMPTOMS: tuple[tuple[str, str], ...] = (
    ("plate never arrived", "the registration the customer said never reached the agent"),
    ("wrong day booked", "a booking was made, on a day the customer did not ask for"),
    ("booked when it should not have", "the customer did not want a booking and got one"),
    ("never booked", "the customer wanted a booking and did not get one"),
    ("note lost", "something the workshop needed to know never reached the booking"),
    ("agent repeated itself", "the same line four times or more"),
    ("customer gave up", "the customer ended the call"),
    ("read out a menu", "three or more times offered in one breath"),
)


def symptom_of(row: dict[str, Any]) -> str | None:
    """The first thing that went wrong on this call, or None if nothing did."""
    if row.get("reg_heard") is False:
        return "plate never arrived"
    if row.get("booked") and not row.get("day_ok"):
        return "wrong day booked"
    if row.get("booked") and not row.get("should_book"):
        return "booked when it should not have"
    if row.get("should_book") and not row.get("booked"):
        return "never booked"
    if row.get("note_ok") is False:
        return "note lost"
    if int(row.get("agent_repeats") or 0) >= 4:
        return "agent repeated itself"
    if row.get("ended_by") == "customer":
        return "customer gave up"
    if int(row.get("options_max") or 0) >= 3:
        return "read out a menu"
    return None


@dataclass
class Cluster:
    symptom: str
    why: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scenarios(self) -> list[str]:
        return sorted({str(row["scenario"]) for row in self.rows})

    def example(self) -> str:
        """The shortest call in the cluster: the cheapest one to fork and the easiest to read."""
        return str(min(self.rows, key=lambda row: int(row.get("turns") or 0))["call"])


@dataclass
class Triage:
    total: int
    clusters: list[Cluster]
    clean: int

    def text(self, limit: int = 10) -> str:
        if not self.clusters:
            return f"{self.total} calls, nothing to fix"
        lines = [
            f"{self.total} calls: {self.clean} came out right, "
            f"{self.total - self.clean} did not, in {len(self.clusters)} groups"
        ]
        for cluster in self.clusters[:limit]:
            share = len(cluster.rows) / self.total
            lines.append(f"\n  {len(cluster.rows):3}  ({share:.0%})  {cluster.symptom} — {cluster.why}")
            lines.append(
                f"       across {len(cluster.scenarios)} scenarios: {', '.join(cluster.scenarios[:6])}"
            )
            lines.append(f"       fork this one first: {cluster.example()}")
        if len(self.clusters) > limit:
            lines.append(f"\n  and {len(self.clusters) - limit} smaller groups")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "clean": self.clean,
            "clusters": [
                {
                    "symptom": c.symptom,
                    "why": c.why,
                    "calls": len(c.rows),
                    "share": len(c.rows) / self.total if self.total else 0.0,
                    "scenarios": c.scenarios,
                    "fork_first": c.example(),
                }
                for c in self.clusters
            ],
        }


def triage(calls: Sequence[Call], known: dict[str, Scenario]) -> Triage:
    """Group the calls that went wrong, biggest group first."""
    rows = [summarize(call, known[call.scenario]) for call in calls if call.scenario in known]
    found: dict[str, Cluster] = {}
    clean = 0
    for row in rows:
        symptom = symptom_of(row)
        if symptom is None:
            clean += 1
            continue
        why = next(text for name, text in SYMPTOMS if name == symptom)
        found.setdefault(symptom, Cluster(symptom, why)).rows.append(row)
    order = Counter({name: len(cluster.rows) for name, cluster in found.items()})
    ranked = [found[name] for name, _ in order.most_common()]
    return Triage(len(rows), ranked, clean)
