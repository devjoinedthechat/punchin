"""Import a transcript in the crudest shape anyone would hand over, and check it grades as wrong.

Run in CI. If importing breaks, an archive stops being usable, and nothing else in the suite notices:
every other test starts from a recording punchin made itself.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from punchin.call import Call
from punchin.cli import main
from punchin.metrics import summarize
from punchin.scenarios import load_scenarios

TRANSCRIPT = """\
Agent: Hej, det er Sofie fra værkstedet. Synet udløber snart. Passer det nu?
Kunde: Ja, det er fint.
Agent: Må jeg få nummerpladen?
Kunde: Det er XY 55 123.
Agent: Hvilken dag passer dig?
Kunde: Torsdag, tak. Og jeg skal have en lånebil.
Agent: Jeg har en tid fredag den 2. oktober klokken 8. Skal jeg booke den?
Kunde: Ja tak.
Agent: Så er den booket. Hej hej.
"""


def run() -> int:
    with tempfile.TemporaryDirectory(prefix="punchin-import-smoke-") as workspace:
        root = Path(workspace)
        source = root / "call.txt"
        source.write_text(TRANSCRIPT)
        calls, scenarios = root / "calls", root / "scenarios"

        imported = main(
            [
                "import",
                str(source),
                "--id",
                "ci-import",
                "--reg",
                "XY 55 123",
                "--day",
                "2026-10-01",
                "--booked",
                "--booked-day",
                "2026-10-02",
                "--extra",
                "lånebil",
                "--out",
                str(calls),
                "--scenarios",
                str(scenarios),
                "-q",
            ]
        )
        if imported != 0:
            print("import failed", file=sys.stderr)
            return 1

        call = Call.load(next(calls.glob("2026*.json")))
        row = summarize(call, load_scenarios(scenarios)["ci-import"])
        checks = {
            "nine turns read": len(call.turns) == 9,
            "the plate reached the scenario": row["reg_ok"] is True,
            "the wrong day is caught": row["day_ok"] is False,
            "so the call is not correct": row["correct"] is False,
            "the courtesy car never reached the note": row["note_ok"] is False,
        }
        for what, held in checks.items():
            print(f"  {'ok  ' if held else 'FAIL'} {what}")
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(run())
