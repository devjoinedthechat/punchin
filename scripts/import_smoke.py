"""Import a transcript in the crudest shape anyone would hand over, and check it grades as wrong.

Run in CI. If importing breaks, an archive stops being usable, and nothing else in the suite notices:
every other test starts from a recording punchin made itself.

The transcript is `examples/dealer-1482.txt`, the same file the README tells a reader to import, so the
documented command and the one CI runs cannot drift apart.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from punchin.call import Call
from punchin.cli import main
from punchin.metrics import summarize
from punchin.scenarios import load_scenarios

TRANSCRIPT = Path(__file__).resolve().parent.parent / "examples" / "dealer-1482.txt"


def run() -> int:
    with tempfile.TemporaryDirectory(prefix="punchin-import-smoke-") as workspace:
        root = Path(workspace)
        source = root / "call.txt"
        source.write_text(TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        calls, scenarios = root / "calls", root / "scenarios"

        imported = main(
            [
                "import",
                str(source),
                "--id",
                "dealer-1482",
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
        row = summarize(call, load_scenarios(scenarios)["dealer-1482"])
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
