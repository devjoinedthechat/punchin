"""punchin: fork a recorded voice-agent call at the turn it went wrong.

Exit codes: 0 success, 1 the run said no (a regression, a fork that did not fix it, a bad flag),
2 something it needed was missing, 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from punchin import __version__
from punchin.adapter import AgentProtocolError
from punchin.commands import (
    cmd_check,
    cmd_dms,
    cmd_doctor,
    cmd_extract,
    cmd_fidelity,
    cmd_fork,
    cmd_import,
    cmd_metrics,
    cmd_player,
    cmd_record,
    cmd_scenarios,
    cmd_show,
    cmd_soundness,
    cmd_triage,
)
from punchin.model import ModelDidNotRun
from punchin.scenarios import (
    SCENARIO_DIR,
)

DEFAULT_OUT = Path(".punchin/calls")


def add_audio_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio", action="store_true", help="speak the customer aloud and hear it back")
    parser.add_argument("--studio", action="store_true", help="skip the phone band; a clean microphone")
    parser.add_argument("--whisper", default="small", help="recogniser size: base loses Danish plates")
    parser.add_argument(
        "--bias",
        action="store_true",
        help="tell the recogniser the call list, the way a production agent would",
    )


def add_scenario_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scenarios",
        default=str(SCENARIO_DIR),
        help="directory of scenarios that outcomes are graded against, beside the built-in corpus",
    )


Commands = Any  # argparse's _SubParsersAction, which is private


def _add_recording(commands: Commands, common: argparse.ArgumentParser) -> None:
    commands.add_parser("scenarios", parents=[common], help="list the corpus").set_defaults(run=cmd_scenarios)

    rec = commands.add_parser("record", parents=[common], help="record a scenario with an agent")
    rec.add_argument("--scenario", default="all", help="a scenario id, or 'all'")
    rec.add_argument("--agent", default="careful", choices=["careful", "careless", "claude-code", "command"])
    rec.add_argument("--model", default="claude-sonnet-5", help="model for --agent claude-code")
    rec.add_argument(
        "--agent-command",
        default="",
        help='your own agent, over JSON: --agent command --agent-command "python agent.py"',
    )
    rec.add_argument("--system-suffix", default="", help="appended to --agent claude-code's prompt")
    rec.add_argument("--out", default=str(DEFAULT_OUT))
    rec.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="record each scenario this many times; a sampled agent needs more than one",
    )
    add_audio_flags(rec)
    rec.add_argument("--snr-db", type=float, default=None, help="mix in car noise at this SNR")
    rec.set_defaults(run=cmd_record)

    fk = commands.add_parser(
        "fork", parents=[common], help="re-run a recorded call from one turn with the change applied"
    )
    fk.add_argument("call", nargs="+", help="one recording, or many to sweep one change across them")
    fk.add_argument(
        "--at",
        default="half",
        help="the agent turn to fork at: a number from `punchin show`, or first, half or last",
    )
    fk.add_argument("--repeat", type=int, default=1, help="attempts, because both sides are stochastic")
    fk.add_argument("--system-suffix", default="", help="the prompt change under test")
    fk.add_argument("--model", default="claude-sonnet-5")
    fk.add_argument(
        "--agent", default="claude-code", choices=["careful", "careless", "claude-code", "command"]
    )
    fk.add_argument("--agent-command", default="", help="your own agent, for --agent command")
    fk.add_argument("--goal", default="extracted", choices=["extracted", "truth"])
    fk.add_argument("--max-usd", type=float, default=2.0, help="stop starting attempts once this is spent")
    fk.add_argument("--out", default=str(DEFAULT_OUT.parent / "forks"))
    fk.add_argument("--json", action="store_true", help="the report as one JSON object")
    add_audio_flags(fk)
    add_scenario_flag(fk)
    fk.set_defaults(run=cmd_fork)


def _add_reading(commands: Commands, common: argparse.ArgumentParser) -> None:
    sh = commands.add_parser("show", parents=[common], help="print a recorded call")
    sh.add_argument("call", nargs="+")
    sh.set_defaults(run=cmd_show)

    met = commands.add_parser("metrics", parents=[common], help="outcome and feel numbers for recorded calls")
    met.add_argument("call", nargs="+")
    met.add_argument("--json", action="store_true", help="one JSON object per call, for a pipeline")
    add_scenario_flag(met)
    met.set_defaults(run=cmd_metrics)

    imp = commands.add_parser(
        "import", parents=[common], help="turn somebody else's transcript into a recording"
    )
    imp.add_argument("transcript", help="JSON, JSONL, or lines like 'Agent: …' and 'Kunde: …'")
    imp.add_argument("--id", required=True, help="a name for this call, used as its scenario id")
    imp.add_argument("--reg", required=True, help="the registration the customer gave")
    imp.add_argument("--day", default=None, help="the day the customer meant, YYYY-MM-DD")
    imp.add_argument("--booked", action="store_true", help="a booking should have been made")
    imp.add_argument("--booked-day", default=None, help="the day actually booked, if it differed")
    imp.add_argument("--booked-reg", default=None, help="the plate actually booked, if it differed")
    imp.add_argument("--booked-time", default="08:00")
    imp.add_argument("--extra", action="append", help="something the workshop needed to know")
    imp.add_argument("--why", default="", help="why this call is worth keeping")
    imp.add_argument("--agent", default="imported", help="what to call the agent that made it")
    imp.add_argument("--out", default=str(DEFAULT_OUT))
    add_scenario_flag(imp)
    imp.set_defaults(run=cmd_import)

    ply = commands.add_parser("player", parents=[common], help="one page that plays two calls side by side")
    ply.add_argument("before")
    ply.add_argument("after")
    ply.add_argument("--out", default=".punchin/player.html")
    ply.add_argument("--title", default=None)
    ply.set_defaults(run=cmd_player)

    chk = commands.add_parser(
        "check", parents=[common], help="fail when a run is worse than the recorded baseline"
    )
    chk.add_argument("call", nargs="+")
    chk.add_argument("--baseline", default=str(DEFAULT_OUT.parent / "baseline.json"))
    chk.add_argument("--update", action="store_true", help="write the baseline from these calls instead")
    chk.add_argument("--json", action="store_true", help="the result as one JSON object")
    add_scenario_flag(chk)
    chk.set_defaults(run=cmd_check)


def _add_judging(commands: Commands, common: argparse.ArgumentParser) -> None:
    """The commands that pass judgement on a run: what went wrong, and whether to believe a fork."""
    tri = commands.add_parser(
        "triage", parents=[common], help="group the calls that went wrong, biggest group first"
    )
    tri.add_argument("call", nargs="+")
    tri.add_argument("--top", type=int, default=10, help="how many groups to print")
    tri.add_argument("--json", action="store_true")
    add_scenario_flag(tri)
    tri.set_defaults(run=cmd_triage)

    snd = commands.add_parser(
        "soundness", parents=[common], help="check that forking says what a full re-run says"
    )
    snd.add_argument("--scenario", required=True, help="a scenario id, or 'all'")
    snd.add_argument("--system-suffix", default="", help="the prompt change whose fork is being checked")
    snd.add_argument("--baseline-suffix", default="", help="the prompt the recording was made with")
    snd.add_argument("--agent", default="claude-code", choices=["claude-code", "command"], help="whose agent")
    snd.add_argument("--agent-command", default="", help="the changed agent, for --agent command")
    snd.add_argument(
        "--baseline-command", default="", help="the agent before the change, for --agent command"
    )
    snd.add_argument("--at", type=int, default=None, help="fork point; the middle of the call by default")
    snd.add_argument("--trials", type=int, default=3)
    snd.add_argument(
        "--model",
        default="claude-sonnet-5",
        help="a weaker model makes the fix land inconsistently, which is what discriminates",
    )
    snd.add_argument("--max-usd", type=float, default=5.0)
    snd.add_argument("--json", action="store_true")
    snd.add_argument("--out", default=str(DEFAULT_OUT.parent / "soundness"))
    add_scenario_flag(snd)
    snd.set_defaults(run=cmd_soundness)


def _add_customer(commands: Commands, common: argparse.ArgumentParser) -> None:
    for name, run, help_text in (
        ("extract", cmd_extract, "read the customer's goal state out of recorded calls"),
        ("fidelity", cmd_fidelity, "teacher-forced: does the pinned customer say what the real one said?"),
    ):
        sub = commands.add_parser(name, parents=[common], help=help_text)
        sub.add_argument("call", nargs="+")
        sub.add_argument("--model", default="claude-sonnet-5")
        if name == "fidelity":
            sub.add_argument("--goal", default="extracted", choices=["extracted", "truth"])
            sub.add_argument("--summary", action="store_true", help="one line per call, then the spread")
            sub.add_argument(
                "--ablate",
                action="store_true",
                help="drop one goal-state field at a time and report what each is worth",
            )
            sub.add_argument(
                "--repeat",
                type=int,
                default=1,
                help="measure this many times; the simulator is sampled, so one run is not a number",
            )
            sub.add_argument(
                "--workers",
                type=int,
                default=4,
                help="turns to measure at once; teacher-forcing makes them independent",
            )
        add_scenario_flag(sub)
        sub.set_defaults(run=run)

    doc = commands.add_parser(
        "doctor", parents=[common], help="what punchin can do on this machine, and what is missing"
    )
    doc.add_argument("--baseline", default="baseline.json")
    add_scenario_flag(doc)
    doc.set_defaults(run=cmd_doctor)

    dms = commands.add_parser(
        "dms", parents=[common], help="the DMS as an MCP server over stdio (what the model calls)"
    )
    dms.add_argument("--state", required=True)
    dms.set_defaults(run=cmd_dms)


def parser() -> argparse.ArgumentParser:
    # `-q` is accepted before or after the subcommand. SUPPRESS on the subcommand copy stops its
    # default clobbering a `-q` that was given first.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="only the report, no progress",
    )

    root = argparse.ArgumentParser(prog="punchin", description=__doc__, parents=[common])
    root.add_argument("--version", action="version", version=f"punchin {__version__}")
    root.set_defaults(quiet=False)
    commands = root.add_subparsers(dest="command", required=True)

    _add_recording(commands, common)
    _add_reading(commands, common)
    _add_judging(commands, common)
    _add_customer(commands, common)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Progress goes to stderr so that stdout stays the report, pipeable and parseable.
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    try:
        return int(args.run(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as missing:
        # Exit 2, the way a shell tool says "bad input". Without the errno, which is noise to a reader.
        print(f"punchin: cannot read {missing.filename}", file=sys.stderr)
        return 2
    except ModelDidNotRun as stopped:
        print(f"punchin: {stopped}", file=sys.stderr)
        return 2
    except (ValueError, AgentProtocolError) as wrong:
        # A mistake in what was asked for, not a bug. A traceback would only bury the sentence.
        print(f"punchin: {wrong}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
