"""punchin: fork a recorded voice-agent call at the turn it went wrong."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Any

from punchin import __version__
from punchin.adapter import CommandAgent
from punchin.agent import Agent, ModelAgent, ScriptedAgent
from punchin.call import Call
from punchin.check import baseline_from, compare, load_baseline, report
from punchin.customer import Customer, ScriptedCustomer
from punchin.dms import TODAY, serve
from punchin.fidelity import teacher_forced
from punchin.fork import Budget, agent_turns, fork
from punchin.goal import extract, score
from punchin.metrics import summarize
from punchin.model import ClaudeCodeModel, Model, ModelDidNotRun
from punchin.player import write as write_player
from punchin.record import record
from punchin.scenarios import BY_ID, SCENARIOS, GoalState, Scenario, vocabulary

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


def agent_for(args: argparse.Namespace) -> Agent:
    """The agent under test. `command` is the one that matters: anybody else's."""
    name = args.agent
    if name == "careful":
        return ScriptedAgent(careful=True)
    if name == "careless":
        return ScriptedAgent(careful=False)
    if name == "claude-code":
        return ModelAgent(ClaudeCodeModel(model=args.model), TODAY, system_suffix=args.system_suffix)
    if name == "command":
        if not args.agent_command:
            raise SystemExit('--agent command needs --agent-command "..."')
        return CommandAgent(shlex.split(args.agent_command), TODAY)
    raise SystemExit(f"unknown agent {name!r}")


def show(call: Call) -> str:
    lines = [f"{call.id}  scenario={call.scenario}  agent={call.agent}  cost=${call.cost_usd:.3f}"]
    for turn in call.turns:
        who = "Agent" if turn.speaker == "agent" else "Kunde"
        took = f"  ({turn.model_ms} ms)" if turn.model_ms else ""
        lines.append(f"  {turn.index:2} {who} {turn.spoken}{took}")
        if turn.heard is not None and turn.heard.strip() != turn.spoken.strip():
            lines.append(f"      heard: {turn.heard}")
        lines.extend(
            f"         -> {c.tool}({c.arguments}) {'ERROR ' + c.error if c.error else ''}"
            for c in turn.tool_calls
        )
    lines.append(f"  fork points (agent turns): {', '.join(str(i) for i in agent_turns(call))}")
    outcome = ", ".join(
        f"{b['reg']} {b['date']} {b['time']}{' note=' + b['note'] if b['note'] else ''}"
        for b in call.bookings
    )
    lines.append(f"  booked: {outcome or 'nothing'}   ended by: {call.notes.get('ended_by')}")
    return "\n".join(lines)


def cmd_scenarios(_: argparse.Namespace) -> int:
    for s in SCENARIOS:
        want = s.expected.day.isoformat() if s.expected.day else "no booking"
        print(f"{s.id:18} {s.pattern:16} {s.vehicle.reg}  expects {want}")
    return 0


def voice(args: argparse.Namespace, inner: Customer, out: Path) -> Customer:
    """`inner`, spoken aloud and heard back, when --audio is on. Shared by `record` and `fork`."""
    if not getattr(args, "audio", False):
        return inner
    from punchin.audio import AudioCustomer, available, recognizer  # noqa: PLC0415 - optional extra

    ready, missing = available()
    if not ready:
        raise SystemExit(f"--audio needs: {missing}")
    return AudioCustomer(
        inner,
        recognizer(args.whisper, vocabulary=vocabulary() if args.bias else None),
        out / "audio",
        over_the_phone=not args.studio,
        snr_db=getattr(args, "snr_db", None),
    )


def customer_for(args: argparse.Namespace, scenario: Scenario, out: Path) -> Customer:
    return voice(args, ScriptedCustomer(scenario), out)


def cmd_record(args: argparse.Namespace) -> int:
    chosen = SCENARIOS if args.scenario == "all" else [BY_ID[args.scenario]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = out / ".dms-state.json"
    for scenario in chosen:
        agent = agent_for(args)
        call = record(scenario, agent, customer_for(args, scenario, out), state, out)
        print(show(call))
        print()
    state.unlink(missing_ok=True)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    for path in args.call:
        print(show(Call.load(Path(path))))
    return 0


def model_for(args: argparse.Namespace) -> Model:
    return ClaudeCodeModel(model=args.model)


def cmd_extract(args: argparse.Namespace) -> int:
    model = model_for(args)
    for path in args.call:
        call = Call.load(Path(path))
        goal, cost = extract(call, model, TODAY)
        print(f"{call.id}  ${cost:.3f}")
        print(f"  {goal.model_dump_json(exclude_defaults=False)}")
        if call.scenario in BY_ID:
            print(f"  vs truth: {score(goal, BY_ID[call.scenario].goal)}")
    return 0


def cmd_fidelity(args: argparse.Namespace) -> int:
    model = model_for(args)
    for path in args.call:
        call = Call.load(Path(path))
        goal: GoalState
        if args.goal == "truth":
            goal = BY_ID[call.scenario].goal
        else:
            goal, _ = extract(call, model, TODAY)
        print(teacher_forced(call, goal, model).text())
        print()
    return 0


BASE_COLUMNS = [
    "scenario",
    "correct",
    "day_ok",
    "turns",
    "options_max",
    "agent_repeats",
    "customer_stalls",
    "ended_by",
    "cost_usd",
]
HEARD_COLUMNS = ["reg_heard", "reg_survived", "lookup_attempts"]


def cmd_metrics(args: argparse.Namespace) -> int:
    calls = [Call.load(Path(path)) for path in args.call]
    rows = [summarize(call, BY_ID[call.scenario]) for call in calls]
    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True, default=str))
        return 0
    keys = list(BASE_COLUMNS)
    if any(key in row for row in rows for key in HEARD_COLUMNS):
        keys += HEARD_COLUMNS  # only there when a recogniser sat in the middle
    print("  ".join(f"{key:>16}" for key in keys))
    for row in rows:
        print("  ".join(f"{str(row.get(key, '-'))[:16]:>16}" for key in keys))
    return 0


def cmd_fork(args: argparse.Namespace) -> int:
    call = Call.load(Path(args.call))
    scenario = BY_ID[call.scenario]
    model = model_for(args)
    goal = scenario.goal if args.goal == "truth" else extract(call, model, TODAY)[0]
    agent = agent_for(args)
    changed = f"system suffix {args.system_suffix!r}" if args.system_suffix else f"agent {agent.name}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = fork(
        call,
        scenario,
        args.at,
        agent=agent,
        goal=goal,
        model=model,
        state_path=out / ".dms-state.json",
        repeat=args.repeat,
        changed=changed,
        budget=Budget(args.max_usd),
        out=out,
        wrap=lambda inner: voice(args, inner, out),
    )
    (out / ".dms-state.json").unlink(missing_ok=True)
    print(report.text())
    return 0 if report.fixed == len(report.attempts) and report.attempts else 1


def cmd_check(args: argparse.Namespace) -> int:
    calls = [Call.load(Path(path)) for path in args.call]
    rows = [summarize(call, BY_ID[call.scenario]) for call in calls]
    baseline_path = Path(args.baseline)
    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline_from(rows), indent=2, sort_keys=True) + "\n")
        print(f"baseline written from {len(rows)} scenarios: {baseline_path}")
        return 0
    if not baseline_path.exists():
        raise SystemExit(f"no baseline at {baseline_path}; write one with `punchin check --update`")
    found = compare(rows, load_baseline(baseline_path))
    print(report(found, len(rows)))
    return 1 if found else 0


def cmd_player(args: argparse.Namespace) -> int:
    before, after = Call.load(Path(args.before)), Call.load(Path(args.after))
    dest = write_player(before, after, Path(args.out), title=args.title)
    print(f"{dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_dms(args: argparse.Namespace) -> int:
    serve(Path(args.state))
    return 0


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
    add_audio_flags(rec)
    rec.add_argument("--snr-db", type=float, default=None, help="mix in car noise at this SNR")
    rec.set_defaults(run=cmd_record)

    fk = commands.add_parser(
        "fork", parents=[common], help="re-run a recorded call from one turn with the change applied"
    )
    fk.add_argument("call")
    fk.add_argument("--at", type=int, required=True, help="the agent turn to fork at (see `punchin show`)")
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
    add_audio_flags(fk)
    fk.set_defaults(run=cmd_fork)


def _add_reading(commands: Commands, common: argparse.ArgumentParser) -> None:
    sh = commands.add_parser("show", parents=[common], help="print a recorded call")
    sh.add_argument("call", nargs="+")
    sh.set_defaults(run=cmd_show)

    met = commands.add_parser("metrics", parents=[common], help="outcome and feel numbers for recorded calls")
    met.add_argument("call", nargs="+")
    met.add_argument("--json", action="store_true", help="one JSON object per call, for a pipeline")
    met.set_defaults(run=cmd_metrics)

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
    chk.set_defaults(run=cmd_check)


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
        sub.set_defaults(run=run)

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
    except (ModelDidNotRun, FileNotFoundError) as stopped:
        print(f"punchin: {stopped}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
