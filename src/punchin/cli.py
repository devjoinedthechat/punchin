"""punchin: fork a recorded voice-agent call at the turn it went wrong."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from punchin import __version__
from punchin.agent import Agent, ModelAgent, ScriptedAgent
from punchin.call import Call
from punchin.customer import Customer, ScriptedCustomer
from punchin.dms import TODAY, serve
from punchin.fidelity import teacher_forced
from punchin.fork import Budget, agent_turns, fork
from punchin.goal import extract, score
from punchin.metrics import summarize
from punchin.model import ClaudeCodeModel, Model
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


def agent_for(name: str, model: str) -> Agent:
    if name == "careful":
        return ScriptedAgent(careful=True)
    if name == "careless":
        return ScriptedAgent(careful=False)
    if name == "claude-code":
        return ModelAgent(ClaudeCodeModel(model=model), TODAY)
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
        agent = agent_for(args.agent, args.model)
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
    agent = ModelAgent(ClaudeCodeModel(model=args.model), TODAY, system_suffix=args.system_suffix)
    changed = f"system suffix {args.system_suffix!r}" if args.system_suffix else f"model {args.model}"
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


def cmd_dms(args: argparse.Namespace) -> int:
    serve(Path(args.state))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="punchin", description=__doc__)
    root.add_argument("--version", action="version", version=f"punchin {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("scenarios", help="list the corpus").set_defaults(run=cmd_scenarios)

    rec = commands.add_parser("record", help="record a scenario with an agent")
    rec.add_argument("--scenario", default="all", help="a scenario id, or 'all'")
    rec.add_argument("--agent", default="careful", choices=["careful", "careless", "claude-code"])
    rec.add_argument("--model", default="claude-sonnet-5", help="model for --agent claude-code")
    rec.add_argument("--out", default=str(DEFAULT_OUT))
    add_audio_flags(rec)
    rec.add_argument("--snr-db", type=float, default=None, help="mix in car noise at this SNR")
    rec.set_defaults(run=cmd_record)

    sh = commands.add_parser("show", help="print a recorded call")
    sh.add_argument("call", nargs="+")
    sh.set_defaults(run=cmd_show)

    for name, run, help_text in (
        ("extract", cmd_extract, "read the customer's goal state out of recorded calls"),
        ("fidelity", cmd_fidelity, "teacher-forced: does the pinned customer say what the real one said?"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("call", nargs="+")
        sub.add_argument("--model", default="claude-sonnet-5")
        if name == "fidelity":
            sub.add_argument("--goal", default="extracted", choices=["extracted", "truth"])
        sub.set_defaults(run=run)

    fk = commands.add_parser("fork", help="re-run a recorded call from one turn with the change applied")
    fk.add_argument("call")
    fk.add_argument("--at", type=int, required=True, help="the agent turn to fork at (see `punchin show`)")
    fk.add_argument("--repeat", type=int, default=1, help="attempts, because both sides are stochastic")
    fk.add_argument("--system-suffix", default="", help="the prompt change under test")
    fk.add_argument("--model", default="claude-sonnet-5")
    fk.add_argument("--goal", default="extracted", choices=["extracted", "truth"])
    fk.add_argument("--max-usd", type=float, default=2.0, help="stop starting attempts once this is spent")
    fk.add_argument("--out", default=str(DEFAULT_OUT.parent / "forks"))
    add_audio_flags(fk)
    fk.set_defaults(run=cmd_fork)

    met = commands.add_parser("metrics", help="outcome and feel numbers for recorded calls")
    met.add_argument("call", nargs="+")
    met.set_defaults(run=cmd_metrics)

    dms = commands.add_parser("dms", help="the DMS as an MCP server over stdio (what the model calls)")
    dms.add_argument("--state", required=True)
    dms.set_defaults(run=cmd_dms)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
