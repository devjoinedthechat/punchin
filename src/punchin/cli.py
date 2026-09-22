"""punchin: fork a recorded voice-agent call at the turn it went wrong."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from punchin import __version__
from punchin.agent import Agent, ModelAgent, ScriptedAgent
from punchin.call import Call
from punchin.customer import ScriptedCustomer
from punchin.dms import TODAY, serve
from punchin.fidelity import teacher_forced
from punchin.goal import extract, score
from punchin.metrics import summarize
from punchin.model import ClaudeCodeModel, Model
from punchin.record import record
from punchin.scenarios import BY_ID, SCENARIOS, GoalState

DEFAULT_OUT = Path(".punchin/calls")


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
        who = "Agent " if turn.speaker == "agent" else "Kunde "
        took = f"  ({turn.model_ms} ms)" if turn.model_ms else ""
        lines.append(f"  {who} {turn.spoken}{took}")
        lines.extend(
            f"         -> {c.tool}({c.arguments}) {'ERROR ' + c.error if c.error else ''}"
            for c in turn.tool_calls
        )
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


def cmd_record(args: argparse.Namespace) -> int:
    chosen = SCENARIOS if args.scenario == "all" else [BY_ID[args.scenario]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = out / ".dms-state.json"
    for scenario in chosen:
        agent = agent_for(args.agent, args.model)
        call = record(scenario, agent, ScriptedCustomer(scenario), state, out)
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


def cmd_metrics(args: argparse.Namespace) -> int:
    rows = [summarize(Call.load(Path(p)), BY_ID[Call.load(Path(p)).scenario]) for p in args.call]
    keys = [
        "scenario",
        "agent",
        "correct",
        "day_ok",
        "note_ok",
        "turns",
        "agent_words_max",
        "options_max",
        "customer_stalls",
        "ended_by",
        "model_ms_mean",
        "cost_usd",
    ]
    print("  ".join(f"{k:>15}" for k in keys))
    for row in rows:
        print("  ".join(f"{str(row.get(k, ''))[:15]:>15}" for k in keys))
    return 0


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
