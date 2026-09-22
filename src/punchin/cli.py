"""punchin: fork a recorded voice-agent call at the turn it went wrong.

Exit codes: 0 success, 1 the run said no (a regression, a fork that did not fix it, a bad flag),
2 something it needed was missing, 130 interrupted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Any

from punchin import __version__
from punchin.adapter import AgentProtocolError, CommandAgent
from punchin.agent import Agent, ModelAgent, ScriptedAgent
from punchin.call import Call
from punchin.check import baseline_from, by_scenario, check, load_baseline
from punchin.customer import Customer, ScriptedCustomer
from punchin.dms import TODAY, normalize_reg, serve
from punchin.fidelity import FIELDS, ablation, across, repeated, teacher_forced
from punchin.fork import Budget, agent_turns, fork
from punchin.goal import extract, score
from punchin.importer import read_call, scenario_for
from punchin.metrics import summarize
from punchin.model import ClaudeCodeModel, Model, ModelDidNotRun
from punchin.player import write as write_player
from punchin.record import record
from punchin.scenarios import (
    BY_ID,
    SCENARIO_DIR,
    SCENARIOS,
    GoalState,
    Scenario,
    load_scenarios,
    ungraded,
    vocabulary,
)
from punchin.soundness import measure, summarise_soundness
from punchin.triage import triage

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
    if getattr(args, "system_suffix", "") and name != "claude-code":
        # A fork's report names the change it applied. Accepting a prompt change for an agent that
        # cannot take one would make the report a lie about a run that tested nothing.
        raise SystemExit(
            f"--system-suffix has no effect on --agent {name}; it is a prompt change, and only "
            f"--agent claude-code takes one. Change your own agent and pass --agent command instead."
        )
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
    for _ in range(max(1, args.repeat)):
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


def known_scenarios(args: argparse.Namespace) -> dict[str, Scenario]:
    return load_scenarios(Path(getattr(args, "scenarios", SCENARIO_DIR)))


def truth_for(known: dict[str, Scenario], call: Call, args: argparse.Namespace) -> Scenario:
    """The outcome this call is graded against, or a sentence saying nobody has stated one."""
    found = known.get(call.scenario)
    if found is None:
        raise SystemExit(ungraded(call.scenario, Path(getattr(args, "scenarios", SCENARIO_DIR))))
    return found


def cmd_extract(args: argparse.Namespace) -> int:
    model = model_for(args)
    for path in args.call:
        call = Call.load(Path(path))
        goal, cost = extract(call, model, TODAY)
        print(f"{call.id}  ${cost:.3f}")
        print(f"  {goal.model_dump_json(exclude_defaults=False)}")
        known = known_scenarios(args)
        if call.scenario in known:
            print(f"  vs truth: {score(goal, known[call.scenario].goal)}")
    return 0


def cmd_fidelity(args: argparse.Namespace) -> int:
    model = model_for(args)
    reports = []
    for path in args.call:
        call = Call.load(Path(path))
        goal: GoalState
        if args.goal == "truth":
            goal = truth_for(known_scenarios(args), call, args).goal
        else:
            goal, _ = extract(call, model, TODAY)

        if args.ablate:
            print(ablation(call, goal, model, FIELDS, times=args.repeat).text())
            continue

        if args.repeat > 1:
            print(repeated(call, goal, model, args.repeat).one_line())
            continue

        report = teacher_forced(call, goal, model)
        reports.append(report)
        print(report.text() if not args.summary else report.one_line())
        if not args.summary:
            print()

    if len(reports) > 1:
        print(across(reports))
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
    known = known_scenarios(args)
    rows = [summarize(call, truth_for(known, call, args)) for call in calls]
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
    agent = agent_for(args)  # before any file is read, so a bad flag fails in a millisecond
    call = Call.load(Path(args.call))
    scenario = truth_for(known_scenarios(args), call, args)
    model = model_for(args)
    goal = scenario.goal if args.goal == "truth" else extract(call, model, TODAY)[0]
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
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(report.text())
    return 0 if report.fixed == len(report.attempts) and report.attempts else 1


def cmd_check(args: argparse.Namespace) -> int:
    calls = [Call.load(Path(path)) for path in args.call]
    known = known_scenarios(args)
    rows = [summarize(call, truth_for(known, call, args)) for call in calls]
    scenarios = len(by_scenario(rows))
    baseline_path = Path(args.baseline)

    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline_from(rows), indent=2, sort_keys=True) + "\n")
        runs = len(rows) // scenarios if scenarios else 0
        print(f"baseline written from {scenarios} scenarios, {runs} run(s) each: {baseline_path}")
        if runs < 2:
            print(
                "  One run each. An agent that is sampled needs more, or this baseline records a "
                "coin toss as if it were a rule."
            )
        return 0

    if not baseline_path.exists():
        raise SystemExit(f"no baseline at {baseline_path}; write one with `punchin check --update`")
    found = check(rows, load_baseline(baseline_path))
    if args.json:
        print(
            json.dumps(
                {
                    "regressions": [{"scenario": r.scenario, "detail": r.detail} for r in found.regressions],
                    "flaky": [
                        {"scenario": f.scenario, "passed": f.passed, "trials": f.trials} for f in found.flaky
                    ],
                    "scenarios": found.scenarios,
                    "trials": found.trials,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(found.text())
    return 1 if found.failed else 0


def cmd_triage(args: argparse.Namespace) -> int:
    known = known_scenarios(args)
    calls = [Call.load(Path(path)) for path in args.call]
    ungradeable = [call.id for call in calls if call.scenario not in known]
    found = triage(calls, known)
    if args.json:
        print(json.dumps(found.as_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(found.text(limit=args.top))
        if ungradeable:
            print(f"\n  {len(ungradeable)} calls had no stated outcome and were left out")
    return 1 if found.clusters else 0


def cmd_soundness(args: argparse.Namespace) -> int:
    known = known_scenarios(args)
    if args.scenario == "all":
        chosen = [s for s in SCENARIOS if s.script]
    elif args.scenario in known:
        chosen = [known[args.scenario]]
    else:
        raise SystemExit(ungraded(args.scenario, Path(args.scenarios)))
    model = model_for(args)
    if args.agent == "command":
        # Somebody else's agent, where the change is a different command rather than a prompt.
        if not (args.agent_command and args.baseline_command):
            raise SystemExit(
                "--agent command needs both --baseline-command (what the recording was made with) "
                "and --agent-command (the changed agent whose fork is being checked)"
            )
        changed: Agent = CommandAgent(shlex.split(args.agent_command), TODAY, name="command:changed")
        plain: Agent = CommandAgent(shlex.split(args.baseline_command), TODAY, name="command:baseline")
        change = args.agent_command
    else:
        if not args.system_suffix:
            raise SystemExit(
                "--system-suffix is the change whose fork is being checked; it is required, "
                "unless you are checking your own agent with --agent command"
            )
        changed = ModelAgent(ClaudeCodeModel(model=args.model), TODAY, system_suffix=args.system_suffix)
        plain = ModelAgent(ClaudeCodeModel(model=args.model), TODAY, system_suffix=args.baseline_suffix)
        change = args.system_suffix
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_usd)

    results = []
    for scenario in chosen:
        found = measure(
            scenario,
            agent_with_change=changed,
            agent_without=plain,
            goal=scenario.goal,
            model=model,
            state_path=out / ".dms-state.json",
            at=args.at,
            trials=args.trials,
            change=change,
            budget=budget,
            out=out,
        )
        results.append(found)
        if not args.json:
            print(found.text())
            print()
    (out / ".dms-state.json").unlink(missing_ok=True)

    if args.json:
        print(json.dumps([f.as_dict() for f in results], ensure_ascii=False, sort_keys=True))
    elif len(results) > 1:
        print(summarise_soundness(results))
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    source = Path(args.transcript)
    call = read_call(source.read_text(), scenario_id=args.id, agent=args.agent)
    day = dt.date.fromisoformat(args.day) if args.day else None
    if args.booked and day is None:
        raise SystemExit("--booked needs --day: a booking that happened has a day it happened on")
    scenario = scenario_for(
        call,
        reg=args.reg,
        day=day,
        booked=args.booked,
        why=args.why,
        extras=list(args.extra or []),
    )
    if args.booked:
        # What the call actually did, so a grader can tell a right booking from a wrong one.
        call.bookings = [
            {
                "reg": normalize_reg(args.booked_reg or args.reg),
                "date": (args.booked_day or args.day),
                "time": args.booked_time,
                "note": "",
            }
        ]

    out, scenarios = Path(args.out), Path(args.scenarios)
    scenarios.mkdir(parents=True, exist_ok=True)
    written = call.save(out)
    (scenarios / f"{scenario.id}.json").write_text(scenario.model_dump_json(indent=2) + "\n")
    spoken = sum(1 for turn in call.turns if turn.speaker == "customer")
    print(f"{written}  ({len(call.turns)} turns, {spoken} from the customer)")
    print(f"{scenarios / f'{scenario.id}.json'}  the outcome it is graded against")
    return 0


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
    fk.add_argument("--json", action="store_true", help="the report as one JSON object")
    add_audio_flags(fk)
    add_scenario_flag(fk)
    fk.set_defaults(run=cmd_fork)


def add_scenario_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scenarios",
        default=str(SCENARIO_DIR),
        help="directory of scenarios that outcomes are graded against, beside the built-in corpus",
    )


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
        add_scenario_flag(sub)
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
