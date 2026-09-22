"""What each command does.

Separated from `cli.py`, which is only about turning a command line into one of these. The two
concerns were one 660-line module and neither was easy to find in it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
from pathlib import Path
from typing import Any

from punchin.adapter import CommandAgent
from punchin.agent import Agent, ModelAgent, ScriptedAgent
from punchin.call import Call
from punchin.check import baseline_from, by_scenario, check, load_baseline
from punchin.customer import Customer, ScriptedCustomer
from punchin.dms import TODAY, normalize_reg, serve
from punchin.doctor import examine, report
from punchin.fidelity import FIELDS, ablation, across, repeated, teacher_forced
from punchin.fork import Budget, Sweep, agent_turns, fork, fork_point
from punchin.goal import extract, score
from punchin.importer import read_call, scenario_for
from punchin.metrics import summarize
from punchin.model import ClaudeCodeModel, Model
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


GUTTER = " " * 12  # what one turn's number and speaker occupy, so continuations line up under the text


def _arguments(arguments: dict[str, Any]) -> str:
    """`date_from=2026-10-01, reg=AB 12 345` rather than a Python dict printed at somebody."""
    return ", ".join(f"{name}={value}" for name, value in arguments.items())


def show(call: Call) -> str:
    """A recording, laid out to be read: turn, speaker, then everything about that turn aligned under it."""
    header = f"{call.id}   {call.agent}"
    if call.cost_usd:
        header += f"   ${call.cost_usd:.3f}"
    lines = [header, ""]

    for turn in call.turns:
        who = "Agent" if turn.speaker == "agent" else "Kunde"
        took = f"   ({turn.model_ms} ms)" if turn.model_ms else ""
        lines.append(f"{turn.index:>3}  {who:<5}  {turn.spoken}{took}")
        if turn.heard is not None and turn.heard.strip() != turn.spoken.strip():
            lines.append(f"{GUTTER}heard  {turn.heard}")
        for made in turn.tool_calls:
            label = "error" if made.error else "calls"
            lines.append(f"{GUTTER}{label}  {made.tool}({_arguments(made.arguments)})")
            if made.error:
                lines.append(f"{GUTTER}       {made.error}")

    booked = ", ".join(
        f"{one['reg']} {one['date']} {one['time']}" + (f" ({one['note']})" if one["note"] else "")
        for one in call.bookings
    )
    lines += [
        "",
        f"{'booked':>8}  {booked or 'nothing'}",
        f"{'ended by':>8}  {call.notes.get('ended_by')}",
        f"{'fork at':>8}  {', '.join(str(i) for i in agent_turns(call))}",
    ]
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
            print(ablation(call, goal, model, FIELDS, times=args.repeat, workers=args.workers).text())
            continue

        if args.repeat > 1:
            print(repeated(call, goal, model, args.repeat, args.workers).one_line())
            continue

        report = teacher_forced(call, goal, model, args.workers)
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
    if len(args.call) > 1:
        return _sweep(args, agent)
    call = Call.load(Path(args.call[0]))
    scenario = truth_for(known_scenarios(args), call, args)
    model = model_for(args)
    goal = scenario.goal if args.goal == "truth" else extract(call, model, TODAY)[0]
    changed = f"system suffix {args.system_suffix!r}" if args.system_suffix else f"agent {agent.name}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = fork(
        call,
        scenario,
        fork_point(call, args.at),
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


def _sweep(args: argparse.Namespace, agent: Agent) -> int:
    """One change against many recordings. Fixing four and breaking one is not an improvement."""
    known = known_scenarios(args)
    model = model_for(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_usd)
    changed = f"system suffix {args.system_suffix!r}" if args.system_suffix else f"agent {agent.name}"
    swept = Sweep(changed)

    for path in args.call:
        call = Call.load(Path(path))
        scenario = truth_for(known, call, args)
        goal = scenario.goal if args.goal == "truth" else extract(call, model, TODAY)[0]
        if budget.exhausted:
            swept.stopped = f"budget of ${budget.limit_usd:.2f} spent after {len(swept.reports)} calls"
            break
        swept.reports.append(
            fork(
                call,
                scenario,
                fork_point(call, args.at),
                agent=agent,
                goal=goal,
                model=model,
                state_path=out / ".dms-state.json",
                repeat=args.repeat,
                changed=changed,
                budget=budget,
                out=out,
                wrap=lambda inner: voice(args, inner, out),
            )
        )
    (out / ".dms-state.json").unlink(missing_ok=True)
    print(json.dumps(swept.as_dict(), ensure_ascii=False, sort_keys=True) if args.json else swept.text())
    return 1 if swept.verdicts["broke"] else 0


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
    # Flags are checked before anything is built. Constructing the model first made a missing
    # --baseline-command report a missing `claude` binary, which is a true sentence about the wrong
    # problem — and only on a machine that has no Claude Code, which is not the one this was written on.
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
    elif not args.system_suffix:
        raise SystemExit(
            "--system-suffix is the change whose fork is being checked; it is required, "
            "unless you are checking your own agent with --agent command"
        )
    else:
        changed = ModelAgent(ClaudeCodeModel(model=args.model), TODAY, system_suffix=args.system_suffix)
        plain = ModelAgent(ClaudeCodeModel(model=args.model), TODAY, system_suffix=args.baseline_suffix)
        change = args.system_suffix

    # The pinned customer needs a model even when both agents are somebody else's.
    model = model_for(args)
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


def cmd_doctor(args: argparse.Namespace) -> int:
    findings = examine(Path(args.scenarios), Path(args.baseline))
    print(report(findings))
    return 1 if any(f.state == "missing" for f in findings) else 0


def cmd_dms(args: argparse.Namespace) -> int:
    serve(Path(args.state))
    return 0
