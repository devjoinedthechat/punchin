# Contributing

```sh
uv sync
uv run pytest -q
uv run ruff format src tests examples scripts
uv run ruff check src tests examples scripts && uv run mypy
uv run python scripts/import_smoke.py      # reading somebody else's transcript still works
```

The suite needs no API key and no `claude` binary: `tests/fixtures/fake_claude.py` stands in for Claude
Code and speaks the same stream-json, so the whole path runs offline and free. The audio tests skip
themselves unless a voice, ffmpeg and faster-whisper are all present, and `-m slow` holds the few that
need a recogniser model on disk.

## What the corpus holds itself to

- **A scenario carries its own answer.** Anything added to `src/punchin/scenarios.py` states the outcome
  that is right, so a grader can be wrong about it. A scenario nobody can fail is not measuring anything.
- **`careful` and `careless` must disagree.** Every grader is checked against the scripted pair before a
  model is run. If both agents score the same, the grader is broken, not the agent.
- **A scripted customer must behave like a person.** The first line whose condition matches is the one
  said, so a catch-all placed above a farewell makes the customer answer "hej hej" with a refusal — and
  then a simulator that rings off correctly is scored as wrong. Order the specific before the general.

## What a number has to clear before it is written down

Every one of these was learned by getting it wrong in this repository first.

- **Run the fair configuration.** A number produced by a setup a practitioner would call rigged is worse
  than no number. Naive decoding loses nearly every plate; a production agent biases the recogniser with
  its call list. Report both and let the gap be the finding.
- **Know the noise before quoting a delta.** Repeated runs of the same call with the same goal state
  spread by 0.13 to 0.23, while the goal-state fields are worth 0.03 to 0.10 each. Use `--repeat`, and
  compare arms **paired by turn** so that turn difficulty — the dominant variance — cancels.
- **Weight by what was measured.** A two-turn call's fidelity moves in steps of 0.50 and is close to a
  coin flip. Pool the turns rather than averaging the calls' averages.
- **Sample the gate too.** `punchin check` takes several runs per scenario and compares rates. A
  single-sample gate fails one build in five on an agent that is right four times in five, and the
  rational response to that is to re-run CI until it passes.
- **Read the transcript before believing the number.** Every wrong conclusion in this project's history
  was a figure taken at face value, and every correction came from opening the call and looking:
  a lost plate that was a decoder setting, "four of six fields are harmful" that was the sampler, a
  simulator that "could not decline" when its refusal matched perfectly and a corpus line was at fault.
- **Only measured numbers go in the README.** If an example is illustrative, it says so.

## The README

Documents what is true now, in the present tense. No status tables, no roadmap, no development
narrative — that history is in git, where it belongs.
