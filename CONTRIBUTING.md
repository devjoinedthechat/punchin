# Contributing

```sh
uv sync
uv run pytest -q
uv run ruff format src tests && uv run ruff check src tests && uv run mypy
```

The test suite needs no API key and no `claude` binary: `tests/fixtures/fake_claude.py` stands in for
Claude Code and speaks the same stream-json, so the whole path runs offline.

A few things this repository holds itself to:

- **A scenario carries its own answer.** Anything added to `src/punchin/scenarios.py` states the outcome
  that is right, so a grader can be wrong about it. A scenario nobody can fail is not measuring anything.
- **`careful` and `careless` must disagree.** Every grader is checked against the scripted pair before a
  model is run. If both agents score the same, the grader is broken, not the agent.
- **Fidelity is measured, never asserted.** A change to the pinned customer is reported with the
  teacher-forced number before and after, on a real recording.
- **No development narrative in the README.** It documents what is true now. History belongs in git.
