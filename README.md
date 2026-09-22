# punchin

**Fork a recorded voice-agent call at the turn it went wrong, and replay the rest with the same customer.**

Every voice-agent testing tool either simulates a generic caller from scratch, or replays the recorded
customer lines word for word and goes incoherent the moment the new agent says something different.
Neither answers the question an operator actually has: *I changed the prompt. Did it fix Tuesday's call?*

punchin replays a recorded call up to turn *k*, lets the changed agent speak from there, and from *k+1*
plays the customer with a simulator pinned to what the original customer wanted, knew and said. The
simulator's fidelity is measured against the real customer's own held-out turns.

## Status

Pre-alpha. What exists today, 2026-09-22:

| | |
|---|---|
| ✅ | A corpus of ten Danish syn-reminder calls, each with its ground truth built in: the plate, the day the customer meant, the outcome that is right. |
| ✅ | A small dealership system with three tools (`lookup_vehicle`, `find_slots`, `book`), callable in-process and as an MCP server over stdio. |
| ✅ | Two scripted agents: `careful` reaches the expected outcome on every scenario; `careless` makes the mistake each scenario is built to catch. Every grader has to tell them apart before a model is run. |
| ✅ | Recording with a model as the agent, through Claude Code in print mode, so no API key is needed. Tool calls are read back exactly as the model made them. |
| ⬜ | Goal-state extraction from a recorded call. |
| ⬜ | The pinned customer, and its teacher-forced fidelity against the real turns. |
| ⬜ | `punchin fork`: prefix from the recording, the changed agent from turn *k*, `--repeat`. |
| ⬜ | Timing and outcome metrics; audio. |

## Try it

```sh
uv sync
uv run punchin scenarios
uv run punchin record --agent careful                     # all ten, free, deterministic
uv run punchin record --agent careless --scenario self-correction
uv run punchin record --agent claude-code --scenario self-correction   # a real model, via Claude Code
```

Recordings land in `.punchin/calls/` as JSON: every turn, every tool call, and what ended up booked.

## Why Danish, why syn

The corpus is Danish because the failure modes that matter in voice are language-specific: a customer
who says *"tirsdag ... nej vent, onsdag"* is a self-correction that a transcript-level check cannot see,
and booking Tuesday is a tool call that succeeds. A syn (MOT) reminder is the simplest real outbound
after-sales call: one vehicle, one date, one booking.

## License

Apache-2.0.
