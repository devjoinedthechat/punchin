<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="" width="96" height="96">
  </picture>
</p>

<h1 align="center">punchin</h1>

<p align="center">
  <b>Fork a recorded voice-agent call at the turn it went wrong.</b><br>
  The prefix replays for free. The rest runs live, against the customer who was really on the call.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.11–3.13">
  <img src="https://img.shields.io/badge/tests-44-brightgreen" alt="44 tests">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0">
  <img src="https://img.shields.io/badge/status-pre--alpha-orange" alt="Status: pre-alpha">
</p>

<p align="center">
  <a href="#try-it">Try it</a> ·
  <a href="#why-this-is-hard">Why this is hard</a> ·
  <a href="#how-a-fork-works">How a fork works</a> ·
  <a href="#is-the-simulated-customer-the-real-one">Fidelity</a> ·
  <a href="#the-corpus">Corpus</a> ·
  <a href="#down-a-phone-line">Audio</a> ·
  <a href="#what-it-does-not-do">Limits</a>
</p>

---

A dealership manager flags a bad call. You listen to it, change a line of the prompt, and now you have
the question every operator of a voice agent has: **did that fix *this* call?**

Nothing answers it today. Simulators generate a fresh synthetic caller and tell you about a call that
never happened. Transcript replay feeds the recorded customer lines back in order, and the moment your
agent asks a different question the recording no longer fits — the conversation degenerates and the
signal is "this is now incoherent".

punchin replays the real call up to the turn you choose, runs your changed agent from there, and plays
the rest of the customer with a simulator **pinned to what the original customer wanted, knew and asked
for**. The turns before the fork are served from the recording and cost nothing.

## The call

An outbound Danish *syn* (MOT) reminder, recorded against a real model. The customer changes her mind
mid-sentence, and the agent handles it — then reads her a menu of five times down the phone:

```
   4 Agent Okay, jeg kan se din Škoda Octavia - synet udløber den 10. oktober. Hvornår kunne du komme forbi?
         -> lookup_vehicle({'reg': 'AB 12 345'})
   5 Kunde Kan jeg få en tid tirsdag? ...nej vent, onsdag. Onsdag er bedre.
   6 Agent Super, onsdag har jeg ledigt klokken 8, halv ni, ti, ét eller halv tre. Passer et af de tidspunkter?
         -> find_slots({'date_from': '2026-09-30', 'date_to': '2026-09-30'})
```

Every tool call succeeded and the right day was booked. No outcome check fires. But nobody holds five
times in their head on a phone call, and the recording shows what it cost: she stalled, and the agent
spent three more turns recovering.

Fork at turn 6 with the fix:

```sh
punchin fork <call> --at 6 --repeat 3 \
  --system-suffix "Tilbyd kun én tid ad gangen. Nævn aldrig flere klokkeslæt i samme replik."
```

```
fork of 20260922-000946-self-correction at turn 6
  changed: system suffix 'Tilbyd kun én tid ad gangen. Nævn aldrig flere klokkeslæt i samme replik.'
  original   correct=True   options_max=5  turns=10  agent_words_max=26  customer_stalls=0
  attempt 1  correct=True   options_max=1  turns=10  agent_words_max=26  customer_stalls=0  live $0.053
  attempt 2  correct=True   options_max=1  turns=12  agent_words_max=26  customer_stalls=0  live $0.062
  attempt 3  correct=True   options_max=1  turns=10  agent_words_max=26  customer_stalls=0  live $0.036
  correct in 3 of 3 attempts (original: True)
  options_max: 5 -> 1 (median of attempts)
  live cost $0.151; the 6 turns before the fork cost nothing
```

The conversation improved and the booking still landed. Turns 0–5 are byte for byte the ones she had,
so nothing about the change is confounded by a different opening.

## Why this is hard

Replaying a conversation is not like replaying a trace. The environment is a person, and what they say
next depends on what your agent just said.

| | What it does at the first divergence | What you learn |
|---|---|---|
| Scripted replay of the transcript | Keeps feeding the recorded lines, which no longer answer the questions being asked | That the conversation is now incoherent |
| A simulated caller from a persona | Starts a new conversation from scratch | About a call that never happened |
| Re-running the whole call live | Pays for every turn again, and the customer is a different one each time | Something, expensively, about a different call |
| **punchin** | Serves the prefix from the recording, then plays *that* customer from their goal state | Whether the change fixes the call you are looking at |

The pinned customer may reveal only what the original customer possessed — their plate, their day,
their constraints, their time of day, in the order they brought them up. Asked something the real
customer never knew, it says so rather than inventing an answer.

## Try it

You need [uv](https://docs.astral.sh/uv/). Everything below runs against a fake dealership system in a
local JSON file: no account, no network, no credentials.

```sh
git clone https://github.com/devjoinedthechat/punchin && cd punchin
uv sync

uv run punchin scenarios                       # the corpus and the outcome each call expects
uv run punchin record --agent careful          # all ten, free and deterministic
uv run punchin record --agent careless --scenario self-correction
uv run punchin metrics .punchin/calls/*.json
```

Spoken, if you are on a Mac with ffmpeg (`uv sync --extra audio`, and the first run downloads a
recogniser):

```sh
uv run punchin record --agent careful --audio --scenario plain-booking
```

With a model as the agent. The backend is **Claude Code in print mode**, so it runs on the login you
already have and needs no API key:

```sh
uv run punchin record   --agent claude-code --scenario self-correction
uv run punchin show     .punchin/calls/<call>.json          # turn indices are the fork points
uv run punchin extract  .punchin/calls/<call>.json
uv run punchin fidelity .punchin/calls/<call>.json --goal truth
uv run punchin fork     .punchin/calls/<call>.json --at 6 --repeat 3 --system-suffix "..."
```

Each turn is one `claude -p` with every built-in tool off and the dealership system attached over MCP,
in a scrubbed environment so the child inherits nothing from the session that launched it. Tool calls
are read back out of its stream-json exactly as the model made them.

## How a fork works

```
turn  0 ───┬─── from the recording, byte for byte, free ────┐
        …  │                                                │
turn  5 ───┴────────────────────────────────────────────────┘
turn  6  the changed agent, live                     ← the only thing under test
turn  7  the customer, pinned to the goal state
turn  8  the changed agent, live
          …until someone ends the call
```

Before the fork runs, the tool calls the recorded agent made in the prefix are re-applied to a fresh
dealership system, so the world at turn 6 is the world the agent found there. `--repeat` runs the whole
thing several times, because both the agent and the customer are stochastic and one good attempt is not
a fix.

Where you fork decides what you are still testing. Fork *after* the self-correction and the agent has to
read past it; fork before it and the trap is gone, because the customer has not said it yet.

## Is the simulated customer the real one?

This is the question the whole tool rests on, so it is measured rather than asserted.

For every customer turn in a recording, the simulator is given the **real** conversation up to the
agent's line before it and asked to write that turn. The real turn is the answer key. The agent's own
randomness never enters, because the agent's lines are always the recorded ones.

```
$ punchin fidelity <call> --goal truth
facts jaccard 0.80, exact 60%, length x1.29, $0.041

   5 ≠ real: Kan jeg få en tid tirsdag? ...nej vent, onsdag. Onsdag er bedre.
       sim: Øh, tirsdag var jeg tænkt... nej, onsdag - onsdag den 30'te. Og gerne så tidligt som muligt.
       real ['day', 'no']  sim ['day', 'no', 'time', 'yes']
```

The simulator self-corrects the way she did, and it is still not her: it volunteered a time preference a
turn earlier than she did, and it says "ja" where she said nothing. That is what 0.80 buys, and the
number is in the repository so a change to the simulator has to move it.

The same run also grades the extraction that produced the goal state, against the corpus truth: on this
call it recovered the plate and the *corrected* Wednesday, not the Tuesday she took back.

## The corpus

Ten Danish *syn*-reminder calls. Each one states the outcome that is right, so a grader can be wrong
about it — the plate, the day, whether a booking should happen at all.

| Scenario | What it is built to catch |
|---|---|
| `self-correction` | "tirsdag ... nej vent, onsdag". Booking Tuesday is a tool call that succeeds. |
| `next-week` | "onsdag i næste uge" is not the nearest Wednesday. |
| `wrong-reg-first` | Two plates in one line; the second is the real one. |
| `already-booked` | The right outcome is no booking. An agent that books anyway succeeded at the wrong thing. |
| `courtesy-car` | A hard condition stated up front, which has to reach the workshop note. |
| `code-switch` | Danish and English in one breath, plus an extra job. |
| `is-it-a-robot` | A direct question about being a machine. |
| `proxy-caller` | The caller is not the owner on file. |
| `hurried` | Everything in the first breath; asking again is the failure. |
| `plain-booking` | The control. Anything a change breaks here is a regression. |

Two scripted agents keep the graders honest: `careful` reaches the expected outcome on all ten, and
`careless` makes each built-in mistake. Both run in CI. If a change makes them agree, the corpus has
stopped measuring anything, and that is a failure before a model is ever run.

## Down a phone line

The corpus can be spoken instead of typed. The customer's line is rendered with the one Danish voice
macOS ships, put through the 8 kHz G.711 mu-law band a telephone call actually uses, and handed to
faster-whisper. **The agent then reads what the recogniser produced and never what was said** — what
was said stays on the turn as the answer key, which is what makes the loss measurable.

```sh
uv run punchin record --agent careful --audio             # down a phone line
uv run punchin record --agent careful --audio --studio    # a clean microphone, for comparison
```

The scripted agent reaches the right outcome on all ten scenarios in text. Once it has to listen:

| | plates that survived | calls that completed | agent repeated itself | customer hung up |
|---|---|---|---|---|
| text | 10 / 10 | 10 / 10 | 1 | 0 / 10 |
| studio microphone | 2 / 10 | 2 / 10 | up to 5 | 8 / 10 |
| through a phone line | **0 / 10** | 0 / 10 | up to 5 | 10 / 10 |

The plate is the whole story, and one digit is enough:

```
   3 Kunde CD 67 890.
      heard: CD 67850
   4 Agent Den kan jeg ikke finde i systemet. Kan du sige nummerpladen igen?
         -> lookup_vehicle({'reg': 'CD67850'}) ERROR no vehicle registered as 'CD67850'
   6 Agent Den kan jeg ikke finde i systemet. Kan du sige nummerpladen igen?
         -> lookup_vehicle({'reg': 'CD67850'}) ERROR no vehicle registered as 'CD67850'
```

Two things fall out of that table which text could not have shown.

**The agent has no fallback.** It is optimal on the text corpus and helpless when an entity fails twice:
it never spells the plate back, never asks for the make instead, never offers a person. The customer
hears the same sentence four times and hangs up. That is a missing path, not a bad prompt, and it only
appears when something can go wrong between a mouth and a tool call.

**Outcome checking scores a disaster as a pass.** `already-booked` comes back `correct=True` through a
phone line, because the right outcome there is no booking and the call collapsed before making one. The
feel columns are the only thing that disagrees: `agent_repeats=5`, `customer_stalls=3`,
`ended_by=customer`.

One caveat, stated plainly: this is **synthesized** Danish, not a human caller, and a TTS voice is a
different — in places harder — distribution for a recogniser. These numbers show the shape of the
failure and the cost of the phone band. They are not an estimate of what a production Danish agent does
with real customers.

## What it does not do

- **No real-time turn-taking.** The loop is turn-driven, and that is the trade that buys forking: you
  cannot serve a prefix and hand control over mid-utterance at the same time. Barge-in, endpointing and
  the silence before a reply need a duplex streaming pipeline, and they are measured well elsewhere
  (EVA-Bench, IHBench, and every commercial voice-testing platform).
- **No telephony.** There is no phone number and no carrier. The part of a phone call that changes the
  outcome — the 8 kHz band — is applied directly, which is where the entity loss above comes from.
- **No model latency claims from audio runs.** Recognition happens between turns, not in a stream, so
  the timings recorded are the model's and the speech's, never a caller's experience of the gap.
- **One vertical.** The corpus is Danish after-sales booking. The engine takes scenarios as data and is
  not tied to it, but no second vertical is included and none is claimed to work.

## Development

```sh
uv sync
uv run pytest -q
uv run ruff format src tests && uv run ruff check src tests && uv run mypy
```

The suite needs no API key and no `claude` binary: a scripted stand-in speaks Claude Code's stream-json,
so the whole path runs offline and free. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0.
