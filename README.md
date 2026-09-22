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
  <a href="https://github.com/devjoinedthechat/punchin/actions/workflows/ci.yml"><img src="https://github.com/devjoinedthechat/punchin/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.11–3.13">
  <img src="https://img.shields.io/badge/tests-192-brightgreen" alt="192 tests">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0">
</p>

<p align="center">
  <a href="#try-it">Try it</a> ·
  <a href="#why-this-is-hard">Why this is hard</a> ·
  <a href="#your-agent-not-this-one">Your agent</a> ·
  <a href="#a-call-you-did-not-record">Import</a> ·
  <a href="#which-call-to-fork">Triage</a> ·
  <a href="#as-a-gate">As a gate</a> ·
  <a href="#how-a-fork-works">How a fork works</a> ·
  <a href="#does-a-fork-tell-the-truth">Soundness</a> ·
  <a href="#is-the-simulated-customer-the-real-one">Fidelity</a> ·
  <a href="#the-corpus">Corpus</a> ·
  <a href="#down-a-phone-line">Audio</a> ·
  <a href="#hearing-it">Player</a> ·
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

An outbound Danish *syn* (MOT) reminder, spoken down a simulated phone line to a real model. The
customer asks for Thursday morning. The recogniser hands the agent a fragment:

```
   5 Kunde Torsdag formiddag ville være godt.
      heard: 2. derform i dag ville være godt.
   6 Agent Den 2. oktober har jeg ledigt klokken 8:00, 8:30, 10:00, 13:00 eller 14:30. Passer et af dem?
         -> find_slots({'date_from': '2026-10-02', 'date_to': '2026-10-02'})
  12 Agent Super, så er det booket! Din Toyota Yaris kommer til syn fredag den 2. oktober klokken 8:00.
         -> book({'slot_id': '20261002-0800', 'reg': 'CD 67 890'})
  13 Kunde Tak, hej hej.
```

The model turned half a word into a plausible date and never questioned it. She wanted Thursday the 1st
and is booked for Friday the 2nd. Every tool call succeeded, so no outcome check fires. The call ends
politely with one stall and no repetition, so no quality metric fires either. Only the corpus, which
knows which day she meant, can tell you this call failed.

Fork at turn 6 with an instruction to read the day back before acting on it:

```sh
punchin fork <call> --at 6 --repeat 3 --audio --bias \
  --system-suffix "Læs altid dagen tilbage til kunden med ugedag og dato, og få et ja, før du leder
                   efter ledige tider. Gæt aldrig en dato ud fra noget, du kun delvist forstod."
```

The prefix is identical, so the agent is handed exactly the same mangled fragment — and this time
refuses to guess:

```
   5 Kunde Torsdag formiddag ville være godt.
      heard: 2. derform i dag ville være godt.          <- the same words, from the recording
   6 Agent Undskyld, jeg er ikke helt sikker på, hvilken dag du mener - kan du sige ugedagen igen?
   7 Kunde Øh nej undskyld, jeg mener den 1. oktober - altså torsdag.
   8 Agent Så det er torsdag den 1. oktober, du gerne vil have bilen ind - er det rigtigt?
   9 Kunde Ja, det er rigtigt, torsdag den 1. oktober.
      heard: Ja, det er rigtigt, to steder den den første ude toper.
  10 Agent Godt, jeg kan se der er ledigt kl. 8:00 den dag - passer det dig?
         -> find_slots({'date_from': '2026-10-01', 'date_to': '2026-10-01'})
```

```
  original   correct=False  options_max=5  turns=14  customer_stalls=1
  attempt 1  correct=True   options_max=1  turns=14  customer_stalls=0  live $0.084
  attempt 2  correct=True   options_max=1  turns=16  customer_stalls=0  live $0.086
  attempt 3  correct=True   options_max=1  turns=14  customer_stalls=0  live $0.072
  correct in 3 of 3 attempts (original: False)
  live cost $0.241; the 6 turns before the fork cost nothing
```

Turn 9 is the part worth keeping. Her confirmation is mangled too — *"to steder den den første ude
toper"* — and it does not matter, because by then the read-back only needs an answer shaped like yes,
not an entity. A sentence that asks the customer to carry the information again is fragile; one that
asks them to confirm it is not. That is a conversational property, not a technical one, and the only way
to find it is to run the same broken audio past a changed agent.

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
uv run punchin doctor                          # what works here, and what to install for the rest

uv run punchin scenarios                       # the corpus and the outcome each call expects
uv run punchin record --agent careful          # all ten, free and deterministic
uv run punchin record --agent careless --scenario self-correction
uv run punchin metrics .punchin/calls/*.json
uv run punchin check   .punchin/calls/*.json   # against the baseline this repository ships
```

Against an agent of your own, which is the point of the thing:

```sh
uv run punchin record --agent command --agent-command "python examples/rule_agent.py"
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

## Your agent, not this one

punchin's own agents exist to show what it does. `--agent command` runs anybody else's: one turn at a
time, a JSON request on stdin and a JSON reply on stdout.

```sh
punchin record --agent command --agent-command "python my_agent.py"
```

```jsonc
// in
{"protocol":1,"today":"2026-09-28","lead":{…},"tools":{"mcp":{…}},"conversation":[{"speaker","text"}]}
// out
{"text":"Må jeg få nummerpladen på bilen?"}
```

`text` is the only field an agent has to produce. It is never asked what tools it called: punchin reads
the dealership system's log before and after the turn, so the calls on the recording are the ones that
really happened. And `conversation` is what the agent *heard* — where a recogniser sat in the middle,
the mangled version, with the truth kept back as the answer key.

[docs/protocol.md](docs/protocol.md) is the full contract and
[examples/rule_agent.py](examples/rule_agent.py) a working implementation, both exercised in CI.

## A call you did not record

The reason to want any of this is an archive of calls that already went wrong, and it never arrives in
punchin's shape. `punchin import` reads the shapes it does arrive in — a JSON array of turns, one JSON
object per line, or a plain `Agent:` / `Kunde:` transcript — and writes a recording.

It also asks for something no transcript contains: **what should have happened.**

```sh
punchin import dealer-1482.txt --id dealer-1482 \
  --reg "XY 55 123" --day 2026-10-01 --booked --booked-day 2026-10-02 --extra "lånebil"
```

```
.punchin/calls/20260922-051808-dealer-1482-imported.json  (9 turns, 4 from the customer)
.punchin/scenarios/dealer-1482.json  the outcome it is graded against
```

That second file is the point. No amount of parsing recovers the day the customer meant; somebody has
to say it, once. From then on the call grades, gates and forks exactly like one punchin recorded, and
`--scenarios` is how every command finds the outcomes you have stated.

```
        scenario           correct            day_ok             turns     agent_repeats
     dealer-1482             False             False                 9                 1
```

She said *torsdag*; it booked Friday the 2nd. Nine turns, no repetition, no stalls, a polite goodbye —
a call that reads perfectly and is wrong, found in an archive rather than in a corpus.

Forking it at turn 6 with a read-back instruction, and one about carrying what she mentioned into the
booking note:

```
   5 Kunde Torsdag, tak. Og jeg skal have en lånebil.
   6 Agent Vi har torsdag den 1. oktober kl. 8:00 ledig. Passer det?
   8 Agent Perfekt, så er du booket til torsdag den 1. oktober kl. 8:00, og jeg har noteret at du
           skal have en lånebil. Vi ses! Farvel.
         -> book({'slot_id': '20261001-0800', 'reg': 'XY 55 123', 'note': 'Lånebil ønskes'})

  correct in 2 of 2 attempts (original: False)
  live cost $0.086; the 6 turns before the fork cost nothing
```

## Which call to fork

An archive has more failures than anyone has attention, and a fork costs a model run either way.
`punchin triage` groups the calls that went wrong by what went wrong, so the work is ordered by how
many customers a fix would reach rather than by which call somebody happened to listen to.

```sh
punchin triage .punchin/calls/*.json
```

```
10 calls: 3 came out right, 7 did not, in 2 groups

    4  (40%)  plate never arrived — the registration the customer said never reached the agent
       across 4 scenarios: already-booked, courtesy-car, hurried, wrong-reg-first
       fork this one first: 20260922-015529-hurried-scripted-careful

    3  (30%)  never booked — the customer wanted a booking and did not get one
       across 3 scenarios: code-switch, plain-booking, proxy-caller
       fork this one first: 20260922-015403-plain-booking-scripted-careful
```

The grouping reads the recording's own facts — what the grader said failed, whether the plate survived
the line, whether the agent repeated itself, who hung up — and deliberately does not ask a model. A
cluster nobody can explain in a sentence is not a cluster anybody can act on. A call showing several
faults is filed under the first one worth fixing, so it is counted once, and each group names the
shortest call in it: the cheapest to fork and the easiest to read.

## As a gate

`punchin check` is the part you put in front of a deploy. It takes the numbers a run produced, holds
them against the numbers of the last good run, and exits non-zero naming every scenario that moved the
wrong way. It calls no model: it reads recordings.

```sh
punchin record --agent careful --repeat 3 --out .punchin/ci   # or your own agent
punchin check  .punchin/ci/*.json --update                    # once, to say what good looks like
punchin check  .punchin/ci/*.json                             # every build after that
```

**`--repeat` is not optional.** An agent is sampled, so a scenario that comes out right four times in
five will fail a single-sample gate one build in five, and the rational response to that is to re-run
CI until it is green — which is the same as having no gate at all. A baseline here records how often a
scenario came out right, not whether it did once, and a regression is a rate that fell:

```
14 regressions across 30 runs of 10 scenarios
  already-booked     agent_repeats: 1 -> 5
  code-switch        note_ok: 100% -> 0% of runs
  next-week          correct: 100% -> 0% of runs
  self-correction    day_ok: 100% -> 0% of runs
  wrong-reg-first    customer_stalls: 0 -> 3
  ...
```

A scenario that disagrees with *itself* inside one run is reported as flaky rather than as passing or
failing. For a voice agent that is a finding and not a nuisance: it means the outcome a customer gets
depends on the sampler.

```
2 scenarios disagreed with themselves — the outcome a customer gets depends on the sampler:
  code-switch: came out right in 2 of 3 runs
  hurried:     came out right in 1 of 3 runs
```

(The two agents punchin ships are deterministic and never go flaky. A model will.)

What counts as worse:

| | rule |
|---|---|
| correctness, the day, the plate, the workshop note | must not fall from true to false |
| times read out in one breath, repeated lines, customer stalls | must not go up |
| turns, lookups | may wander a little before they count |
| a metric only one side has | never a regression |

The first row is ground truth: the corpus knows the day she meant. The second row is **hand-written
proxies** — cheap, deterministic, and wrong in ways you can read off the source. Use them for movement
on one call across a change, not as a score for an agent.

This repository gates itself on [baseline.json](baseline.json) in CI.

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

## Does a fork tell the truth?

Everything above rests on one thing: that forking a recording at turn *k* says what a full live re-run
would have said. That is a claim about the tool, and there is a specific reason to doubt it — a
simulated customer can be more helpful than the real one was, which would make every fix look like it
worked.

A fork differs from a live run in two ways at once, so `punchin soundness` separates them:

```sh
punchin soundness --scenario self-correction --at 4 --trials 3 \
  --baseline-suffix "…the prompt the recording was made with…" \
  --system-suffix   "…the change being tested…"
```

| arm | what it runs | what it isolates |
|---|---|---|
| live, full re-run | the scripted customer, from the first turn | the ground truth |
| fork, scripted customer | the recorded prefix, then the same script | the fork mechanism alone |
| fork, pinned customer | the recorded prefix, then the simulator | what punchin actually does |

`fork-scripted` against `live` is whether replaying a prefix changes the answer — a gap there is a bug
in the tool. `fork-pinned` against `fork-scripted` is whether the simulator changes it — a gap there is
a limit of simulation. They need different fixes, so they are reported apart.

On `self-correction`, forked at turn 4, with a baseline prompt that really does book the day she took
back and a change that tells the agent to take her correction and read the day back:

```
  the recording being forked was wrong

  live, full re-run          correct 3/3  (44%-100%)   <- ground truth
  fork, scripted customer    correct 3/3  (44%-100%)
  fork, pinned customer      correct 3/3  (44%-100%)   <- what punchin does

  the fork mechanism moves the answer by +0%
  the pinned customer moves it by       +0%
  forks and live runs agreed on every one of 3 trials
```

Read the intervals before the point estimates, and read this next part before either. Three trials
cannot tell a sound fork from one that is ten percent optimistic. Both the baseline failing and the fix
working are unambiguous here, so every arm sits at its ceiling and the run agrees without having had
much chance to do anything else. The case that would really test it is one where the fix works *some*
of the time — 2 of 3 against 1 of 3 — and engineering that on purpose is hard.

So the number is not the artifact. The check is — and it runs against **your** agent, not only against
punchin's. For an agent of your own the change under test is a different command rather than a different
prompt:

```sh
punchin soundness --scenario self-correction --at 4 --trials 3 --agent command \
  --baseline-command "python my_agent.py" \
  --agent-command    "python my_agent_fixed.py"
```

Only the simulated customer costs anything; both agents are yours and run for free. That is the whole
claim this tool rests on, offered as something you can falsify on your own code rather than as a
reassurance in a README.

## Is the simulated customer the real one?

This is the question the whole tool rests on, so it is measured rather than asserted.

For every customer turn in a recording, the simulator is given the **real** conversation up to the
agent's line before it and asked to write that turn. The real turn is the answer key. The agent's own
randomness never enters, because the agent's lines are always the recorded ones.

```
$ punchin fidelity .punchin/calls/*.json --goal truth --summary

  …self-correction   jaccard 0.87  exact 80%  length x1.51  helps +0.40  over 5 turns
  …wrong-reg-first   jaccard 0.70  exact 40%  length x3.88  helps +0.00  over 5 turns
  …                                                         (eight more)

10 calls, 47 customer turns
  per turn (pooled): jaccard 0.69, exact 51%, volunteers +0.09 facts a turn beyond the real customer
  per call:          mean 0.67, median 0.70, range 0.50-0.87
```

### Is she as difficult as the real one was?

Overlap cannot tell you. A simulated customer can match every fact the real one gave **and hand over
three more**, and an agent talking to a more forthcoming customer has an easier job than the agent on
the real call did. That is how a fix comes to look better than it is.

So *volunteers* is measured separately: per turn, the facts the customer offered that the agent had not
asked for, against what the real customer offered in the same place. Zero is as forthcoming as the woman
on the recording. Above zero means forks run against someone easier.

It is **+0.09** — one extra fact every eleven turns. Small, and pointing the same way as the
[soundness check](#does-a-fork-tell-the-truth), which found no gap at all.

**Read the pooled line, not the per-call one.** Averaging call averages weighs a two-turn call like a
ten-turn one, and a two-turn score moves in steps of 0.50. `already-booked` swinging between 0.00 and
0.50 is that, not a simulator that cannot refuse: its refusal turn matches every time.

**The simulator is two to four times more verbose** than the customer it plays, on every call. Some of
that is this corpus, whose customers answer in one or two words by construction (*"Ja."*, *"Onsdag."*).
On a recording made by a model, where the agent's lines are conversational, the same measurement gives
0.80. Read 0.69 as a floor, and use the number for what it is good at: a change to the simulator,
measured before and after on the same recordings.

### What each part of the goal state is worth

```sh
punchin fidelity <call> --goal truth --ablate --repeat 3
```

drops one part of the goal state at a time, measures each arm several times, and compares them turn by
turn rather than on their averages:

```
  everything           jaccard 0.74  spread 0.11  (0.70 0.81 0.72)

  field                 paired   ± s.e.  turns   verdict
  manner                +0.104    0.031     15   carries its weight
  reveals               +0.018    0.040     15   not distinguishable from zero
  mood                  -0.007    0.036     15   not distinguishable from zero

  Paired by turn, so turn difficulty cancels.
```

The repeats are not optional politeness, and neither is the pairing. Four runs of one call with the
identical goal state came back **0.80, 0.70, 0.83, 0.70** — a spread of 0.13, and 0.23 counting a fifth.
The fields are worth 0.03 to 0.10 each. Comparing the arms' averages therefore cannot see them at all: a
single-run ablation of that call produced a confident-looking table showing four of six fields as
actively harmful, every delta of which was inside the sampler.

Pairing fixes that properly rather than by brute force. Most of the variance is turn difficulty — a turn
where the customer says *"Ja."* scores differently from one where she corrects herself, whatever the
goal state says — and that difficulty is **identical on both sides of the comparison**. Taking the
difference turn by turn cancels it, so a field worth 0.10 becomes visible through 0.13 of run-to-run
spread instead of drowning in it.

A field whose paired difference sits inside two standard errors of zero is reported as *not
distinguishable from zero*. One that stays there is a field the simulator was never using, and it should
come out of the schema rather than sit there looking principled.

Budget for it. Seven arms at three repeats over a five-turn call is a hundred model calls, and each one
is a separate process — tens of minutes and a few tenths of a dollar even with `--workers`. This is a
question you answer when you change the goal state, not something to run in CI.

The same run also grades the extraction that produced the goal state, against the corpus truth: on the
headline call it recovered the plate and the *corrected* Wednesday, not the Tuesday she took back.

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
faster-whisper. **The agent then reads what the recogniser produced and never what was said** — what was
said stays on the turn as the answer key, which is what makes the loss measurable.

```sh
uv run punchin record --agent careful --audio --bias      # a phone line, recogniser told the call list
uv run punchin record --agent careful --audio             # the same line, told nothing
uv run punchin record --agent careful --audio --studio    # a clean microphone
```

### What the number plate survives

A dealership calling about a service knows which cars it is ringing before the phone rings, so the call
list is something a production agent has and a naive one ignores. Passing it to the decoder is the
difference between reading a plate and inventing one. Ten plates, spoken and put through the phone band:

| decoding | plates read correctly | prompt text appearing in transcripts |
|---|---|---|
| nothing passed to the decoder | 1 / 10 | — |
| call list as hotwords | 7 / 10 | none |
| call list as hotwords and a short prompt | 8 / 10 | none |
| that list padded with weekdays and opening hours | 5 / 10 | **12 turns** |

The last row is the one to keep. `initial_prompt` conditions the decoder as if it were speech that came
just before, so a long one is something the decoder can plausibly continue — and does. A customer saying
*"Tirsdag."* came back as *"En samtale om en tidligste."*: a fragment of the prompt itself. The call
list alone, around 120 characters, never leaked once.

So the vocabulary always goes to `hotwords`, and only into the prompt while it stays under a length
budget — a constant in `audio.py`, with the measurement in its comment.

Two cautions on these figures. They come from **synthesized** Danish, not a human caller, and a TTS
voice is a different and in places harder distribution for a recogniser. And rewording the prompt by a
few words moves the end-to-end result by a plate either way, so the step change from 1 is the finding,
not the digit.

### What the call survives

The scripted agent reaches the right outcome on all ten scenarios in text. Over a phone line, with the
call list passed to the decoder, three of ten still do. Where the calls die separates a missing path
from a path that never triggers, and the two agents fail differently enough to be worth both.

**The scripted agent has no fallback at all.** When a plate fails to resolve it asks for the plate
again, in the same words, four or five times, and the customer hangs up. It never spells the plate back,
never asks for the make instead, never offers a person. That is a missing path rather than a bad prompt,
and it cannot appear in a text corpus, where the plate always arrives.

**The model has the ladder and climbs it.** Given the plate no amount of decoder biasing recovers,
sonnet-5 asked for it again, then *"bogstav for bogstav"*, then for just the first two letters, and when
none of that arrived it ended the call politely and said it would ring back another time. That is the
right behaviour, and the right outcome is no booking, so nothing here needs fixing.

**The failure that matters is the one that never reaches the ladder.** In the call at the top of this
page, the same model recovered gracefully from a mis-heard *time* — *"Beklager, jeg hørte dig ikke helt
tydeligt"* — and then silently invented a *date* from audio that was just as bad. A half-heard time
sounds like noise. A half-heard day still looks like a day. A fallback triggers on confusion, and this
failure produces confidence, which is why reading the day back beats any amount of prompting about
uncertainty.

**Outcome checking scores a disaster as a pass.** `already-booked` comes back `correct=True` over a
phone line, because the right outcome there is no booking and the call collapsed before making one. The
feel columns are the only thing that disagrees: `agent_repeats=5`, `customer_stalls=3`,
`ended_by=customer`. The inverse of the headline call, where the feel columns were clean and the outcome
was wrong. Neither kind of check finds both, which is why punchin keeps ground truth as well.

## Hearing it

A table says a call got worse. `punchin player` builds one page that plays both calls side by side,
with the audio embedded, so it opens from disk and can be attached to a bug report without a server.

```sh
punchin player <before> <after> --out call.html
```

Turns served from the recording are dimmed and the fork point is marked, so it is obvious which part of
the second call is the change and which part is the same conversation. Where a recogniser sat between
the customer and the agent, both lines are shown: what she said, and under it what arrived.

## What it does not do

**No real-time turn-taking.** The loop is turn-driven, which is the trade that buys forking: you cannot
serve a prefix and hand over control mid-utterance at once. Barge-in, endpointing and the gap before a
reply need a duplex pipeline, and are measured well elsewhere — EVA-Bench, IHBench, and every commercial
voice-testing platform.

**No telephony.** No phone number, no carrier. The part of a call that changes the outcome, the 8 kHz
band, is applied directly.

**No latency a caller would recognise.** Recognition runs between turns, not in a stream, so a recorded
timing is the model's and the speech's, never the silence someone waited through.

**Audio wants a Mac.** Off macOS it falls back to espeak-ng and runs, but espeak's Danish is not
intelligible to the recogniser — *"Det er AB 12 345."* comes back as *"Vi er med til at tjekke på en
annen tema før"*, biased or not. It keeps the code exercised and says so at startup. Every figure here
comes from the macOS `Sara` voice.

**One vertical.** Danish after-sales booking. Scenarios are data and the engine is not tied to them, but
no second vertical ships and none is claimed to work.

**The feel metrics are uncalibrated.** No human has rated the same calls. Good for movement across a
change, not for scoring an agent.

**Soundness is checked, not established.** Three trials, one scenario, every arm at its ceiling. A fix
that works only some of the time would test it properly, and engineering one deliberately is unsolved.

**Fidelity is noisier than what it measures.** Repeated runs of one call spread 0.13 to 0.23; the
goal-state fields are worth 0.03 to 0.10 each. Pairing and `--repeat` make the gap visible rather than
closing it, so no field can be called worthless yet.

**A scenario's outcome is somebody's judgement.** An imported call is graded against what a person said
should have happened — explicit rather than inferred, which also means a wrong judgement grades
confidently wrong.

## Development

```sh
uv sync
uv run pytest -q
uv run ruff format src tests examples scripts
uv run ruff check src tests examples scripts && uv run mypy
uv run python scripts/import_smoke.py     # reading somebody else's transcript still works
```

The suite needs no API key and no `claude` binary: a scripted stand-in speaks Claude Code's stream-json,
so the whole path runs offline and free. The audio tests skip themselves unless `say`, ffmpeg and
faster-whisper are all present, and `-m slow` holds the two that need a recogniser model on disk.

```
punchin doctor             what works on this machine, and what to install for the rest
punchin scenarios          the corpus, and the outcome each call expects
punchin import             somebody else's transcript, plus the outcome you say was right
punchin triage             group the calls that went wrong, biggest group first
punchin soundness          check that forking says what a full re-run says
punchin record             run a scenario, optionally spoken and over a phone band
punchin show               a recording, with its fork points and what was heard
punchin metrics            outcome and feel numbers, as a table or --json
punchin check              fail when a run is worse than the baseline
punchin fork               re-run a recording from one turn with a change applied
punchin extract            the customer's goal state, read out of a recording
punchin fidelity           teacher-forced: is the simulated customer the real one?
                           --summary for a spread over many calls, --ablate for what each
                           part of the goal state is worth
punchin player             one page that plays two calls side by side
punchin dms                the dealership system as an MCP server, for the model to call
```

Progress goes to stderr and the report to stdout, so a command can be piped; `-q` silences the
progress. A model that never ran (a logged-out client) is an error rather than an agent turn, and a
transient failure is retried. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0.
