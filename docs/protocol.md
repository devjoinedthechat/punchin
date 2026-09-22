# The agent protocol

punchin records, forks and gates an agent it knows nothing about. This is everything it needs from one.

An agent is a command. punchin runs it once per turn, writes a JSON request to its standard input, and
reads a JSON reply from its standard output. There is no session and no state to keep: each turn carries
the whole conversation so far, so the command may exit between turns, and a crash costs one turn rather
than a run.

```sh
punchin record --agent command --agent-command "python my_agent.py"
```

[`examples/rule_agent.py`](../examples/rule_agent.py) is a working implementation, exercised in CI.

## The request

```jsonc
{
  "protocol": 1,
  "today": "2026-09-28",
  "today_spoken": "mandag den 28. september",
  "lead": { "owner": "Mette Kjær", "syn_due": "2026-10-10" },
  "tools": {
    "mcp": { "mcpServers": { "dms": { "command": "…", "args": ["…"] } } },
    "state_path": "/…/.punchin/calls/.dms-state.json"
  },
  "conversation": [
    { "speaker": "agent",    "text": "Hej, det er Sofie fra værkstedet…" },
    { "speaker": "customer", "text": "Det er AB-12300-354." }
  ]
}
```

- `lead` is why the call is happening: the customer on file and when their inspection runs out.
- `tools.mcp` is an MCP client config for the dealership system, ready to write to a file and pass to
  any MCP client. `tools.state_path` is the same system's state, for an agent that would rather not
  speak MCP.
- `conversation` is **what the agent heard**, never what was said. When a recogniser sat between the
  customer and the agent, this holds the recogniser's version, mangled entities and all. What she
  actually said stays on the recording as the answer key and is never shown to the agent.

An agent that does not understand `protocol` should exit non-zero saying so, rather than guess.

## The reply

```jsonc
{ "text": "Må jeg få nummerpladen på bilen?", "cost_usd": 0.004 }
```

`text` is the only required field: the line the agent says out loud, nothing else. `cost_usd` is
optional and is summed into the call's cost. Anything else is ignored, so the reply may carry your own
fields.

punchin reads the last JSON object on stdout, so an agent that logs to stdout still works.

## Tool calls

**The agent is never asked what it called.** punchin reads the dealership system's own log before and
after the turn, and the difference is what that turn did. So the tool calls on a recording are the ones
that really happened, whether or not the agent would have reported them accurately, and an agent has
nothing to implement here.

The consequence worth knowing: anything your agent does to the dealership system during a turn is
attributed to that turn, including retries and calls it made and then ignored. That is usually what you
want, and it is always the truth.

## Ending the call

End the reply's `text` with `[FARVEL]` to hang up after that line. The marker is stripped before the
line is recorded, graded or spoken, so it never reaches the customer.

## Failing

| What punchin sees | What it does |
|---|---|
| exit code other than 0 | stops the run, quoting the last of stderr |
| no JSON object on stdout | stops the run, quoting what it got |
| a reply without a string `text` | stops the run, quoting the reply |
| no reply within the timeout (120s) | stops the run |

A stopped run still writes the recording of everything that was said before it stopped, so a failure
halfway through a call is inspectable rather than lost.

## Forking somebody else's agent

`punchin fork` works the same way, with one rule: `--system-suffix` is refused for `--agent command`.
A fork's report names the change under test, and punchin cannot apply a prompt change to an agent whose
prompt it does not own. Change your agent, and fork with the changed command.
