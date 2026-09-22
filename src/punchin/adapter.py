"""Bring your own agent.

The agents in `punchin.agent` exist to show what the tool does. They are not the agent anyone wants to
test. `CommandAgent` runs an agent that lives outside this package, one turn at a time, over a JSON
request on stdin and a JSON reply on stdout — so any stack that can read JSON can be recorded, forked
and gated.

The contract is deliberately one-sided: the agent is asked for a line and answers with a line. It is
never asked what tools it called, because it would have to be trusted about that. punchin reads the
dealership system's own log before and after the turn instead, so the tool calls on the recording are
the ones that really happened.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from typing import Any

from punchin.agent import AgentTurn, Lead, mcp_config, say_date
from punchin.call import Call, ToolCall
from punchin.dms import Dms

PROTOCOL = 1


class AgentProtocolError(RuntimeError):
    """The command did not hold up its end of the contract."""


def request_for(call: Call, lead: Lead, dms: Dms, today: dt.date) -> dict[str, Any]:
    """What the agent is told. The conversation is what the agent HEARD, never what was said."""
    return {
        "protocol": PROTOCOL,
        "today": today.isoformat(),
        "today_spoken": say_date(today),
        "lead": {"owner": lead.owner, "syn_due": lead.syn_due.isoformat()},
        "tools": {"mcp": mcp_config(dms), "state_path": str(dms.path)},
        "conversation": [{"speaker": turn.speaker, "text": turn.as_heard} for turn in call.turns],
    }


def _reply_from(stdout: str) -> dict[str, Any]:
    """The last JSON object the command printed, so a chatty agent's logging does not break it."""
    for line in reversed([line for line in stdout.splitlines() if line.strip().startswith("{")]):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AgentProtocolError(f"no JSON object on stdout; got {stdout.strip()[:300]!r}")


class CommandAgent:
    """An agent in another process, asked for one turn at a time."""

    def __init__(
        self,
        command: list[str],
        today: dt.date,
        *,
        timeout_s: float = 120.0,
        name: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("an agent command cannot be empty")
        self.command = command
        self.today = today
        self.timeout_s = timeout_s
        self.name = name or f"command:{command[0].rsplit('/', 1)[-1]}"

    def respond(self, call: Call, lead: Lead, dms: Dms) -> AgentTurn:
        before = len(dms.load().calls)
        payload = json.dumps(request_for(call, lead, dms, self.today), ensure_ascii=False)
        started = time.monotonic()
        try:
            done = subprocess.run(  # noqa: S603 - the command is the user's own agent
                self.command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            raise AgentProtocolError(f"{self.name} did not answer in {self.timeout_s:.0f}s") from expired
        elapsed = int((time.monotonic() - started) * 1000)
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip()[-400:]
            raise AgentProtocolError(f"{self.name} exited {done.returncode}: {detail}")

        reply = _reply_from(done.stdout)
        text = reply.get("text")
        if not isinstance(text, str):
            raise AgentProtocolError(f"{self.name} replied without a string `text`: {reply!r}")

        # Whatever the agent says it did, this is what the dealership system recorded.
        made = dms.load().calls[before:]
        calls = [
            ToolCall(
                tool=str(entry["tool"]),
                arguments=dict(entry.get("arguments") or {}),
                result=entry.get("result"),
                error=entry.get("error"),
            )
            for entry in made
        ]
        return AgentTurn(text.strip(), calls, elapsed, float(reply.get("cost_usd") or 0.0))
