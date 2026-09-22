"""Stands in for `claude -p ... --output-format stream-json`: answers the three prompts punchin sends
(an agent turn with tools, a goal-state extraction, a pinned customer line) with rules instead of a model,
and prints Claude Code's stream-json events."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def goal_state(prompt: str) -> dict[str, Any]:
    """A rule-based reading of the transcript: the last plate and the last weekday the customer said."""
    from punchin.scenarios import next_weekday, regs_mentioned, weekdays_mentioned

    customer = " ".join(line[6:] for line in prompt.splitlines() if line.startswith("Kunde:"))
    regs, days = regs_mentioned(customer), weekdays_mentioned(customer)
    weeks = 1 if re.search(r"næste uge|ikke i denne uge", customer, re.I) else 0
    declined = re.search(r"allerede (booket|bestilt)", customer, re.I)
    wants = None if declined or not days else next_weekday(days[-1], weeks_ahead=weeks)
    extras = [e for e in ("lånebil", "stor service") if e.split()[0] in customer.lower()]
    return {
        "intent": "book syn" if wants else "no booking",
        "reg": regs[-1] if regs else "",
        "wants_day": wants.isoformat() if wants else None,
        "constraints": [],
        "extras": extras,
        "formality": "informal",
        "mood": "neutral",
        "reveals": ["reg", "day"],
    }


def customer_line(prompt: str) -> str:
    """Answers the agent's last line from the goal-state block in the prompt, and nothing else."""

    def field(name: str) -> str:
        found = re.search(rf"- {name}: (.*)", prompt)
        return found.group(1) if found else ""

    agents = [line[7:] for line in prompt.splitlines() if line.startswith("Agent:")]
    heard = agents[-1].lower() if agents else ""
    if re.search(r"farvel|hej hej|vi ses", heard):
        return "Tak, hej. [LÆGGER PÅ]"
    if "nummerplade" in heard:
        return f"Det er {field('nummerplade')}."
    if re.search(r"hvilken dag|hvornår|passer dig", heard):
        return f"{field('ønsket dag').split()[0].capitalize()}."
    if re.search(r"skal jeg booke|passer|klokken|kl\.", heard):
        return "Ja tak."
    return "Ja."


async def main(argv: list[str]) -> None:
    prompt = argv[argv.index("-p") + 1]
    leaked = sorted(k for k in os.environ if k.startswith(("CLAUDE", "VSCODE", "MCP_", "ANTHROPIC_")))

    if "--json-schema" in argv:
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        out = goal_state(prompt) if "reg" in schema.get("properties", {}) else {"echo": prompt}
        emit(
            {
                "type": "result",
                "subtype": "success",
                "result": json.dumps(out),
                "structured_output": out,
                "num_turns": 1,
                "total_cost_usd": 0.001,
            }
        )
        return

    if "--mcp-config" not in argv:
        asked_for_a_line = "Skriv kundens næste replik" in prompt
        text = customer_line(prompt) if asked_for_a_line else f"plain answer leaked={leaked}"
        emit(
            {
                "type": "result",
                "subtype": "success",
                "result": text,
                "num_turns": 1,
                "total_cost_usd": 0.002,
            }
        )
        return

    config = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text())["mcpServers"]["dms"]
    reg = re.search(r"\b[A-Z]{2} ?\d{2} ?\d{3}\b", prompt)
    if reg is None:
        raise SystemExit("no plate in the prompt")
    params = StdioServerParameters(command=config["command"], args=config["args"])
    async with Client(params) as client:
        result = await client.call_tool("lookup_vehicle", {"reg": reg.group(0)})
    text = "".join(getattr(b, "text", "") for b in result.content)
    call = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "mcp__dms__lookup_vehicle",
        "input": {"reg": reg.group(0)},
    }
    emit({"type": "assistant", "message": {"content": [call]}})
    answer = {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": text}]}
    emit({"type": "user", "message": {"content": [answer]}})
    emit(
        {
            "type": "result",
            "subtype": "success",
            "result": f"Tak, det er din bil. leaked={leaked}",
            "num_turns": 2,
            "total_cost_usd": 0.0123,
            "modelUsage": {"claude-sonnet-5": {}},
        }
    )


anyio.run(main, sys.argv)
