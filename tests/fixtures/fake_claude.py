"""Stands in for `claude -p ... --output-format stream-json`: reads the prompt and MCP config, calls the
DMS server over stdio the way Claude Code would, and prints Claude Code's stream-json events."""

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


async def main(argv: list[str]) -> None:
    prompt = argv[argv.index("-p") + 1]
    leaked = sorted(k for k in os.environ if k.startswith(("CLAUDE", "VSCODE", "MCP_", "ANTHROPIC_")))
    if "--json-schema" in argv:
        emit(
            {
                "type": "result",
                "subtype": "success",
                "result": "",
                "structured_output": {"echo": prompt},
                "num_turns": 1,
                "total_cost_usd": 0.001,
            }
        )
        return
    if "--mcp-config" not in argv:
        emit(
            {
                "type": "result",
                "subtype": "success",
                "result": f"plain answer leaked={leaked}",
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
