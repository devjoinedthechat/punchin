"""The Claude Code backend end to end, spending nothing: a fake binary and a real MCP server over stdio."""

import sys
from pathlib import Path

from punchin.agent import mcp_config
from punchin.dms import TODAY, Dms, Vehicle, fresh
from punchin.model import ClaudeCodeModel

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"


def test_tool_calls_come_back_from_stream_json_and_no_session_variables_leak(tmp_path: Path) -> None:
    dms = Dms(tmp_path / "state.json")
    dms.save(
        fresh(
            [Vehicle(reg="AB12345", make="Škoda", model="Octavia", year=2019, owner="Mette", syn_due=TODAY)]
        )
    )
    model = ClaudeCodeModel([sys.executable, str(FAKE)])
    done = model.complete("system", "Kunde: Det er AB 12 345.", mcp=mcp_config(dms))
    assert [(c.tool, c.error) for c in done.tool_calls] == [("lookup_vehicle", None)]
    assert done.tool_calls[0].result["model"] == "Octavia"
    assert done.cost_usd == 0.0123
    assert done.model == "claude-sonnet-5"
    assert "leaked=[]" in done.text
    assert dms.load().calls[0]["tool"] == "lookup_vehicle"  # the server really ran against the state file


def test_a_plain_completion_and_a_structured_one() -> None:
    model = ClaudeCodeModel([sys.executable, str(FAKE)])
    assert ClaudeCodeModel([sys.executable, str(FAKE)]).complete("s", "hej").text.startswith("plain answer")
    assert model.complete("s", "hej", schema={"type": "object"}).structured == {"echo": "hej"}
