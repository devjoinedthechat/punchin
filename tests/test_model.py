"""The Claude Code backend end to end, spending nothing: a fake binary and a real MCP server over stdio."""

import sys
from pathlib import Path

import pytest

from punchin.agent import mcp_config
from punchin.dms import TODAY, Dms, Vehicle, fresh
from punchin.model import ClaudeCodeModel, ModelDidNotRun

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


def test_a_logged_out_client_is_a_failure_not_an_agent_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude Code answers `success` with "Not logged in" and no model usage. Recorded as a turn, the
    agent would say that to a customer, so it has to raise instead."""
    monkeypatch.setenv("FAKE_CLAUDE_LOGGED_OUT", "1")
    with pytest.raises(ModelDidNotRun, match="logged in"):
        ClaudeCodeModel([sys.executable, str(FAKE)]).complete("s", "hej")


def test_a_logged_out_client_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_LOGGED_OUT", "1")
    tries = tmp_path / "tries"
    monkeypatch.setenv("FAKE_CLAUDE_COUNT", str(tries))
    model = ClaudeCodeModel([sys.executable, str(FAKE)], attempts=3, backoff_s=0)
    with pytest.raises(ModelDidNotRun):
        model.complete("s", "hej")


def test_a_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One flaky call must not kill a three-attempt fork."""
    monkeypatch.setenv("FAKE_CLAUDE_FLAKY_ONCE", str(tmp_path / "failed-once"))
    model = ClaudeCodeModel([sys.executable, str(FAKE)], attempts=3, backoff_s=0)
    assert model.complete("s", "hej").text.startswith("plain answer")
    assert (tmp_path / "failed-once").exists()


def test_a_failure_that_never_clears_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ClaudeCodeModel(
        [sys.executable, "-c", "raise SystemExit('always broken')"], attempts=2, backoff_s=0
    )
    with pytest.raises(RuntimeError, match="without a result"):
        model.complete("s", "hej")
