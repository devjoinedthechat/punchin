"""A model behind one completion at a time, and Claude Code in print mode as the first backend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from punchin.call import ToolCall

# Left out of the child's environment: a parent Claude Code session's variables would make the child
# think it is nested inside it, and ANTHROPIC_* would point it at a key or endpoint meant for the parent.
_INHERITED = ("CLAUDE", "VSCODE", "MCP_", "ANTHROPIC_")


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    structured: Any = None
    cost_usd: float = 0.0
    model: str = ""
    elapsed_ms: int = 0


class Model(Protocol):
    name: str

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        mcp: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Completion: ...


def find_claude() -> str | None:
    """The `claude` command, or the one bundled with the newest Claude Code VS Code extension."""
    if found := shutil.which("claude"):
        return found
    bundled = sorted(
        Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")
    )
    return str(bundled[-1]) if bundled else None


class ClaudeCodeModel:
    """One completion per `claude -p`, with the login Claude Code already has; no API key.

    Built-in tools are all off (`--tools ""`). Tools arrive only through `mcp`, an MCP config, and the
    calls come back from the stream-json output exactly as the model made them. Nothing is written to
    the user's session history, and every call is capped by `--max-budget-usd`.
    """

    def __init__(
        self,
        command: list[str] | None = None,
        model: str = "claude-sonnet-5",
        *,
        max_turns: int = 8,
        budget_usd: float = 0.50,
        timeout_s: float = 240.0,
    ) -> None:
        if command is None:
            found = find_claude()
            if found is None:
                raise RuntimeError("no `claude` command found; install Claude Code or pass command=[...]")
            command = [found]
        self.command = command
        self.model = model
        self.max_turns = max_turns
        self.budget_usd = budget_usd
        self.timeout_s = timeout_s
        self.name = f"claude-code:{model}"

    def _arguments(
        self, system: str, prompt: str, config: Path | None, schema: dict[str, Any] | None
    ) -> list[str]:
        args = [
            *self.command,
            "-p",
            prompt,
            "--model",
            self.model,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--permission-prompts",
            "none",
            "--max-turns",
            str(self.max_turns),
            "--max-budget-usd",
            f"{self.budget_usd:.2f}",
            "--setting-sources",
            "project",
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if config is not None:
            args += ["--mcp-config", str(config), "--strict-mcp-config", "--allowedTools", "mcp__dms"]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]
        return args

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        mcp: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        env = {k: v for k, v in os.environ.items() if not k.startswith(_INHERITED)}
        with tempfile.TemporaryDirectory(prefix="punchin-claude-code-") as cwd:
            config = None
            if mcp is not None:
                config = Path(cwd) / "mcp.json"
                config.write_text(json.dumps(mcp))
            started = time.monotonic()
            done = subprocess.run(  # noqa: S603 - the command is Claude Code, chosen by the caller
                self._arguments(system, prompt, config, schema),
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            elapsed = int((time.monotonic() - started) * 1000)
        completion = self._read(done.stdout, done.stderr)
        completion.elapsed_ms = elapsed
        return completion

    @staticmethod
    def _events(stdout: str) -> Iterator[dict[str, Any]]:
        for line in stdout.splitlines():
            if line.strip().startswith("{"):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _read(self, stdout: str, stderr: str) -> Completion:
        pending: dict[str, tuple[str, dict[str, Any]]] = {}
        calls: list[ToolCall] = []
        final: dict[str, Any] | None = None
        for event in self._events(stdout):
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = str(block.get("name", "")).removeprefix("mcp__dms__")
                    pending[block["id"]] = (name, dict(block.get("input") or {}))
                elif block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    name, arguments = pending.pop(block["tool_use_id"])
                    text = _text(block.get("content"))
                    try:
                        result: Any = json.loads(text)
                    except json.JSONDecodeError:
                        result = text
                    failed = bool(block.get("is_error"))
                    calls.append(
                        ToolCall(
                            tool=name, arguments=arguments, result=result, error=text if failed else None
                        )
                    )
            if event.get("type") == "result":
                final = event
        for name, arguments in pending.values():
            calls.append(ToolCall(tool=name, arguments=arguments, error="no result before the session ended"))
        if final is None:
            detail = stderr.strip()[-500:] or stdout[-500:]
            raise RuntimeError(f"Claude Code ended without a result: {detail}")
        if final.get("is_error"):
            raise RuntimeError(f"Claude Code failed: {final.get('result') or final.get('subtype')}")
        text = str(final.get("result") or "")
        structured = final.get("structured_output")
        if structured is None and text.lstrip().startswith(("{", "[")):
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        used = ", ".join((final.get("modelUsage") or {}).keys()) or self.model
        return Completion(text, calls, structured, float(final.get("total_cost_usd") or 0.0), used)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""
