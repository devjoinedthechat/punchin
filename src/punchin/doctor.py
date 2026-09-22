"""What works on this machine, and what to do about what does not.

punchin leans on things it did not install: a Claude Code login for the model, ffmpeg and a Danish
voice and a recogniser for audio, a scenario directory for anything imported. Each of those fails in
the middle of a run otherwise, several minutes in, which is the worst moment to find out.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from punchin.audio import DANISH_VOICE, ESPEAK_IS_A_TOY
from punchin.check import FORMAT as BASELINE_FORMAT
from punchin.model import find_claude
from punchin.scenarios import SCENARIO_DIR, load_scenarios

State = Literal["ok", "warn", "missing"]
MARK = {"ok": "ok  ", "warn": "warn", "missing": "no  "}


@dataclass
class Finding:
    what: str
    state: State
    detail: str
    fix: str = ""

    def text(self) -> str:
        line = f"  {MARK[self.state]} {self.what:22} {self.detail}"
        return line if not self.fix or self.state == "ok" else f"{line}\n       {self.fix}"


def _model() -> list[Finding]:
    found = find_claude()
    if found is None:
        return [
            Finding(
                "model backend",
                "missing",
                "no `claude` command",
                "install Claude Code, or drive punchin with --agent command instead",
            )
        ]
    # Not run here: starting it costs a model call, and a doctor should be free.
    return [Finding("model backend", "ok", "Claude Code found", "")]


def _audio() -> list[Finding]:
    findings: list[Finding] = []

    if shutil.which("say"):
        listed = shutil.which("say") and DANISH_VOICE
        findings.append(
            Finding("voice", "ok", f"macOS say, {listed}")
            if _has_danish_voice()
            else Finding(
                "voice",
                "missing",
                f"macOS say has no {DANISH_VOICE} voice",
                "System Settings, Accessibility, Spoken Content, Manage Voices, Danish",
            )
        )
    elif shutil.which("espeak-ng"):
        findings.append(Finding("voice", "warn", "espeak-ng only", ESPEAK_IS_A_TOY))
    else:
        findings.append(Finding("voice", "missing", "no text-to-speech", "install espeak-ng, or use a Mac"))

    findings.append(
        Finding("ffmpeg", "ok", "found")
        if shutil.which("ffmpeg")
        else Finding("ffmpeg", "missing", "not on PATH", "brew install ffmpeg / apt install ffmpeg")
    )

    try:
        import faster_whisper  # noqa: F401, PLC0415

        findings.append(Finding("recogniser", "ok", "faster-whisper installed"))
    except ImportError:
        findings.append(
            Finding("recogniser", "missing", "faster-whisper not installed", "uv sync --extra audio")
        )
    return findings


def _has_danish_voice() -> bool:
    from punchin.audio import _voices  # noqa: PLC0415 - reading the cache the audio module keeps

    return DANISH_VOICE in _voices()


def _corpus(scenarios: Path, baseline: Path) -> list[Finding]:
    findings = []
    try:
        known = load_scenarios(scenarios)
        extra = len(known) - len(load_scenarios(None))
        findings.append(
            Finding("scenarios", "ok", f"{len(known)} known ({extra} from {scenarios})")
            if extra
            else Finding("scenarios", "ok", f"{len(known)} built in, none in {scenarios}")
        )
    except ValueError as bad:
        findings.append(Finding("scenarios", "missing", str(bad)[:80], "fix or remove that file"))

    if not baseline.is_file():
        findings.append(Finding("baseline", "warn", f"none at {baseline}", "punchin check <calls> --update"))
    else:
        import json  # noqa: PLC0415

        try:
            written = json.loads(baseline.read_text()).get("format")
        except (ValueError, OSError):
            # A doctor that raises is the worst kind of doctor. Unreadable is simply not a baseline.
            written = None
        findings.append(
            Finding("baseline", "ok", f"{baseline}, format {written}")
            if written == BASELINE_FORMAT
            else Finding(
                "baseline",
                "warn",
                f"{baseline} is format {written}, this punchin reads {BASELINE_FORMAT}",
                "punchin check <calls> --update",
            )
        )
    return findings


def examine(scenarios: Path = SCENARIO_DIR, baseline: Path = Path("baseline.json")) -> list[Finding]:
    return [*_model(), *_audio(), *_corpus(scenarios, baseline)]


def report(findings: list[Finding]) -> str:
    lines = ["what punchin can do on this machine", ""]
    lines.extend(finding.text() for finding in findings)
    broken = [f for f in findings if f.state == "missing"]
    warned = [f for f in findings if f.state == "warn"]
    lines.append("")
    if not broken and not warned:
        lines.append("  everything it needs is here")
    else:
        if broken:
            lines.append(f"  {len(broken)} missing: {', '.join(f.what for f in broken)}")
        if warned:
            lines.append(f"  {len(warned)} usable but worth knowing: {', '.join(f.what for f in warned)}")
        lines.append("  text-only recording, forking and gating work without any of the audio ones.")
    return "\n".join(lines)
