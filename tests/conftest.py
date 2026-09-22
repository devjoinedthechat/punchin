"""Shared rules for the whole suite.

The important one: no test may reach the real model backend. Every path that needs a model is given
`tests/fixtures/fake_claude.py`, which speaks the same stream-json for nothing. A test that forgets and
constructs `ClaudeCodeModel()` with no command would otherwise silently spend money and run at the speed
of the network — which happened once, and took a sweep test from 14 seconds to 85.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make finding the real `claude` binary an error, so forgetting the stand-in fails loudly."""

    def refuse() -> str | None:
        raise AssertionError(
            "a test tried to use the real Claude Code backend. Pass the stand-in instead: "
            "ClaudeCodeModel([sys.executable, str(FAKE)]), or patch punchin.commands.model_for."
        )

    monkeypatch.setattr("punchin.model.find_claude", refuse)
