"""What punchin can do on this machine, and whether it says so honestly."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from punchin.cli import main
from punchin.doctor import Finding, examine, report


def test_it_looks_at_everything_punchin_leans_on(tmp_path: Path) -> None:
    found = {f.what for f in examine(tmp_path / "scenarios", tmp_path / "b.json")}
    assert found == {"model backend", "voice", "ffmpeg", "recogniser", "scenarios", "baseline"}


def test_a_machine_with_nothing_on_it_says_what_to_install(tmp_path: Path) -> None:
    with patch("shutil.which", return_value=None), patch("punchin.doctor.find_claude", return_value=None):
        findings = examine(tmp_path / "scenarios", tmp_path / "b.json")
    by_name = {f.what: f for f in findings}
    assert by_name["model backend"].state == "missing"
    assert "--agent command" in by_name["model backend"].fix
    assert by_name["voice"].state == "missing"
    assert by_name["ffmpeg"].state == "missing"
    assert "install" in by_name["ffmpeg"].fix

    printed = report(findings)
    assert "missing" in printed
    # and the thing that still works without any of it
    assert "text-only recording, forking and gating work" in printed


def test_espeak_alone_is_usable_but_flagged(tmp_path: Path) -> None:
    with (
        patch("shutil.which", lambda n: None if n == "say" else "/usr/bin/" + n),
        patch("punchin.doctor.find_claude", return_value="/usr/bin/claude"),
    ):
        voice = next(f for f in examine(tmp_path, tmp_path / "b.json") if f.what == "voice")
    assert voice.state == "warn"
    assert "mean nothing" in voice.fix


def test_a_baseline_from_another_version_is_flagged_not_trusted(tmp_path: Path) -> None:
    stale = tmp_path / "b.json"
    stale.write_text(json.dumps({"format": 99, "scenarios": {}}))
    baseline = next(f for f in examine(tmp_path, stale) if f.what == "baseline")
    assert baseline.state == "warn"
    assert "--update" in baseline.fix


def test_a_directory_where_a_baseline_should_be_is_reported_not_raised(tmp_path: Path) -> None:
    """A doctor that raises is the worst kind of doctor."""
    baseline = next(f for f in examine(tmp_path, tmp_path) if f.what == "baseline")
    assert baseline.state == "warn"
    assert "--update" in baseline.fix


def test_scenarios_on_disk_are_counted(tmp_path: Path) -> None:
    from punchin.scenarios import BY_ID

    (tmp_path / "mine.json").write_text(
        BY_ID["plain-booking"].model_copy(update={"id": "mine"}).model_dump_json()
    )
    scenarios = next(f for f in examine(tmp_path, tmp_path / "b.json") if f.what == "scenarios")
    assert scenarios.state == "ok"
    assert "1 from" in scenarios.detail


def test_a_broken_scenario_file_is_named(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text('{"id": "x"}')
    scenarios = next(f for f in examine(tmp_path, tmp_path / "b.json") if f.what == "scenarios")
    assert scenarios.state == "missing"


def test_the_command_exits_nonzero_only_when_something_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    code = main(["doctor", "--scenarios", str(tmp_path), "--baseline", str(tmp_path / "b.json"), "-q"])
    printed = capsys.readouterr().out
    assert "what punchin can do on this machine" in printed
    assert code in (0, 1)  # depends on the machine, and either is a correct answer

    with patch("shutil.which", return_value=None), patch("punchin.doctor.find_claude", return_value=None):
        assert main(["doctor", "--scenarios", str(tmp_path), "--baseline", str(tmp_path), "-q"]) == 1


def test_an_ok_finding_does_not_print_a_fix() -> None:
    assert "do this" not in Finding("x", "ok", "fine", "do this").text()
    assert "do this" in Finding("x", "missing", "broken", "do this").text()
