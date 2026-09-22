"""Extraction and teacher-forced fidelity, end to end on the fake backend."""

import sys
from pathlib import Path

from punchin.agent import ScriptedAgent
from punchin.customer import ScriptedCustomer
from punchin.dms import TODAY
from punchin.fidelity import teacher_forced
from punchin.goal import extract, score
from punchin.model import ClaudeCodeModel
from punchin.record import record
from punchin.scenarios import BY_ID

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"


def test_extraction_recovers_the_scenario_truth_and_fidelity_scores_every_customer_turn(
    tmp_path: Path,
) -> None:
    scenario = BY_ID["self-correction"]
    call = record(scenario, ScriptedAgent(careful=True), ScriptedCustomer(scenario), tmp_path / "s.json")
    model = ClaudeCodeModel([sys.executable, str(FAKE)])
    goal, cost = extract(call, model, TODAY)
    assert score(goal, scenario.goal)["reg"] and score(goal, scenario.goal)["day"]
    assert cost == 0.001
    report = teacher_forced(call, goal, model)
    assert len(report.turns) == sum(t.speaker == "customer" for t in call.turns)
    assert report.turns[1].simulated == "Det er AB 12 345."  # asked for the plate, gave the plate
    assert report.turns[1].exact
    assert 0.0 <= report.mean_jaccard <= 1.0
    assert "≠" in report.text() or "=" in report.text()
