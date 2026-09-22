from pathlib import Path

from punchin.agent import ScriptedAgent
from punchin.customer import ScriptedCustomer
from punchin.metrics import options_offered, outcome, summarize
from punchin.record import record
from punchin.scenarios import BY_ID


def test_a_menu_of_times_is_counted() -> None:
    assert (
        options_offered(
            "Super, onsdag har jeg ledigt klokken 8, halv ni, ti, ét eller halv tre. Passer et af dem?"
        )
        == 5
    )
    assert options_offered("Jeg har en tid onsdag den 30. september kl. 08:00. Skal jeg booke den?") == 1
    assert options_offered("Hvilken dag passer dig?") == 0


def test_outcome_tells_the_careful_and_careless_agents_apart(tmp_path: Path) -> None:
    scenario = BY_ID["self-correction"]
    good = record(scenario, ScriptedAgent(careful=True), ScriptedCustomer(scenario), tmp_path / "s.json")
    bad = record(scenario, ScriptedAgent(careful=False), ScriptedCustomer(scenario), tmp_path / "s.json")
    assert outcome(good, scenario)["correct"] is True
    assert outcome(bad, scenario) == {
        "booked": True,
        "should_book": True,
        "day_ok": False,
        "reg_ok": True,
        "note_ok": True,
        "correct": False,
    }
    row = summarize(good, scenario)
    assert row["options_max"] == 1
    assert row["customer_stalls"] == 0
    assert row["model_ms_mean"] is None  # scripted: there was no model, so no latency is claimed
