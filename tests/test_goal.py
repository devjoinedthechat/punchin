import datetime as dt

from punchin.goal import facts, score
from punchin.scenarios import BY_ID, next_weekday

GOAL = BY_ID["self-correction"].goal


def test_a_line_reveals_the_facts_it_contains_and_nothing_else() -> None:
    assert facts(GOAL, "Det er AB 12 345.") == {"reg"}
    assert facts(GOAL, "Kan jeg få en tid tirsdag? ...nej vent, onsdag.") == {"day", "no"}
    assert facts(GOAL, "Kan jeg få en tid tirsdag?") == set()  # the wrong day is not the customer's fact
    assert facts(GOAL, "Ja tak, det passer.") == {"yes", "bye"}
    assert facts(BY_ID["courtesy-car"].goal, "Jeg skal have en lånebil imens.") == {
        "constraint:skal have lånebil",
        "extra:lånebil",
    }


def test_scoring_against_the_truth() -> None:
    same = GOAL.model_copy()
    assert score(same, GOAL) == {"reg": True, "day": True, "extras_recall": 1.0, "formality": True}
    wrong = GOAL.model_copy(update={"wants_day": next_weekday("tirsdag"), "reg": "AB12346"})
    assert score(wrong, GOAL)["day"] is False
    assert score(wrong, GOAL)["reg"] is False
    assert isinstance(GOAL.wants_day, dt.date)
