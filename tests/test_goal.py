import datetime as dt

from punchin.goal import facts, score
from punchin.scenarios import BY_ID, next_weekday

GOAL = BY_ID["self-correction"].goal


def test_a_line_reveals_the_facts_it_contains_and_nothing_else() -> None:
    assert facts(GOAL, "Det er AB 12 345.") == {"reg"}
    assert facts(GOAL, "Kan jeg få en tid tirsdag? ...nej vent, onsdag.") == {"day", "no"}
    assert facts(GOAL, "Kan jeg få en tid tirsdag?") == set()  # the wrong day is not the customer's fact
    assert facts(GOAL, "Ja tak, det passer.") == {"yes"}
    assert facts(BY_ID["courtesy-car"].goal, "Jeg skal have en lånebil imens.") == {
        "constraint:skal have lånebil",
        "extra:lånebil",
    }


def test_a_greeting_is_not_a_farewell() -> None:
    """'Hej' opens a Danish call as often as it ends one, and 'tak' is politeness anywhere in it."""
    assert "bye" not in facts(GOAL, "Ja, det er mig. Hej.")
    assert "bye" not in facts(GOAL, "Ja tak, det passer.")
    assert facts(GOAL, "Tak, hej hej.") == {"bye"}
    assert facts(GOAL, "Fint, farvel.") == {"yes", "bye"}


def test_a_turn_that_is_only_a_time_preference_is_not_an_empty_turn() -> None:
    """Against an empty set a faithful line scores zero; that was the metric, not the simulator."""
    assert "time" in facts(GOAL, "Bare den tidligste, klokken 8.")
    assert "time" in facts(GOAL, "Øh ja, klokken 8 passer fint.")
    assert "time" not in facts(BY_ID["proxy-caller"].goal, "Klokken 8.")  # that customer stated no preference


def test_scoring_against_the_truth() -> None:
    same = GOAL.model_copy()
    assert score(same, GOAL) == {"reg": True, "day": True, "extras_recall": 1.0, "formality": True}
    wrong = GOAL.model_copy(update={"wants_day": next_weekday("tirsdag"), "reg": "AB12346"})
    assert score(wrong, GOAL)["day"] is False
    assert score(wrong, GOAL)["reg"] is False
    assert isinstance(GOAL.wants_day, dt.date)


def test_a_polite_refusal_is_a_refusal_and_not_also_an_agreement() -> None:
    """ "Nej tak, det er fint" is one thing. Tagging it as both was costing a real call half its score."""
    assert facts(GOAL, "Nej tak, det er fint.") == {"no"}
    assert facts(GOAL, "Ja, det er fint.") == {"yes"}
    assert facts(GOAL, "Nej.") == {"no"}
    assert facts(GOAL, "Fint.") == {"yes"}
    # whichever came first is what the line is doing
    assert facts(GOAL, "Ja, men ikke i denne uge.") == {"yes", "next_week"}


def test_what_the_agent_asked_for_is_read_off_its_line() -> None:
    from punchin.goal import solicited

    assert "reg" in solicited("Må jeg få nummerpladen?")
    assert "day" in solicited("Hvilken dag passer dig?")
    assert "time" in solicited("Jeg har en tid kl. 8. Passer det?")
    assert solicited("Så er den booket. Hej hej.") == {"bye"}


def test_a_customer_who_answers_more_than_the_question_is_volunteering() -> None:
    """The measure the soundness check needs: overlap cannot see a simulator that over-helps."""
    from punchin.goal import volunteered

    asked_day = "Hvilken dag passer dig?"
    assert volunteered(GOAL, asked_day, "Onsdag.") == set()
    assert volunteered(GOAL, asked_day, "Onsdag, og gerne klokken 8.") == {"time"}
    # answering the question asked is never volunteering, however much of it there is
    assert volunteered(GOAL, "Må jeg få nummerpladen?", "Det er AB 12 345.") == set()
    # and a plate given before anyone asked is
    assert volunteered(GOAL, "Passer det nu?", "Ja, det er AB 12 345.") == {"reg"}


def test_asking_whether_it_is_a_robot_is_never_counted_as_volunteering() -> None:
    """Some things a customer says whenever they like, and they say nothing about difficulty."""
    from punchin.goal import volunteered

    assert volunteered(GOAL, "Passer det nu?", "Er det en robot jeg taler med?") == set()
