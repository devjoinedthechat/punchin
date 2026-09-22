"""Where a call was lost, and where it was still savable."""

import datetime as dt

from punchin.call import Call
from punchin.curve import Curve, Point
from punchin.scenarios import BY_ID

SCENARIO = BY_ID["self-correction"]


class _Report:
    """A fork result with a known before and after, without running anything."""

    def __init__(self, passed: int, trials: int, was: bool = False) -> None:
        self.fixed, self._trials, self._was = passed, trials, was
        self.live_cost_usd = 0.0

    @property
    def attempts(self) -> list[None]:
        return [None] * self._trials

    @property
    def before(self) -> dict:
        return {"correct": self._was}


def _curve(shape: list[tuple[int, int, int]], was: bool = False) -> Curve:
    call = Call(id="c", scenario=SCENARIO.id, agent="a", customer="k", started_at=dt.datetime.now(dt.UTC))
    curve = Curve(call, SCENARIO, "a change")
    curve.points = [Point(at, _Report(p, t, was)) for at, p, t in shape]  # type: ignore[arg-type]
    return curve


def test_a_turn_where_every_attempt_failed_is_where_the_call_goes_astray() -> None:
    curve = _curve([(0, 2, 2), (2, 2, 2), (4, 0, 2), (6, 0, 2)])
    assert curve.always_lost == [4, 6]
    assert curve.first_lost == 4  # nothing after it recovers
    assert "goes astray" in curve.text()
    assert "point of no return" in curve.text()


def test_a_turn_where_attempts_disagree_is_where_the_outcome_is_decided() -> None:
    curve = _curve([(0, 2, 2), (2, 1, 2), (4, 2, 2)])
    assert curve.undecided == [2]
    assert "decided there" in curve.text()


def test_a_curve_that_rises_is_measuring_the_recording_not_the_agent() -> None:
    """Forking late is an easier test: the prefix has already made the decisions."""
    curve = _curve([(0, 0, 2), (2, 1, 2), (4, 2, 2)])
    assert curve.carried_by_prefix is True
    assert "a late fork tests less" in curve.text()
    assert curve.first_lost is None  # turn 0 failed, but later turns recover


def test_a_flat_curve_says_no_single_turn_decided_it() -> None:
    curve = _curve([(0, 2, 2), (2, 2, 2), (4, 2, 2)])
    assert curve.always_lost == []
    assert curve.undecided == []
    assert curve.carried_by_prefix is False
    assert "no single turn decided this call" in curve.text()


def test_a_change_that_never_works_says_so() -> None:
    curve = _curve([(0, 0, 2), (2, 0, 2), (4, 0, 2)])
    assert curve.last_saved is None
    assert curve.first_lost == 0
    assert "does not address this call" in curve.text()


def test_the_numbers_survive_as_json() -> None:
    found = _curve([(0, 2, 2), (2, 0, 2)], was=True).as_dict()
    assert found["was_correct"] is True
    assert found["points"] == [{"at": 0, "passed": 2, "trials": 2}, {"at": 2, "passed": 0, "trials": 2}]
    assert found["always_lost"] == [2]
    assert "carried_by_prefix" in found


def test_a_point_knows_whether_it_settled() -> None:
    assert Point(0, _Report(2, 2)).decided is True  # type: ignore[arg-type]
    assert Point(0, _Report(0, 2)).decided is True  # type: ignore[arg-type]
    assert Point(0, _Report(1, 2)).decided is False  # type: ignore[arg-type]


def test_an_empty_curve_says_nothing_rather_than_guessing() -> None:
    assert "nothing measured" in _curve([]).text()
