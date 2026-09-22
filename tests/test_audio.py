"""The audio layer, on this machine only: it needs `say`, ffmpeg and faster-whisper."""

from pathlib import Path

import pytest

from punchin.audio import available, duration_ms, speak, telephone, with_noise
from punchin.call import Call, Turn
from punchin.customer import CustomerTurn, ScriptedCustomer
from punchin.dms import normalize_reg
from punchin.metrics import entities
from punchin.scenarios import BY_ID, regs_mentioned

READY, MISSING = available()
pytestmark = pytest.mark.skipif(not READY, reason=MISSING)

SCENARIO = BY_ID["self-correction"]


def test_a_line_becomes_a_wav_a_phone_line_narrows_and_noise_lengthens_nothing(tmp_path: Path) -> None:
    clean = speak("Det er AB 12 345.", tmp_path / "clean.wav")
    assert clean.exists()
    spoken_ms = duration_ms(clean)
    assert 800 < spoken_ms < 6000  # a short sentence, said at a normal pace

    called = telephone(clean, tmp_path / "phone.wav")
    assert duration_ms(called) == pytest.approx(spoken_ms, abs=60)

    noisy = with_noise(called, tmp_path / "noisy.wav", snr_db=15)
    assert duration_ms(noisy) == pytest.approx(spoken_ms, abs=60)
    assert noisy.stat().st_size > 0


def test_what_was_said_stays_the_answer_key_and_only_what_was_heard_reaches_the_agent() -> None:
    call = Call(id="c", scenario="s", agent="a", customer="c", started_at=_when())
    call.turns = [
        Turn(index=0, speaker="agent", text="Hvad er nummerpladen?", started_at=_when(), ended_at=_when()),
        Turn(
            index=1,
            speaker="customer",
            text="Det er AB 12 345.",
            started_at=_when(),
            ended_at=_when(),
            heard="Det er AB-12300-354.",
        ),
    ]
    assert call.turns[1].spoken == "Det er AB 12 345."
    assert call.turns[1].as_heard == "Det er AB-12300-354."
    assert "Kunde: Det er AB-12300-354." in call.transcript()
    assert "Kunde: Det er AB 12 345." in call.transcript(heard=False)


def test_a_plate_lost_in_the_recogniser_shows_up_as_a_lost_plate() -> None:
    call = Call(id="c", scenario="self-correction", agent="a", customer="c", started_at=_when())
    call.turns = [
        Turn(
            index=0,
            speaker="customer",
            text="Det er AB 12 345.",
            started_at=_when(),
            ended_at=_when(),
            heard="Det er AB-12300-354.",
        )
    ]
    scored = entities(call, SCENARIO)
    assert scored["reg_spoken"] is True
    assert scored["reg_heard"] is False  # she said it; the agent never got it
    assert scored["reg_survived"] is False


@pytest.mark.slow
def test_the_danish_voice_and_the_recogniser_agree_on_a_plain_sentence(tmp_path: Path) -> None:
    """A sanity check on the pair, not on the corpus: a simple line has to survive."""
    from punchin.audio import Recognizer

    wav = speak("Jeg vil gerne booke en tid på onsdag.", tmp_path / "plain.wav")
    heard = Recognizer("small").hear(wav)
    assert "onsdag" in heard.lower()


def test_a_biased_recogniser_says_so_in_its_name_and_an_unbiased_one_does_not() -> None:
    from punchin.audio import Recognizer

    assert Recognizer("small").name == "whisper:small"
    assert Recognizer("small", vocabulary="AB 12 345").name == "whisper:small+bias"


def test_the_vocabulary_covers_every_closed_set_a_call_contains_not_only_the_plates() -> None:
    """A weekday is as closed a set as a plate. Leaving it out loses the day after the plate survived."""
    from punchin.scenarios import SCENARIOS, call_list, spell_plate, vocabulary

    assert spell_plate("AB12345") == "AB 12 345"
    listed = call_list()
    assert all(spell_plate(scenario.vehicle.reg) in listed for scenario in SCENARIOS)
    assert listed.count(",") == len(SCENARIOS) - 1

    full = vocabulary()
    assert listed in full
    assert all(day in full for day in ("mandag", "tirsdag", "onsdag", "torsdag", "fredag"))
    assert "formiddag" in full and "klokken 8" in full


@pytest.mark.slow
def test_the_call_list_recovers_a_plate_the_phone_line_destroyed(tmp_path: Path) -> None:
    """Naive decoding loses this plate; telling the recogniser the call list gets it back."""
    from punchin.audio import Recognizer, speak, telephone
    from punchin.scenarios import call_list

    wav = telephone(speak("Det er CD 67 890.", tmp_path / "p.wav"), tmp_path / "p-phone.wav")
    assert "CD67890" not in regs_mentioned(Recognizer("small").hear(wav))
    assert "CD67890" in regs_mentioned(Recognizer("small", vocabulary=call_list()).hear(wav))


def test_the_scripted_customer_still_answers_when_nothing_is_spoken() -> None:
    answer = ScriptedCustomer(SCENARIO).respond(
        Call(id="c", scenario="s", agent="a", customer="c", started_at=_when())
    )
    assert isinstance(answer, CustomerTurn)
    assert answer.audio is None and answer.heard is None


def test_plates_are_read_out_of_a_line_however_the_recogniser_punctuated_them() -> None:
    assert regs_mentioned("Det er AB 12.345") == ["AB12345"]
    assert regs_mentioned("Det er AB-12-345.") == ["AB12345"]
    assert normalize_reg("T.A.B. 12.345") != "AB12345"  # a leading letter is a different plate


def _when():
    import datetime as dt

    return dt.datetime.now(dt.UTC)
