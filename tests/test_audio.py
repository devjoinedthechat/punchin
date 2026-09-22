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

SCENARIO = BY_ID["self-correction"]


@pytest.mark.skipif(not READY, reason=MISSING)
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
@pytest.mark.skipif(not READY, reason=MISSING)
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


def test_the_vocabulary_is_the_call_list_and_stays_that_narrow() -> None:
    """Widening it to weekdays and opening hours was measured and cost more than it bought."""
    from punchin.scenarios import SCENARIOS, call_list, spell_plate, vocabulary

    assert spell_plate("AB12345") == "AB 12 345"
    listed = call_list()
    assert all(spell_plate(scenario.vehicle.reg) in listed for scenario in SCENARIOS)
    assert listed.count(",") == len(SCENARIOS) - 1
    assert vocabulary() == listed
    assert "onsdag" not in vocabulary()


def test_a_long_vocabulary_is_kept_out_of_the_decoder_prompt() -> None:
    """A long initial_prompt comes back as the transcript on a short turn. Hotwords always; prompt only
    while it is short enough that the decoder cannot plausibly continue it."""
    from punchin.audio import PROMPT_BUDGET, Recognizer
    from punchin.scenarios import vocabulary

    assert Recognizer("small").bias() == {}

    short = Recognizer("small", vocabulary=vocabulary()).bias()
    assert len(vocabulary()) <= PROMPT_BUDGET
    assert short["hotwords"] == vocabulary()
    assert vocabulary() in short["initial_prompt"]

    padded = Recognizer("small", vocabulary="x" * (PROMPT_BUDGET + 1)).bias()
    assert "hotwords" in padded
    assert "initial_prompt" not in padded


@pytest.mark.slow
@pytest.mark.skipif(not READY, reason=MISSING)
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


@pytest.mark.skipif(not READY, reason=MISSING)
def test_the_linux_voice_path_runs_when_espeak_is_what_is_there(tmp_path: Path) -> None:
    """The fallback a Linux user gets. Untested code is not a fallback, it is a hope."""
    import shutil
    from unittest.mock import patch

    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng is not installed")

    with patch("punchin.audio.speaker", return_value="espeak-ng"):
        wav = speak("Det er AB 12 345.", tmp_path / "e.wav")
        assert wav.exists()
        assert 500 < duration_ms(wav) < 8000
        assert duration_ms(telephone(wav, tmp_path / "e-phone.wav")) == pytest.approx(
            duration_ms(wav), abs=60
        )


def test_which_engine_spoke_is_on_the_customers_name(tmp_path: Path) -> None:
    """Two engines are two distributions to a recogniser; their numbers must not be mixed."""
    from unittest.mock import patch

    from punchin.audio import AudioCustomer, Recognizer

    inner = ScriptedCustomer(SCENARIO)
    for engine in ("say", "espeak-ng"):
        with patch("punchin.audio.speaker", return_value=engine):
            named = AudioCustomer(inner, Recognizer("small"), tmp_path).name
            assert engine in named, named


def test_a_machine_with_no_voice_says_what_to_install(tmp_path: Path) -> None:
    from unittest.mock import patch

    with patch("punchin.audio.speaker", return_value=None):
        ready, missing = available()
        assert ready is False
        assert "espeak-ng" in missing
        with pytest.raises(RuntimeError, match="espeak-ng"):
            speak("Hej.", tmp_path / "x.wav")


def test_choosing_espeak_warns_that_its_numbers_mean_nothing(caplog) -> None:
    """A fallback that produces unreadable audio is worse than none if nobody is told."""
    import logging
    import shutil as sh
    from unittest.mock import patch

    from punchin.audio import ESPEAK_IS_A_TOY, _voices, speaker

    if not sh.which("espeak-ng"):
        pytest.skip("espeak-ng is not installed")

    _voices.cache_clear()
    with (
        patch("shutil.which", lambda name: None if name == "say" else "/usr/bin/espeak-ng"),
        caplog.at_level(logging.WARNING),
    ):
        assert speaker() == "espeak-ng"
    _voices.cache_clear()
    assert ESPEAK_IS_A_TOY in caplog.text
    assert "mean nothing" in ESPEAK_IS_A_TOY
