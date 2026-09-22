"""Speaking a turn and hearing it back.

The customer's line is rendered to speech, put through the band a telephone call actually uses, and
handed to a recogniser. The agent then reads what the recogniser produced, not what was said. What
was said stays on the turn as the answer key, which is what makes entity loss measurable rather than
merely suspected.

Everything here is local: macOS `say` for the voice, ffmpeg for the line, and faster-whisper for the
recogniser. No account and no key.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from punchin.call import Call
from punchin.customer import Customer, CustomerTurn

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

DANISH_VOICE = "Sara"  # the one da_DK voice macOS ships
WORDS_PER_MINUTE = 180
# `base` mangles Danish plates past recognition; `small` reads them correctly in studio conditions
# and still loses them down a phone line, which is the interesting case.
DEFAULT_MODEL = "small"
SAMPLE_RATE = 16000


def available() -> tuple[bool, str]:
    """Whether this machine can speak and listen, and what is missing if it cannot."""
    if not shutil.which("say"):
        return False, "no `say` command (macOS only)"
    if not shutil.which("ffmpeg"):
        return False, "no ffmpeg on PATH"
    try:
        import faster_whisper  # noqa: F401, PLC0415
    except ImportError:
        return False, "faster-whisper is not installed: uv sync --extra audio"
    if DANISH_VOICE not in _voices():
        return False, f"no {DANISH_VOICE} voice installed (System Settings, Spoken Content)"
    return True, ""


@cache
def _voices() -> frozenset[str]:
    listed = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False)  # noqa: S607
    return frozenset(line.split()[0] for line in listed.stdout.splitlines() if line.strip())


def _ffmpeg(*arguments: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *arguments], check=True)  # noqa: S603, S607


def speak(text: str, dest: Path, *, voice: str = DANISH_VOICE, rate: int = WORDS_PER_MINUTE) -> Path:
    """Render one line as 16 kHz mono, the way a headset would capture it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = dest.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(raw), text], check=True)  # noqa: S603, S607
    _ffmpeg("-i", str(raw), "-ar", str(SAMPLE_RATE), "-ac", "1", str(dest))
    raw.unlink(missing_ok=True)
    return dest


def telephone(src: Path, dest: Path) -> Path:
    """8 kHz G.711 mu-law and back again: what a phone line does to a voice before anyone hears it."""
    narrow = dest.with_name(f"{dest.stem}-ulaw.wav")
    _ffmpeg("-i", str(src), "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(narrow))
    _ffmpeg("-i", str(narrow), "-ar", str(SAMPLE_RATE), "-ac", "1", str(dest))
    narrow.unlink(missing_ok=True)
    return dest


def with_noise(src: Path, dest: Path, *, snr_db: float = 15.0) -> Path:
    """Mix in the low rumble of a car, at a signal-to-noise ratio in decibels."""
    gain = 10 ** (-snr_db / 20)
    _ffmpeg(
        "-i",
        str(src),
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=color=brown:sample_rate={SAMPLE_RATE}",
        "-filter_complex",
        f"[1:a]volume={gain:.5f}[n];[0:a][n]amix=inputs=2:duration=first[out]",
        "-map",
        "[out]",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        str(dest),
    )
    return dest


def duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:  # "rb" so the reader, not the writer, is in hand
        return round(1000 * handle.getnframes() / handle.getframerate())


class Recognizer:
    """faster-whisper, loaded once and kept."""

    def __init__(self, size: str = DEFAULT_MODEL, *, language: str = "da") -> None:
        self.size = size
        self.language = language
        self.name = f"whisper:{size}"
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel  # noqa: PLC0415 - only the audio path needs it

            self._model = WhisperModel(self.size, device="cpu", compute_type="int8")
        return self._model

    def hear(self, path: Path) -> str:
        segments, _ = self.model.transcribe(str(path), language=self.language, beam_size=5)
        return " ".join(str(segment.text).strip() for segment in segments).strip()


@cache
def recognizer(size: str = DEFAULT_MODEL, language: str = "da") -> Recognizer:
    """One recogniser per size, so a run over the corpus loads the model once rather than per call."""
    return Recognizer(size, language=language)


class AudioCustomer:
    """A text customer, spoken aloud and heard back through a recogniser."""

    def __init__(
        self,
        inner: Customer,
        recognizer: Recognizer,
        out: Path,
        *,
        voice: str = DANISH_VOICE,
        rate: int = WORDS_PER_MINUTE,
        over_the_phone: bool = True,
        snr_db: float | None = None,
    ) -> None:
        self.inner = inner
        self.recognizer = recognizer
        self.out = out
        self.voice = voice
        self.rate = rate
        self.over_the_phone = over_the_phone
        self.snr_db = snr_db
        line = "phone" if over_the_phone else "studio"
        self.name = f"{inner.name}|{voice}|{line}|{recognizer.name}"

    def respond(self, call: Call) -> CustomerTurn | None:
        said = self.inner.respond(call)
        if said is None:
            return None
        stem = f"{call.id}-turn{len(call.turns):02d}"
        wav = speak(said.text, self.out / f"{stem}.wav", voice=self.voice, rate=self.rate)
        if self.over_the_phone:
            wav = telephone(wav, self.out / f"{stem}-phone.wav")
        if self.snr_db is not None:
            wav = with_noise(wav, self.out / f"{stem}-noisy.wav", snr_db=self.snr_db)
        return CustomerTurn(
            said.text, audio=str(wav), heard=self.recognizer.hear(wav), audio_ms=duration_ms(wav)
        )
