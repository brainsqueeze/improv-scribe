"""
scripts/generate_fixtures.py — Generate synthetic test WAV fixtures.

Creates mono float32 WAV files of pure sine tones at known MIDI pitches,
used by the test suite to validate the pitch estimation and note tracking
pipeline end-to-end without requiring a physical instrument.

Each fixture is a sequence of notes with known onset times, making it
possible to assert exact expected MIDI values in tests.

Usage
-----
    python scripts/generate_fixtures.py

Output files are written to tests/fixtures/.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
SAMPLE_RATE = 44100
AMPLITUDE = 0.7   # below clipping, above noise gate default (0.01)


def midi_to_hz(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def sine_tone(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine wave with a brief exponential decay (pluck envelope)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Exponential decay simulates a plucked string
    envelope = np.exp(-3.0 * t / duration_s)
    return (AMPLITUDE * envelope * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert float32 → int16 for broad compatibility
    data_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        wf.writeframes(data_int16.tobytes())
    print(f"  Written: {path}  ({len(audio)/sr:.2f}s)")


# ---------------------------------------------------------------------------
# Fixture definitions
# ---------------------------------------------------------------------------

def make_single_note(midi: int, duration_s: float = 1.0) -> None:
    """Single sustained note."""
    note_name = f"midi_{midi:03d}"
    audio = sine_tone(midi_to_hz(midi), duration_s)
    write_wav(FIXTURES_DIR / f"single_{note_name}.wav", audio)


def make_scale(midi_notes: list[int], note_dur: float = 0.4, gap_dur: float = 0.05) -> str:
    """A sequence of notes with short silences between them."""
    chunks: list[np.ndarray] = []
    for midi in midi_notes:
        chunks.append(sine_tone(midi_to_hz(midi), note_dur))
        chunks.append(silence(gap_dur))
    audio = np.concatenate(chunks)
    name = f"scale_{midi_notes[0]}_{midi_notes[-1]}"
    write_wav(FIXTURES_DIR / f"{name}.wav", audio)
    return name


def make_chromatic_scale_guitar() -> None:
    """E2 chromatic scale up 12 frets (standard guitar low E string)."""
    # E2 = MIDI 40, up 12 semitones to E3
    notes = list(range(40, 53))
    make_scale(notes)
    print(f"  Guitar chromatic: MIDI {notes[0]}–{notes[-1]}")


def make_chromatic_scale_bass() -> None:
    """E1 chromatic scale up 12 frets (standard bass low E string)."""
    # E1 = MIDI 28
    notes = list(range(28, 41))
    make_scale(notes)
    print(f"  Bass chromatic: MIDI {notes[0]}–{notes[-1]}")


def make_simple_melody_guitar() -> None:
    """E-G-A-B-D melody at quarter notes, 120 BPM."""
    # Common pentatonic notes in guitar range
    notes = [40, 43, 45, 47, 50]  # E2 G2 A2 B2 D3
    audio_parts: list[np.ndarray] = []
    for midi in notes:
        audio_parts.append(sine_tone(midi_to_hz(midi), 0.5))
        audio_parts.append(silence(0.02))
    audio = np.concatenate(audio_parts)
    write_wav(FIXTURES_DIR / "melody_guitar_pentatonic.wav", audio)
    print(f"  Guitar pentatonic melody: {notes}")


def make_silence_fixture() -> None:
    """Pure silence — should produce zero NoteEvents."""
    write_wav(FIXTURES_DIR / "silence.wav", silence(2.0))


def make_noise_fixture() -> None:
    """White noise — noise gate should block; few/no notes expected."""
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(SAMPLE_RATE * 2) * 0.005).astype(np.float32)
    write_wav(FIXTURES_DIR / "noise_below_gate.wav", noise)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating fixtures → {FIXTURES_DIR}")

    # Single notes — used for unit-level pitch accuracy tests
    for midi in [40, 45, 52, 64, 69]:   # E2 A2 E3 E4 A4
        make_single_note(midi, duration_s=1.5)

    # Bass notes
    for midi in [28, 33, 35, 40]:   # E1 A1 B1 E2
        make_single_note(midi, duration_s=1.5)

    # Sequences
    make_chromatic_scale_guitar()
    make_chromatic_scale_bass()
    make_simple_melody_guitar()

    # Edge cases
    make_silence_fixture()
    make_noise_fixture()

    print(f"\nAll fixtures written to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
