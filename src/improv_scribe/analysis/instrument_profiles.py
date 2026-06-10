"""
analysis/instrument_profiles.py — Per-instrument physical constraints.

These profiles are used to:
  1. Constrain pitch search range in pYIN / CREPE (avoids octave errors).
  2. Select the correct clef and 8va transposition in the score builder.
  3. Tune the noise gate threshold (bass fundamentals are louder at low freqs).

Reference
---------
Guitar standard tuning: E2–e4 open strings, up to ~D6 at 22nd fret.
Bass standard scale: E1–G2 open strings, practical top ~D4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Instrument(StrEnum):
    GUITAR = "guitar"
    BASS = "bass"


@dataclass(frozen=True)
class InstrumentProfile:
    name: str
    instrument: Instrument

    # Physical frequency bounds for pitch estimation
    freq_min_hz: float
    freq_max_hz: float

    # MIDI note bounds (inclusive) for score sanity-checking
    midi_min: int   # lowest open string
    midi_max: int   # highest reachable fret

    # music21 clef string — passed directly to music21.clef.clefFromString()
    clef: str

    # Guitar sounds an octave lower than written.
    # Set transpose_semitones=-12 to write in the traditional guitar octave.
    transpose_semitones: int

    # Maximum simultaneous notes emitted per chord cluster on the
    # basic-pitch path. Bass is monophonic in the MVP scope: its long-ringing
    # strings produce strong (amp ≥ 0.6) octave/unison re-detections during
    # later notes that would otherwise form false dyads.
    max_polyphony: int = 6

    # Noise gate RMS override (None → use global config default)
    noise_gate_rms_override: float | None = None


PROFILES: dict[Instrument, InstrumentProfile] = {
    Instrument.GUITAR: InstrumentProfile(
        name="Guitar (standard)",
        instrument=Instrument.GUITAR,
        freq_min_hz=73.42,    # D2 — 2 semitones below E2 for CREPE headroom
        freq_max_hz=1174.66,  # D6
        midi_min=40,          # E2 (sounding)
        midi_max=86,          # D6 (sounding) — matches freq_max_hz and the
                              # 22nd fret on the high E string (64 + 22)
        clef="treble8vb",     # 8vb treble: written an octave above sounding pitch
        transpose_semitones=-12,
        max_polyphony=6,
    ),
    Instrument.BASS: InstrumentProfile(
        name="Bass Guitar (standard scale)",
        instrument=Instrument.BASS,
        freq_min_hz=38.89,    # D1 — 2 semitones below E1 for CREPE headroom
        freq_max_hz=293.66,   # D4 (sounding)
        midi_min=28,          # E1 (sounding)
        midi_max=62,          # D4 (sounding)
        clef="bass8vb",       # 8vb bass: written an octave above sounding pitch
        transpose_semitones=-12,
        max_polyphony=1,      # monophonic MVP scope (see field docstring)
        noise_gate_rms_override=0.015,
    ),
}


def get_profile(instrument: Instrument | str) -> InstrumentProfile:
    """Return the profile for *instrument*, accepting enum or string."""
    if isinstance(instrument, str):
        instrument = Instrument(instrument.lower())
    return PROFILES[instrument]
