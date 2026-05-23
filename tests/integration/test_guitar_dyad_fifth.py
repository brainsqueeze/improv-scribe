"""End-to-end pipeline regression tests for:
    samples/guitar/chords/6_string_electric_perfect_fifths.mp3

Six perfect-fifth intervals — basic-pitch detects 3 as dyads, 3 as singletons
(one member of the fifth registers below the 0.65 amplitude floor). See spec
§13.5 for the exact captured ground truth (validated against the live Phase 2
pipeline in Task 14).
"""

from __future__ import annotations

import os

import music21.chord
import music21.note
import numpy as np
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "chords" / "6_string_electric_perfect_fifths.mp3"
INSTRUMENT = Instrument.GUITAR
EXPECTED_DURATION_S = 12.30

_BACKEND = os.getenv("ATS_PITCH_BACKEND", "crepe")

# Per-backend: only basic_pitch can detect chord events; CREPE/pyin are
# monophonic and will produce singletons for the same sample (best-effort
# mono interpretation). Phase 2 ships with basic_pitch as the new default
# in Task 15; for now, basic_pitch ground truth is what matters.
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (47,),       # B2 only
        (41, 48),    # F2+C3
        (50,),       # D3 only
        (45, 52),    # A2+E3
        (47,),       # B2 only
        (48, 55),    # C3+G3
    ],
    # CREPE/pyin are monophonic — they produce singleton events. Skip-listed
    # (calibration not blocking Phase 2; can fill in post-merge).
    "crepe": [],
    "pyin":  [],
}
EXPECTED_MIDI_TUPLES = EXPECTED_MIDI_TUPLES_BY_BACKEND[_BACKEND]
NOTE_COUNT = len(EXPECTED_MIDI_TUPLES)


# ---------------------------------------------------------------------------
# Fixture chain — runs the full pipeline once for this module
# ---------------------------------------------------------------------------

(
    audio,
    pitch_result,
    onsets,
    note_events,
    tempo_result,
    quantized_notes,
    score,
    tab_assignments,
) = make_pipeline_fixtures(SAMPLE_PATH, INSTRUMENT)


# ---------------------------------------------------------------------------
# Stage: audio
# ---------------------------------------------------------------------------

class TestAudio:
    def test_audio_shape(self, audio):
        y, _ = audio
        assert y.ndim == 1
        assert len(y) > 0
        assert y.dtype == np.float32

    def test_audio_duration(self, audio):
        y, sr = audio
        duration_s = len(y) / sr
        assert abs(duration_s - EXPECTED_DURATION_S) <= 1.0


# ---------------------------------------------------------------------------
# Stage: note_events
# ---------------------------------------------------------------------------

class TestNoteEvents:
    def test_note_count(self, note_events):
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND!r}")
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} events, got {len(note_events)}: "
            f"{[tuple(e.midi_notes) for e in note_events]}"
        )

    def test_note_midi_tuples_match(self, note_events):
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND!r}")
        actual = [tuple(e.midi_notes) for e in note_events]
        assert actual == EXPECTED_MIDI_TUPLES


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_chord_emission(self, score):
        """For each chord event (len(midi_notes) > 1) in the ground truth,
        the score should have a music21.chord.Chord at that position."""
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND!r}")
        chord_count = sum(1 for t in EXPECTED_MIDI_TUPLES if len(t) > 1)
        if chord_count == 0:
            pytest.skip(f"Backend {_BACKEND!r} produces no chords on this sample")

        chords_in_score = list(score.recurse().getElementsByClass(music21.chord.Chord))
        assert len(chords_in_score) == chord_count, (
            f"Expected {chord_count} Chord objects, found {len(chords_in_score)}"
        )


# ---------------------------------------------------------------------------
# Stage: tab_assignments
# ---------------------------------------------------------------------------

class TestTabAssignments:
    def test_chord_tab_uses_distinct_strings(self, tab_assignments):
        """Every chord-shape assignment must use distinct strings."""
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND!r}")
        for assignment in tab_assignments:
            if assignment is None or len(assignment) <= 1:
                continue
            strings = [s for s, _f in assignment]
            assert len(set(strings)) == len(strings), (
                f"Chord assignment {assignment} reuses a string"
            )
