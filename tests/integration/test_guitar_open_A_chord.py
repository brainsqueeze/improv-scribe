"""End-to-end pipeline regression test for:
    samples/guitar/chords/6_string_electric_open_A_chord.mp3

User strums an open A major chord (A2 E3 A3 C#4 E4)
six times. With cluster-aware amplitude filtering the
pipeline recovers 5-6 members per strum, including the
spectrally-verified low E2 the player catches
(see docs/precision_audit_basic_pitch.md).

Ground truth re-derived 2026-06-10 after the precision audit.
"""

from __future__ import annotations

import os

import music21.chord
import numpy as np
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "chords" / "6_string_electric_open_A_chord.mp3"
INSTRUMENT = Instrument.GUITAR
EXPECTED_DURATION_S = 12.30

_BACKEND = os.getenv("ATS_PITCH_BACKEND", "basic_pitch")

EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        # E2 (40) is spectrally verified as physically sounding in every
        # strum (the player catches the low E string); E4 falls below the
        # relative floor on 4 of 6 strums (raw amp 0.34-0.42).
        (40, 45, 52, 57, 61),
        (40, 45, 52, 57, 61),
        (40, 45, 52, 57, 61, 64),
        (40, 45, 52, 57, 61),
        (40, 45, 52, 57, 61, 64),
        (40, 61),         # final partial strum near end of file
    ],
    "crepe": [],
    "pyin":  [],
}
EXPECTED_MIDI_TUPLES = EXPECTED_MIDI_TUPLES_BY_BACKEND[_BACKEND]
NOTE_COUNT = len(EXPECTED_MIDI_TUPLES)


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


class TestScore:
    def test_score_chord_emission(self, score):
        """Each cluster with len(midi_notes) > 1 should emit a music21.chord.Chord."""
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND!r}")
        chord_count = sum(1 for t in EXPECTED_MIDI_TUPLES if len(t) > 1)
        if chord_count == 0:
            pytest.skip(f"Backend {_BACKEND!r} produces no chords on this sample")

        chords_in_score = list(score.recurse().getElementsByClass(music21.chord.Chord))
        assert len(chords_in_score) == chord_count, (
            f"Expected {chord_count} Chord objects, found {len(chords_in_score)}"
        )


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
