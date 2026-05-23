"""
tests/integration/test_bass_line_in.py

End-to-end pipeline regression tests for:
    samples/bass/4_string_bass_line_in.mp3

The sample plays each open string of a calibrated 4-string bass (line-in)
from low to high: E1 A1 D2 G2. One note per string, four total.
"""

from __future__ import annotations

import os

import music21.clef
import music21.note
import numpy as np

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

_BACKEND = os.getenv("ATS_PITCH_BACKEND", "basic_pitch")

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

SAMPLE_PATH = SAMPLE_ROOT / "bass" / "4_string_bass_line_in.mp3"
INSTRUMENT = Instrument.BASS
EXPECTED_DURATION_S = 12.3

# basic-pitch produces 5 raw events: the four open strings plus one sympathetic
# E1 detection at 6.749s (amp 0.738). In Phase 2, onset clustering groups the
# G3 onset (6.668s, amp 0.774) with the sympathetic E1 (6.749s, 81ms gap) into
# one chord event [28, 43], reducing the total event count to 4.
NOTE_COUNT_BY_BACKEND: dict[str, int] = {
    "crepe":       4,
    "pyin":        4,
    "basic_pitch": 4,
}
NOTE_COUNT = NOTE_COUNT_BY_BACKEND[_BACKEND]

# Concert (sounding) MIDI, low string → high string: E1 A1 D2 G2
# basic-pitch Phase 2: sympathetic E1 (6.749s) clusters with G3 (6.668s) into
# chord [28, 43]. midi_note returns the lowest pitch (28 = E1).
EXPECTED_MIDI_BY_BACKEND: dict[str, list[int]] = {
    "crepe":       [28, 33, 38, 43],
    "pyin":        [28, 33, 38, 43],
    "basic_pitch": [28, 33, 38, 28],   # 4th event is chord [28,43]; midi_note → lowest = 28
}
EXPECTED_MIDI = EXPECTED_MIDI_BY_BACKEND[_BACKEND]

# Notes are written at concert pitch (bass8vb clef carries the octave offset).
# basic-pitch Phase 2 (Tasks 5+6): chord [28,43] is serialised as a
# music21.chord.Chord, so both pitches appear in the score.
EXPECTED_WRITTEN_MIDI_BY_BACKEND: dict[str, list[int]] = {
    "crepe":       [28, 33, 38, 43],
    "pyin":        [28, 33, 38, 43],
    "basic_pitch": [28, 28, 33, 38, 43],  # chord [28,43] emits both pitches; 28 appears twice
}
EXPECTED_WRITTEN_MIDI = EXPECTED_WRITTEN_MIDI_BY_BACKEND[_BACKEND]

# Tab: every open string → ((string_idx, fret=0),), 0-based from lowest string.
# Phase 2: each assignment is a tuple of (string, fret) pairs; mono → length-1 tuple.
# basic-pitch Phase 2: last event is chord [28,43]; chord-aware DP assigns E1→string 0
# and G2→string 3, both open (fret 0).
EXPECTED_TAB_BY_BACKEND: dict[str, list[tuple[tuple[int, int], ...]]] = {
    "crepe":       [((0, 0),), ((1, 0),), ((2, 0),), ((3, 0),)],
    "pyin":        [((0, 0),), ((1, 0),), ((2, 0),), ((3, 0),)],
    "basic_pitch": [((0, 0),), ((1, 0),), ((2, 0),), ((0, 0), (3, 0))],  # chord [28,43] → strings 0+3
}
EXPECTED_TAB = EXPECTED_TAB_BY_BACKEND[_BACKEND]

# Clef: "bass8vb" → sign='F', octaveChange=-1
EXPECTED_CLEF_SIGN = "F"

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

    def test_audio_sample_rate(self, audio):
        _, sr = audio
        assert sr == 44100

    def test_audio_duration(self, audio):
        y, sr = audio
        duration_s = len(y) / sr
        assert abs(duration_s - EXPECTED_DURATION_S) <= 2.0, (
            f"Expected ~{EXPECTED_DURATION_S}s, got {duration_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Stage: pitch_result
# ---------------------------------------------------------------------------

class TestPitchResult:
    def test_pitch_result_has_data(self, pitch_result):
        # CREPE/pyin populate frames; basic-pitch populates bp_notes.
        assert pitch_result.frames or pitch_result.bp_notes

    def test_pitch_frequency_range(self, pitch_result):
        if not pitch_result.frames:
            import pytest
            pytest.skip("basic-pitch does not produce frame-level data")
        profile = get_profile(INSTRUMENT)
        for frame in pitch_result.voiced_frames:
            assert profile.freq_min_hz <= frame.freq_hz <= profile.freq_max_hz, (
                f"Frame freq {frame.freq_hz:.1f} Hz outside [{profile.freq_min_hz}, {profile.freq_max_hz}]"
            )


# ---------------------------------------------------------------------------
# Stage: onsets
# ---------------------------------------------------------------------------

ONSET_COUNT_MIN = 4
ONSET_COUNT_MAX = 6


class TestOnsets:
    def test_onset_count(self, onsets):
        assert ONSET_COUNT_MIN <= len(onsets) <= ONSET_COUNT_MAX, (
            f"Expected {ONSET_COUNT_MIN}–{ONSET_COUNT_MAX} onsets, got {len(onsets)}"
        )

    def test_onsets_sorted(self, onsets):
        times = [o.time_s for o in onsets]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# Stage: note_events
# ---------------------------------------------------------------------------

class TestNoteEvents:
    def test_note_count(self, note_events):
        # This is the primary "all notes detected" gate.
        # If this fails: print note_events to inspect what the pipeline detected.
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} NoteEvents, got {len(note_events)}: "
            f"{[e.midi_notes[0] for e in note_events]}"
        )

    def test_note_pitches(self, note_events):
        # midi_note is already rounded to int; ±0.5 is effectively exact match
        # for calibrated open-string recordings.
        for event, expected in zip(note_events, EXPECTED_MIDI, strict=True):
            assert abs(event.midi_notes[0] - expected) <= 0.5, (
                f"Expected MIDI {expected}, got {event.midi_notes[0]} "
                f"({event.frequencies_hz[0]:.1f} Hz)"
            )

    def test_notes_in_instrument_range(self, note_events):
        profile = get_profile(INSTRUMENT)
        for event in note_events:
            assert profile.midi_min <= event.midi_notes[0] <= profile.midi_max, (
                f"MIDI {event.midi_notes[0]} outside instrument range "
                f"[{profile.midi_min}, {profile.midi_max}]"
            )


# ---------------------------------------------------------------------------
# Stage: tempo_result
# ---------------------------------------------------------------------------

class TestTempoResult:
    def test_tempo_positive(self, tempo_result):
        assert 40.0 <= tempo_result.bpm <= 250.0, (
            f"BPM {tempo_result.bpm} outside [40, 250]"
        )


# ---------------------------------------------------------------------------
# Stage: quantized_notes
# ---------------------------------------------------------------------------

class TestQuantizedNotes:
    def test_quantized_note_count(self, quantized_notes):
        non_rests = [n for n in quantized_notes if not n.is_rest]
        assert len(non_rests) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} non-rest QuantizedNotes, "
            f"got {len(non_rests)} (total with rests: {len(quantized_notes)})"
        )

    def test_quantized_pitches_unchanged(self, quantized_notes, note_events):
        # Quantizer must not alter pitch — only timing.
        quantized_midis = [n.midi_notes[0] for n in quantized_notes if not n.is_rest]
        event_midis = [e.midi_notes[0] for e in note_events]
        assert quantized_midis == event_midis


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_clef(self, score):
        part = score.parts[0]
        clefs = list(part.recurse().getElementsByClass(music21.clef.Clef))
        assert len(clefs) >= 1, "No clef found in score part"
        clef_obj = clefs[0]
        assert clef_obj.sign == EXPECTED_CLEF_SIGN, (
            f"Expected clef sign '{EXPECTED_CLEF_SIGN}', got '{clef_obj.sign}'"
        )
        assert clef_obj.octaveChange == -1, (
            f"Expected octaveChange -1 (8vb), got {clef_obj.octaveChange}"
        )

    def test_score_written_pitches(self, score):
        import music21.chord
        part = score.parts[0]
        written_midis: list[int] = []
        for el in part.recurse().notes:
            if isinstance(el, music21.chord.Chord):
                written_midis.extend(p.midi for p in el.pitches)
            else:
                written_midis.append(el.pitch.midi)
        written_midis = sorted(written_midis)
        assert written_midis == sorted(EXPECTED_WRITTEN_MIDI), (
            f"Written MIDIs {written_midis} != expected {sorted(EXPECTED_WRITTEN_MIDI)}"
        )


# ---------------------------------------------------------------------------
# Stage: tab_assignments
# ---------------------------------------------------------------------------

class TestTabAssignments:
    def test_tab_length(self, tab_assignments, quantized_notes):
        assert len(tab_assignments) == len(quantized_notes)

    def test_tab_all_fret_zero(self, tab_assignments):
        for assignment in tab_assignments:
            if assignment is not None:
                for _s, fret in assignment:
                    assert fret == 0, (
                        f"Open string expected fret 0, got {fret}"
                    )

    def test_tab_exact_string_assignments(self, tab_assignments):
        # Compare in onset order (low string played first = ascending MIDI order)
        non_none = [a for a in tab_assignments if a is not None]
        assert non_none == EXPECTED_TAB, (
            f"Tab assignments {non_none} != expected {EXPECTED_TAB}"
        )
