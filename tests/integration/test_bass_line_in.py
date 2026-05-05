"""
tests/integration/test_bass_line_in.py

End-to-end pipeline regression tests for:
    samples/bass/4_string_bass_line_in.mp3

The sample plays each open string of a calibrated 4-string bass (line-in)
from low to high: E1 A1 D2 G2. One note per string, four total.
"""

from __future__ import annotations

import numpy as np
import music21.clef
import music21.note

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

SAMPLE_PATH = SAMPLE_ROOT / "bass" / "4_string_bass_line_in.mp3"
INSTRUMENT = Instrument.BASS
NOTE_COUNT = 4
EXPECTED_DURATION_S = 12.3

# Concert (sounding) MIDI, low string → high string: E1 A1 D2 G2
EXPECTED_MIDI = [28, 33, 38, 43]

# Written MIDI: bass transpose_semitones=-12 → written = midi_note + 12
# Produces: E2 A2 D3 G3
EXPECTED_WRITTEN_MIDI = [40, 45, 50, 55]

# Tab: every open string → (string_idx, fret=0), 0-based from lowest string
EXPECTED_TAB = [(0, 0), (1, 0), (2, 0), (3, 0)]

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
    def test_pitch_voiced_frames_nonempty(self, pitch_result):
        assert len(pitch_result.voiced_frames) > 0

    def test_pitch_frequency_range(self, pitch_result):
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
            f"{[e.midi_note for e in note_events]}"
        )

    def test_note_pitches(self, note_events):
        # midi_note is already rounded to int; ±0.5 is effectively exact match
        # for calibrated open-string recordings.
        for event, expected in zip(note_events, EXPECTED_MIDI):
            assert abs(event.midi_note - expected) <= 0.5, (
                f"Expected MIDI {expected}, got {event.midi_note} "
                f"({event.frequency_hz:.1f} Hz)"
            )

    def test_notes_in_instrument_range(self, note_events):
        profile = get_profile(INSTRUMENT)
        for event in note_events:
            assert profile.midi_min <= event.midi_note <= profile.midi_max, (
                f"MIDI {event.midi_note} outside instrument range "
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
        quantized_midis = [n.midi_note for n in quantized_notes if not n.is_rest]
        event_midis = [e.midi_note for e in note_events]
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
        part = score.parts[0]
        notes = list(part.recurse().getElementsByClass(music21.note.Note))
        written_midis = sorted([n.pitch.midi for n in notes])
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
                _, fret = assignment
                assert fret == 0, (
                    f"Open string expected fret 0, got {fret}"
                )

    def test_tab_exact_string_assignments(self, tab_assignments):
        # Compare in onset order (low string played first = ascending MIDI order)
        non_none = [a for a in tab_assignments if a is not None]
        assert non_none == EXPECTED_TAB, (
            f"Tab assignments {non_none} != expected {EXPECTED_TAB}"
        )
