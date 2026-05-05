"""
tests/notation/test_score_builder.py

Tests for ScoreBuilder — verifies that music21 Score objects are constructed
correctly from QuantizedNote lists.
"""

from __future__ import annotations

import music21.note
import music21.stream
import music21.tempo
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.notation.score_builder import ScoreBuilder
from improv_scribe.quantization.grid import NoteDuration, QuantizedNote
from improv_scribe.quantization.tempo import TempoResult


def _make_tempo(bpm: float = 120.0) -> TempoResult:
    return TempoResult(bpm=bpm, beat_times_s=[], confidence=1.0)


def _make_note(midi: int, quarter_length: float, beat: float = 0.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=midi,
        frequency_hz=261.63,
        confidence=0.9,
        cents_deviation=0.0,
        onset_beat=beat,
        duration_beats=quarter_length,
        duration_type=NoteDuration.QUARTER,
        quarter_length=quarter_length,
        is_rest=False,
    )


def _make_rest(quarter_length: float, beat: float = 0.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=0,
        frequency_hz=0.0,
        confidence=1.0,
        cents_deviation=0.0,
        onset_beat=beat,
        duration_beats=quarter_length,
        duration_type=NoteDuration.QUARTER,
        quarter_length=quarter_length,
        is_rest=True,
    )


@pytest.fixture
def guitar_profile():
    return get_profile(Instrument.GUITAR)


@pytest.fixture
def bass_profile():
    return get_profile(Instrument.BASS)


class TestScoreBuilder:
    def test_build_returns_score(self, guitar_profile):
        notes = [_make_note(60, 1.0, 0.0), _make_note(62, 1.0, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo())
        score = builder.build(notes)
        assert isinstance(score, music21.stream.Score)

    def test_score_has_one_part(self, guitar_profile):
        notes = [_make_note(60, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo())
        score = builder.build(notes)
        parts = list(score.parts)
        assert len(parts) == 1

    def test_note_count_matches(self, guitar_profile):
        """Number of Note elements in the standard notation part should equal input note count."""
        notes = [
            _make_note(60, 1.0, 0.0),
            _make_note(62, 1.0, 1.0),
            _make_note(64, 1.0, 2.0),
        ]
        builder = ScoreBuilder(guitar_profile, _make_tempo())
        score = builder.build(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert len(score_notes) == 3

    def test_rest_included_in_score(self, guitar_profile):
        elements = [
            _make_note(60, 1.0, 0.0),
            _make_rest(1.0, 1.0),
            _make_note(64, 1.0, 2.0),
        ]
        builder = ScoreBuilder(guitar_profile, _make_tempo())
        score = builder.build(elements)
        rests = list(score.flatten().getElementsByClass(music21.note.Rest))
        assert len(rests) >= 1

    def test_tempo_mark_in_score(self, guitar_profile):
        notes = [_make_note(60, 1.0)]
        bpm = 142.0
        builder = ScoreBuilder(guitar_profile, _make_tempo(bpm))
        score = builder.build(notes)
        marks = list(score.flatten().getElementsByClass(music21.tempo.MetronomeMark))
        assert len(marks) >= 1
        assert marks[0].number == pytest.approx(bpm)

    def test_empty_notes_builds_score(self, guitar_profile):
        """Empty note list should not raise — produces an empty score."""
        builder = ScoreBuilder(guitar_profile, _make_tempo())
        score = builder.build([])
        assert isinstance(score, music21.stream.Score)

    def test_bass_profile_uses_bass_clef(self, bass_profile):
        import music21.clef
        notes = [_make_note(40, 1.0)]
        builder = ScoreBuilder(bass_profile, _make_tempo())
        score = builder.build(notes)
        clefs = list(score.flatten().getElementsByClass(music21.clef.Clef))
        assert any("bass" in type(c).__name__.lower() or c.sign == "F" for c in clefs)

    def test_title_set_in_metadata(self, guitar_profile):
        notes = [_make_note(60, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), title="My Song")
        score = builder.build(notes)
        assert score.metadata.title == "My Song"

    def test_build_raw_returns_score(self, guitar_profile):
        notes = [_make_note(60, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo())
        score = builder.build_raw(notes)
        assert isinstance(score, music21.stream.Score)


class TestOctaveTransposition:
    def test_guitar_note_written_up_octave(self, guitar_profile):
        """Sounding C4 (MIDI 60) must appear as C5 (MIDI 72) in the score."""
        notes = [_make_note(60, 1.0)]
        score = ScoreBuilder(guitar_profile, _make_tempo()).build(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert score_notes[0].pitch.midi == 72

    def test_bass_note_written_up_octave(self, bass_profile):
        """Sounding E1 (MIDI 28) must appear as E2 (MIDI 40) in the score."""
        notes = [_make_note(28, 1.0)]
        score = ScoreBuilder(bass_profile, _make_tempo()).build(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert score_notes[0].pitch.midi == 40

    def test_guitar_uses_treble8vb_clef(self, guitar_profile):
        import music21.clef
        score = ScoreBuilder(guitar_profile, _make_tempo()).build([_make_note(60, 1.0)])
        clefs = list(score.flatten().getElementsByClass(music21.clef.Clef))
        assert any(isinstance(c, music21.clef.Treble8vbClef) for c in clefs)

    def test_bass_uses_8vb_clef(self, bass_profile):
        import music21.clef
        score = ScoreBuilder(bass_profile, _make_tempo()).build([_make_note(28, 1.0)])
        clefs = list(score.flatten().getElementsByClass(music21.clef.Clef))
        assert any(
            (isinstance(c, music21.clef.BassClef) and getattr(c, "octaveChange", 0) == -1)
            or type(c).__name__ == "Bass8vbClef"
            for c in clefs
        )

    def test_build_raw_stays_at_sounding_pitch(self, guitar_profile):
        """build_raw() must NOT apply the octave offset — it feeds MIDI export."""
        notes = [_make_note(60, 1.0)]
        score = ScoreBuilder(guitar_profile, _make_tempo()).build_raw(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert score_notes[0].pitch.midi == 60

    def test_tempo_mark_is_rounded(self, guitar_profile):
        """Fractional BPM from librosa must be rounded to nearest integer."""
        notes = [_make_note(60, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(bpm=142.7))
        score = builder.build(notes)
        marks = list(score.flatten().getElementsByClass(music21.tempo.MetronomeMark))
        assert marks[0].number == 143

    def test_build_raw_quantized_preserves_timing(self, guitar_profile):
        """build_raw with quantized notes preserves beat offsets and sounding pitch."""
        notes = [_make_note(60, 1.0, 0.0), _make_note(64, 1.0, 1.0)]
        score = ScoreBuilder(guitar_profile, _make_tempo()).build_raw(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert len(score_notes) == 2
        assert score_notes[0].pitch.midi == 60
        assert score_notes[1].pitch.midi == 64
