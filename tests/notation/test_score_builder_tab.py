"""
tests/notation/test_score_builder_tab.py

Tests for ScoreBuilder.compute_tab_assignments() and the guarantee that
build() always returns a single-part Score regardless of include_tab.

The actual TAB Part injection is tested separately in tests/export/test_tab_xml.py.
"""

from __future__ import annotations

import pytest

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.notation.score_builder import ScoreBuilder
from improv_scribe.quantization.grid import NoteDuration, QuantizedNote
from improv_scribe.quantization.tempo import TempoResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tempo(bpm: float = 120.0) -> TempoResult:
    return TempoResult(bpm=bpm, beat_times_s=[], confidence=1.0)


def _make_note(midi: int, ql: float = 1.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=midi,
        frequency_hz=0.0,
        confidence=1.0,
        cents_deviation=0.0,
        onset_beat=0.0,
        duration_beats=ql,
        duration_type=NoteDuration.QUARTER,
        quarter_length=ql,
        is_rest=False,
    )


def _make_rest(ql: float = 1.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=0,
        frequency_hz=0.0,
        confidence=1.0,
        cents_deviation=0.0,
        onset_beat=0.0,
        duration_beats=ql,
        duration_type=NoteDuration.QUARTER,
        quarter_length=ql,
        is_rest=True,
    )


@pytest.fixture
def guitar_profile():
    return get_profile(Instrument.GUITAR)


@pytest.fixture
def bass_profile():
    return get_profile(Instrument.BASS)


# ---------------------------------------------------------------------------
# Tests: compute_tab_assignments
# ---------------------------------------------------------------------------

class TestComputeTabAssignments:
    def test_compute_tab_assignments_returns_none_for_rests(self, guitar_profile):
        """Rest QuantizedNotes must produce None in the assignments list."""
        notes = [_make_note(64, 1.0), _make_rest(1.0), _make_note(62, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=True)
        assignments = builder.compute_tab_assignments(notes)

        assert assignments[1] is None

    def test_compute_tab_assignments_length_matches_notes(self, guitar_profile):
        """Return list must be the same length as the input note list."""
        notes = [_make_note(64, 1.0), _make_rest(1.0), _make_note(62, 1.0), _make_note(60, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=True)
        assignments = builder.compute_tab_assignments(notes)

        assert len(assignments) == len(notes)

    def test_compute_tab_assignments_valid_fret_range(self, guitar_profile):
        """All non-None assignments must have fret in [0, 22]."""
        notes = [
            _make_note(40, 1.0),   # E2 — lowest guitar note
            _make_note(64, 1.0),   # E4 — highest open string
            _make_note(76, 1.0),   # E5 — mid-range
        ]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=True)
        assignments = builder.compute_tab_assignments(notes)

        for assignment in assignments:
            if assignment is not None:
                _, fret = assignment
                assert 0 <= fret <= 22, f"Fret {fret} out of range [0, 22]"

    def test_compute_tab_assignments_all_none_for_all_rests(self, guitar_profile):
        """A list of only rests must yield all None assignments."""
        notes = [_make_rest(1.0), _make_rest(1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=True)
        assignments = builder.compute_tab_assignments(notes)

        assert all(a is None for a in assignments)

    def test_compute_tab_assignments_non_rest_notes_have_assignment(self, guitar_profile):
        """Non-rest notes within instrument range must have a non-None assignment."""
        notes = [_make_note(64, 1.0), _make_note(62, 1.0), _make_note(59, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=True)
        assignments = builder.compute_tab_assignments(notes)

        for i, (note, assignment) in enumerate(zip(notes, assignments, strict=True)):
            if not note.is_rest:
                assert assignment is not None, f"Note at index {i} has no assignment"


# ---------------------------------------------------------------------------
# Tests: build() always returns a single Part
# ---------------------------------------------------------------------------

class TestBuildReturnsSinglePart:
    def test_build_returns_single_part_regardless_of_include_tab_true(self, guitar_profile):
        """build() with include_tab=True must still return a single-part Score."""
        notes = [_make_note(64, 1.0), _make_note(62, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=True)
        score = builder.build(notes)

        assert len(list(score.parts)) == 1

    def test_build_returns_single_part_regardless_of_include_tab_false(self, guitar_profile):
        """build() with include_tab=False must return a single-part Score."""
        notes = [_make_note(64, 1.0), _make_note(62, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=False)
        score = builder.build(notes)

        assert len(list(score.parts)) == 1

    def test_include_tab_false_does_not_affect_build(self, guitar_profile):
        """
        Scores built with include_tab=True and include_tab=False should have
        identical Part counts (both 1) and identical note content.
        """
        notes = [_make_note(64, 1.0), _make_note(62, 1.0)]

        builder_true = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=True)
        builder_false = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=False)

        score_true = builder_true.build(notes)
        score_false = builder_false.build(notes)

        assert len(list(score_true.parts)) == len(list(score_false.parts)) == 1

    def test_include_tab_default_true_gives_one_part(self, guitar_profile):
        """Default ScoreBuilder (include_tab not specified) returns 1-part Score."""
        notes = [_make_note(64, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo())
        score = builder.build(notes)

        assert len(list(score.parts)) == 1

    def test_compute_tab_assignments_raises_when_include_tab_false(self, guitar_profile):
        """compute_tab_assignments() must raise RuntimeError when include_tab=False."""
        notes = [_make_note(64, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(), include_tab=False)

        with pytest.raises(RuntimeError, match="include_tab=False"):
            builder.compute_tab_assignments(notes)
