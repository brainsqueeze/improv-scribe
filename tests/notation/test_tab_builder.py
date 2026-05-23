"""
tests/notation/test_tab_builder.py

Unit tests for tab_builder — fret assignment via dynamic programming.
Includes both singleton (mono) tests and chord-aware tests (Phase 2).
"""

from __future__ import annotations

from improv_scribe.analysis.instrument_profiles import Instrument
from improv_scribe.notation.tab_builder import (
    BASS_TUNING,
    GUITAR_TUNING,
    MAX_FRET,
    assign_frets,
    get_candidates,
    get_chord_shapes,
)
from improv_scribe.quantization.grid import NoteDuration, QuantizedNote

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qn(midi_notes: tuple[int, ...], is_rest: bool = False) -> QuantizedNote:
    n = len(midi_notes) if not is_rest else 0
    return QuantizedNote(
        midi_notes=() if is_rest else midi_notes,
        frequencies_hz=() if is_rest else (440.0,) * n,
        confidences=() if is_rest else (0.8,) * n,
        cents_deviations=() if is_rest else (0.0,) * n,
        onset_beat=0.0,
        duration_beats=1.0,
        duration_type=NoteDuration.QUARTER,
        quarter_length=1.0,
        is_rest=is_rest,
    )


def _make_note(midi: int, quarter_length: float = 1.0) -> QuantizedNote:
    """Mono note helper — back-compat with pre-Phase-2 tests."""
    return _qn((midi,))


def _make_rest(quarter_length: float = 1.0) -> QuantizedNote:
    """Rest helper — back-compat with pre-Phase-2 tests."""
    return _qn((), is_rest=True)


# ---------------------------------------------------------------------------
# get_candidates
# ---------------------------------------------------------------------------

class TestGetCandidates:
    def test_note_playable_on_multiple_strings(self):
        # MIDI 45 = A2; on guitar: string 0 (E2, open=40) fret 5, string 1 (A2, open=45) fret 0
        candidates = get_candidates(45, GUITAR_TUNING)
        assert (0, 5) in candidates
        assert (1, 0) in candidates

    def test_lowest_open_string_guitar(self):
        # MIDI 40 = E2 = open string 0 on guitar, fret 0 only
        candidates = get_candidates(40, GUITAR_TUNING)
        assert candidates == [(0, 0)]

    def test_lowest_open_string_bass(self):
        # MIDI 28 = E1 = open string 0 on bass, fret 0 only
        candidates = get_candidates(28, BASS_TUNING)
        assert candidates == [(0, 0)]

    def test_note_outside_range_returns_empty(self):
        # MIDI 0 is way below any guitar string open tuning
        candidates = get_candidates(0, GUITAR_TUNING)
        assert candidates == []

    def test_note_above_max_fret_excluded(self):
        # A note requiring fret > 22 on every string should yield empty
        # Highest guitar string is E4 (MIDI 64); fret 23 = MIDI 87
        candidates = get_candidates(87, GUITAR_TUNING)
        # String 5 (E4=64): fret=23 → excluded; lower strings also exceed 22 or are < 0
        for _string_idx, fret in candidates:
            assert 0 <= fret <= MAX_FRET

    def test_max_fret_boundary_included(self):
        # String 0 (E2=40) fret 22 → MIDI 62; should be a valid candidate
        candidates = get_candidates(62, GUITAR_TUNING)
        assert (0, 22) in candidates

    def test_fret_zero_boundary_included(self):
        # Open strings are fret 0 — already tested above, confirm the boundary value
        candidates = get_candidates(43, BASS_TUNING)
        # G2 = MIDI 43; bass string 3 (G2=43) → fret 0
        assert (3, 0) in candidates

    def test_bass_note_candidates(self):
        # MIDI 33 = A1; bass string 1 (A1=33) fret 0, string 0 (E1=28) fret 5
        candidates = get_candidates(33, BASS_TUNING)
        assert (1, 0) in candidates
        assert (0, 5) in candidates

    def test_candidates_ordered_low_string_first(self):
        # get_candidates iterates tuning in order (low string = index 0 first)
        candidates = get_candidates(50, GUITAR_TUNING)
        string_indices = [s for s, _f in candidates]
        assert string_indices == sorted(string_indices)


# ---------------------------------------------------------------------------
# assign_frets — singleton (mono) tests
# ---------------------------------------------------------------------------

class TestAssignFrets:
    def test_rest_returns_none(self):
        notes = [_make_rest()]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result == [None]

    def test_all_rests_returns_all_none(self):
        notes = [_make_rest(), _make_rest(), _make_rest()]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result == [None, None, None]

    def test_rest_in_sequence_gets_none(self):
        # Rest embedded between notes
        notes = [_make_note(40), _make_rest(), _make_note(45)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result[1] is None
        assert result[0] is not None
        assert result[2] is not None

    def test_single_note_returns_valid_assignment(self):
        notes = [_make_note(40)]  # E2 — open string 0 on guitar
        result = assign_frets(notes, Instrument.GUITAR)
        assert len(result) == 1
        assert result[0] == ((0, 0),)

    def test_prefers_lower_fret_for_first_note(self):
        # MIDI 45 = A2 on guitar: candidates are (0, 5) and (1, 0).
        # All first-note candidates start with cost 0; tie-break picks fret 0.
        notes = [_make_note(45)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result[0] is not None
        shape = result[0]
        assert len(shape) == 1
        _string_idx, fret = shape[0]
        assert fret == 0, (
            f"Expected fret 0 (lowest fret), got {fret}. "
            "Tie-break should prefer lower fret regardless of candidate order."
        )

        # MIDI 50 = D3 on guitar: (0,10),(1,5),(2,0). Lowest fret (0 on string 2) must win.
        notes_b = [_make_note(50)]
        result_b = assign_frets(notes_b, Instrument.GUITAR)
        assert result_b[0] is not None
        _string_idx_b, fret_b = result_b[0][0]
        assert fret_b == 0, (
            f"Expected fret 0 (lowest fret) for MIDI 50, got {fret_b}."
        )

    def test_fallback_for_note_outside_range(self):
        # MIDI 0 has no candidates on guitar → fallback ((0, 0),)
        notes = [_make_note(0)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result[0] == ((0, 0),)

    def test_ascending_scale_stays_in_position(self):
        # E minor pentatonic starting at E2 (MIDI 40): 40,43,45,47,50,52
        midi_notes = [40, 43, 45, 47, 50, 52]
        notes = [_make_note(m) for m in midi_notes]
        result = assign_frets(notes, Instrument.GUITAR)

        assert len(result) == len(notes)
        assert all(r is not None for r in result)

        # All frets should stay low (position 0–5 is natural for this phrase)
        frets = [shape[0][1] for shape in result]
        assert max(frets) <= 7  # loose bound — must not jump to high positions

    def test_minimize_position_shift(self):
        # MIDI 40 on guitar: only (0, 0). MIDI 45: (0, 5) and (1, 0).
        # After (0,0) for MIDI 40, (1,0) costs |0-0|=0; (0,5) costs |5-0|=5 → (1,0) wins.
        notes = [_make_note(40), _make_note(45)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result[0] == ((0, 0),)
        assert result[1] == ((1, 0),)

    def test_output_length_matches_input(self):
        notes = [_make_note(40), _make_rest(), _make_note(45), _make_note(50)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert len(result) == len(notes)

    def test_bass_instrument(self):
        # E1 (MIDI 28) on bass: open string 0, fret 0
        notes = [_make_note(28)]
        result = assign_frets(notes, Instrument.BASS)
        assert result[0] == ((0, 0),)

    def test_empty_notes_returns_empty(self):
        result = assign_frets([], Instrument.GUITAR)
        assert result == []

    def test_all_assignments_within_fret_bounds(self):
        midi_notes = [40, 45, 47, 50, 52, 55, 57, 59]
        notes = [_make_note(m) for m in midi_notes]
        result = assign_frets(notes, Instrument.GUITAR)
        for r in result:
            if r is not None:
                for _s, fret in r:
                    assert 0 <= fret <= MAX_FRET

    # -----------------------------------------------------------------------
    # Issue 1 regression — DP initialisation must not include fret cost
    # -----------------------------------------------------------------------

    def test_zero_shift_path_beats_low_fret_start(self):
        """
        Regression test for the dp[0] initialisation bug.

        The objective is sum(|fret[i] - fret[i-1]|); the fret value of the
        FIRST note must NOT contribute to the DP cost.

        Counter-example:
          Note 0: MIDI 50 — guitar candidates (0,10),(1,5),(2,0)
          Note 1: MIDI 60 — guitar candidates (0,20),(1,15),(2,10),(3,5),(4,1)

          Path A: note0=(0,10), note1=(2,10) — shift = |10-10| = 0
          Path B: note0=(2,0),  note1=(4,1)  — shift = |1-0|   = 1

        True minimum-shift path is A (shift 0).
        """
        notes = [_make_note(50), _make_note(60)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result[0] is not None
        assert result[1] is not None
        fret0 = result[0][0][1]
        fret1 = result[1][0][1]
        assert abs(fret1 - fret0) == 0, (
            f"Expected zero-shift path, got fret0={fret0}, fret1={fret1}, "
            f"shift={abs(fret1-fret0)}. "
            f"Old (buggy) code would pick fret0=0,fret1=1 (shift=1)."
        )

    # -----------------------------------------------------------------------
    # Issue 2 regression — mid-sequence no-candidate note must not cascade
    # -----------------------------------------------------------------------

    def test_out_of_range_note_does_not_cascade_fallback(self):
        """
        A note with no candidates in the middle of a sequence must receive the
        ((0,0),) fallback while the notes before and after it are still assigned
        valid fret positions from the DP (not the fallback).
        """
        notes = [
            _make_note(40),   # E2 — valid: (0, 0)
            _make_note(0),    # MIDI 0 — out of range, fallback ((0, 0),)
            _make_note(45),   # A2 — valid: string 0 fret 5 or string 1 fret 0
        ]
        result = assign_frets(notes, Instrument.GUITAR)
        assert len(result) == 3

        # Middle note gets fallback
        assert result[1] == ((0, 0),)

        # First note is valid (only candidate is (0,0) for E2)
        assert result[0] == ((0, 0),)

        # Third note must be a VALID assignment for MIDI 45: (0,5) or (1,0)
        assert result[2] in {((0, 5),), ((1, 0),)}, (
            f"Expected valid fret assignment for note 2, got {result[2]}"
        )


# ---------------------------------------------------------------------------
# get_chord_shapes
# ---------------------------------------------------------------------------

class TestGetChordShapes:
    """get_chord_shapes() enumerates all (string, fret) assignments where
    chord members occupy distinct strings."""

    def test_singleton_returns_candidates_wrapped_in_tuples(self):
        # E2 (MIDI 40) is only playable on the lowest string of guitar (open)
        shapes = get_chord_shapes((40,), GUITAR_TUNING)
        assert len(shapes) == 1
        assert shapes[0] == ((0, 0),)

    def test_dyad_returns_distinct_string_pairs(self):
        # E2 (40) on string 0, B2 (47) playable on string 0 or string 1.
        # No-string-conflict: only (0,0) + (1,2) is valid (string 0 is taken).
        shapes = get_chord_shapes((40, 47), GUITAR_TUNING)
        # Verify all returned shapes use distinct strings
        for shape in shapes:
            strings = [s for s, _f in shape]
            assert len(set(strings)) == len(strings)

    def test_unplayable_chord_returns_empty(self):
        # Three identical MIDI E2s: only 1 string can play E2 -> no
        # conflict-free shape exists.
        shapes = get_chord_shapes((40, 40, 40), GUITAR_TUNING)
        assert shapes == []

    def test_out_of_range_member_returns_empty(self):
        # MIDI 5 is below guitar's range; member has no candidates.
        shapes = get_chord_shapes((5, 60), GUITAR_TUNING)
        assert shapes == []

    def test_shapes_are_canonical_string_sorted(self):
        shapes = get_chord_shapes((40, 47), GUITAR_TUNING)
        for shape in shapes:
            strings = [s for s, _f in shape]
            assert strings == sorted(strings)


# ---------------------------------------------------------------------------
# assign_frets — chord-aware tests (Phase 2)
# ---------------------------------------------------------------------------

class TestAssignFretsChordAware:
    """assign_frets() returns a tuple of (string, fret) pairs per QuantizedNote.
    Singletons get length-1 tuples; chords get length-N tuples with distinct strings."""

    def test_singleton_returns_length_1_tuple(self):
        result = assign_frets([_qn((40,))], Instrument.GUITAR)
        assert result == [((0, 0),)]

    def test_chord_returns_length_n_tuple_with_distinct_strings(self):
        result = assign_frets([_qn((40, 47, 52))], Instrument.GUITAR)
        assert len(result) == 1
        assert result[0] is not None
        shape = result[0]
        assert len(shape) == 3
        # Distinct strings
        strings = [s for s, _f in shape]
        assert len(set(strings)) == 3

    def test_rest_returns_none(self):
        result = assign_frets([_qn((), is_rest=True)], Instrument.GUITAR)
        assert result == [None]

    def test_mono_path_equivalent_to_pre_phase2(self):
        """A sequence of mono notes produces the same result as Phase 0 / Phase 1.
        Specifically: open low E -> low A -> open D should map to
        ((0,0), (1,0), (2,0)) — open string fingerings, no movement."""
        result = assign_frets(
            [_qn((40,)), _qn((45,)), _qn((50,))],
            Instrument.GUITAR,
        )
        assert result == [((0, 0),), ((1, 0),), ((2, 0),)]

    def test_chord_followed_by_mono_uses_consistent_dp(self):
        """A chord followed by a single note shouldn't crash; both are assigned."""
        result = assign_frets(
            [_qn((40, 47)), _qn((52,))],
            Instrument.GUITAR,
        )
        assert len(result) == 2
        # Both non-rest -> both non-None
        assert result[0] is not None
        assert result[1] is not None

    def test_unplayable_chord_falls_back_gracefully(self):
        """A chord with no conflict-free shape (e.g. three E2s) falls back
        to a shape rather than crashing. The exact fallback can be a
        singleton ((0, 0),) or a playable subset, but it must not crash."""
        result = assign_frets([_qn((40, 40, 40))], Instrument.GUITAR)
        assert result[0] is not None
        # No crash is the key assertion.

    def test_bass_chord(self):
        """E1+A1 dyad on bass: E1 (28) on string 0 open, A1 (33) on string 1 open."""
        result = assign_frets([_qn((28, 33))], Instrument.BASS)
        assert result[0] is not None
        shape = result[0]
        assert len(shape) == 2
        strings = sorted(s for s, _f in shape)
        assert strings == [0, 1]
        # Both should be fret 0 (open strings)
        for _s, f in shape:
            assert f == 0
