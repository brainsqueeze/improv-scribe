"""
tests/notation/test_tab_builder.py

Unit tests for tab_builder — fret assignment via dynamic programming.
"""

from __future__ import annotations

from audio_to_sheet.analysis.instrument_profiles import Instrument
from audio_to_sheet.notation.tab_builder import (
    BASS_TUNING,
    GUITAR_TUNING,
    MAX_FRET,
    assign_frets,
    get_candidates,
)
from audio_to_sheet.quantization.grid import NoteDuration, QuantizedNote

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_note(midi: int, quarter_length: float = 1.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=midi,
        frequency_hz=440.0,
        confidence=0.95,
        cents_deviation=0.0,
        onset_beat=0.0,
        duration_beats=quarter_length,
        duration_type=NoteDuration.QUARTER,
        quarter_length=quarter_length,
        is_rest=False,
    )


def _make_rest(quarter_length: float = 1.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=0,
        frequency_hz=0.0,
        confidence=1.0,
        cents_deviation=0.0,
        onset_beat=0.0,
        duration_beats=quarter_length,
        duration_type=NoteDuration.QUARTER,
        quarter_length=quarter_length,
        is_rest=True,
    )


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
# assign_frets
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
        assert result[0] == (0, 0)

    def test_prefers_lower_fret_for_first_note(self):
        # MIDI 45 = A2 on guitar: candidates are (0, 5) and (1, 0).
        # All first-note candidates start with cost 0; tie-break picks fret 0.
        # Sub-case A: candidates naturally ordered low-string-first → (0,5) then (1,0).
        notes = [_make_note(45)]
        result = assign_frets(notes, Instrument.GUITAR)
        _string_idx, fret = result[0]
        assert fret == 0, (
            f"Expected fret 0 (lowest fret), got {fret}. "
            "Tie-break should prefer lower fret regardless of candidate order."
        )

        # Sub-case B: verify the choice is independent of iteration order by
        # exercising a note whose only two candidates differ in fret but share
        # equal transition cost.  MIDI 50 = D3 on guitar: (0,10),(1,5),(2,0).
        # With a single note all costs are 0; the lowest fret (0 on string 2) must win.
        notes_b = [_make_note(50)]
        result_b = assign_frets(notes_b, Instrument.GUITAR)
        _string_idx_b, fret_b = result_b[0]
        assert fret_b == 0, (
            f"Expected fret 0 (lowest fret) for MIDI 50, got {fret_b}."
        )

    def test_fallback_for_note_outside_range(self):
        # MIDI 0 has no candidates on guitar → fallback (0, 0)
        notes = [_make_note(0)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result[0] == (0, 0)

    def test_ascending_scale_stays_in_position(self):
        # E minor pentatonic starting at E2 (MIDI 40): 40,43,45,47,50,52
        # DP should keep the hand in low position rather than jumping up strings
        midi_notes = [40, 43, 45, 47, 50, 52]
        notes = [_make_note(m) for m in midi_notes]
        result = assign_frets(notes, Instrument.GUITAR)

        assert len(result) == len(notes)
        assert all(r is not None for r in result)

        # All frets should stay low (position 0–5 is natural for this phrase)
        frets = [fret for _s, fret in result]
        assert max(frets) <= 7  # loose bound — must not jump to high positions

    def test_minimize_position_shift(self):
        # Two notes where one candidate is far away and one is close.
        # MIDI 40 on guitar: only (0, 0).
        # MIDI 45 on guitar: (0, 5) and (1, 0).
        # After assigning (0, 0) to MIDI 40, DP should prefer (0, 5) over (1, 0)
        # because |5-0| = 5 vs starting at (1, 0) which would be a different string
        # but the fret cost is |0-0| = 0 — actually (1, 0) is cheaper.
        # Verify the algorithm picks the lower total cost assignment.
        notes = [_make_note(40), _make_note(45)]
        result = assign_frets(notes, Instrument.GUITAR)
        # First note must be (0, 0) — only candidate
        assert result[0] == (0, 0)
        # Second note: (1, 0) costs |0-0|=0; (0, 5) costs |5-0|=5 → (1, 0) wins
        assert result[1] == (1, 0)

    def test_output_length_matches_input(self):
        notes = [_make_note(40), _make_rest(), _make_note(45), _make_note(50)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert len(result) == len(notes)

    def test_bass_instrument(self):
        # E1 (MIDI 28) on bass: open string 0, fret 0
        notes = [_make_note(28)]
        result = assign_frets(notes, Instrument.BASS)
        assert result[0] == (0, 0)

    def test_empty_notes_returns_empty(self):
        result = assign_frets([], Instrument.GUITAR)
        assert result == []

    def test_all_assignments_within_fret_bounds(self):
        midi_notes = [40, 45, 47, 50, 52, 55, 57, 59]
        notes = [_make_note(m) for m in midi_notes]
        result = assign_frets(notes, Instrument.GUITAR)
        for r in result:
            if r is not None:
                _s, fret = r
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

        OLD (buggy) code initialised dp[0] with fret number, so:
          old cost A = 10 + 0 = 10,  old cost B = 0 + 1 = 1  → old picks B  (WRONG)

        NEW (fixed) code initialises dp[0] = 0 for all candidates, so:
          new cost A = 0 + 0 = 0,   new cost B = 0 + 1 = 1  → new picks A  (CORRECT)

        We verify the fixed code produces a zero-shift assignment.
        """
        # Note 0: MIDI 50 — (0,10),(1,5),(2,0)
        # Note 1: MIDI 60 — (0,20),(1,15),(2,10),(3,5),(4,1)
        # Zero-shift path: (0,10)→(2,10).  Nonzero-shift paths all have shift≥1.
        notes = [_make_note(50), _make_note(60)]
        result = assign_frets(notes, Instrument.GUITAR)
        assert result[0] is not None
        assert result[1] is not None
        _s0, fret0 = result[0]
        _s1, fret1 = result[1]
        # The zero-shift path must win; any nonzero shift means the bug is present
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
        (0,0) fallback while the notes before and after it are still assigned
        valid fret positions from the DP (not the (0,0) fallback).

        Sequence: [valid_note, out_of_range_note, valid_note]
        MIDI 0 has no candidates on guitar.
        """
        notes = [
            _make_note(40),   # E2 — valid: (0, 0)
            _make_note(0),    # MIDI 0 — out of range, fallback (0, 0)
            _make_note(45),   # A2 — valid: string 0 fret 5 or string 1 fret 0
        ]
        result = assign_frets(notes, Instrument.GUITAR)
        assert len(result) == 3

        # Middle note gets fallback
        assert result[1] == (0, 0)

        # First note is valid (only candidate is (0,0) for E2)
        assert result[0] == (0, 0)

        # Third note must be a VALID assignment, not (0,0) fallback.
        # Valid candidates for MIDI 45: (0,5) and (1,0).
        assert result[2] in {(0, 5), (1, 0)}, (
            f"Expected valid fret assignment for note 2, got {result[2]}"
        )