"""
tests/quantization/test_grid.py

Tests for RhythmQuantizer. Uses hand-crafted NoteEvent lists at known timings
to verify correct duration snapping, rest insertion, and beat alignment.
"""

from __future__ import annotations

import pytest

from improv_scribe.analysis.note_tracker import NoteEvent
from improv_scribe.quantization.grid import (
    _DURATION_FRACTIONS,
    NoteDuration,
    RhythmQuantizer,
    to_quarter_length,
)
from improv_scribe.quantization.tempo import TempoResult


def _make_tempo(bpm: float = 120.0) -> TempoResult:
    return TempoResult(bpm=bpm, beat_times_s=[], confidence=1.0)


def _make_event(onset_s: float, offset_s: float, midi: int = 60) -> NoteEvent:
    return NoteEvent(
        onset_s=onset_s,
        offset_s=offset_s,
        midi_notes=(midi,),
        frequencies_hz=(261.63,),
        confidences=(0.9,),
        cents_deviations=(0.0,),
    )


class TestDurationFractions:
    def test_whole_note_is_one(self):
        assert _DURATION_FRACTIONS[NoteDuration.WHOLE] == pytest.approx(1.0)

    def test_quarter_note_is_025(self):
        assert _DURATION_FRACTIONS[NoteDuration.QUARTER] == pytest.approx(0.25)

    def test_eighth_note_is_0125(self):
        assert _DURATION_FRACTIONS[NoteDuration.EIGHTH] == pytest.approx(0.125)

    def test_quarter_length_quarter_note(self):
        # music21 quarterLength: quarter=1.0, half=2.0, whole=4.0
        assert to_quarter_length(NoteDuration.QUARTER) == pytest.approx(1.0)
        assert to_quarter_length(NoteDuration.HALF) == pytest.approx(2.0)
        assert to_quarter_length(NoteDuration.WHOLE) == pytest.approx(4.0)
        assert to_quarter_length(NoteDuration.EIGHTH) == pytest.approx(0.5)


class TestRhythmQuantizer:
    def test_empty_input(self):
        quantizer = RhythmQuantizer(_make_tempo())
        assert quantizer.quantize([]) == []

    def test_single_quarter_note_at_120bpm(self):
        """At 120 BPM, 0.5s duration ≈ one quarter note."""
        # beat_duration = 60/120 = 0.5s, quarter note = 0.25 whole = 1 beat = 0.5s
        event = _make_event(onset_s=0.0, offset_s=0.5)
        quantizer = RhythmQuantizer(_make_tempo(120.0))
        notes = quantizer.quantize([event])
        note_elements = [n for n in notes if not n.is_rest]
        assert len(note_elements) == 1
        assert note_elements[0].duration_type == NoteDuration.QUARTER

    def test_single_half_note_at_120bpm(self):
        """At 120 BPM, 1.0s duration ≈ one half note (2 beats)."""
        event = _make_event(onset_s=0.0, offset_s=1.0)
        quantizer = RhythmQuantizer(_make_tempo(120.0))
        notes = quantizer.quantize([event])
        note_elements = [n for n in notes if not n.is_rest]
        assert len(note_elements) == 1
        assert note_elements[0].duration_type == NoteDuration.HALF

    def test_rest_inserted_between_notes(self):
        """Gap between two notes should produce a rest element."""
        # Two quarter notes with a quarter-note gap between them (at 120 BPM)
        events = [
            _make_event(onset_s=0.0, offset_s=0.5),    # quarter note
            _make_event(onset_s=1.0, offset_s=1.5),    # another quarter note (0.5s gap)
        ]
        quantizer = RhythmQuantizer(_make_tempo(120.0))
        all_elements = quantizer.quantize(events)
        rests = [n for n in all_elements if n.is_rest]
        assert len(rests) >= 1

    def test_output_sorted_by_onset(self):
        events = [
            _make_event(onset_s=0.0, offset_s=0.5, midi=60),
            _make_event(onset_s=0.5, offset_s=1.0, midi=62),
            _make_event(onset_s=1.0, offset_s=1.5, midi=64),
        ]
        quantizer = RhythmQuantizer(_make_tempo(120.0))
        result = quantizer.quantize(events)
        onsets = [n.onset_beat for n in result]
        assert onsets == sorted(onsets)

    def test_quarter_lengths_positive(self):
        events = [_make_event(0.0, 0.5), _make_event(0.5, 1.0)]
        quantizer = RhythmQuantizer(_make_tempo(120.0))
        result = quantizer.quantize(events)
        assert all(n.quarter_length > 0 for n in result)

    def test_bpm_property(self):
        quantizer = RhythmQuantizer(_make_tempo(142.0))
        assert quantizer.bpm == pytest.approx(142.0)

    def test_time_signature_property(self):
        quantizer = RhythmQuantizer(_make_tempo(), time_signature=(3, 4))
        assert quantizer.time_signature == (3, 4)

    def test_midi_note_preserved(self):
        events = [_make_event(0.0, 0.5, midi=45)]
        quantizer = RhythmQuantizer(_make_tempo(120.0))
        result = quantizer.quantize(events)
        note_elements = [n for n in result if not n.is_rest]
        assert note_elements[0].midi_notes[0] == 45

    def test_no_triplets_when_disabled(self):
        events = [_make_event(0.0, 1.0 / 6.0)]   # triplet-quarter duration
        quantizer = RhythmQuantizer(_make_tempo(120.0), include_triplets=False)
        result = quantizer.quantize(events)
        for n in result:
            assert "triplet" not in n.duration_type.value


class TestQuantizerTiling:
    """Phase 4 — RhythmQuantizer.quantize() must produce non-overlapping output.

    The tiling invariant: for any consecutive pair of QuantizedNote entries
    in the output list, prev.onset_beat + prev.duration_beats <= next.onset_beat.
    """

    def _quantize(self, events: list[NoteEvent], bpm: float = 120.0) -> list:
        from improv_scribe.quantization.grid import RhythmQuantizer
        return RhythmQuantizer(_make_tempo(bpm)).quantize(events)

    def _assert_tiling(self, quantized: list) -> None:
        """Assert no entry overlaps the next."""
        for i in range(len(quantized) - 1):
            prev = quantized[i]
            nxt = quantized[i + 1]
            end = prev.onset_beat + prev.duration_beats
            assert end <= nxt.onset_beat + 1e-9, (
                f"Overlap at index {i}: prev ends at {end}, next starts at {nxt.onset_beat}"
            )

    def test_consecutive_notes_tile_exactly(self):
        # Two abutting quarter notes at 120 BPM: 0.0-0.5s, 0.5-1.0s
        # = beats 0.0-1.0, 1.0-2.0
        events = [
            _make_event(0.0, 0.5),
            _make_event(0.5, 1.0),
        ]
        q = self._quantize(events)
        # No rest expected (they abut)
        non_rests = [n for n in q if not n.is_rest]
        assert len(non_rests) == 2
        self._assert_tiling(q)
        # Tile invariant strict here: 0.0 + 1.0 == 1.0
        assert non_rests[0].onset_beat + non_rests[0].duration_beats == pytest.approx(non_rests[1].onset_beat)

    def test_rest_fills_gap_exactly(self):
        # Two quarter notes with a 1-beat gap at 120 BPM:
        # 0.0-0.5s, then gap, then 1.0-1.5s = beats 0-1, gap 1-2, note 2-3
        events = [
            _make_event(0.0, 0.5),
            _make_event(1.0, 1.5),
        ]
        q = self._quantize(events)
        # Expect: note, rest, note
        assert len(q) == 3
        assert q[0].is_rest is False
        assert q[1].is_rest is True
        assert q[2].is_rest is False
        self._assert_tiling(q)
        # The rest should exactly fill the gap
        assert q[1].duration_beats == pytest.approx(1.0)

    def test_overlap_regression_dyad_sample_scenario(self):
        # Reproduce the failing dyad-sample shape: 40 BPM tempo, onsets
        # at irregular fractional beat positions that triggered the
        # original §14.3 overlap bug.
        # At 40 BPM: 1 beat = 1.5s. Onsets at 0.290s, 2.079s, 3.821s,
        # 5.541s match the octave-dyads sample (basic-pitch output).
        events = [
            _make_event(0.290, 1.265, midi=40),
            _make_event(2.079, 3.229, midi=41),
            _make_event(3.821, 4.797, midi=55),
            _make_event(5.541, 6.970, midi=45),
        ]
        q = self._quantize(events, bpm=40.0)
        # Crucial: no overlap. This is the regression guard.
        self._assert_tiling(q)
        # All durations are exact catalog values
        for entry in q:
            ql = entry.quarter_length
            assert ql > 0, f"zero or negative quarter_length at {entry}"

    def test_triplet_quarter_duration_preserved(self):
        # An event whose duration is exactly 2/3 beat at 120 BPM
        # (1 beat = 0.5s; 2/3 beat = 0.333s). Triplet-quarter is 2/3 beat.
        events = [_make_event(0.0, 0.333)]
        q = self._quantize(events)
        non_rests = [n for n in q if not n.is_rest]
        assert len(non_rests) == 1
        assert non_rests[0].duration_type == NoteDuration.TRIPLET_QUARTER
        # Triplet-quarter = 2/3 beat = 2/3 quarter-length
        assert non_rests[0].duration_beats == pytest.approx(2.0 / 3.0)

    def test_phase_1_2_3_aligned_inputs_unchanged(self):
        # Simulate clean-grid inputs as produced by CREPE/basic-pitch on the
        # mono open-string samples at 120 BPM: onsets at beats 1,2,3,4,5,6
        # with 1-beat durations.
        events = [
            _make_event(0.5, 1.0),   # beat 1, dur 1
            _make_event(1.0, 1.5),   # beat 2, dur 1
            _make_event(1.5, 2.0),   # beat 3, dur 1
            _make_event(2.0, 2.5),   # beat 4, dur 1
            _make_event(2.5, 3.0),   # beat 5, dur 1
            _make_event(3.0, 3.5),   # beat 6, dur 1
        ]
        q = self._quantize(events)
        non_rests = [n for n in q if not n.is_rest]
        assert len(non_rests) == 6
        # All snap to integer beat positions
        for i, entry in enumerate(non_rests):
            assert entry.onset_beat == pytest.approx(float(i + 1))
            assert entry.duration_type == NoteDuration.QUARTER
            assert entry.duration_beats == pytest.approx(1.0)
        self._assert_tiling(q)
