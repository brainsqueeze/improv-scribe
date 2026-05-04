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
        midi_note=midi,
        frequency_hz=261.63,
        confidence=0.9,
        cents_deviation=0.0,
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
        assert note_elements[0].midi_note == 45

    def test_no_triplets_when_disabled(self):
        events = [_make_event(0.0, 1.0 / 6.0)]   # triplet-quarter duration
        quantizer = RhythmQuantizer(_make_tempo(120.0), include_triplets=False)
        result = quantizer.quantize(events)
        for n in result:
            assert "triplet" not in n.duration_type.value
