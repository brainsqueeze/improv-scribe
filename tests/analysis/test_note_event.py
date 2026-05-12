"""Unit tests for NoteEvent's chord-capable shape.

These tests lock down the data-model contract that the rest of the
polyphonic detection pipeline relies on. Phase 0 keeps back-compat
properties so existing single-pitch consumers see no behaviour change.
"""

from __future__ import annotations

import pytest

from improv_scribe.analysis.note_tracker import NoteEvent, _merge_consecutive_same_pitch


def _make(midi_notes: tuple[int, ...] = (60,), **overrides) -> NoteEvent:
    """Helper: build a NoteEvent with sensible defaults for testing."""
    n = len(midi_notes)
    defaults = {
        "onset_s": 0.0,
        "offset_s": 1.0,
        "midi_notes": midi_notes,
        "frequencies_hz": tuple(440.0 * 2 ** ((m - 69) / 12) for m in midi_notes),
        "confidences": (0.9,) * n,
        "cents_deviations": (0.0,) * n,
    }
    defaults.update(overrides)
    return NoteEvent(**defaults)


class TestNoteEventShape:
    def test_singleton_construction(self):
        event = _make(midi_notes=(60,))
        assert event.midi_notes == (60,)
        assert len(event.frequencies_hz) == 1
        assert len(event.confidences) == 1
        assert len(event.cents_deviations) == 1

    def test_chord_construction(self):
        event = _make(midi_notes=(60, 64, 67))   # C major triad
        assert event.midi_notes == (60, 64, 67)
        assert len(event.frequencies_hz) == 3
        assert len(event.confidences) == 3
        assert len(event.cents_deviations) == 3

    def test_is_chord_property(self):
        assert _make(midi_notes=(60,)).is_chord is False
        assert _make(midi_notes=(60, 64)).is_chord is True
        assert _make(midi_notes=(60, 64, 67)).is_chord is True

    def test_duration_s(self):
        event = _make(onset_s=0.5, offset_s=2.5)
        assert event.duration_s == pytest.approx(2.0)

    def test_duration_s_clamps_to_zero(self):
        event = _make(onset_s=2.0, offset_s=1.0)
        assert event.duration_s == 0.0


class TestNoteEventBackCompatProperties:
    """Phase 0 keeps these properties for callers that haven't migrated yet.
    Phase 2 removes them; the migration completion check is a grep for `.midi_note`
    and `.frequency_hz` returning zero hits in `src/`."""

    def test_midi_note_returns_first_element(self):
        assert _make(midi_notes=(60,)).midi_note == 60
        assert _make(midi_notes=(60, 64, 67)).midi_note == 60   # lowest

    def test_frequency_hz_returns_first_element(self):
        event = _make(midi_notes=(69,))   # A4 = 440 Hz
        assert event.frequency_hz == pytest.approx(440.0)

    def test_confidence_returns_mean(self):
        event = _make(midi_notes=(60, 64), confidences=(0.8, 0.6))
        assert event.confidence == pytest.approx(0.7)

    def test_cents_deviation_returns_first_element(self):
        event = _make(midi_notes=(60, 64), cents_deviations=(5.0, -3.0))
        assert event.cents_deviation == pytest.approx(5.0)


class TestMergeConsecutiveSamePitch:
    """The merge helper collapses back-to-back same-pitch events caused by
    spurious re-onsets on sustained notes.

    Phase 0 changes: comparison is now over full midi_notes tuples (chord
    identity), parallel-tuple arithmetic for averaged fields, and a separate
    gap threshold for chord events (200 ms) vs mono events (600 ms).
    """

    def test_merges_consecutive_singletons_close_in_time(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60,),
                   frequencies_hz=(261.6,), confidences=(0.9,))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60,),
                   frequencies_hz=(262.0,), confidences=(0.85,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 1
        assert merged[0].onset_s == 0.0
        assert merged[0].offset_s == 1.0
        assert merged[0].midi_notes == (60,)
        # Frequencies averaged element-wise
        assert merged[0].frequencies_hz[0] == pytest.approx((261.6 + 262.0) / 2)
        assert merged[0].confidences[0] == pytest.approx((0.9 + 0.85) / 2)

    def test_does_not_merge_singletons_with_large_gap(self):
        # Mono gap > 600 ms must NOT merge
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60,))
        e2 = _make(onset_s=1.5, offset_s=2.0, midi_notes=(60,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_does_not_merge_different_singletons(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60,))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(64,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_merges_consecutive_chords_with_identical_pitches(self):
        # Same chord, < 200 ms gap -> merge
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64, 67),
                   frequencies_hz=(261.6, 329.6, 392.0),
                   confidences=(0.9, 0.85, 0.8),
                   cents_deviations=(0.0, 0.0, 0.0))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60, 64, 67),
                   frequencies_hz=(262.0, 330.0, 392.5),
                   confidences=(0.85, 0.8, 0.75),
                   cents_deviations=(0.0, 0.0, 0.0))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 1
        assert merged[0].midi_notes == (60, 64, 67)
        assert merged[0].frequencies_hz[0] == pytest.approx((261.6 + 262.0) / 2)
        assert merged[0].frequencies_hz[1] == pytest.approx((329.6 + 330.0) / 2)
        assert merged[0].frequencies_hz[2] == pytest.approx((392.0 + 392.5) / 2)

    def test_does_not_merge_chords_with_different_pitches(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60, 67))   # one note differs
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_does_not_merge_chord_to_singleton(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_chord_gap_threshold_is_tighter_than_mono(self):
        # Chord gap of 300 ms (eighth notes at 100 BPM) must NOT merge
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64))
        e2 = _make(onset_s=0.8, offset_s=1.3, midi_notes=(60, 64))   # gap = 300 ms
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_empty_list_returns_empty(self):
        assert _merge_consecutive_same_pitch([]) == []

    def test_single_event_returns_single(self):
        e = _make(midi_notes=(60,))
        merged = _merge_consecutive_same_pitch([e])
        assert merged == [e]

    def test_merges_chain_of_three_consecutive_singletons(self):
        # Three identical-pitch events within the gap must collapse to one.
        # Regression guard: the iterative merge must keep extending the
        # running event, not just merge pairs.
        e1 = _make(onset_s=0.0, offset_s=0.4, midi_notes=(60,))
        e2 = _make(onset_s=0.5, offset_s=0.9, midi_notes=(60,))
        e3 = _make(onset_s=1.0, offset_s=1.4, midi_notes=(60,))
        merged = _merge_consecutive_same_pitch([e1, e2, e3])
        assert len(merged) == 1
        assert merged[0].onset_s == 0.0
        assert merged[0].offset_s == 1.4
        assert merged[0].midi_notes == (60,)
