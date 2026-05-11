"""Unit tests for NoteEvent's chord-capable shape.

These tests lock down the data-model contract that the rest of the
polyphonic detection pipeline relies on. Phase 0 keeps back-compat
properties so existing single-pitch consumers see no behaviour change.
"""

from __future__ import annotations

import pytest

from improv_scribe.analysis.note_tracker import NoteEvent


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
