"""Unit tests for QuantizedNote's chord-capable shape (Phase 0)."""

from __future__ import annotations

import pytest

from improv_scribe.quantization.grid import NoteDuration, QuantizedNote


def _make_qn(
    midi_notes: tuple[int, ...] = (60,),
    is_rest: bool = False,
    **overrides,
) -> QuantizedNote:
    """Helper: build a QuantizedNote with sensible defaults for testing."""
    n = len(midi_notes) if not is_rest else 0
    defaults = {
        "midi_notes": () if is_rest else midi_notes,
        "frequencies_hz": () if is_rest else (440.0,) * n,
        "confidences": () if is_rest else (0.9,) * n,
        "cents_deviations": () if is_rest else (0.0,) * n,
        "onset_beat": 0.0,
        "duration_beats": 1.0,
        "duration_type": NoteDuration.QUARTER,
        "quarter_length": 1.0,
        "is_rest": is_rest,
    }
    defaults.update(overrides)
    return QuantizedNote(**defaults)


class TestQuantizedNoteShape:
    def test_singleton_construction(self):
        qn = _make_qn(midi_notes=(60,))
        assert qn.midi_notes == (60,)
        assert qn.is_rest is False

    def test_chord_construction(self):
        qn = _make_qn(midi_notes=(60, 64, 67))
        assert qn.midi_notes == (60, 64, 67)
        assert len(qn.frequencies_hz) == 3
        assert len(qn.confidences) == 3

    def test_rest_has_empty_tuples(self):
        qn = _make_qn(is_rest=True)
        assert qn.is_rest is True
        assert qn.midi_notes == ()
        assert qn.frequencies_hz == ()
        assert qn.confidences == ()
        assert qn.cents_deviations == ()


class TestQuantizedNoteBackCompatProperties:
    """Removed in Phase 2."""

    def test_midi_note_returns_first_element(self):
        assert _make_qn(midi_notes=(60,)).midi_note == 60
        assert _make_qn(midi_notes=(60, 64, 67)).midi_note == 60

    def test_midi_note_returns_zero_for_rest(self):
        # Existing rest convention is midi_note=0.
        assert _make_qn(is_rest=True).midi_note == 0

    def test_frequency_hz_returns_first_element(self):
        assert _make_qn(midi_notes=(60,), frequencies_hz=(261.6,)).frequency_hz == pytest.approx(261.6)

    def test_frequency_hz_returns_zero_for_rest(self):
        assert _make_qn(is_rest=True).frequency_hz == 0.0

    def test_confidence_returns_mean(self):
        qn = _make_qn(midi_notes=(60, 64), confidences=(0.8, 0.6))
        assert qn.confidence == pytest.approx(0.7)

    def test_confidence_returns_one_for_rest(self):
        # Existing rest convention is confidence=1.0.
        assert _make_qn(is_rest=True).confidence == 1.0

    def test_cents_deviation_returns_first_element(self):
        qn = _make_qn(midi_notes=(60, 64), cents_deviations=(5.0, -3.0))
        assert qn.cents_deviation == pytest.approx(5.0)

    def test_cents_deviation_returns_zero_for_rest(self):
        # Rest convention: cents_deviation=0.0.
        assert _make_qn(is_rest=True).cents_deviation == 0.0
