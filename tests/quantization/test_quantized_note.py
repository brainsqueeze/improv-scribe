"""Unit tests for QuantizedNote's chord-capable shape."""

from __future__ import annotations

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
