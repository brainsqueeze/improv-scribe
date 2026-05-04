"""
tests/quantization/test_tempo.py

Tests for TempoEstimator. Constructs synthetic NoteEvent lists at regular
intervals corresponding to known BPM values and verifies detection accuracy.
"""

from __future__ import annotations

import pytest

from improv_scribe.analysis.note_tracker import NoteEvent
from improv_scribe.config import AppConfig
from improv_scribe.quantization.tempo import TempoEstimator


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


def _events_at_bpm(bpm: float, n_notes: int = 8, start: float = 0.0) -> list[NoteEvent]:
    """Create evenly-spaced NoteEvents at *bpm* quarter-note pulse."""
    beat_s = 60.0 / bpm
    events = []
    for i in range(n_notes):
        onset = start + i * beat_s
        events.append(NoteEvent(
            onset_s=onset,
            offset_s=onset + beat_s * 0.9,
            midi_note=60,
            frequency_hz=261.63,
            confidence=0.9,
            cents_deviation=0.0,
        ))
    return events


class TestTempoEstimator:
    @pytest.mark.parametrize("bpm", [60.0, 90.0, 120.0, 140.0, 180.0])
    def test_known_bpm_within_10_percent(self, config, bpm):
        """Detected BPM should be within 10% of the true BPM for clean pulse input."""
        events = _events_at_bpm(bpm, n_notes=8)
        estimator = TempoEstimator(config)
        result = estimator.estimate(events)
        ratio = result.bpm / bpm
        assert 0.9 <= ratio <= 1.1, (
            f"BPM={bpm}: detected={result.bpm:.1f}, ratio={ratio:.3f}"
        )

    def test_single_event_returns_default_bpm(self, config):
        """With fewer than 2 events, should return default 120 BPM."""
        events = _events_at_bpm(120.0, n_notes=1)
        estimator = TempoEstimator(config)
        result = estimator.estimate(events)
        assert result.bpm == pytest.approx(120.0)

    def test_empty_events_returns_default(self, config):
        estimator = TempoEstimator(config)
        result = estimator.estimate([])
        assert result.bpm == pytest.approx(120.0)
        assert result.confidence == pytest.approx(0.0)

    def test_bpm_clamped_to_config_range(self, config):
        """BPM should always fall within [tempo_min, tempo_max]."""
        # Absurdly fast notes → should be clamped to max
        events = _events_at_bpm(500.0, n_notes=8)
        estimator = TempoEstimator(config)
        result = estimator.estimate(events)
        assert config.tempo_min <= result.bpm <= config.tempo_max

    def test_confidence_between_0_and_1(self, config):
        events = _events_at_bpm(120.0, n_notes=8)
        estimator = TempoEstimator(config)
        result = estimator.estimate(events)
        assert 0.0 <= result.confidence <= 1.0

    def test_ioi_fallback_for_two_events(self, config):
        """With exactly 2 events, uses IOI median fallback."""
        events = _events_at_bpm(100.0, n_notes=2)
        estimator = TempoEstimator(config)
        result = estimator.estimate(events)
        # IOI = 0.6s → 100 BPM; allow generous tolerance for this path
        assert abs(result.bpm - 100.0) < 15.0
