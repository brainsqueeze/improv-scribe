"""
tests/analysis/test_pitch.py

Tests for PitchEstimator (pYIN backend).

Uses synthetic sine-wave signals at known frequencies so results are
deterministic. We allow ±1 semitone tolerance on the median detected pitch,
which accounts for pYIN's small systematic bias on attack transients.

Signal duration notes
---------------------
pYIN's voiced probability is low for the first few frames (attack transient
+ model warm-up). Low-frequency notes (E2=82 Hz, bass E1=41 Hz) need a
longer signal so the steady-state portion dominates the voiced-frame median.
Rule of thumb: use at least 20 full periods worth of signal.
  E2  (82 Hz): 20 / 82 ≈ 0.24s → use 2.0s
  A2 (110 Hz): 20 / 110 ≈ 0.18s → use 1.5s
  E1  (41 Hz): 20 / 41 ≈ 0.49s → use 3.0s
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.pitch import PitchEstimator, PitchResult
from improv_scribe.config import AppConfig

SR = 44100
AMPLITUDE = 0.7


def _sine(midi: int, duration_s: float) -> np.ndarray:
    """Pure sine with exponential decay envelope at the exact frequency for *midi*."""
    freq = 440.0 * 2 ** ((midi - 69) / 12.0)
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    # Slow decay so the note stays well above noise gate for most of its duration
    envelope = np.exp(-1.0 * t / duration_s)
    return (AMPLITUDE * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _min_duration_for_midi(midi: int) -> float:
    """Return a conservative minimum signal duration for reliable pYIN detection."""
    freq = 440.0 * 2 ** ((midi - 69) / 12.0)
    # At least 30 full periods, minimum 1.5s
    return max(30.0 / freq, 1.5)


def _median_voiced_midi(result: PitchResult) -> float:
    """Median MIDI note of all voiced frames."""
    voiced = result.voiced_frames
    assert voiced, "No voiced frames detected — check signal amplitude or confidence threshold"
    freqs = [f.freq_hz for f in voiced if math.isfinite(f.freq_hz) and f.freq_hz > 0]
    assert freqs, "No finite voiced frequencies"
    midi_values = [69.0 + 12.0 * math.log2(f / 440.0) for f in freqs]
    return float(np.median(midi_values))


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def guitar_profile():
    return get_profile(Instrument.GUITAR)


@pytest.fixture
def bass_profile():
    return get_profile(Instrument.BASS)


class TestPitchEstimatorPyin:

    @pytest.mark.parametrize("midi_note", [40, 45, 52, 64, 69])
    def test_guitar_single_note_within_one_semitone(self, config, guitar_profile, midi_note):
        """Detected median pitch should be within ±1 semitone of ground truth."""
        duration = _min_duration_for_midi(midi_note)
        audio = _sine(midi_note, duration)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, guitar_profile)
        detected = _median_voiced_midi(result)
        assert abs(detected - midi_note) <= 1.0, (
            f"MIDI {midi_note}: detected median {detected:.2f}, "
            f"delta={abs(detected - midi_note):.2f}"
        )

    @pytest.mark.parametrize("midi_note", [28, 33, 35, 40])
    def test_bass_single_note_within_one_semitone(self, config, bass_profile, midi_note):
        """Bass detection — lower frequencies need larger frame_length (auto-computed)."""
        duration = _min_duration_for_midi(midi_note)
        audio = _sine(midi_note, duration)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, bass_profile)
        detected = _median_voiced_midi(result)
        assert abs(detected - midi_note) <= 1.0, (
            f"Bass MIDI {midi_note}: detected {detected:.2f}, "
            f"delta={abs(detected - midi_note):.2f}"
        )

    def test_silence_produces_no_voiced_frames(self, config, guitar_profile):
        """Pure silence should produce zero voiced frames."""
        audio = np.zeros(SR * 2, dtype=np.float32)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, guitar_profile)
        assert len(result.voiced_frames) == 0

    def test_sub_noise_gate_signal_mostly_unvoiced(self, config, guitar_profile):
        """Very low amplitude noise should produce few voiced frames."""
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(SR * 2) * 0.001).astype(np.float32)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(noise, guitar_profile)
        voiced_ratio = len(result.voiced_frames) / max(len(result.frames), 1)
        assert voiced_ratio < 0.1, f"Noise yielded {voiced_ratio:.1%} voiced frames"

    def test_out_of_range_note_not_voiced_by_guitar_profile(self, config, guitar_profile):
        """A note below guitar fmin should not produce voiced frames within guitar range."""
        # E0 = MIDI 16 ≈ 20.6 Hz — well below guitar fmin (82.41 Hz)
        freq = 440.0 * 2 ** ((16 - 69) / 12.0)
        t = np.linspace(0, 2.0, SR * 2, endpoint=False)
        audio = (0.7 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, guitar_profile)
        # pYIN constrained to [fmin, fmax] — should not find voiced fundamental below fmin
        in_range_voiced = [
            f for f in result.voiced_frames
            if f.freq_hz >= guitar_profile.freq_min_hz
        ]
        assert len(in_range_voiced) == 0

    def test_unknown_backend_raises(self, config, guitar_profile):
        with pytest.raises(ValueError, match="Unknown pitch backend"):
            PitchEstimator(config, backend="nonexistent")

    def test_pitch_result_has_frames(self, config, guitar_profile):
        audio = _sine(69, duration_s=1.0)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, guitar_profile)
        assert len(result.frames) > 0

    def test_pitch_result_times_are_monotonic(self, config, guitar_profile):
        audio = _sine(52, duration_s=1.5)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, guitar_profile)
        times = [f.time_s for f in result.frames]
        assert times == sorted(times)

    def test_confidence_in_unit_interval(self, config, guitar_profile):
        audio = _sine(45, duration_s=1.5)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, guitar_profile)
        for frame in result.frames:
            assert 0.0 <= frame.confidence <= 1.0, (
                f"Confidence out of range: {frame.confidence}"
            )

    def test_voiced_frames_subset_of_all_frames(self, config, guitar_profile):
        audio = _sine(60, duration_s=1.5)
        estimator = PitchEstimator(config, backend="pyin")
        result = estimator.estimate(audio, guitar_profile)
        assert len(result.voiced_frames) <= len(result.frames)

    def test_backend_name_property(self, config):
        estimator = PitchEstimator(config, backend="pyin")
        assert estimator.backend_name == "pyin"


"""Unit tests for the basic-pitch pitch backend wrapper (Phase 1)."""


class TestBasicPitchNote:
    """The BasicPitchNote dataclass captures one event from basic-pitch.predict()."""

    def test_construction(self):
        from improv_scribe.analysis.pitch import BasicPitchNote
        ev = BasicPitchNote(start_s=0.10, end_s=0.50, midi=60, amplitude=0.80)
        assert ev.start_s == pytest.approx(0.10)
        assert ev.end_s == pytest.approx(0.50)
        assert ev.midi == 60
        assert ev.amplitude == pytest.approx(0.80)

    def test_duration_s_property(self):
        from improv_scribe.analysis.pitch import BasicPitchNote
        ev = BasicPitchNote(start_s=0.10, end_s=0.50, midi=60, amplitude=0.80)
        assert ev.duration_s == pytest.approx(0.40)


class TestPitchResultBpNotes:
    """PitchResult gains an optional bp_notes field used by basic-pitch backend."""

    def test_default_is_none(self):
        result = PitchResult(frames=[], bp_notes=None, sample_rate=44100, hop_length=512)
        assert result.bp_notes is None

    def test_can_carry_bp_notes(self):
        from improv_scribe.analysis.pitch import BasicPitchNote
        notes = [
            BasicPitchNote(start_s=0.0, end_s=0.5, midi=60, amplitude=0.8),
            BasicPitchNote(start_s=0.5, end_s=1.0, midi=64, amplitude=0.7),
        ]
        result = PitchResult(frames=[], bp_notes=notes, sample_rate=44100, hop_length=512)
        assert result.bp_notes is not None
        assert len(result.bp_notes) == 2
        assert result.bp_notes[0].midi == 60


def _fake_predict_returns(note_events: list[tuple]) -> object:
    """Build a fake predict() return value: (model_output, midi_data, note_events)."""
    # model_output and midi_data are not consumed by our wrapper
    return ({}, None, note_events)


class TestBasicPitchBackend:
    """The basic-pitch backend wrapper unpacks predict() and applies filtering.

    All tests mock basic_pitch.inference.predict so they run without exercising
    the real model (and without requiring basic-pitch to be installed in CI).
    """

    def _make_config(self):
        from improv_scribe.config import AppConfig
        return AppConfig()

    def _make_profile(self):
        from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
        return get_profile(Instrument.GUITAR)

    def test_unpacks_note_events_into_bp_notes(self):
        from improv_scribe.analysis.pitch import _BasicPitchBackend  # noqa: PLC0415

        # 3 events: 2 within range + amplitude, 1 below amplitude floor
        fake_events = [
            (0.10, 0.50, 60, 0.80, [1, 1, 1]),   # OK
            (0.50, 0.90, 64, 0.70, [1, 1, 1]),   # OK
            (0.90, 1.30, 67, 0.30, [1, 1, 1]),   # below floor (0.50)
        ]
        audio = np.zeros(44100, dtype=np.float32)

        with patch("basic_pitch.inference.predict", return_value=_fake_predict_returns(fake_events)):
            backend = _BasicPitchBackend()
            result = backend.estimate(
                audio=audio,
                sample_rate=44100,
                profile=self._make_profile(),
                config=self._make_config(),
            )

        assert result.bp_notes is not None
        assert len(result.bp_notes) == 2   # third event filtered out
        assert {n.midi for n in result.bp_notes} == {60, 64}
        for n in result.bp_notes:
            assert n.amplitude >= 0.50

    def test_filters_out_of_range_notes(self):
        from improv_scribe.analysis.pitch import _BasicPitchBackend  # noqa: PLC0415

        # MIDI 20 is below guitar's midi_min=40; MIDI 110 is above midi_max=98
        fake_events = [
            (0.1, 0.5, 20, 0.80, [1]),    # below guitar range
            (0.5, 0.9, 60, 0.80, [1]),    # in range
            (0.9, 1.3, 110, 0.80, [1]),   # above guitar range
        ]
        audio = np.zeros(44100, dtype=np.float32)

        with patch("basic_pitch.inference.predict", return_value=_fake_predict_returns(fake_events)):
            backend = _BasicPitchBackend()
            result = backend.estimate(
                audio=audio,
                sample_rate=44100,
                profile=self._make_profile(),
                config=self._make_config(),
            )

        assert result.bp_notes is not None
        assert len(result.bp_notes) == 1
        assert result.bp_notes[0].midi == 60

    def test_filters_very_short_notes(self):
        from improv_scribe.analysis.pitch import _BasicPitchBackend  # noqa: PLC0415

        # min_note_duration_s default = 0.050 s
        fake_events = [
            (0.10, 0.13, 60, 0.80, [1]),   # 30 ms — too short
            (0.50, 0.90, 64, 0.80, [1]),   # 400 ms — OK
        ]
        audio = np.zeros(44100, dtype=np.float32)

        with patch("basic_pitch.inference.predict", return_value=_fake_predict_returns(fake_events)):
            backend = _BasicPitchBackend()
            result = backend.estimate(
                audio=audio,
                sample_rate=44100,
                profile=self._make_profile(),
                config=self._make_config(),
            )

        assert result.bp_notes is not None
        assert len(result.bp_notes) == 1
        assert result.bp_notes[0].midi == 64

    def test_frames_is_empty_list_not_none(self):
        """PitchResult.frames stays an empty list (not None) so existing
        code that reads .voiced_frames or len(.frames) doesn't crash."""
        from improv_scribe.analysis.pitch import _BasicPitchBackend  # noqa: PLC0415

        audio = np.zeros(44100, dtype=np.float32)
        with patch("basic_pitch.inference.predict", return_value=_fake_predict_returns([])):
            backend = _BasicPitchBackend()
            result = backend.estimate(
                audio=audio,
                sample_rate=44100,
                profile=self._make_profile(),
                config=self._make_config(),
            )

        assert result.frames == []
        assert result.voiced_frames == []
        assert result.bp_notes == []
