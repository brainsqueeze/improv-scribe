"""
tests/analysis/test_note_tracker.py

Tests for NoteTracker: verifies that onset+pitch frames are correctly
assembled into NoteEvents with accurate MIDI values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.note_tracker import NoteTracker, hz_to_midi
from improv_scribe.analysis.onset import Onset
from improv_scribe.analysis.pitch import BasicPitchNote, PitchFrame, PitchResult
from improv_scribe.config import AppConfig


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def guitar_profile():
    return get_profile(Instrument.GUITAR)


def _make_pitch_result(frames: list[tuple[float, float, float]]) -> PitchResult:
    """
    Construct a PitchResult from (time_s, freq_hz, confidence) tuples.
    A frame is voiced if confidence >= 0.5.
    """
    pitch_frames = [
        PitchFrame(
            time_s=t,
            freq_hz=f if conf >= 0.5 else float("nan"),
            confidence=conf,
            is_voiced=conf >= 0.5 and math.isfinite(f),
        )
        for t, f, conf in frames
    ]
    return PitchResult(frames=pitch_frames, sample_rate=44100, hop_length=512)


class TestHzToMidi:
    def test_a4_is_69(self):
        midi, cents = hz_to_midi(440.0)
        assert midi == 69
        assert abs(cents) < 0.01

    def test_e2_is_40(self):
        midi, _ = hz_to_midi(82.41)
        assert midi == 40

    def test_zero_returns_zero(self):
        midi, cents = hz_to_midi(0.0)
        assert midi == 0

    def test_cents_within_50(self):
        # Any frequency should produce |cents| <= 50
        for freq in [100.0, 220.0, 440.0, 880.0, 1760.0]:
            _, cents = hz_to_midi(freq)
            assert abs(cents) <= 50.0


class TestNoteTracker:
    def test_single_note(self, config, guitar_profile):
        """One onset, multiple voiced frames → one NoteEvent."""
        # E2 = 82.41 Hz = MIDI 40
        frames = [(t, 82.41, 0.9) for t in np.arange(0.0, 1.0, 0.01)]
        pitch_result = _make_pitch_result(frames)
        onsets = [Onset(time_s=0.0, strength=1.0)]

        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets)

        assert len(events) == 1
        assert events[0].midi_notes[0] == 40
        assert events[0].onset_s == pytest.approx(0.0)

    def test_two_notes(self, config, guitar_profile):
        """Two onsets → two NoteEvents with correct pitches."""
        # E2 for first note, A2 for second
        e2_hz = 82.41
        a2_hz = 110.00
        frames = (
            [(t, e2_hz, 0.9) for t in np.arange(0.0, 0.5, 0.01)]
            + [(t, a2_hz, 0.9) for t in np.arange(0.5, 1.0, 0.01)]
        )
        pitch_result = _make_pitch_result(frames)
        onsets = [
            Onset(time_s=0.0, strength=1.0),
            Onset(time_s=0.5, strength=1.0),
        ]

        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets)

        assert len(events) == 2
        assert events[0].midi_notes[0] == 40   # E2
        assert events[1].midi_notes[0] == 45   # A2

    def test_no_onsets_returns_empty(self, config, guitar_profile):
        frames = [(0.1, 440.0, 0.9)]
        pitch_result = _make_pitch_result(frames)
        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets=[])
        assert events == []

    def test_no_voiced_frames_returns_empty(self, config, guitar_profile):
        frames = [(0.1, float("nan"), 0.1)]  # unvoiced
        pitch_result = _make_pitch_result(frames)
        onsets = [Onset(time_s=0.0, strength=1.0)]
        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets)
        assert events == []

    def test_out_of_range_note_filtered(self, config, guitar_profile):
        """Notes outside the instrument's MIDI range should be dropped."""
        # MIDI 10 = ~14 Hz — far below guitar minimum (40)
        freq = 440.0 * 2 ** ((10 - 69) / 12.0)
        frames = [(t, freq, 0.9) for t in np.arange(0.0, 1.0, 0.01)]
        pitch_result = _make_pitch_result(frames)
        onsets = [Onset(time_s=0.0, strength=1.0)]
        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets)
        assert events == []

    def test_chunk_offset_applied(self, config, guitar_profile):
        """chunk_offset_s should shift all event times."""
        frames = [(t, 82.41, 0.9) for t in np.arange(0.0, 0.5, 0.01)]
        pitch_result = _make_pitch_result(frames)
        onsets = [Onset(time_s=0.0, strength=1.0)]
        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets, chunk_offset_s=10.0)
        assert events[0].onset_s == pytest.approx(10.0)

    def test_events_sorted_by_onset(self, config, guitar_profile):
        """Output must be sorted ascending by onset_s."""
        e2_hz, a2_hz, d3_hz = 82.41, 110.00, 146.83
        frames = (
            [(t, e2_hz, 0.9) for t in np.arange(0.0, 0.3, 0.01)]
            + [(t, a2_hz, 0.9) for t in np.arange(0.3, 0.6, 0.01)]
            + [(t, d3_hz, 0.9) for t in np.arange(0.6, 1.0, 0.01)]
        )
        pitch_result = _make_pitch_result(frames)
        onsets = [
            Onset(time_s=0.0, strength=1.0),
            Onset(time_s=0.3, strength=0.9),
            Onset(time_s=0.6, strength=0.8),
        ]
        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets)
        onset_times = [e.onset_s for e in events]
        assert onset_times == sorted(onset_times)

    def test_note_event_duration_positive(self, config, guitar_profile):
        frames = [(t, 82.41, 0.9) for t in np.arange(0.0, 0.5, 0.01)]
        pitch_result = _make_pitch_result(frames)
        onsets = [Onset(time_s=0.0, strength=1.0)]
        tracker = NoteTracker(config, guitar_profile)
        events = tracker.process(pitch_result, onsets)
        assert all(e.duration_s > 0 for e in events)


# ---------------------------------------------------------------------------
# Task 7 — basic-pitch dispatch path
# ---------------------------------------------------------------------------

def _bp_pitch_result(notes: list[BasicPitchNote]) -> PitchResult:
    """Helper: build a PitchResult carrying basic-pitch notes."""
    return PitchResult(
        frames=[],
        sample_rate=44100,
        hop_length=512,
        bp_notes=notes,
    )


class TestNoteTrackerBasicPitch:
    """When `pitch_result.bp_notes is not None`, NoteTracker.process() takes
    the basic-pitch path: one BasicPitchNote becomes one singleton NoteEvent.
    No onset clustering in Phase 1 (Phase 2 adds it for chord support).
    """

    def _config(self):
        return AppConfig()

    def _profile(self):
        return get_profile(Instrument.GUITAR)

    def test_empty_bp_notes_returns_empty(self):
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([])
        events = tracker.process(result, onsets=[])
        assert events == []

    def test_one_bp_note_becomes_one_singleton_event(self):
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.10, end_s=0.50, midi=60, amplitude=0.80),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        e = events[0]
        assert e.onset_s == pytest.approx(0.10)
        assert e.offset_s == pytest.approx(0.50)
        assert e.midi_notes == (60,)
        assert e.confidences == (pytest.approx(0.80),)
        assert e.cents_deviations == (0.0,)

    def test_multiple_bp_notes_become_separate_singletons(self):
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.10, end_s=0.50, midi=60, amplitude=0.80),
            BasicPitchNote(start_s=0.50, end_s=0.90, midi=64, amplitude=0.70),
            BasicPitchNote(start_s=0.90, end_s=1.30, midi=67, amplitude=0.60),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 3
        assert [tuple(e.midi_notes) for e in events] == [(60,), (64,), (67,)]

    def test_events_sorted_by_onset(self):
        """basic-pitch can emit events out of time order — NoteTracker sorts."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.90, end_s=1.30, midi=67, amplitude=0.6),
            BasicPitchNote(start_s=0.10, end_s=0.50, midi=60, amplitude=0.8),
            BasicPitchNote(start_s=0.50, end_s=0.90, midi=64, amplitude=0.7),
        ])
        events = tracker.process(result, onsets=[])
        assert [e.onset_s for e in events] == [0.10, 0.50, 0.90]

    def test_chunk_offset_applied(self):
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.10, end_s=0.50, midi=60, amplitude=0.8),
        ])
        events = tracker.process(result, onsets=[], chunk_offset_s=5.0)
        assert events[0].onset_s == pytest.approx(5.10)
        assert events[0].offset_s == pytest.approx(5.50)

    def test_dispatches_on_bp_notes_not_frames(self):
        """If bp_notes is not None (even empty list), basic-pitch path is taken
        regardless of whether onsets are passed in."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([])
        # Onsets are non-empty but ignored on the basic-pitch path
        events = tracker.process(result, onsets=[Onset(time_s=0.1, strength=1.0)])
        assert events == []


class TestNoteTrackerBasicPitchClustering:
    """Phase 2 — basic-pitch's flat note events get clustered into chord events.

    Clustering rule: a new cluster opens when current.start_s exceeds the
    earliest member of the cluster currently being built by more than the
    ONSET_GROUPING_WINDOW_MS (default 100 ms).

    Cluster members are sorted by midi ascending; duplicate MIDI within a
    cluster keep the highest-amplitude detection (basic-pitch can emit two
    near-simultaneous events for the same pitch).
    """

    def _config(self):
        return AppConfig()

    def _profile(self):
        return get_profile(Instrument.GUITAR)

    def test_two_close_events_become_one_dyad_chord(self):
        """Two events 12 ms apart -> one chord NoteEvent with midi_notes=(40, 52)."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.290, end_s=1.265, midi=40, amplitude=0.66),
            BasicPitchNote(start_s=0.302, end_s=1.370, midi=52, amplitude=0.78),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        e = events[0]
        assert e.midi_notes == (40, 52)
        assert e.is_chord is True
        # Onset = earliest member's start_s; offset = latest member's end_s
        assert e.onset_s == pytest.approx(0.290)
        assert e.offset_s == pytest.approx(1.370)
        # Frequencies/confidences/cents parallel to midi_notes
        assert len(e.frequencies_hz) == 2
        assert len(e.confidences) == 2
        assert e.confidences == (pytest.approx(0.66), pytest.approx(0.78))

    def test_events_outside_window_become_separate_events(self):
        """Two events 200 ms apart -> two separate singleton NoteEvents."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.80),
            BasicPitchNote(start_s=0.200, end_s=0.700, midi=52, amplitude=0.75),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 2
        assert events[0].midi_notes == (40,)
        assert events[1].midi_notes == (52,)
        assert events[0].is_chord is False
        assert events[1].is_chord is False

    def test_midi_notes_sorted_ascending(self):
        """A cluster received in arbitrary order yields midi_notes sorted ascending."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.302, end_s=1.000, midi=52, amplitude=0.78),
            BasicPitchNote(start_s=0.290, end_s=1.000, midi=40, amplitude=0.66),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 52)   # not (52, 40)

    def test_earliest_anchor_caps_cluster_width(self):
        """Anchor is 0.000, window is 100ms. Event at 0.080 joins (80ms < 100ms).
        Event at 0.140 starts a NEW cluster (140ms - 0ms > 100ms)."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=1.000, midi=40, amplitude=0.70),
            BasicPitchNote(start_s=0.080, end_s=1.000, midi=47, amplitude=0.70),
            BasicPitchNote(start_s=0.140, end_s=1.000, midi=52, amplitude=0.70),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 2
        assert events[0].midi_notes == (40, 47)
        assert events[1].midi_notes == (52,)

    def test_three_member_cluster(self):
        """All three members within 100 ms of the anchor cluster together."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=1.000, midi=40, amplitude=0.80),
            BasicPitchNote(start_s=0.030, end_s=1.000, midi=47, amplitude=0.75),
            BasicPitchNote(start_s=0.075, end_s=1.000, midi=52, amplitude=0.70),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 47, 52)
        assert events[0].is_chord is True

    def test_duplicate_midi_in_cluster_kept_at_highest_amplitude(self):
        """basic-pitch can emit two events for the same pitch within the
        cluster window; we keep one (the higher-amplitude one)."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.55),
            BasicPitchNote(start_s=0.020, end_s=0.500, midi=40, amplitude=0.75),  # same pitch, higher amp
            BasicPitchNote(start_s=0.030, end_s=0.500, midi=47, amplitude=0.65),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 47)
        # The kept E2 detection is the 0.75 one, not the 0.55 one.
        # confidences[0] corresponds to midi=40 (sorted ascending), so:
        assert events[0].confidences[0] == pytest.approx(0.75)

    def test_relative_floor_drops_quiet_member(self):
        """Within a cluster, members below POLYPHONIC_RELATIVE_FLOOR * max_amp
        get dropped. Default ratio = 0.5; max_amp 0.85, so anything < 0.425
        is dropped."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.85),
            BasicPitchNote(start_s=0.030, end_s=0.500, midi=47, amplitude=0.40),  # 0.40 < 0.425 -> dropped
            BasicPitchNote(start_s=0.060, end_s=0.500, midi=52, amplitude=0.70),  # 0.70 > 0.425 -> kept
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 52)   # 47 dropped

    def test_singleton_cluster_unaffected_by_relative_floor(self):
        """A cluster with one member has max_amp == its own amp, so the
        relative floor is trivially satisfied and the singleton survives."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.66),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40,)

    def test_chord_clusters_in_temporal_order(self):
        """Multiple chord clusters across the timeline emerge sorted by onset_s."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=2.0, end_s=3.0, midi=45, amplitude=0.70),
            BasicPitchNote(start_s=2.02, end_s=3.0, midi=57, amplitude=0.70),
            BasicPitchNote(start_s=0.0, end_s=1.0, midi=40, amplitude=0.80),
            BasicPitchNote(start_s=0.02, end_s=1.0, midi=52, amplitude=0.75),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 2
        assert events[0].midi_notes == (40, 52)
        assert events[0].onset_s == pytest.approx(0.0)
        assert events[1].midi_notes == (45, 57)
        assert events[1].onset_s == pytest.approx(2.0)
