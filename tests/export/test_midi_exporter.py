"""Unit tests for MIDIExporter raw chord support (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from improv_scribe.analysis.note_tracker import NoteEvent
from improv_scribe.config import AppConfig
from improv_scribe.export.midi_exporter import MIDIExporter
from improv_scribe.quantization.tempo import TempoResult


def test_raw_export_chord_emits_all_members(tmp_path: Path):
    """A chord NoteEvent with midi_notes=(60, 64, 67) must produce
    three note_on events at the same tick (and three note_offs)."""
    try:
        import mido  # noqa: PLC0415, F401
    except ImportError:
        pytest.skip("mido not installed")

    config = AppConfig()
    exporter = MIDIExporter(config)

    chord_event = NoteEvent(
        onset_s=0.0,
        offset_s=1.0,
        midi_notes=(60, 64, 67),   # C major triad
        frequencies_hz=(261.6, 329.6, 392.0),
        confidences=(0.8, 0.8, 0.8),
        cents_deviations=(0.0, 0.0, 0.0),
    )

    out_path = tmp_path / "out.mid"
    exporter.raw_from_events(
        [chord_event],
        TempoResult(bpm=120, beat_times_s=[], confidence=0.9),
        out_path,
    )

    import mido  # noqa: PLC0415
    mid = mido.MidiFile(str(out_path))
    # Collect note_on events with velocity > 0
    note_ons = [
        msg for track in mid.tracks for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ]
    notes_played = sorted(msg.note for msg in note_ons)
    assert notes_played == [60, 64, 67]


def test_raw_export_singleton_event_still_works(tmp_path: Path):
    """Regression guard: a singleton NoteEvent still produces exactly one note_on/note_off."""
    try:
        import mido  # noqa: PLC0415, F401
    except ImportError:
        pytest.skip("mido not installed")

    config = AppConfig()
    exporter = MIDIExporter(config)

    event = NoteEvent(
        onset_s=0.0,
        offset_s=1.0,
        midi_notes=(60,),
        frequencies_hz=(261.6,),
        confidences=(0.8,),
        cents_deviations=(0.0,),
    )

    out_path = tmp_path / "out.mid"
    exporter.raw_from_events(
        [event],
        TempoResult(bpm=120, beat_times_s=[], confidence=0.9),
        out_path,
    )

    import mido  # noqa: PLC0415
    mid = mido.MidiFile(str(out_path))
    note_ons = [
        msg for track in mid.tracks for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ]
    assert len(note_ons) == 1
    assert note_ons[0].note == 60
