"""
export/midi_exporter.py — Exports a music21 Score to a MIDI file.

Supports both quantized (standard notation timing) and raw (float-second)
NoteEvent lists. The raw path bypasses music21's Score object entirely and
writes a MIDI file directly via the `mido` library for maximum timing fidelity.
"""

from __future__ import annotations

from pathlib import Path

import music21.stream

from improv_scribe.analysis.note_tracker import NoteEvent
from improv_scribe.config import AppConfig
from improv_scribe.quantization.tempo import TempoResult


class MIDIExporter:
    """
    Exports to MIDI in two modes:

    quantized_from_score(score, path)
        Writes a music21 Score to MIDI — preserves quantized rhythm grid.

    raw_from_events(events, tempo_result, path)
        Writes NoteEvents with original float-second timings using mido.
        Suitable for DAW import where exact timing is preferred over notation.

    Parameters
    ----------
    config : AppConfig
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Quantized path (via music21)
    # ------------------------------------------------------------------

    def quantized_from_score(
        self, score: music21.stream.Score, output_path: Path
    ) -> Path:
        """
        Write *score* to a MIDI file preserving quantized rhythm.

        Parameters
        ----------
        score : music21.stream.Score
        output_path : Path

        Returns
        -------
        Path — resolved MIDI output path
        """
        output_path = Path(output_path).with_suffix(".mid")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(output_path))
        return output_path

    # ------------------------------------------------------------------
    # Raw timing path (via mido)
    # ------------------------------------------------------------------

    def raw_from_events(
        self,
        events: list[NoteEvent],
        tempo_result: TempoResult,
        output_path: Path,
        channel: int = 0,
        velocity: int = 80,
    ) -> Path:
        """
        Write raw-timed NoteEvents directly to MIDI without quantization.

        Parameters
        ----------
        events : list[NoteEvent]
            Sorted by onset_s.
        tempo_result : TempoResult
            BPM used to set the MIDI tempo track.
        output_path : Path
        channel : int
            MIDI channel (0-indexed). 0 = piano; 9 = drums.
        velocity : int
            Note velocity (0–127).

        Returns
        -------
        Path — resolved MIDI output path
        """
        try:
            import mido  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Raw MIDI export requires `pip install mido`. "
                "Quantized export (quantized_from_score) works without it."
            ) from exc

        output_path = Path(output_path).with_suffix(".mid")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ticks_per_beat = 480
        bpm = tempo_result.bpm
        tempo_us = int(60_000_000 / bpm)   # microseconds per beat

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

        note_track = mido.MidiTrack()
        mid.tracks.append(note_track)

        # Build flat list of (time_s, msg_type, note) and sort by time.
        # Chord events: one note_on/note_off per chord member at the same tick.
        messages: list[tuple[float, str, int]] = []
        for event in events:
            for midi in event.midi_notes:
                messages.append((event.onset_s, "note_on", midi))
                messages.append((event.offset_s, "note_off", midi))
        messages.sort(key=lambda m: (m[0], 0 if m[1] == "note_on" else 1))

        def s_to_ticks(t: float) -> int:
            beats = t * bpm / 60.0
            return int(round(beats * ticks_per_beat))

        prev_ticks = 0
        for time_s, msg_type, note in messages:
            abs_ticks = s_to_ticks(time_s)
            delta = max(0, abs_ticks - prev_ticks)
            vel = velocity if msg_type == "note_on" else 0
            note_track.append(
                mido.Message(msg_type, channel=channel, note=note, velocity=vel, time=delta)
            )
            prev_ticks = abs_ticks

        mid.save(str(output_path))
        return output_path
