"""
quantization/grid.py — Rhythm quantization and duration grid-snapping.

Given a set of NoteEvents with raw float onset/offset times (seconds) and a
tempo estimate (BPM), this module:

1. Converts all times to beat units (quarter notes = 1 beat).
2. Snaps onsets and offsets to the nearest subdivision grid point.
3. Assigns a standard music notation duration (whole, half, quarter, 8th, 16th).
4. Inserts rests where the gap between notes exceeds a threshold.
5. Optionally passes through raw (unquantized) NoteEvents for MIDI export.

Duration model
--------------
We define durations as fractions of a whole note. Standard values:

    whole  = 1.0
    half   = 0.5
    quarter = 0.25
    8th    = 0.125
    16th   = 0.0625
    triplet-quarter = 1/6  ≈ 0.1667
    triplet-8th     = 1/12 ≈ 0.0833

music21 uses its own quarterLength system (quarter note = 1.0). We convert:
    music21_quarterLength = duration_whole_fraction * 4.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from improv_scribe.analysis.note_tracker import NoteEvent
from improv_scribe.quantization.tempo import TempoResult

# ---------------------------------------------------------------------------
# Duration catalogue
# ---------------------------------------------------------------------------

class NoteDuration(StrEnum):
    """Standard music notation durations, named for display."""
    WHOLE      = "whole"
    HALF       = "half"
    DOTTED_HALF = "dotted-half"
    QUARTER    = "quarter"
    DOTTED_QUARTER = "dotted-quarter"
    EIGHTH     = "eighth"
    DOTTED_EIGHTH = "dotted-eighth"
    SIXTEENTH  = "sixteenth"
    TRIPLET_QUARTER = "triplet-quarter"
    TRIPLET_EIGHTH  = "triplet-eighth"


# Fraction of a whole note for each duration
_DURATION_FRACTIONS: dict[NoteDuration, float] = {
    NoteDuration.WHOLE:           1.0,
    NoteDuration.HALF:            0.5,
    NoteDuration.DOTTED_HALF:     0.75,
    NoteDuration.QUARTER:         0.25,
    NoteDuration.DOTTED_QUARTER:  0.375,
    NoteDuration.EIGHTH:          0.125,
    NoteDuration.DOTTED_EIGHTH:   0.1875,
    NoteDuration.SIXTEENTH:       0.0625,
    NoteDuration.TRIPLET_QUARTER: 1.0 / 6.0,
    NoteDuration.TRIPLET_EIGHTH:  1.0 / 12.0,
}

# music21 quarterLength = whole_fraction * 4
def to_quarter_length(duration: NoteDuration) -> float:
    return _DURATION_FRACTIONS[duration] * 4.0


@dataclass
class QuantizedNote:
    """A NoteEvent after rhythm quantization."""
    midi_note: int
    frequency_hz: float
    confidence: float
    cents_deviation: float

    onset_beat: float        # beat position (quarter note = 1)
    duration_beats: float    # duration in beats

    duration_type: NoteDuration
    quarter_length: float    # music21 quarterLength

    is_rest: bool = False    # True for inserted rest notes


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

_SORTED_DURATIONS = sorted(
    _DURATION_FRACTIONS.items(), key=lambda kv: kv[1], reverse=True
)


class RhythmQuantizer:
    """
    Snaps NoteEvents to a rhythmic grid derived from a TempoResult.

    Parameters
    ----------
    tempo_result : TempoResult
    time_signature : tuple[int, int]
        Numerator, denominator. Default (4, 4).
    include_triplets : bool
        Whether to include triplet durations in the grid (default True).
    smallest_duration : NoteDuration
        Finest grid subdivision. Default = SIXTEENTH.
    """

    def __init__(
        self,
        tempo_result: TempoResult,
        time_signature: tuple[int, int] = (4, 4),
        include_triplets: bool = True,
        smallest_duration: NoteDuration = NoteDuration.SIXTEENTH,
    ) -> None:
        self._bpm = tempo_result.bpm
        self._time_sig = time_signature
        self._beat_duration_s = 60.0 / self._bpm  # seconds per quarter note

        # Build the set of candidate durations
        min_frac = _DURATION_FRACTIONS[smallest_duration]
        self._candidates: list[tuple[NoteDuration, float]] = [
            (dur, frac)
            for dur, frac in _SORTED_DURATIONS
            if frac >= min_frac - 1e-9 and (include_triplets or "triplet" not in dur.value)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, events: list[NoteEvent]) -> list[QuantizedNote]:
        """
        Quantize raw NoteEvents into grid-aligned QuantizedNotes.

        Parameters
        ----------
        events : list[NoteEvent]
            Sorted by onset_s.

        Returns
        -------
        list[QuantizedNote]
            Notes and rests in score order.
        """
        if not events:
            return []

        quantized: list[QuantizedNote] = []
        prev_end_beat = 0.0

        for event in events:
            onset_beat = self._s_to_beat(event.onset_s)
            offset_beat = self._s_to_beat(event.offset_s)
            raw_dur_beats = max(offset_beat - onset_beat, self._min_dur_beats())

            # Snap onset to nearest grid point
            snapped_onset = self._snap_to_grid(onset_beat)
            snapped_dur = self._snap_duration(raw_dur_beats)
            dur_type, dur_frac = snapped_dur
            dur_beats = dur_frac * 4.0   # whole note = 4 beats

            # Insert rest if gap from previous note
            gap = snapped_onset - prev_end_beat
            if gap > self._min_dur_beats() * 0.5:
                rest = self._make_rest(prev_end_beat, gap)
                if rest is not None:
                    quantized.append(rest)

            quantized.append(QuantizedNote(
                midi_note=event.midi_note,
                frequency_hz=event.frequency_hz,
                confidence=event.confidence,
                cents_deviation=event.cents_deviation,
                onset_beat=snapped_onset,
                duration_beats=dur_beats,
                duration_type=dur_type,
                quarter_length=to_quarter_length(dur_type),
            ))

            prev_end_beat = snapped_onset + dur_beats

        return quantized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _s_to_beat(self, time_s: float) -> float:
        """Convert seconds to beat position (quarter note = 1 beat)."""
        return time_s / self._beat_duration_s

    def _min_dur_beats(self) -> float:
        """Minimum duration in beats = smallest candidate duration * 4."""
        min_frac = min(frac for _, frac in self._candidates)
        return min_frac * 4.0

    def _snap_to_grid(self, beat: float) -> float:
        """Snap a beat position to the nearest grid point."""
        grid_size = self._min_dur_beats()
        return round(beat / grid_size) * grid_size

    def _snap_duration(self, raw_beats: float) -> tuple[NoteDuration, float]:
        """Find the closest standard duration to raw_beats."""
        raw_frac = raw_beats / 4.0   # convert beats → whole-note fraction
        best_dur, best_frac, best_dist = self._candidates[0][0], self._candidates[0][1], np.inf
        for dur, frac in self._candidates:
            dist = abs(frac - raw_frac)
            if dist < best_dist:
                best_dur, best_frac, best_dist = dur, frac, dist
        return best_dur, best_frac

    def _make_rest(self, start_beat: float, gap_beats: float) -> QuantizedNote | None:
        """Create a rest QuantizedNote for a gap between notes."""
        dur_type, dur_frac = self._snap_duration(gap_beats / 4.0 * 4.0)
        if _DURATION_FRACTIONS[dur_type] < self._min_dur_beats() / 4.0 - 1e-9:
            return None
        return QuantizedNote(
            midi_note=0,
            frequency_hz=0.0,
            confidence=1.0,
            cents_deviation=0.0,
            onset_beat=start_beat,
            duration_beats=_DURATION_FRACTIONS[dur_type] * 4.0,
            duration_type=dur_type,
            quarter_length=to_quarter_length(dur_type),
            is_rest=True,
        )

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def time_signature(self) -> tuple[int, int]:
        return self._time_sig
