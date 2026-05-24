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
from fractions import Fraction
from math import gcd

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
    """A NoteEvent after rhythm quantization, chord-capable.

    Pitch fields are tuples parallel to ``midi_notes``. Singleton tuples for
    monophonic notes, length-N tuples for chords, empty tuples for rests.

    Parameters
    ----------
    midi_notes : tuple[int, ...]
        MIDI notes, sorted ascending. Empty tuple for rests.
    frequencies_hz : tuple[float, ...]
        Parallel to midi_notes. Empty for rests.
    confidences : tuple[float, ...]
        Parallel to midi_notes. Empty for rests.
    cents_deviations : tuple[float, ...]
        Parallel to midi_notes. Empty for rests.
    onset_beat : float
        Beat position (quarter note = 1).
    duration_beats : float
        Duration in beats.
    duration_type : NoteDuration
        Standard music notation duration name.
    quarter_length : float
        music21 quarterLength.
    is_rest : bool
        True for inserted rest entries.
    """
    midi_notes: tuple[int, ...]
    frequencies_hz: tuple[float, ...]
    confidences: tuple[float, ...]
    cents_deviations: tuple[float, ...]

    onset_beat: float
    duration_beats: float
    duration_type: NoteDuration
    quarter_length: float

    is_rest: bool = False


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

        # Build the set of candidate durations (sorted ascending by duration)
        min_frac = _DURATION_FRACTIONS[smallest_duration]
        self._candidates: list[tuple[NoteDuration, float]] = sorted(
            (
                (dur, frac)
                for dur, frac in _SORTED_DURATIONS
                if frac >= min_frac - 1e-9 and (include_triplets or "triplet" not in dur.value)
            ),
            key=lambda dur_frac: dur_frac[1],
        )

        # Common grid: GCD of all candidate beat values. Every candidate
        # duration is an integer multiple of this grid, so snapping both
        # onsets and offsets to it guarantees the tiling invariant.
        # With triplets: GCD(1/4, 1/3) = 1/12 beat ≈ 0.0833.
        # Without triplets: GCD of regular beats = 1/4 beat = 0.25.
        beat_fracs = [Fraction(frac).limit_denominator(48) * 4 for _, frac in self._candidates]
        grid_frac = beat_fracs[0]
        for bf in beat_fracs[1:]:
            grid_frac = Fraction(gcd(grid_frac.numerator * bf.denominator,
                                     bf.numerator * grid_frac.denominator),
                                 grid_frac.denominator * bf.denominator)
        self._grid_beats: float = float(grid_frac)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, events: list[NoteEvent]) -> list[QuantizedNote]:
        """Quantize raw NoteEvents into grid-aligned QuantizedNotes.

        Phase 4 algorithm: snap both onset and offset to ``self._grid_beats``.
        Pick the largest catalog NoteDuration that fits within the snapped
        duration; the chosen catalog duration may be ≤ the raw snapped
        duration (note shortens slightly to fit) but never longer (no
        overlap). Rests fill the gap between the previous note's chosen
        end and the next note's snapped onset using the same rule.

        Tiling invariant (asserted by tests): for consecutive entries,
        ``prev.onset_beat + prev.duration_beats <= next.onset_beat``.

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
            snapped_onset = self._snap_to_grid(self._s_to_beat(event.onset_s))
            # Clamp: never place a note before the previous note has ended.
            # Round-to-nearest can push an onset backwards past prev_end_beat;
            # in that case advance to the next grid cell at or after prev_end_beat.
            if snapped_onset < prev_end_beat - 1e-9:
                snapped_onset = (
                    round(prev_end_beat / self._grid_beats) * self._grid_beats
                )

            snapped_offset = self._snap_to_grid(self._s_to_beat(event.offset_s))
            # Ensure at least one grid cell of duration
            if snapped_offset < snapped_onset + self._grid_beats:
                snapped_offset = snapped_onset + self._grid_beats

            snapped_dur_beats = snapped_offset - snapped_onset
            dur_type, dur_beats = self._largest_fitting_duration(snapped_dur_beats)
            # Note's chosen end is `snapped_onset + dur_beats`. This may be
            # ≤ snapped_offset (we shrink to a catalog value); the leftover
            # is absorbed into the rest after this note.

            note_end = snapped_onset + dur_beats

            # Insert rest if there's a gap from the previous note's end
            # to this note's snapped onset.
            gap = snapped_onset - prev_end_beat
            if gap >= self._grid_beats - 1e-9:
                rest = self._make_rest(prev_end_beat, gap)
                if rest is not None:
                    quantized.append(rest)

            quantized.append(QuantizedNote(
                midi_notes=event.midi_notes,
                frequencies_hz=event.frequencies_hz,
                confidences=event.confidences,
                cents_deviations=event.cents_deviations,
                onset_beat=snapped_onset,
                duration_beats=dur_beats,
                duration_type=dur_type,
                quarter_length=to_quarter_length(dur_type),
            ))

            prev_end_beat = note_end

        return quantized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _s_to_beat(self, time_s: float) -> float:
        """Convert seconds to beat position (quarter note = 1 beat)."""
        return time_s / self._beat_duration_s

    def _snap_to_grid(self, beat: float) -> float:
        """Snap a beat position to the nearest ``self._grid_beats`` multiple."""
        return round(beat / self._grid_beats) * self._grid_beats

    def _largest_fitting_duration(
        self, dur_beats: float
    ) -> tuple[NoteDuration, float]:
        """Return the largest catalog (NoteDuration, beat_value) that fits.

        "Fits" means ``catalog_beat_value <= dur_beats + 1e-9``. If no catalog
        value fits (i.e. ``dur_beats < smallest catalog``), returns the
        smallest catalog value — this is the only case where the chosen
        duration exceeds the requested duration (needed to avoid emitting
        a zero-duration note, which music21 rejects).
        """
        # _candidates is sorted ascending by fraction.
        # Walk descending to find the largest that fits.
        smallest = self._candidates[0]
        for dur, frac in reversed(self._candidates):
            beat_value = frac * 4.0
            if beat_value <= dur_beats + 1e-9:
                return dur, beat_value
        # Nothing fits — return smallest (must expand to avoid zero-duration)
        return smallest[0], smallest[1] * 4.0

    def _make_rest(self, start_beat: float, gap_beats: float) -> QuantizedNote | None:
        """Create a rest QuantizedNote for a gap between notes.

        Returns ``None`` if the gap is smaller than ``self._grid_beats``.
        The rest's chosen duration is the largest catalog value that fits
        the gap; any leftover (gap minus chosen duration) is absorbed as
        quantization noise — the next note's snapped onset is unchanged.
        """
        if gap_beats < self._grid_beats - 1e-9:
            return None
        dur_type, dur_beats = self._largest_fitting_duration(gap_beats)
        # If the smallest catalog value is larger than the gap, no rest fits —
        # absorb the gap as quantization noise rather than creating an overlap.
        if dur_beats > gap_beats + 1e-9:
            return None
        return QuantizedNote(
            midi_notes=(),
            frequencies_hz=(),
            confidences=(),
            cents_deviations=(),
            onset_beat=start_beat,
            duration_beats=dur_beats,
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
