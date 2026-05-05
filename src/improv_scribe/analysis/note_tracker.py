"""
analysis/note_tracker.py — Combines onset times + pitch frames into NoteEvents.

Strategy
--------
1. For each detected onset, find all voiced pitch frames that fall within
   the note's active region (from this onset to the next onset or silence).
2. Take the median f0 across those frames (robust to attack transient artefacts).
3. Convert Hz → MIDI note number (round to nearest semitone).
4. Emit a NoteEvent with onset_s, offset_s (= next onset or last voiced frame),
   midi_note, and mean confidence.

MIDI conversion
---------------
  midi = 69 + 12 * log2(f / 440)

This is rounded to the nearest integer. Pitch bend / microtonal deviation is
recorded as `cents_deviation` for future use (e.g. expressive MIDI export).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from improv_scribe.analysis.instrument_profiles import InstrumentProfile
from improv_scribe.analysis.onset import Onset
from improv_scribe.analysis.pitch import PitchResult
from improv_scribe.config import AppConfig

# ---------------------------------------------------------------------------
# NoteEvent — the central currency of the pipeline
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """
    A single detected note with timing and pitch.

    All times are in seconds relative to the start of the recorded session.
    """
    onset_s: float          # note start time
    offset_s: float         # note end time (next onset or last voiced frame)
    midi_note: int          # nearest semitone (0–127)
    frequency_hz: float     # median f0 across active frames
    confidence: float       # mean voiced-probability across active frames
    cents_deviation: float  # deviation from equal temperament (±50¢)

    # Future: list[int] for chord support
    # Future: velocity (from onset strength or RMS)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.offset_s - self.onset_s)

    def __repr__(self) -> str:
        return (
            f"NoteEvent(midi={self.midi_note}, "
            f"onset={self.onset_s:.3f}s, "
            f"dur={self.duration_s:.3f}s, "
            f"conf={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Hz ↔ MIDI helpers
# ---------------------------------------------------------------------------

def hz_to_midi(freq_hz: float) -> tuple[int, float]:
    """
    Convert frequency to (midi_note, cents_deviation).

    Returns
    -------
    midi_note : int
        Nearest MIDI note number.
    cents_deviation : float
        Signed deviation from equal temperament in cents (−50 to +50).
    """
    if freq_hz <= 0 or math.isnan(freq_hz):
        return 0, 0.0
    midi_float = 69.0 + 12.0 * math.log2(freq_hz / 440.0)
    midi_note = int(round(midi_float))
    cents = (midi_float - midi_note) * 100.0
    return midi_note, cents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maximum silence between two same-pitch events to treat as a single note.
# Spurious re-onsets caused by harmonic evolution appear within 600 ms;
# intentional repeated notes at ≥80 BPM have gaps ≥375 ms but are typically
# accompanied by a fresh attack, so we use a conservative 600 ms ceiling.
_MERGE_GAP_S: float = 0.600


def _merge_consecutive_same_pitch(events: list[NoteEvent]) -> list[NoteEvent]:
    """Merge back-to-back NoteEvents with identical pitch if the gap is small.

    Handles phantom re-onsets that appear when onset_detect fires on harmonic
    evolution of a sustained note (e.g. the B3 string triggering twice).
    """
    if not events:
        return events

    merged: list[NoteEvent] = [events[0]]
    for current in events[1:]:
        prev = merged[-1]
        gap = current.onset_s - prev.offset_s
        if current.midi_note == prev.midi_note and gap <= _MERGE_GAP_S:
            # Extend previous event to cover the full duration of both
            merged[-1] = NoteEvent(
                onset_s=prev.onset_s,
                offset_s=current.offset_s,
                midi_note=prev.midi_note,
                frequency_hz=(prev.frequency_hz + current.frequency_hz) / 2.0,
                confidence=(prev.confidence + current.confidence) / 2.0,
                cents_deviation=(prev.cents_deviation + current.cents_deviation) / 2.0,
            )
        else:
            merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# NoteTracker
# ---------------------------------------------------------------------------

class NoteTracker:
    """
    Assembles NoteEvent objects from onset + pitch data.

    Parameters
    ----------
    config : AppConfig
    profile : InstrumentProfile
        Used to validate MIDI range of detected notes.

    Usage
    -----
    Call process() after each analysis chunk. The tracker is stateless
    across chunks for MVP (batch analysis of a full recording session).
    For future real-time streaming, state would be carried across chunks.
    """

    def __init__(self, config: AppConfig, profile: InstrumentProfile) -> None:
        self._config = config
        self._profile = profile

    def process(
        self,
        pitch_result: PitchResult,
        onsets: list[Onset],
        chunk_offset_s: float = 0.0,
    ) -> list[NoteEvent]:
        """
        Produce NoteEvents from a chunk's pitch + onset data.

        Parameters
        ----------
        pitch_result : PitchResult
        onsets : list[Onset]
        chunk_offset_s : float
            Add this to all times so they are session-absolute, not chunk-relative.

        Returns
        -------
        list[NoteEvent]  — sorted by onset_s
        """
        if not onsets:
            return []

        voiced = pitch_result.voiced_frames
        if not voiced:
            return []

        events: list[NoteEvent] = []
        onset_times = [o.time_s for o in onsets]

        for i, onset in enumerate(onsets):
            t_start = onset.time_s
            t_end = onset_times[i + 1] if i + 1 < len(onset_times) else voiced[-1].time_s

            # Collect voiced frames within this note's time window
            active_frames = [
                f for f in voiced
                if t_start <= f.time_s < t_end
            ]

            if not active_frames:
                continue

            freqs = np.array([f.freq_hz for f in active_frames])
            freqs = freqs[np.isfinite(freqs)]
            if len(freqs) == 0:
                continue

            median_freq = float(np.median(freqs))
            mean_conf = float(np.mean([f.confidence for f in active_frames]))

            midi_note, cents_dev = hz_to_midi(median_freq)

            # Reject notes outside instrument range
            if not (self._profile.midi_min <= midi_note <= self._profile.midi_max):
                continue

            events.append(NoteEvent(
                onset_s=t_start + chunk_offset_s,
                offset_s=t_end + chunk_offset_s,
                midi_note=midi_note,
                frequency_hz=median_freq,
                confidence=mean_conf,
                cents_deviation=cents_dev,
            ))

        sorted_events = sorted(events, key=lambda e: e.onset_s)
        return _merge_consecutive_same_pitch(sorted_events)
