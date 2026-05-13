"""
analysis/note_tracker.py — Combines onset times + pitch frames into NoteEvents.

Strategy
--------
1. For each detected onset, find all voiced pitch frames that fall within
   the note's active region (from this onset to the next onset or silence).
2. Take the median f0 across those frames (robust to attack transient artefacts).
3. Convert Hz → MIDI note number (round to nearest semitone).
4. Emit a NoteEvent with onset_s, offset_s (= next onset or last voiced frame),
   midi_notes (singleton tuple in the monophonic case), and mean confidence.

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
from improv_scribe.analysis.pitch import BasicPitchNote, PitchResult
from improv_scribe.config import AppConfig

# ---------------------------------------------------------------------------
# NoteEvent — the central currency of the pipeline
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """A detected note or chord event with timing and pitch.

    All times are in seconds relative to the start of the recorded session.
    A monophonic detection emits singleton tuples (length 1). Chord
    detections emit tuples of length 2+, with `midi_notes` sorted ascending
    so chord identity is canonical.

    Phase 0 of the polyphonic migration introduces the tuple fields and
    keeps single-element back-compat properties for callers that have not
    yet been migrated. The properties will be removed in Phase 2.

    Parameters
    ----------
    onset_s : float
        Note start time.
    offset_s : float
        Note end time (next onset or last voiced frame).
    midi_notes : tuple[int, ...]
        MIDI note numbers (0–127), sorted ascending. Empty tuple for rests
        (rests are represented in QuantizedNote, not NoteEvent — NoteEvent
        always has at least one pitch).
    frequencies_hz : tuple[float, ...]
        Median f0 across active frames, parallel to midi_notes.
    confidences : tuple[float, ...]
        Mean voiced-probability across active frames, parallel to midi_notes.
    cents_deviations : tuple[float, ...]
        Deviation from equal temperament (-50 to +50), parallel to midi_notes.
    """
    onset_s: float
    offset_s: float
    midi_notes: tuple[int, ...]
    frequencies_hz: tuple[float, ...]
    confidences: tuple[float, ...]
    cents_deviations: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.offset_s - self.onset_s)

    @property
    def is_chord(self) -> bool:
        return len(self.midi_notes) > 1

    # ------------------------------------------------------------------
    # Back-compat shims — removed in Phase 2
    # ------------------------------------------------------------------

    @property
    def midi_note(self) -> int:
        """Lowest MIDI note. Back-compat shim — prefer `midi_notes[0]`."""
        return self.midi_notes[0]

    @property
    def frequency_hz(self) -> float:
        """First-pitch frequency. Back-compat shim — prefer `frequencies_hz[0]`."""
        return self.frequencies_hz[0]

    @property
    def confidence(self) -> float:
        """Mean confidence across chord members. Back-compat shim."""
        return sum(self.confidences) / len(self.confidences)

    @property
    def cents_deviation(self) -> float:
        """First-pitch cents deviation. Back-compat shim."""
        return self.cents_deviations[0]

    def __repr__(self) -> str:
        if self.is_chord:
            return (
                f"NoteEvent(midi={list(self.midi_notes)}, "
                f"onset={self.onset_s:.3f}s, "
                f"dur={self.duration_s:.3f}s, "
                f"conf={self.confidence:.2f})"
            )
        return (
            f"NoteEvent(midi={self.midi_notes[0]}, "
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
# Octave-error correction
# ---------------------------------------------------------------------------

# Fraction of CREPE frames in a note window that must be near the sub-octave
# for the frame-based correction to fire.
_SUBOCTAVE_FRAME_FRACTION: float = 0.15

# Frequency tolerance for "near sub-octave" check (±2 semitones ≈ ratio 1.122).
_SUBOCTAVE_SEMITONE_TOLERANCE: float = 2.0

# Minimum ratio of sub-octave spectral energy to detected-frequency energy for
# the spectral fallback to conclude the detected pitch is a harmonic error.
# CREPE on mic'd acoustic guitar can lock onto the 2nd harmonic (e.g. E3 instead
# of E2) with full confidence; in that case zero CREPE frames land near the true
# fundamental, so the frame-based check fails.  The spectral check looks directly
# at the raw audio: the true fundamental always has acoustic energy even when
# weaker than its harmonics, while a non-harmonic sub-octave (e.g. G2 when
# playing G3) will have near-zero energy.  10 % is conservative enough to avoid
# false corrections on correctly detected notes.
_SPECTRAL_SUBOCTAVE_RATIO: float = 0.20

# Tighter semitone tolerance used only by the spectral check (±1 semitone).
# The frame-based check uses _SUBOCTAVE_SEMITONE_TOLERANCE (±2 semitones) because
# CREPE frames can drift slightly from the true pitch.  The FFT bin for the
# correct fundamental is accurate, so ±1 semitone is sufficient and prevents
# adjacent open strings (spaced 5 semitones apart) from bleeding into the band.
# Example: G2 (98 Hz) ±1 st = [92.5, 103.8 Hz]; A2 (110 Hz) stays outside,
# while G2 ±2 st = [87.4, 110.2 Hz] accidentally includes A2.
_SPECTRAL_SEMITONE_TOLERANCE: float = 1.0

# Minimum RMS of the audio window before the spectral check is considered
# reliable.  Background noise in a near-silent window (e.g. the decaying tail
# of a sustained note) produces roughly equal spectral energy at all frequencies,
# making the sub-octave ratio meaningless.  0.01 matches the noise gate RMS floor
# used during live capture, so the spectral check only fires on windows where an
# active guitar signal is present.
_MIN_SIGNAL_RMS: float = 0.01


def _spectral_sub_octave_present(
    audio_window: np.ndarray,
    freq_hz: float,
    sample_rate: int,
) -> bool:
    """Return True if freq_hz/2 has notable acoustic energy in *audio_window*.

    Computes the magnitude spectrum and compares energy in a ±2-semitone band
    around freq_hz/2 vs freq_hz.  A ratio ≥ _SPECTRAL_SUBOCTAVE_RATIO indicates
    that freq_hz is likely a harmonic and the true fundamental is freq_hz/2.

    Parameters
    ----------
    audio_window : np.ndarray
        Raw audio samples for the note's onset window.
    freq_hz : float
        Frequency CREPE reported (potentially a harmonic error).
    sample_rate : int

    Returns
    -------
    bool
    """
    if len(audio_window) < 256:
        return False

    if float(np.sqrt(np.mean(audio_window ** 2))) < _MIN_SIGNAL_RMS:
        return False

    sub_hz = freq_hz / 2.0
    # Need at least 4 complete periods of sub_hz for reliable bin resolution.
    min_samples = max(256, int(4 * sample_rate / sub_hz))
    n_fft = 1
    while n_fft < min_samples:
        n_fft <<= 1
    n_fft = min(n_fft, len(audio_window))

    window_fn = np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(audio_window[:n_fft] * window_fn))
    freqs_arr = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    tol = 2.0 ** (_SPECTRAL_SEMITONE_TOLERANCE / 12.0)

    def band_energy(target_hz: float) -> float:
        mask = (freqs_arr >= target_hz / tol) & (freqs_arr <= target_hz * tol)
        return float(np.sum(spectrum[mask] ** 2))

    energy_f = band_energy(freq_hz)
    if energy_f == 0.0:
        return False

    return band_energy(sub_hz) / energy_f >= _SPECTRAL_SUBOCTAVE_RATIO


def _correct_octave_error(
    median_freq: float,
    active_frames_hz: np.ndarray,
    profile: InstrumentProfile,
    audio_window: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> float:
    """Return *median_freq / 2* when sub-octave evidence is present.

    CREPE often reports the 2nd harmonic instead of the fundamental on mic'd
    acoustic guitar (e.g. A3 instead of A2).  Two checks are applied in order:

    1. **Frame check** — if ≥15 % of voiced CREPE frames fall within ±2 semitones
       of the sub-octave, the fundamental was partially detected.
    2. **Spectral fallback** — if the frame check fails (CREPE was 100 % confident
       in the wrong octave), inspect the raw audio spectrum.  The true fundamental
       always has some acoustic energy; a non-harmonic sub-octave (e.g. G2 when
       playing G3) does not.

    Parameters
    ----------
    median_freq : float
        Median pitch across voiced frames for this note window (Hz).
    active_frames_hz : np.ndarray
        Array of per-frame frequencies from CREPE (finite values only).
    profile : InstrumentProfile
        Used to validate that the corrected frequency is within instrument range.
    audio_window : np.ndarray | None
        Raw audio samples for the onset window.  Required for the spectral
        fallback; if None only the frame check is applied.
    sample_rate : int | None
        Required when *audio_window* is provided.

    Returns
    -------
    float
        Corrected frequency (Hz).
    """
    sub_hz = median_freq / 2.0
    if not (profile.freq_min_hz <= sub_hz <= profile.freq_max_hz):
        return median_freq

    # Frame-based check: some CREPE frames are near the true fundamental.
    tol = 2.0 ** (_SUBOCTAVE_SEMITONE_TOLERANCE / 12.0)
    near_sub = np.sum(
        (active_frames_hz >= sub_hz / tol) & (active_frames_hz <= sub_hz * tol)
    )
    if near_sub / len(active_frames_hz) >= _SUBOCTAVE_FRAME_FRACTION:
        return sub_hz

    # Spectral fallback: CREPE was fully confident in the wrong octave; check
    # the raw audio for sub-octave acoustic energy.
    # Limited to notes strictly below profile.midi_min + 24 (two octaves above
    # the lowest open string).  On acoustic guitars, body resonance creates
    # enough sub-octave energy near the highest open string to trigger a false
    # correction — e.g. E3 body resonance while E4 is played on a mic'd guitar.
    if audio_window is not None and sample_rate is not None:
        detected_midi, _ = hz_to_midi(median_freq)
        if detected_midi < profile.midi_min + 24:
            if _spectral_sub_octave_present(audio_window, median_freq, sample_rate):
                return sub_hz

    return median_freq


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

# Maximum silence between two same-pitch single-note events to treat as one note.
# Spurious re-onsets caused by harmonic evolution appear within 600 ms;
# intentional repeated notes at >=80 BPM have gaps >=375 ms but are accompanied
# by a fresh attack, so we use a conservative 600 ms ceiling.
_MERGE_GAP_S: float = 0.600

# Tighter threshold for chord events. Eighth-note strums at 100 BPM are 300 ms
# apart and must NOT merge into one held chord.
_MERGE_GAP_CHORD_S: float = 0.200


def _avg_tuples(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    """Element-wise average of two parallel tuples.

    Pre-condition: len(a) == len(b). Used by the merge helper after the
    caller verifies tuple equality of midi_notes (which guarantees parallel
    structure).
    """
    return tuple((x + y) / 2.0 for x, y in zip(a, b, strict=True))


def _merge_consecutive_same_pitch(events: list[NoteEvent]) -> list[NoteEvent]:
    """Merge back-to-back NoteEvents whose midi_notes are identical.

    Handles phantom re-onsets caused by onset_detect firing on harmonic
    evolution of a sustained note (e.g. the B3 string triggering twice).

    The merge gap threshold differs by chord size:
    - Singleton (mono) events: 600 ms — covers harmonic-evolution
      false re-onsets on a decaying single note.
    - Chord events: 200 ms — tighter, because eighth-note repeated chords
      at 100 BPM are 300 ms apart and must NOT merge.
    """
    if not events:
        return events

    merged: list[NoteEvent] = [events[0]]
    for current in events[1:]:
        prev = merged[-1]
        gap = current.onset_s - prev.offset_s
        same_pitches = current.midi_notes == prev.midi_notes
        threshold = _MERGE_GAP_CHORD_S if prev.is_chord else _MERGE_GAP_S
        if same_pitches and gap <= threshold:
            merged[-1] = NoteEvent(
                onset_s=prev.onset_s,
                offset_s=current.offset_s,
                midi_notes=prev.midi_notes,
                frequencies_hz=_avg_tuples(prev.frequencies_hz, current.frequencies_hz),
                confidences=_avg_tuples(prev.confidences, current.confidences),
                cents_deviations=_avg_tuples(prev.cents_deviations, current.cents_deviations),
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
        audio: np.ndarray | None = None,
    ) -> list[NoteEvent]:
        """Produce NoteEvents from a chunk's pitch + onset data.

        Dispatches on whether the PitchResult carries basic-pitch's pre-assembled
        note events (`bp_notes`) or frame-level f0 data (`frames`). Phase 1
        basic-pitch path emits one singleton NoteEvent per BasicPitchNote — chord
        clustering will be added in Phase 2.

        Parameters
        ----------
        pitch_result : PitchResult
        onsets : list[Onset]
        chunk_offset_s : float
            Add this to all times so they are session-absolute, not chunk-relative.
        audio : np.ndarray | None
            Raw audio samples (full recording or chunk, 1-D float32).  When
            provided, enables the spectral octave-error fallback in addition to
            the frame-based check.  Strongly recommended for live mic recordings.

        Returns
        -------
        list[NoteEvent]
            Sorted by onset_s.
        """
        if pitch_result.bp_notes is not None:
            return self._process_basic_pitch(pitch_result.bp_notes, chunk_offset_s)
        return self._process_frame_based(pitch_result, onsets, chunk_offset_s, audio)

    def _process_frame_based(
        self,
        pitch_result: PitchResult,
        onsets: list[Onset],
        chunk_offset_s: float = 0.0,
        audio: np.ndarray | None = None,
    ) -> list[NoteEvent]:
        """Assemble NoteEvents from onset times + per-frame f0 data (pYIN/CREPE path).

        This is the original monophonic pipeline logic, preserved verbatim from
        before the basic-pitch dispatch was introduced.

        Parameters
        ----------
        pitch_result : PitchResult
        onsets : list[Onset]
        chunk_offset_s : float
            Add this to all times so they are session-absolute, not chunk-relative.
        audio : np.ndarray | None
            Raw audio samples for the spectral octave-error fallback.

        Returns
        -------
        list[NoteEvent]
            Sorted by onset_s.
        """
        if not onsets:
            return []

        voiced = pitch_result.voiced_frames
        if not voiced:
            return []

        sr = self._config.sample_rate
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

            # Extract raw audio window for spectral octave correction fallback.
            audio_window: np.ndarray | None = None
            if audio is not None:
                s0 = max(0, int(t_start * sr))
                s1 = min(len(audio), int(t_end * sr))
                if s1 > s0:
                    audio_window = audio[s0:s1]

            median_freq = float(np.median(freqs))
            median_freq = _correct_octave_error(
                median_freq, freqs, self._profile,
                audio_window=audio_window,
                sample_rate=sr,
            )
            mean_conf = float(np.mean([f.confidence for f in active_frames]))

            midi_note, cents_dev = hz_to_midi(median_freq)

            # Reject notes outside instrument range
            if not (self._profile.midi_min <= midi_note <= self._profile.midi_max):
                continue

            events.append(NoteEvent(
                onset_s=t_start + chunk_offset_s,
                offset_s=t_end + chunk_offset_s,
                midi_notes=(midi_note,),
                frequencies_hz=(median_freq,),
                confidences=(mean_conf,),
                cents_deviations=(cents_dev,),
            ))

        sorted_events = sorted(events, key=lambda e: e.onset_s)
        return _merge_consecutive_same_pitch(sorted_events)

    def _process_basic_pitch(
        self,
        bp_notes: list[BasicPitchNote],
        chunk_offset_s: float,
    ) -> list[NoteEvent]:
        """Convert basic-pitch's pre-assembled notes into singleton NoteEvents.

        Phase 1: one BasicPitchNote => one singleton NoteEvent. No onset
        clustering. No octave-error correction (basic-pitch already does its
        own polyphonic spectral analysis; layering the existing
        _correct_octave_error over its output is risky — see spec §3.2).

        Output is sorted by onset_s ascending. Same-pitch deduplication uses
        the existing _merge_consecutive_same_pitch helper, which on singleton
        chord-equality is behaviourally identical to pre-Phase-0 mono semantics.

        Parameters
        ----------
        bp_notes : list[BasicPitchNote]
            Pre-assembled note events from basic-pitch's predict() output.
        chunk_offset_s : float
            Add this to all times so they are session-absolute, not chunk-relative.

        Returns
        -------
        list[NoteEvent]
            Sorted by onset_s.
        """
        events: list[NoteEvent] = []
        for bp in bp_notes:
            events.append(NoteEvent(
                onset_s=bp.start_s + chunk_offset_s,
                offset_s=bp.end_s + chunk_offset_s,
                midi_notes=(bp.midi,),
                # MIDI -> 440-tuned frequency: 440 * 2**((midi - 69)/12)
                frequencies_hz=(440.0 * 2.0 ** ((bp.midi - 69) / 12.0),),
                confidences=(bp.amplitude,),
                cents_deviations=(0.0,),
            ))

        sorted_events = sorted(events, key=lambda e: e.onset_s)
        return _merge_consecutive_same_pitch(sorted_events)
