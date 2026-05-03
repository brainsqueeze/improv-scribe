"""
quantization/tempo.py — Tempo estimation from note onset times.

Uses librosa.beat.beat_track on a synthetic onset strength signal reconstructed
from NoteEvent onset times. This avoids re-running audio analysis but still
leverages librosa's battle-tested dynamic programming beat tracker.

Minimum reliable input: ~4 beats. Short clips (< 2 s) will produce unreliable BPM.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from audio_to_sheet.analysis.note_tracker import NoteEvent
from audio_to_sheet.config import AppConfig


@dataclass
class TempoResult:
    bpm: float
    beat_times_s: list[float]   # estimated beat grid positions
    confidence: float           # heuristic: lower STD of inter-onset intervals → higher


class TempoEstimator:
    """
    Estimates tempo from a list of NoteEvents.

    Two strategies (automatically selected):

    1. **librosa beat tracker** (preferred when ≥ 4 events):
       Reconstructs an impulse onset_envelope at the configured hop_length
       resolution, then runs librosa's beat tracker.

    2. **Inter-onset interval median** (fallback for very short clips):
       Computes median IOI and converts to BPM. Less reliable for compound
       rhythms but works when there are only 2–3 events.

    Parameters
    ----------
    config : AppConfig
    """

    MIN_EVENTS_FOR_BEAT_TRACK = 4

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def estimate(self, events: list[NoteEvent]) -> TempoResult:
        """
        Estimate tempo from note events.

        Parameters
        ----------
        events : list[NoteEvent]
            Must be sorted by onset_s. At least 2 events required.

        Returns
        -------
        TempoResult
        """
        if len(events) < 2:
            return TempoResult(bpm=120.0, beat_times_s=[], confidence=0.0)

        onset_times = np.array([e.onset_s for e in events])

        if len(events) >= self.MIN_EVENTS_FOR_BEAT_TRACK:
            return self._beat_track(onset_times)
        else:
            return self._ioi_median(onset_times)

    # ------------------------------------------------------------------
    # Private strategies
    # ------------------------------------------------------------------

    def _beat_track(self, onset_times: np.ndarray) -> TempoResult:
        import librosa  # lazy import

        sr = self._config.sample_rate
        hop = self._config.hop_length

        # Build sparse onset envelope from onset times
        duration_s = float(onset_times[-1]) + 1.0
        n_frames = int(np.ceil(duration_s * sr / hop))
        onset_env = np.zeros(n_frames, dtype=np.float32)
        for t in onset_times:
            frame_idx = int(round(t * sr / hop))
            if 0 <= frame_idx < n_frames:
                onset_env[frame_idx] = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tempo_arr, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop,
                bpm=None,
                start_bpm=120.0,
                tightness=100.0,
                trim=False,
            )

        bpm = float(np.atleast_1d(tempo_arr)[0])
        bpm = float(np.clip(bpm, self._config.tempo_min, self._config.tempo_max))

        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop).tolist()

        # Confidence: lower CV of inter-onset intervals → higher confidence
        ioi = np.diff(onset_times)
        cv = float(np.std(ioi) / (np.mean(ioi) + 1e-9))
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

        return TempoResult(bpm=bpm, beat_times_s=beat_times, confidence=confidence)

    def _ioi_median(self, onset_times: np.ndarray) -> TempoResult:
        ioi = np.diff(onset_times)
        median_ioi = float(np.median(ioi))
        bpm = float(np.clip(60.0 / median_ioi, self._config.tempo_min, self._config.tempo_max))
        return TempoResult(bpm=bpm, beat_times_s=[], confidence=0.3)
