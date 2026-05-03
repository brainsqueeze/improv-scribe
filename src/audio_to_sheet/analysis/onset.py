"""
analysis/onset.py — Note onset detection.

Uses librosa's onset detection pipeline (spectral flux + peak picking) to
find the start times of individual notes. Onset times are later paired with
pitch frames by note_tracker.py to form NoteEvents.

Onset detection is separate from pitch detection because:
  1. The attack transient of a plucked string often has an ambiguous pitch.
  2. We want onset times at higher temporal precision than pitch frames.
  3. They can be run in parallel if needed in a future async pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_to_sheet.config import AppConfig


@dataclass
class Onset:
    """A single detected onset."""
    time_s: float       # onset time in seconds (from chunk start)
    strength: float     # onset strength at this time (arbitrary units)


class OnsetDetector:
    """
    Wraps librosa's onset detection with guitar/bass-tuned defaults.

    Parameters
    ----------
    config : AppConfig

    Notes
    -----
    `librosa.onset.onset_detect` uses the spectral flux of a mel-spectrogram
    by default, which works well for transient-rich plucked string instruments.

    `backtrack=True` snaps detected peaks back to the nearest energy trough,
    giving more accurate physical onset positions.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def detect(self, audio: np.ndarray) -> list[Onset]:
        """
        Detect note onsets in *audio* (1-D float32, mono).

        Returns
        -------
        list[Onset]
            Sorted by time_s ascending.
        """
        import librosa  # lazy import

        sr = self._config.sample_rate
        hop = self._config.hop_length

        # onset_strength returns the novelty function
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=hop,
        )

        # Peak-pick with backtrack to physical onset position
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop,
            backtrack=True,
            units="frames",
        )

        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

        onsets: list[Onset] = []
        for t, frame in zip(onset_times, onset_frames, strict=True):
            strength = float(onset_env[frame]) if frame < len(onset_env) else 0.0
            onsets.append(Onset(time_s=float(t), strength=strength))

        return onsets
