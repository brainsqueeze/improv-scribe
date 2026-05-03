"""
capture/noise_gate.py — Energy-threshold noise gate.

Prevents silent/noise-floor blocks from entering the pitch pipeline.
Uses a hold timer to avoid clipping sustain on quiet notes.
"""

from __future__ import annotations

import numpy as np

from audio_to_sheet.config import AppConfig


class NoiseGate:
    """
    Simple RMS-based noise gate with hold time.

    The gate is OPEN (passes audio) when the block RMS exceeds the threshold,
    or when the hold timer is still active from the previous open state.

    Parameters
    ----------
    config : AppConfig
    """

    def __init__(self, config: AppConfig) -> None:
        self._threshold = config.noise_gate_rms
        hold_samples = int(config.noise_gate_hold_ms * 1e-3 * config.sample_rate)
        self._hold_samples = hold_samples
        self._hold_counter = 0   # remaining hold samples

    def process(self, block: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Apply gate to a block of samples.

        Returns
        -------
        gated : np.ndarray
            The block (unmodified if open, zeroed if closed).
        is_open : bool
            True if the gate passed the signal.
        """
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms >= self._threshold:
            self._hold_counter = self._hold_samples
            return block, True

        if self._hold_counter > 0:
            self._hold_counter = max(0, self._hold_counter - len(block))
            return block, True

        return np.zeros_like(block), False

    def reset(self) -> None:
        """Reset hold counter — call when stopping a recording session."""
        self._hold_counter = 0
