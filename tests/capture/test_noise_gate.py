"""
tests/capture/test_noise_gate.py

Tests for NoiseGate — verifies threshold gating and hold-time behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from improv_scribe.capture.noise_gate import NoiseGate
from improv_scribe.config import AppConfig


@pytest.fixture
def config() -> AppConfig:
    cfg = AppConfig()
    cfg.noise_gate_rms = 0.01
    cfg.noise_gate_hold_ms = 80.0
    return cfg


def _loud_block(n: int = 512, amp: float = 0.5) -> np.ndarray:
    """Block whose RMS is well above the gate threshold."""
    return np.full(n, amp, dtype=np.float32)


def _quiet_block(n: int = 512, amp: float = 0.001) -> np.ndarray:
    """Block whose RMS is well below the gate threshold."""
    return np.full(n, amp, dtype=np.float32)


class TestNoiseGate:
    def test_loud_block_passes(self, config):
        gate = NoiseGate(config)
        block = _loud_block()
        gated, is_open = gate.process(block)
        assert is_open is True
        np.testing.assert_array_equal(gated, block)

    def test_quiet_block_blocked(self, config):
        gate = NoiseGate(config)
        block = _quiet_block()
        gated, is_open = gate.process(block)
        assert is_open is False
        assert np.all(gated == 0.0)

    def test_hold_keeps_gate_open_after_loud(self, config):
        """After a loud block, quiet blocks within hold time should still pass."""
        gate = NoiseGate(config)
        gate.process(_loud_block())   # opens gate and starts hold timer
        _, is_open = gate.process(_quiet_block())
        assert is_open is True

    def test_hold_expires_after_enough_quiet_blocks(self, config):
        """After enough quiet blocks exceeding hold duration, gate should close."""
        gate = NoiseGate(config)
        gate.process(_loud_block())

        # Drain hold timer — each block = 512 samples @ 44100 Hz ≈ 11.6ms
        # Hold = 80ms → need ~7 blocks
        hold_samples = int(config.noise_gate_hold_ms * 1e-3 * config.sample_rate)
        block_size = 512
        n_drain = (hold_samples // block_size) + 2  # +2 for margin

        is_open = True
        for _ in range(n_drain):
            _, is_open = gate.process(_quiet_block(block_size))

        assert is_open is False

    def test_reset_closes_hold(self, config):
        """After reset(), quiet blocks should be gated immediately."""
        gate = NoiseGate(config)
        gate.process(_loud_block())   # open + start hold
        gate.reset()
        _, is_open = gate.process(_quiet_block())
        assert is_open is False

    def test_rms_at_threshold(self, config):
        """Block whose RMS is at or above threshold should pass.

        Note: a constant float32 block set to exactly `threshold` may have
        RMS fractionally below due to float32 representation of 0.01.
        We therefore test with a value guaranteed to be >= threshold after
        float32 round-trip, i.e. threshold * (1 + epsilon).
        """
        gate = NoiseGate(config)
        threshold = config.noise_gate_rms
        # Use a value that is unambiguously above threshold after fp32 rounding
        value = np.float32(threshold * 1.01)
        block = np.full(512, value, dtype=np.float32)
        rms = float(np.sqrt(np.mean(block ** 2)))
        assert rms >= threshold, "Precondition: test block RMS must be >= threshold"
        _, is_open = gate.process(block)
        assert is_open is True
