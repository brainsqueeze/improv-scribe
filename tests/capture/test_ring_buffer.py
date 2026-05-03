"""
tests/capture/test_ring_buffer.py

Tests for RingBuffer — verifies correct FIFO/circular behaviour, thread safety,
and correct output from read_last() under various write/read patterns.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from audio_to_sheet.capture.audio_input import RingBuffer


class TestRingBuffer:
    def test_initial_state_is_zeros(self):
        buf = RingBuffer(capacity_samples=1024)
        data = buf.read_last(1024)
        assert np.all(data == 0.0)

    def test_write_and_read_exact_capacity(self):
        buf = RingBuffer(capacity_samples=8)
        data = np.arange(8, dtype=np.float32)
        buf.write(data)
        result = buf.read_last(8)
        np.testing.assert_array_almost_equal(result, data)

    def test_read_fewer_than_written(self):
        buf = RingBuffer(capacity_samples=16)
        data = np.arange(16, dtype=np.float32)
        buf.write(data)
        result = buf.read_last(4)
        # Should return the last 4 elements
        np.testing.assert_array_almost_equal(result, data[-4:])

    def test_wrap_around_overwrites_oldest(self):
        """Writing more than capacity should wrap and overwrite oldest samples."""
        buf = RingBuffer(capacity_samples=8)
        first = np.ones(8, dtype=np.float32)
        second = np.full(8, 2.0, dtype=np.float32)
        buf.write(first)
        buf.write(second)
        result = buf.read_last(8)
        np.testing.assert_array_almost_equal(result, second)

    def test_partial_write_then_read(self):
        buf = RingBuffer(capacity_samples=16)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf.write(data)
        result = buf.read_last(4)
        np.testing.assert_array_almost_equal(result, data)

    def test_multiple_small_writes(self):
        buf = RingBuffer(capacity_samples=16)
        for i in range(4):
            buf.write(np.array([float(i)], dtype=np.float32))
        result = buf.read_last(4)
        expected = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_n_written_monotonically_increases(self):
        buf = RingBuffer(capacity_samples=32)
        assert buf.n_written == 0
        buf.write(np.ones(10, dtype=np.float32))
        assert buf.n_written == 10
        buf.write(np.ones(5, dtype=np.float32))
        assert buf.n_written == 15

    def test_capacity_property(self):
        buf = RingBuffer(capacity_samples=512)
        assert buf.capacity == 512

    def test_read_more_than_capacity_clamped(self):
        """read_last(n > capacity) should return at most capacity samples."""
        buf = RingBuffer(capacity_samples=8)
        buf.write(np.ones(8, dtype=np.float32))
        result = buf.read_last(1000)
        assert len(result) == 8

    def test_thread_safety_concurrent_writes(self):
        """Multiple writer threads should not corrupt the buffer."""
        buf = RingBuffer(capacity_samples=4096)
        errors = []

        def writer(value: float) -> None:
            try:
                for _ in range(100):
                    buf.write(np.full(16, value, dtype=np.float32))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(float(i),)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # Buffer should be readable without exception
        result = buf.read_last(256)
        assert len(result) == 256

    def test_read_returns_copy(self):
        """Modifying returned array must not affect internal buffer."""
        buf = RingBuffer(capacity_samples=8)
        buf.write(np.ones(8, dtype=np.float32))
        result = buf.read_last(8)
        result[:] = 99.0
        result2 = buf.read_last(8)
        assert np.all(result2 == 1.0)
