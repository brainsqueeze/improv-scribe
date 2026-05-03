"""
capture/audio_input.py — Audio device management and streaming.

Wraps sounddevice for CoreAudio (macOS) access. Provides:
  - list_devices()       : human-readable device table
  - RingBuffer           : lock-free circular float32 buffer
  - AudioStream          : context-manager stream with callback injection

sounddevice is imported lazily inside functions that require it so that
RingBuffer and DeviceInfo are importable in test environments without
PortAudio installed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    host_api_name: str

    def __str__(self) -> str:
        return (
            f"[{self.index:2d}] {self.name!r:40s} "
            f"ch={self.max_input_channels}  "
            f"sr={int(self.default_sample_rate)}  "
            f"api={self.host_api_name}"
        )


def list_devices() -> list[DeviceInfo]:
    """Return all input-capable audio devices visible to sounddevice / CoreAudio."""
    import sounddevice as sd  # lazy: needs PortAudio
    raw = sd.query_devices()
    host_apis = sd.query_hostapis()
    devices: list[DeviceInfo] = []
    for idx, dev in enumerate(raw):
        if dev["max_input_channels"] < 1:
            continue
        api_name = host_apis[dev["hostapi"]]["name"]
        devices.append(
            DeviceInfo(
                index=idx,
                name=dev["name"],
                max_input_channels=dev["max_input_channels"],
                default_sample_rate=dev["default_samplerate"],
                host_api_name=api_name,
            )
        )
    return devices


def print_devices() -> None:
    """Pretty-print all input devices to stdout."""
    devices = list_devices()
    if not devices:
        print("No input devices found.")
        return
    print("Available input devices:")
    for d in devices:
        print(f"  {d}")


# ---------------------------------------------------------------------------
# Ring buffer  (no PortAudio dependency — always importable)
# ---------------------------------------------------------------------------

class RingBuffer:
    """
    Thread-safe circular buffer for float32 mono audio.

    Writer (audio callback) and reader (analysis thread) operate independently.
    The buffer never blocks the writer — oldest samples are overwritten when full.

    Parameters
    ----------
    capacity_samples : int
        Total buffer length in samples.
    """

    def __init__(self, capacity_samples: int) -> None:
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write_pos = 0
        self._n_written = 0          # total samples ever written (monotonic)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write interface (called from audio callback — keep allocation-free)
    # ------------------------------------------------------------------

    def write(self, data: np.ndarray) -> None:
        """Write *data* (1-D float32) into the ring, overwriting oldest if full."""
        n = len(data)
        with self._lock:
            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos : end] = data
            else:
                # wrap-around
                first = self._capacity - self._write_pos
                self._buf[self._write_pos :] = data[:first]
                self._buf[: n - first] = data[first:]
            self._write_pos = (self._write_pos + n) % self._capacity
            self._n_written += n

    # ------------------------------------------------------------------
    # Read interface (called from analysis thread)
    # ------------------------------------------------------------------

    def read_last(self, n_samples: int) -> np.ndarray:
        """Return a *copy* of the last *n_samples* written."""
        n_samples = min(n_samples, self._capacity)
        with self._lock:
            out = np.empty(n_samples, dtype=np.float32)
            end = self._write_pos
            start = end - n_samples
            if start >= 0:
                out[:] = self._buf[start:end]
            else:
                # wrap
                first = -start
                out[:first] = self._buf[self._capacity + start :]
                out[first:] = self._buf[:end]
            return out

    @property
    def n_written(self) -> int:
        """Total samples written since creation."""
        with self._lock:
            return self._n_written

    @property
    def capacity(self) -> int:
        return self._capacity


# ---------------------------------------------------------------------------
# Audio stream  (requires PortAudio via sounddevice)
# ---------------------------------------------------------------------------

# Callback type: receives (block: np.ndarray[float32, shape=(N,)]) -> None
AudioCallback = Callable[[np.ndarray], None]


class AudioStream:
    """
    Wraps a sounddevice InputStream for mono float32 capture.

    Supports multiple subscriber callbacks so the GUI waveform widget and the
    analysis pipeline can both consume the same stream independently.

    Usage
    -----
    >>> stream = AudioStream(config, device_index=2)
    >>> stream.add_callback(my_fn)
    >>> with stream:
    ...     time.sleep(5)   # record for 5 s
    """

    def __init__(self, config: Any, device_index: int | None = None) -> None:
        """
        Parameters
        ----------
        config : AppConfig
        device_index : int | None
            sounddevice device index. None = system default input.
        """
        self._config = config
        self._device_index = device_index
        self._callbacks: list[AudioCallback] = []
        self._ring = RingBuffer(
            int(config.sample_rate * config.ring_buffer_seconds)
        )
        self._stream: Any | None = None

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def add_callback(self, fn: AudioCallback) -> None:
        """Register a function to be called with each audio block."""
        self._callbacks.append(fn)

    def remove_callback(self, fn: AudioCallback) -> None:
        self._callbacks.remove(fn)

    # ------------------------------------------------------------------
    # Ring buffer access
    # ------------------------------------------------------------------

    @property
    def ring_buffer(self) -> RingBuffer:
        return self._ring

    # ------------------------------------------------------------------
    # Internal sounddevice callback
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: Any,
    ) -> None:
        if status:
            print(f"[AudioStream] sounddevice status: {status}")
        block = indata[:, 0].copy()   # take channel 0 → mono float32
        self._ring.write(block)
        for cb in self._callbacks:
            try:
                cb(block)
            except Exception as exc:  # noqa: BLE001
                print(f"[AudioStream] callback error: {exc}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd  # lazy: needs PortAudio
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            blocksize=self._config.block_size,
            device=self._device_index,
            channels=1,
            dtype="float32",
            callback=self._sd_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def __enter__(self) -> "AudioStream":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
