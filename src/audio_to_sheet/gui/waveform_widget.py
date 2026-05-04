"""
gui/waveform_widget.py — Real-time waveform display widget.

Uses pyqtgraph's PlotWidget with an OpenGL-accelerated line plot for
low-latency waveform rendering. Updated from the audio callback thread
via a Qt signal (thread-safe).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class _Signaller(QObject):
    """Thin QObject to bridge the audio callback thread → Qt main thread."""
    new_block = pyqtSignal(object)   # emits np.ndarray


class WaveformWidget(QWidget):
    """
    Scrolling oscilloscope-style waveform display.

    Displays the last `window_s` seconds of audio at the configured sample rate.

    Parameters
    ----------
    sample_rate : int
    window_s : float
        Seconds of audio history to display.
    """

    def __init__(self, sample_rate: int, window_s: float = 2.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._window_samples = int(window_s * sample_rate)
        self._buffer = np.zeros(self._window_samples, dtype=np.float32)

        self._signaller = _Signaller()
        self._signaller.new_block.connect(self._on_new_block)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(antialias=True)
        self._plot_widget = pg.PlotWidget(background="#1a1a2e")
        self._plot_widget.setLabel("left", "Amplitude")
        self._plot_widget.setLabel("bottom", "Time (s)")
        self._plot_widget.setYRange(-1.0, 1.0)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMouseEnabled(x=False, y=False)

        x_axis = np.linspace(-self._window_samples / self._sample_rate, 0, self._window_samples)
        pen = pg.mkPen(color="#00d4ff", width=1)
        self._curve = self._plot_widget.plot(x_axis, self._buffer, pen=pen)

        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Audio callback interface (called from non-Qt thread)
    # ------------------------------------------------------------------

    def push_block(self, block: np.ndarray) -> None:
        """Thread-safe: emit signal to update waveform from audio callback."""
        self._signaller.new_block.emit(block)

    # ------------------------------------------------------------------
    # Qt slot (main thread)
    # ------------------------------------------------------------------

    def _on_new_block(self, block: np.ndarray) -> None:
        n = len(block)
        if n >= self._window_samples:
            self._buffer[:] = block[-self._window_samples:]
        else:
            self._buffer[:-n] = self._buffer[n:]
            self._buffer[-n:] = block

        self._curve.setData(self._buffer)

    def reset(self) -> None:
        """Clear the display."""
        self._buffer[:] = 0.0
        self._curve.setData(self._buffer)
