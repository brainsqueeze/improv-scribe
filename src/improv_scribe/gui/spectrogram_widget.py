"""
gui/spectrogram_widget.py — Scrolling CQT spectrogram widget.

Uses a Constant-Q Transform (CQT) for display because it produces a
perceptually linear pitch axis (each octave = fixed pixel height),
which is ideal for guitar/bass where low-frequency content is important.

CQT is computed on each incoming chunk (accumulated from audio blocks)
and the result is appended to a scrolling image buffer rendered by pyqtgraph.

Note: CQT is more expensive than STFT. Computation is done on a background
thread to avoid GUI jank. The image buffer is updated via Qt signal.
"""

from __future__ import annotations

import threading

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

# CQT parameters
N_BINS = 84          # 7 octaves * 12 bins/octave
BINS_PER_OCTAVE = 12
FMIN = 40.0          # ~E1 (bass low string)
HOP_LENGTH = 512
DISPLAY_COLS = 300   # number of time columns in the scrolling image


class _Signaller(QObject):
    new_column = pyqtSignal(object)   # emits np.ndarray of shape (N_BINS,)


class SpectrogramWidget(QWidget):
    """
    Scrolling CQT spectrogram, frequency on Y axis (FMIN at bottom).

    Parameters
    ----------
    sample_rate : int
    chunk_size : int
        Number of samples accumulated before computing a CQT column.
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_size: int = 4096,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

        self._accumulator: list[np.ndarray] = []
        self._accumulated_samples = 0

        self._image_data = np.zeros((N_BINS, DISPLAY_COLS), dtype=np.float32)

        self._signaller = _Signaller()
        self._signaller.new_column.connect(self._on_new_column)
        self._lock = threading.Lock()

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget(background="#1a1a2e")
        self._plot_widget.setLabel("left", "Pitch (MIDI-approx)")
        self._plot_widget.setLabel("bottom", "Time →")
        self._plot_widget.setMouseEnabled(x=False, y=False)

        self._img = pg.ImageItem()
        self._plot_widget.addItem(self._img)

        # Colormap: dark blue → cyan → yellow
        colormap = pg.colormap.get("CET-L9")
        self._img.setColorMap(colormap) # type: ignore

        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Audio callback interface
    # ------------------------------------------------------------------

    def push_block(self, block: np.ndarray) -> None:
        """Accumulate audio blocks; trigger CQT when chunk_size reached."""
        with self._lock:
            self._accumulator.append(block.copy())
            self._accumulated_samples += len(block)
            if self._accumulated_samples >= self._chunk_size:
                chunk = np.concatenate(self._accumulator)
                self._accumulator.clear()
                self._accumulated_samples = 0

        if self._accumulated_samples == 0:  # we just cleared it
            threading.Thread(
                target=self._compute_cqt_column,
                args=(chunk,),
                daemon=True,
            ).start()

    def _compute_cqt_column(self, chunk: np.ndarray) -> None:
        """Compute one CQT column and emit signal (runs in background thread)."""
        try:
            import librosa  # lazy import
            cqt = np.abs(
                librosa.cqt(
                    chunk,
                    sr=self._sample_rate,
                    hop_length=HOP_LENGTH,
                    fmin=FMIN,
                    n_bins=N_BINS,
                    bins_per_octave=BINS_PER_OCTAVE,
                )
            )
            # Collapse time axis to single column (mean across chunk)
            col = np.mean(cqt, axis=1).astype(np.float32)
            # Log-magnitude for display
            col = librosa.amplitude_to_db(col, ref=np.max) if np.max(col) > 0 else col
            self._signaller.new_column.emit(col)
        except Exception:  # noqa: BLE001
            pass  # silently skip bad chunks

    # ------------------------------------------------------------------
    # Qt slot (main thread)
    # ------------------------------------------------------------------

    def _on_new_column(self, col: np.ndarray) -> None:
        # Scroll left and append new column on right
        self._image_data[:, :-1] = self._image_data[:, 1:]
        self._image_data[:, -1] = col
        self._img.setImage(self._image_data.T, autoLevels=True)

    def reset(self) -> None:
        """Clear the spectrogram."""
        self._image_data[:] = 0.0
        self._img.setImage(self._image_data.T)
        with self._lock:
            self._accumulator.clear()
            self._accumulated_samples = 0
