"""
gui/transport.py — Transport control bar (Record / Stop / Export).

Emits Qt signals for state transitions. The MainWindow connects these to
the audio pipeline and export pipeline.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from audio_to_sheet.analysis.instrument_profiles import Instrument
from audio_to_sheet.capture.audio_input import DeviceInfo


class TransportBar(QWidget):
    """
    Horizontal bar containing:
      - Device selector (QComboBox)
      - Instrument selector (Guitar / Bass)
      - Backend selector (pYIN / CREPE)
      - Rhythm mode selector (Auto-tempo / Raw)
      - Record button
      - Stop button
      - Export PDF button
      - Export MIDI button
      - Status label
    """

    # Signals
    record_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    export_pdf_requested = pyqtSignal()
    export_midi_requested = pyqtSignal()
    device_changed = pyqtSignal(int)           # device index
    instrument_changed = pyqtSignal(str)       # Instrument value string
    backend_changed = pyqtSignal(str)          # 'pyin' or 'crepe'
    rhythm_mode_changed = pyqtSignal(str)      # 'auto' or 'raw'

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # -- Device selector --
        layout.addWidget(QLabel("Input:"))
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(200)
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self._device_combo)

        layout.addSpacing(12)

        # -- Instrument --
        layout.addWidget(QLabel("Instrument:"))
        self._instrument_combo = QComboBox()
        self._instrument_combo.addItems(["Guitar", "Bass"])
        self._instrument_combo.currentTextChanged.connect(
            lambda t: self.instrument_changed.emit(t.lower())
        )
        layout.addWidget(self._instrument_combo)

        layout.addSpacing(12)

        # -- Backend --
        layout.addWidget(QLabel("Pitch:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["pYIN", "CREPE"])
        self._backend_combo.currentTextChanged.connect(
            lambda t: self.backend_changed.emit(t.lower())
        )
        layout.addWidget(self._backend_combo)

        layout.addSpacing(12)

        # -- Rhythm mode --
        layout.addWidget(QLabel("Rhythm:"))
        self._rhythm_combo = QComboBox()
        self._rhythm_combo.addItems(["Auto-tempo", "Raw"])
        self._rhythm_combo.currentTextChanged.connect(
            lambda t: self.rhythm_mode_changed.emit("auto" if "auto" in t.lower() else "raw")
        )
        layout.addWidget(self._rhythm_combo)

        layout.addStretch()

        # -- Transport buttons --
        self._record_btn = QPushButton("⏺  Record")
        self._record_btn.setStyleSheet("QPushButton { color: #ff4444; font-weight: bold; }")
        self._record_btn.clicked.connect(self.record_requested)
        layout.addWidget(self._record_btn)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(self._stop_btn)

        layout.addSpacing(8)

        self._pdf_btn = QPushButton("📄 Export PDF")
        self._pdf_btn.setEnabled(False)
        self._pdf_btn.clicked.connect(self.export_pdf_requested)
        layout.addWidget(self._pdf_btn)

        self._midi_btn = QPushButton("🎵 Export MIDI")
        self._midi_btn.setEnabled(False)
        self._midi_btn.clicked.connect(self.export_midi_requested)
        layout.addWidget(self._midi_btn)

        layout.addSpacing(8)

        self._status_label = QLabel("Ready")
        self._status_label.setMinimumWidth(180)
        layout.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Device list management
    # ------------------------------------------------------------------

    def populate_devices(self, devices: list[DeviceInfo]) -> None:
        """Populate device combo from a list of DeviceInfo objects."""
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._devices = devices
        for d in devices:
            self._device_combo.addItem(f"[{d.index}] {d.name}", userData=d.index)
        self._device_combo.blockSignals(False)
        if devices:
            self.device_changed.emit(devices[0].index)

    def _on_device_changed(self, combo_idx: int) -> None:
        if combo_idx >= 0 and combo_idx < self._device_combo.count():
            dev_index = self._device_combo.itemData(combo_idx)
            if dev_index is not None:
                self.device_changed.emit(int(dev_index))

    # ------------------------------------------------------------------
    # State transitions (called by MainWindow)
    # ------------------------------------------------------------------

    def set_recording(self, is_recording: bool) -> None:
        self._record_btn.setEnabled(not is_recording)
        self._stop_btn.setEnabled(is_recording)
        self._device_combo.setEnabled(not is_recording)
        self._instrument_combo.setEnabled(not is_recording)

    def set_has_result(self, has_result: bool) -> None:
        self._pdf_btn.setEnabled(has_result)
        self._midi_btn.setEnabled(has_result)

    def set_status(self, message: str) -> None:
        self._status_label.setText(message)

    # ------------------------------------------------------------------
    # Current selections (for MainWindow to read)
    # ------------------------------------------------------------------

    @property
    def selected_instrument(self) -> str:
        return self._instrument_combo.currentText().lower()

    @property
    def selected_backend(self) -> str:
        return self._backend_combo.currentText().lower()

    @property
    def selected_rhythm_mode(self) -> str:
        t = self._rhythm_combo.currentText().lower()
        return "auto" if "auto" in t else "raw"
