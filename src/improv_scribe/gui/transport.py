"""
gui/transport.py — Transport control bar (Record / Stop / Export).

Emits Qt signals for state transitions. The MainWindow connects these to
the audio pipeline and export pipeline.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from improv_scribe.capture.audio_input import DeviceInfo
from improv_scribe.gui.flow_layout import FlowLayout
from improv_scribe.gui.theme import ACCENT_LIGHT, TEXT, make_icon, make_pixmap


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
    backend_changed = pyqtSignal(str)          # 'pyin' or 'crepe' or 'basic-pitch
    rhythm_mode_changed = pyqtSignal(str)      # 'auto' or 'raw'

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _make_field(
        self, label: str, icon_name: str, combo: QComboBox, min_width: int
    ) -> QFrame:
        """Wrap *combo* in a labelled card with a leading icon."""
        field = QFrame()
        field.setObjectName("Field")
        field.setMinimumWidth(min_width)
        row = QHBoxLayout(field)
        row.setContentsMargins(12, 6, 10, 6)
        row.setSpacing(10)

        icon_label = QLabel()
        icon_label.setPixmap(make_pixmap(icon_name, ACCENT_LIGHT, 18))
        icon_label.setFixedWidth(20)
        row.addWidget(icon_label)

        col = QVBoxLayout()
        col.setSpacing(0)
        caption = QLabel(label.upper())
        caption.setObjectName("FieldLabel")
        col.addWidget(caption)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(6)
        col.addWidget(combo)
        row.addLayout(col, 1)
        return field

    @staticmethod
    def _set_field_enabled(field: QFrame, enabled: bool) -> None:
        field.setProperty("disabledField", not enabled)
        field.style().unpolish(field)
        field.style().polish(field)

    def _setup_ui(self) -> None:
        # Wrapping layout: controls reflow onto extra rows on narrow windows.
        layout = FlowLayout(self, margin=20, h_spacing=10, v_spacing=10)

        # -- Device selector --
        self._device_combo = QComboBox()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._device_field = self._make_field("Input", "mic", self._device_combo, 220)
        self._device_combo.setMinimumContentsLength(22)  # device names run long
        layout.addWidget(self._device_field)

        # -- Instrument --
        self._instrument_combo = QComboBox()
        self._instrument_combo.addItems(["Guitar", "Bass"])
        self._instrument_combo.currentTextChanged.connect(
            lambda t: self.instrument_changed.emit(t.lower())
        )
        self._instrument_field = self._make_field(
            "Instrument", "guitar", self._instrument_combo, 130
        )
        layout.addWidget(self._instrument_field)

        # -- Backend --
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["CREPE", "Basic-pitch", "pYIN"])
        self._backend_combo.currentTextChanged.connect(
            lambda t: self.backend_changed.emit(t.lower())
        )
        layout.addWidget(self._make_field("Pitch", "wave", self._backend_combo, 140))

        # -- Rhythm mode --
        self._rhythm_combo = QComboBox()
        self._rhythm_combo.addItems(["Auto-tempo", "Raw"])
        self._rhythm_combo.currentTextChanged.connect(
            lambda t: self.rhythm_mode_changed.emit("auto" if "auto" in t.lower() else "raw")
        )
        layout.addWidget(self._make_field("Rhythm", "clock", self._rhythm_combo, 140))

        # -- Transport buttons (kept together so they wrap as a group) --
        actions = QWidget()
        actions_row = QHBoxLayout(actions)
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(10)

        self._record_btn = QPushButton("Record")
        self._record_btn.setObjectName("Record")
        self._record_btn.setIcon(make_icon("record", "#ffffff", 14))
        self._record_btn.clicked.connect(self.record_requested)
        actions_row.addWidget(self._record_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setIcon(make_icon("stop", TEXT, 16))
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested)
        actions_row.addWidget(self._stop_btn)

        self._pdf_btn = QPushButton("Export PDF")
        self._pdf_btn.setObjectName("Export")
        self._pdf_btn.setIcon(make_icon("pdf", TEXT, 18))
        self._pdf_btn.setEnabled(False)
        self._pdf_btn.clicked.connect(self.export_pdf_requested)
        actions_row.addWidget(self._pdf_btn)

        self._midi_btn = QPushButton("Export MIDI")
        self._midi_btn.setObjectName("Export")
        self._midi_btn.setIcon(make_icon("midi", TEXT, 18))
        self._midi_btn.setEnabled(False)
        self._midi_btn.clicked.connect(self.export_midi_requested)
        actions_row.addWidget(self._midi_btn)
        layout.addWidget(actions)

        # Retained for API compatibility; MainWindow reports status in the
        # status bar, so this label is not shown.
        self._status_label = QLabel("Ready")
        self._status_label.hide()

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
            self._device_combo.setItemData(
                self._device_combo.count() - 1, d.name, Qt.ItemDataRole.ToolTipRole
            )
        self._device_combo.blockSignals(False)
        if devices:
            self.device_changed.emit(devices[0].index)

    def _on_device_changed(self, combo_idx: int) -> None:
        self._device_combo.setToolTip(self._device_combo.itemText(combo_idx))
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
        self._set_field_enabled(self._device_field, not is_recording)
        self._set_field_enabled(self._instrument_field, not is_recording)

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

    def set_backend(self, backend: str) -> None:
        """Set the backend combo to display *backend* (internal name).

        Maps internal backend identifiers ('pyin', 'crepe', 'basic_pitch') to
        their display labels and updates the combo. Triggers
        ``currentTextChanged`` if the value differs from the current selection.
        """
        label_map = {"pyin": "pYIN", "crepe": "CREPE", "basic_pitch": "Basic-pitch"}
        label = label_map.get(backend.lower())
        if label is not None:
            self._backend_combo.setCurrentText(label)

    @property
    def selected_rhythm_mode(self) -> str:
        t = self._rhythm_combo.currentText().lower()
        return "auto" if "auto" in t else "raw"
