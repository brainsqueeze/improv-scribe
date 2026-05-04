"""
gui/main_window.py — Top-level application window.

Layout
------
┌────────────────────────────────────────────────────────┐
│  TransportBar (device, instrument, record/stop/export) │
├────────────────────────────────────────────────────────┤
│  WaveformWidget (top half)                             │
├────────────────────────────────────────────────────────┤
│  SpectrogramWidget (bottom half)                       │
├────────────────────────────────────────────────────────┤
│  Status bar                                            │
└────────────────────────────────────────────────────────┘

Pipeline ownership
------------------
MainWindow owns and coordinates:
  - AudioStream        (capture)
  - NoiseGate          (capture)
  - PitchEstimator     (analysis)
  - OnsetDetector      (analysis)
  - NoteTracker        (analysis)
  - TempoEstimator     (quantization)
  - RhythmQuantizer    (quantization)
  - ScoreBuilder       (notation)
  - PDFExporter        (export)
  - MIDIExporter       (export)

Recording state machine
-----------------------
IDLE → RECORDING → PROCESSING → DONE → IDLE
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.note_tracker import NoteTracker
from improv_scribe.analysis.onset import OnsetDetector
from improv_scribe.analysis.pitch import PitchEstimator
from improv_scribe.capture.audio_input import AudioStream, list_devices
from improv_scribe.capture.noise_gate import NoiseGate
from improv_scribe.config import AppConfig
from improv_scribe.export.midi_exporter import MIDIExporter
from improv_scribe.export.pdf_exporter import PDFExporter
from improv_scribe.gui.spectrogram_widget import SpectrogramWidget
from improv_scribe.gui.transport import TransportBar
from improv_scribe.gui.waveform_widget import WaveformWidget
from improv_scribe.notation.score_builder import ScoreBuilder
from improv_scribe.quantization.grid import RhythmQuantizer
from improv_scribe.quantization.tempo import TempoEstimator


class _PipelineSignaller(QObject):
    """Signals emitted by the background processing thread → main thread."""
    processing_done = pyqtSignal(object, object)   # (score, events)
    processing_failed = pyqtSignal(str)


class MainWindow(QMainWindow):
    """
    Primary application window.

    Parameters
    ----------
    config : AppConfig
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._stream: AudioStream | None = None
        self._noise_gate = NoiseGate(config)
        self._recorded_blocks: list[np.ndarray] = []
        self._is_recording = False
        self._last_score = None
        self._last_events = None
        self._last_quantized_notes = None
        self._last_tab_assignments = None
        self._last_profile: object = None
        self._current_device_index: int | None = None
        self._current_instrument = Instrument.GUITAR
        self._rhythm_mode = "auto"

        self._signaller = _PipelineSignaller()
        self._signaller.processing_done.connect(self._on_processing_done)
        self._signaller.processing_failed.connect(self._on_processing_failed)

        self._setup_ui()
        self._populate_devices()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("Audio → Sheet Music")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Transport bar
        self._transport = TransportBar()
        self._transport.record_requested.connect(self._on_record)
        self._transport.stop_requested.connect(self._on_stop)
        self._transport.export_pdf_requested.connect(self._on_export_pdf)
        self._transport.export_midi_requested.connect(self._on_export_midi)
        self._transport.device_changed.connect(self._on_device_changed)
        self._transport.instrument_changed.connect(self._on_instrument_changed)
        self._transport.backend_changed.connect(self._on_backend_changed)
        self._transport.rhythm_mode_changed.connect(self._on_rhythm_mode_changed)
        root_layout.addWidget(self._transport)

        # Waveform + spectrogram in a vertical splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._waveform = WaveformWidget(sample_rate=self._config.sample_rate)
        splitter.addWidget(self._waveform)

        self._spectrogram = SpectrogramWidget(sample_rate=self._config.sample_rate)
        splitter.addWidget(self._spectrogram)

        splitter.setSizes([250, 350])
        root_layout.addWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — select a device and press Record.")

    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------

    def _populate_devices(self) -> None:
        devices = list_devices()
        self._transport.populate_devices(devices)

    def _on_device_changed(self, index: int) -> None:
        self._current_device_index = index

    def _on_instrument_changed(self, instrument_str: str) -> None:
        self._current_instrument = Instrument(instrument_str)
        profile = get_profile(self._current_instrument)
        # Update noise gate threshold if instrument provides an override
        if profile.noise_gate_rms_override is not None:
            self._noise_gate = NoiseGate.__new__(NoiseGate)
            self._noise_gate._threshold = profile.noise_gate_rms_override
            self._noise_gate._hold_samples = int(
                self._config.noise_gate_hold_ms * 1e-3 * self._config.sample_rate
            )
            self._noise_gate._hold_counter = 0

    def _on_backend_changed(self, backend: str) -> None:
        self._config.pitch_backend = backend.replace("pyin", "pyin").replace("crepe", "crepe")

    def _on_rhythm_mode_changed(self, mode: str) -> None:
        self._rhythm_mode = mode

    # ------------------------------------------------------------------
    # Transport controls
    # ------------------------------------------------------------------

    def _on_record(self) -> None:
        self._recorded_blocks.clear()
        self._noise_gate.reset()
        self._waveform.reset()
        self._spectrogram.reset()

        self._stream = AudioStream(self._config, device_index=self._current_device_index)
        self._stream.add_callback(self._audio_callback)
        self._stream.start()

        self._is_recording = True
        self._transport.set_recording(True)
        self._transport.set_has_result(False)
        self._status_bar.showMessage("● Recording…")

    def _on_stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream = None

        self._is_recording = False
        self._transport.set_recording(False)
        self._status_bar.showMessage("Processing…")

        # Run analysis pipeline in background thread
        blocks_copy = list(self._recorded_blocks)
        instrument = self._current_instrument
        rhythm_mode = self._rhythm_mode
        threading.Thread(
            target=self._run_pipeline,
            args=(blocks_copy, instrument, rhythm_mode),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Audio callback (runs in sounddevice thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, block: np.ndarray) -> None:
        gated, is_open = self._noise_gate.process(block)
        if is_open:
            self._recorded_blocks.append(gated.copy())

        # Always push to display widgets (raw, not gated)
        self._waveform.push_block(block)
        self._spectrogram.push_block(block)

    # ------------------------------------------------------------------
    # Background analysis pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        blocks: list[np.ndarray],
        instrument: Instrument,
        rhythm_mode: str,
    ) -> None:
        """Runs in a background thread. Emits signal when done."""
        try:
            if not blocks:
                self._signaller.processing_failed.emit("No audio recorded.")
                return

            audio = np.concatenate(blocks).astype(np.float32)
            profile = get_profile(instrument)
            config = self._config

            # 1. Pitch estimation
            estimator = PitchEstimator(config)
            pitch_result = estimator.estimate(audio, profile)
            if config.debug_pitch:
                estimator.flush_debug_csv()

            # 2. Onset detection
            onset_detector = OnsetDetector(config)
            onsets = onset_detector.detect(audio)

            # 3. Assemble NoteEvents
            tracker = NoteTracker(config, profile)
            events = tracker.process(pitch_result, onsets)

            if not events:
                self._signaller.processing_failed.emit(
                    "No notes detected. Check input level and instrument selection."
                )
                return

            # 4. Tempo estimation
            tempo_estimator = TempoEstimator(config)
            tempo_result = tempo_estimator.estimate(events)

            # 5. Rhythm quantization
            if rhythm_mode == "auto":
                quantizer = RhythmQuantizer(tempo_result)
                quantized_notes = quantizer.quantize(events)
                score_builder = ScoreBuilder(profile, tempo_result)
                score = score_builder.build(quantized_notes)
                # Store for tab injection at export time (set before signal fires)
                self._last_quantized_notes = quantized_notes
                self._last_tab_assignments = score_builder.compute_tab_assignments(quantized_notes)
                self._last_profile = profile
            else:
                # Raw mode: build a minimal score for MIDI (no PDF grid)
                score_builder = ScoreBuilder(profile, tempo_result)
                score = score_builder.build_raw([])  # placeholder
                score = None  # signal to export as raw MIDI only

            self._signaller.processing_done.emit(score, events)

        except Exception as exc:  # noqa: BLE001
            self._signaller.processing_failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Pipeline result slots (main thread)
    # ------------------------------------------------------------------

    def _on_processing_done(self, score: object, events: object) -> None:
        self._last_score = score
        self._last_events = events
        has_score = score is not None
        self._transport.set_has_result(True)
        n = len(events) if events else 0
        mode_str = "auto-tempo" if has_score else "raw timing"
        self._status_bar.showMessage(
            f"Done — {n} notes detected ({mode_str}). Ready to export."
        )

    def _on_processing_failed(self, message: str) -> None:
        self._status_bar.showMessage(f"Error: {message}")
        QMessageBox.warning(self, "Processing Failed", message)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_pdf(self) -> None:
        if self._last_score is None:
            QMessageBox.information(
                self, "No Score",
                "PDF export requires auto-tempo mode. Re-record with Rhythm set to Auto-tempo."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", str(self._config.output_dir / "transcription.pdf"),
            "PDF Files (*.pdf)"
        )
        if not path:
            return

        self._status_bar.showMessage("Exporting PDF…")
        try:
            exporter = PDFExporter(self._config)
            out = exporter.export(
                self._last_score, Path(path),
                tab_notes=self._last_quantized_notes,
                tab_assignments=self._last_tab_assignments,
                tab_profile=self._last_profile,
            )
            self._status_bar.showMessage(f"PDF saved → {out}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Error", str(exc))
            self._status_bar.showMessage("PDF export failed.")

    def _on_export_midi(self) -> None:
        if self._last_events is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save MIDI", str(self._config.output_dir / "transcription.mid"),
            "MIDI Files (*.mid)"
        )
        if not path:
            return

        self._status_bar.showMessage("Exporting MIDI…")
        try:
            exporter = MIDIExporter(self._config)
            from improv_scribe.quantization.tempo import TempoEstimator
            tempo_estimator = TempoEstimator(self._config)
            tempo_result = tempo_estimator.estimate(self._last_events)

            if self._last_score is not None and self._rhythm_mode == "auto":
                out = exporter.quantized_from_score(self._last_score, Path(path))
            else:
                out = exporter.raw_from_events(self._last_events, tempo_result, Path(path))

            self._status_bar.showMessage(f"MIDI saved → {out}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Error", str(exc))
            self._status_bar.showMessage("MIDI export failed.")

    def closeEvent(self, event: object) -> None:
        """Ensure stream is stopped on window close."""
        if self._stream:
            self._stream.stop()
        super().closeEvent(event)  # type: ignore[arg-type]
