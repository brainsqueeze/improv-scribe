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

import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
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
from improv_scribe.gui.score_widget import ScorePanel
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
    render_done = pyqtSignal(int, object)          # (render token, list[Path])
    render_failed = pyqtSignal(int, str)           # (render token, message)


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
        self._badges: list[QLabel] = []
        self._info_label = QLabel()

        self._signaller = _PipelineSignaller()
        self._signaller.processing_done.connect(self._on_processing_done)
        self._signaller.processing_failed.connect(self._on_processing_failed)
        self._signaller.render_done.connect(self._on_render_done)
        self._signaller.render_failed.connect(self._on_render_failed)
        self._render_token = 0
        self._render_dir: Path | None = None

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
        central.setObjectName("Root")
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
        self._transport.set_backend(self._config.pitch_backend)
        root_layout.addWidget(self._transport)

        # View switch (Live | Score)
        root_layout.addWidget(self._build_view_strip())

        # Live view: waveform + spectrogram cards in a vertical splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setContentsMargins(20, 0, 20, 0)
        splitter.setChildrenCollapsible(False)

        self._waveform = WaveformWidget(sample_rate=self._config.sample_rate)
        splitter.addWidget(
            self._make_card("Waveform", "last 2.0 s", self._waveform, badge=True)
        )

        self._spectrogram = SpectrogramWidget(sample_rate=self._config.sample_rate)
        splitter.addWidget(
            self._make_card("Spectrogram", "CQT · E1 – E7", self._spectrogram, badge=True)
        )
        splitter.setSizes([250, 350])

        # Score view: rendered notation + TAB
        self._score_panel = ScorePanel()
        score_host = QWidget()
        score_layout = QVBoxLayout(score_host)
        score_layout.setContentsMargins(20, 0, 20, 0)
        score_layout.addWidget(self._score_panel)

        self._views = QStackedWidget()
        self._views.addWidget(splitter)      # 0 = live
        self._views.addWidget(score_host)    # 1 = score
        root_layout.addWidget(self._views, 1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.setSizeGripEnabled(False)
        self._status_bar.addPermanentWidget(self._info_label)
        self._refresh_info_label()
        self._status_bar.showMessage("Ready — select a device and press Record.")

    def _build_view_strip(self) -> QWidget:
        """Live | Score segmented switch plus a one-line hint."""
        strip = QWidget()
        row = QHBoxLayout(strip)
        row.setContentsMargins(20, 0, 20, 14)
        row.setSpacing(16)

        bar = QFrame()
        bar.setObjectName("SegBar")
        bar_row = QHBoxLayout(bar)
        bar_row.setContentsMargins(4, 4, 4, 4)
        bar_row.setSpacing(4)
        self._live_btn = QPushButton("Live")
        self._score_btn = QPushButton("Score")
        group = QButtonGroup(self)
        group.setExclusive(True)
        for idx, btn in enumerate((self._live_btn, self._score_btn)):
            btn.setObjectName("Seg")
            btn.setCheckable(True)
            group.addButton(btn, idx)
            bar_row.addWidget(btn)
        self._live_btn.setChecked(True)
        self._score_btn.setEnabled(False)
        group.idClicked.connect(self._views_set_index)
        row.addWidget(bar)

        self._view_hint = QLabel("Score renders after you press Stop.")
        self._view_hint.setObjectName("Hint")
        row.addWidget(self._view_hint, 1)
        return strip

    def _views_set_index(self, index: int) -> None:
        self._views.setCurrentIndex(index)

    def _show_view(self, index: int) -> None:
        (self._live_btn, self._score_btn)[index].setChecked(True)
        self._views.setCurrentIndex(index)

    def _make_card(
        self, title: str, subtitle: str, body: QWidget, badge: bool = False
    ) -> QFrame:
        """Wrap *body* in a titled card; optionally add a LIVE/IDLE badge."""
        card = QFrame()
        card.setObjectName("Card")
        col = QVBoxLayout(card)
        col.setContentsMargins(18, 14, 18, 14)
        col.setSpacing(8)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("CardTitle")
        s = QLabel(subtitle)
        s.setObjectName("CardSub")
        head.addWidget(t)
        head.addWidget(s)
        head.addStretch()
        if badge:
            b = QLabel("IDLE")
            b.setObjectName("Badge")
            b.setProperty("kind", "idle")
            self._badges.append(b)
            head.addWidget(b)
        col.addLayout(head)
        col.addWidget(body, 1)
        return card

    def _set_live_badges(self, live: bool) -> None:
        for b in self._badges:
            b.setText("LIVE" if live else "IDLE")
            b.setProperty("kind", "live" if live else "idle")
            b.style().unpolish(b)
            b.style().polish(b)

    def _refresh_info_label(self) -> None:
        backend = self._config.pitch_backend.replace("_", "-").title()
        self._info_label.setText(
            f"{self._config.sample_rate / 1000:g} kHz · {backend} · "
            f"{self._current_instrument.value.title()}"
        )

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
        self._refresh_info_label()
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
        self._config.pitch_backend = (
            backend
                .replace("pyin", "pyin")
                .replace("crepe", "crepe")
                .replace("basic-pitch", "basic_pitch")
        )
        self._refresh_info_label()

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
        self._set_live_badges(True)
        self._invalidate_render()
        self._score_panel.show_message("Record and press Stop to render the score.")
        self._score_btn.setEnabled(False)
        self._show_view(0)
        self._view_hint.setText("Score renders after you press Stop.")
        self._status_bar.showMessage("● Recording…")

    def _on_stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream = None

        self._is_recording = False
        self._transport.set_recording(False)
        self._set_live_badges(False)
        self._status_bar.showMessage("Processing…")
        self._score_btn.setEnabled(True)
        self._score_panel.show_rendering()
        self._show_view(1)
        self._view_hint.setText("Analysing your recording, then rendering the score.")

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
            events = tracker.process(pitch_result, onsets, audio=audio)

            if not events:
                # pYIN's HMM-based voicing detector collapses on low-SNR
                # signals (e.g. acoustic guitar via laptop mic).  Surface a
                # more actionable message when that's likely the cause.
                if config.pitch_backend == "pyin":
                    voiced_count = len(pitch_result.voiced_frames)
                    duration_s = len(audio) / config.sample_rate
                    if voiced_count < max(10, int(duration_s)):
                        self._signaller.processing_failed.emit(
                            f"pYIN found only {voiced_count} voiced frames "
                            f"over {duration_s:.1f}s. This usually means low "
                            "signal-to-noise ratio. pYIN is best for clean "
                            "line-in signals; switch to 'Basic-pitch' or "
                            "'CREPE' for noisy mic input."
                        )
                        return
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
                _NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                def _midi_name(m: int) -> str:
                    return _NOTE_NAMES[m % 12] + str(m // 12 - 1)
                def _format_note_names(qn) -> str:
                    """Format chord-aware note display: 'E4' or 'E4/G4/B4'."""
                    if qn.is_rest:
                        return "rest"
                    return "/".join(_midi_name(m) for m in qn.midi_notes)
                _debug = [
                    (a, _format_note_names(n)) if a is not None else (None, "rest")
                    for n, a in zip(quantized_notes, self._last_tab_assignments, strict=True)
                ]
                print("[TAB DEBUG] (assignment, note):", _debug)
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
        if has_score:
            self._status_bar.showMessage(
                f"Done — {n} notes detected ({mode_str}). Rendering score…"
            )
            self._view_hint.setText("MuseScore is rendering your score.")
            self._start_render(score)
        else:
            self._status_bar.showMessage(
                f"Done — {n} notes detected ({mode_str}). Ready to export."
            )
            self._score_panel.show_message(
                "Score preview needs Auto-tempo rhythm mode. "
                "Raw timing exports to MIDI only."
            )
            self._view_hint.setText("No score in Raw rhythm mode.")

    def _on_processing_failed(self, message: str) -> None:
        self._status_bar.showMessage(f"Error: {message}")
        self._score_panel.show_message("No score — processing failed.")
        self._view_hint.setText("Processing failed.")
        QMessageBox.warning(self, "Processing Failed", message)

    # ------------------------------------------------------------------
    # In-app score rendering (runs after Stop, in a background thread)
    # ------------------------------------------------------------------

    def _invalidate_render(self) -> None:
        """Discard any in-flight render and delete its temp pages."""
        self._render_token += 1
        self._cleanup_render_dir()

    def _cleanup_render_dir(self) -> None:
        if self._render_dir is not None:
            shutil.rmtree(self._render_dir, ignore_errors=True)
            self._render_dir = None

    def _start_render(self, score: object) -> None:
        self._invalidate_render()
        token = self._render_token
        out_dir = Path(tempfile.mkdtemp(prefix="ats_score_"))
        self._render_dir = out_dir
        self._score_panel.show_rendering()
        threading.Thread(
            target=self._run_render,
            args=(
                token,
                score,
                out_dir,
                self._last_quantized_notes,
                self._last_tab_assignments,
                self._last_profile,
            ),
            daemon=True,
        ).start()

    def _run_render(
        self,
        token: int,
        score: object,
        out_dir: Path,
        tab_notes: object,
        tab_assignments: object,
        tab_profile: object,
    ) -> None:
        """Background thread: MusicXML → MuseScore → SVG pages."""
        try:
            pages = PDFExporter(self._config).export_svg_pages(
                score,  # type: ignore[arg-type]
                out_dir,
                tab_notes=tab_notes,  # type: ignore[arg-type]
                tab_assignments=tab_assignments,  # type: ignore[arg-type]
                tab_profile=tab_profile,  # type: ignore[arg-type]
            )
            self._signaller.render_done.emit(token, pages)
        except Exception as exc:  # noqa: BLE001
            self._signaller.render_failed.emit(token, str(exc))

    def _on_render_done(self, token: int, pages: object) -> None:
        if token != self._render_token:
            return  # stale: a new recording started
        self._score_panel.show_pages(list(pages))  # type: ignore[call-overload]
        self._view_hint.setText(
            "Rendered from this recording. Export PDF saves the same pages."
        )
        self._status_bar.showMessage("Score ready. Ready to export.")

    def _on_render_failed(self, token: int, message: str) -> None:
        if token != self._render_token:
            return
        first_line = message.strip().splitlines()[0] if message.strip() else "unknown error"
        self._score_panel.show_message(
            f"Couldn't render the score preview.\n{first_line}\n"
            "MIDI export is still available."
        )
        self._view_hint.setText("Score preview unavailable.")
        self._status_bar.showMessage("Score preview failed — exports still available.")

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
        self._cleanup_render_dir()
        super().closeEvent(event)  # type: ignore[arg-type]
