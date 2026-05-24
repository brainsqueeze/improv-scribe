"""Phase 4 PDF render smoke test.

Regression guard for the spec §14.3 quantizer overlap bug: runs the full
pipeline on the open G chord sample, hands the result to PDFExporter,
asserts the PDF file is written and non-trivial.

basic_pitch only: this test relies on chord-event detection. CREPE/pyin
skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.note_tracker import NoteTracker
from improv_scribe.analysis.onset import OnsetDetector
from improv_scribe.analysis.pitch import PitchEstimator
from improv_scribe.config import AppConfig
from improv_scribe.export.pdf_exporter import PDFExporter
from improv_scribe.notation.score_builder import ScoreBuilder
from improv_scribe.quantization.grid import RhythmQuantizer
from improv_scribe.quantization.tempo import TempoEstimator
from tests.integration.conftest import SAMPLE_ROOT

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "chords" / "6_string_electric_open_G_chord.mp3"


def test_open_g_chord_renders_to_pdf(tmp_path: Path):
    """End-to-end: sample → quantizer → ScoreBuilder → PDFExporter.

    Note: tabs are intentionally skipped. Tab injection for chords is Phase 5.
    """
    backend = os.getenv("ATS_PITCH_BACKEND", "basic_pitch")
    if backend != "basic_pitch":
        pytest.skip(f"PDF render test requires basic_pitch backend (got {backend})")

    config = AppConfig()
    profile = get_profile(Instrument.GUITAR)
    y, _ = librosa.load(str(SAMPLE_PATH), sr=44100, mono=True)

    pitch_result = PitchEstimator(config, backend="basic_pitch").estimate(y, profile)
    onsets = OnsetDetector(config).detect(y)
    events = NoteTracker(config, profile).process(pitch_result, onsets, audio=y)
    assert len(events) > 0, "expected basic_pitch to detect at least one event"

    tempo = TempoEstimator(config).estimate(events)
    quantized = RhythmQuantizer(tempo).quantize(events)
    builder = ScoreBuilder(profile, tempo, title="Phase 4 smoke test")
    score = builder.build(quantized)

    pdf_path = tmp_path / "open_G_chord.pdf"
    out = PDFExporter(config).export(score, pdf_path)
    assert out.exists(), f"PDF was not written to {out}"
    assert out.stat().st_size > 5000, (
        f"PDF size {out.stat().st_size} bytes is suspiciously small "
        "(MuseScore may have produced an empty or broken file)"
    )
