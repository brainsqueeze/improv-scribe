"""
tests/integration/conftest.py

Shared infrastructure for sample-based integration tests.

make_pipeline_fixtures() returns a tuple of module-scoped pytest fixtures.
Assign the returned tuple at module level in each test file — pytest discovers
each name as a fixture for that module.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.note_tracker import NoteTracker
from improv_scribe.analysis.onset import OnsetDetector
from improv_scribe.analysis.pitch import PitchEstimator
from improv_scribe.config import AppConfig
from improv_scribe.notation.score_builder import ScoreBuilder
from improv_scribe.quantization.grid import RhythmQuantizer
from improv_scribe.quantization.tempo import TempoEstimator

# Path to the project-level samples/ directory
# __file__ = tests/integration/conftest.py  →  parents[2] = project root
SAMPLE_ROOT = Path(__file__).parents[2] / "samples"


def make_pipeline_fixtures(sample_path: Path, instrument: Instrument):
    """
    Build a module-scoped pipeline fixture chain for one sample file.

    Parameters
    ----------
    sample_path : Path
        Absolute path to the audio sample.
    instrument : Instrument
        Instrument enum value — selects the correct InstrumentProfile.

    Returns
    -------
    tuple of eight pytest.fixture functions:
        (audio, pitch_result, onsets, note_events,
         tempo_result, quantized_notes, score, tab_assignments)

    Each fixture calls only the previous stage's fixture as input.
    A failure at stage N surfaces as ERROR on all downstream fixtures.
    """
    _config = AppConfig()
    _profile = get_profile(instrument)

    @pytest.fixture(scope="module")
    def audio():
        y, sr = librosa.load(str(sample_path), sr=44100, mono=True)
        return y, sr

    @pytest.fixture(scope="module")
    def pitch_result(audio):
        y, _ = audio
        estimator = PitchEstimator(_config, backend="pyin")
        return estimator.estimate(y, _profile)

    @pytest.fixture(scope="module")
    def onsets(audio):
        y, _ = audio
        detector = OnsetDetector(_config)
        return detector.detect(y)

    @pytest.fixture(scope="module")
    def note_events(pitch_result, onsets):
        tracker = NoteTracker(_config, _profile)
        return tracker.process(pitch_result, onsets)

    @pytest.fixture(scope="module")
    def tempo_result(note_events):
        estimator = TempoEstimator(_config)
        return estimator.estimate(note_events)

    @pytest.fixture(scope="module")
    def quantized_notes(note_events, tempo_result):
        quantizer = RhythmQuantizer(tempo_result)
        return quantizer.quantize(note_events)

    @pytest.fixture(scope="module")
    def score(quantized_notes, tempo_result):
        builder = ScoreBuilder(_profile, tempo_result)
        return builder.build(quantized_notes)

    @pytest.fixture(scope="module")
    def tab_assignments(quantized_notes, tempo_result):
        builder = ScoreBuilder(_profile, tempo_result)
        return builder.compute_tab_assignments(quantized_notes)

    return (
        audio,
        pitch_result,
        onsets,
        note_events,
        tempo_result,
        quantized_notes,
        score,
        tab_assignments,
    )
