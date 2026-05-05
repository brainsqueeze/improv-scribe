# Sample-Based Integration Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end regression tests that run the full `improv_scribe` pipeline on real audio samples and assert correctness at each stage.

**Architecture:** Four test modules (one per sample file) each define a chain of `scope="module"` pytest fixtures that build the pipeline stage-by-stage. A shared factory in `conftest.py` eliminates duplication. Individual test functions assert on the output of exactly one stage; a fixture failure at stage N propagates as `ERROR` (not `FAIL`) to downstream tests, immediately identifying the broken stage.

**Tech Stack:** pytest (module-scoped fixtures), librosa (audio loading), pYIN pitch backend, music21 (score inspection).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `tests/integration/__init__.py` | Makes `tests.integration` an importable package |
| Create | `tests/integration/conftest.py` | `SAMPLE_ROOT` path + `make_pipeline_fixtures()` factory |
| Create | `tests/integration/test_guitar_acoustic_line_in.py` | 18 tests for `samples/guitar/6_string_acoustic_line_in.mp3` |
| Create | `tests/integration/test_guitar_acoustic_mic.py` | 18 tests for `samples/guitar/6_string_acoustic_mic.mp3` |
| Create | `tests/integration/test_guitar_electric_line_in.py` | 18 tests for `samples/guitar/6_string_electric_line_in.mp3` |
| Create | `tests/integration/test_bass_line_in.py` | 18 tests for `samples/bass/4_string_bass_line_in.mp3` |

---

## Task 1: Integration test infrastructure

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/conftest.py`

- [ ] **Step 1: Create the package init file**

```python
# tests/integration/__init__.py
```

(Empty file — makes `tests.integration` importable so test modules can do `from tests.integration.conftest import ...`.)

- [ ] **Step 2: Create conftest.py with SAMPLE_ROOT and the fixture factory**

```python
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
```

- [ ] **Step 3: Smoke-check the conftest imports**

```bash
conda run -n auto-sheet-music python -c "
from tests.integration.conftest import make_pipeline_fixtures, SAMPLE_ROOT
print('SAMPLE_ROOT:', SAMPLE_ROOT)
print('exists:', SAMPLE_ROOT.exists())
"
```

Expected output:
```
SAMPLE_ROOT: /path/to/project/samples
exists: True
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/__init__.py tests/integration/conftest.py
git commit -m "test: add integration test infrastructure (conftest + fixture factory)"
```

---

## Task 2: Guitar — acoustic line-in tests

**Files:**
- Create: `tests/integration/test_guitar_acoustic_line_in.py`

- [ ] **Step 1: Create the test file**

```python
"""
tests/integration/test_guitar_acoustic_line_in.py

End-to-end pipeline regression tests for:
    samples/guitar/6_string_acoustic_line_in.mp3

The sample plays each open string of a calibrated acoustic guitar (line-in)
from low to high: E2 A2 D3 G3 B3 E4. One note per string, six total.
"""

from __future__ import annotations

import numpy as np
import music21.clef
import music21.note

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "6_string_acoustic_line_in.mp3"
INSTRUMENT = Instrument.GUITAR
NOTE_COUNT = 6
EXPECTED_DURATION_S = 12.3

# Concert (sounding) MIDI, low string → high string: E2 A2 D3 G3 B3 E4
EXPECTED_MIDI = [40, 45, 50, 55, 59, 64]

# Written MIDI: guitar transpose_semitones=-12 → written = midi_note + 12
# Produces: E3 A3 D4 G4 B4 E5
EXPECTED_WRITTEN_MIDI = [52, 57, 62, 67, 71, 76]

# Tab: every open string → (string_idx, fret=0), 0-based from lowest string
EXPECTED_TAB = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]

# Clef: "treble8vb" → sign='G', octaveChange=-1
EXPECTED_CLEF_SIGN = "G"

# ---------------------------------------------------------------------------
# Fixture chain — runs the full pipeline once for this module
# ---------------------------------------------------------------------------

(
    audio,
    pitch_result,
    onsets,
    note_events,
    tempo_result,
    quantized_notes,
    score,
    tab_assignments,
) = make_pipeline_fixtures(SAMPLE_PATH, INSTRUMENT)

# ---------------------------------------------------------------------------
# Stage: audio
# ---------------------------------------------------------------------------

class TestAudio:
    def test_audio_shape(self, audio):
        y, _ = audio
        assert y.ndim == 1
        assert len(y) > 0
        assert y.dtype == np.float32

    def test_audio_sample_rate(self, audio):
        _, sr = audio
        assert sr == 44100

    def test_audio_duration(self, audio):
        y, sr = audio
        duration_s = len(y) / sr
        assert abs(duration_s - EXPECTED_DURATION_S) <= 2.0, (
            f"Expected ~{EXPECTED_DURATION_S}s, got {duration_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Stage: pitch_result
# ---------------------------------------------------------------------------

class TestPitchResult:
    def test_pitch_voiced_frames_nonempty(self, pitch_result):
        assert len(pitch_result.voiced_frames) > 0

    def test_pitch_frequency_range(self, pitch_result):
        profile = get_profile(INSTRUMENT)
        for frame in pitch_result.voiced_frames:
            assert profile.freq_min_hz <= frame.freq_hz <= profile.freq_max_hz, (
                f"Frame freq {frame.freq_hz:.1f} Hz outside [{profile.freq_min_hz}, {profile.freq_max_hz}]"
            )


# ---------------------------------------------------------------------------
# Stage: onsets
# ---------------------------------------------------------------------------

class TestOnsets:
    def test_onset_count(self, onsets):
        assert NOTE_COUNT <= len(onsets) <= NOTE_COUNT + 2, (
            f"Expected {NOTE_COUNT}–{NOTE_COUNT + 2} onsets, got {len(onsets)}"
        )

    def test_onsets_sorted(self, onsets):
        times = [o.time_s for o in onsets]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# Stage: note_events
# ---------------------------------------------------------------------------

class TestNoteEvents:
    def test_note_count(self, note_events):
        # This is the primary "all notes detected" gate.
        # If this fails: print note_events to inspect what the pipeline detected.
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} NoteEvents, got {len(note_events)}: "
            f"{[e.midi_note for e in note_events]}"
        )

    def test_note_pitches(self, note_events):
        # midi_note is already rounded to int; ±0.5 is effectively exact match
        # for calibrated open-string recordings.
        for event, expected in zip(note_events, EXPECTED_MIDI):
            assert abs(event.midi_note - expected) <= 0.5, (
                f"Expected MIDI {expected}, got {event.midi_note} "
                f"({event.frequency_hz:.1f} Hz)"
            )

    def test_notes_in_instrument_range(self, note_events):
        profile = get_profile(INSTRUMENT)
        for event in note_events:
            assert profile.midi_min <= event.midi_note <= profile.midi_max, (
                f"MIDI {event.midi_note} outside instrument range "
                f"[{profile.midi_min}, {profile.midi_max}]"
            )


# ---------------------------------------------------------------------------
# Stage: tempo_result
# ---------------------------------------------------------------------------

class TestTempoResult:
    def test_tempo_positive(self, tempo_result):
        assert 40.0 <= tempo_result.bpm <= 250.0, (
            f"BPM {tempo_result.bpm} outside [40, 250]"
        )


# ---------------------------------------------------------------------------
# Stage: quantized_notes
# ---------------------------------------------------------------------------

class TestQuantizedNotes:
    def test_quantized_note_count(self, quantized_notes):
        non_rests = [n for n in quantized_notes if not n.is_rest]
        assert len(non_rests) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} non-rest QuantizedNotes, "
            f"got {len(non_rests)} (total with rests: {len(quantized_notes)})"
        )

    def test_quantized_pitches_unchanged(self, quantized_notes, note_events):
        # Quantizer must not alter pitch — only timing.
        quantized_midis = [n.midi_note for n in quantized_notes if not n.is_rest]
        event_midis = [e.midi_note for e in note_events]
        assert quantized_midis == event_midis


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_clef(self, score):
        part = score.parts[0]
        clefs = list(part.recurse().getElementsByClass(music21.clef.Clef))
        assert len(clefs) >= 1, "No clef found in score part"
        clef_obj = clefs[0]
        assert clef_obj.sign == EXPECTED_CLEF_SIGN, (
            f"Expected clef sign '{EXPECTED_CLEF_SIGN}', got '{clef_obj.sign}'"
        )
        assert clef_obj.octaveChange == -1, (
            f"Expected octaveChange -1 (8vb), got {clef_obj.octaveChange}"
        )

    def test_score_written_pitches(self, score):
        part = score.parts[0]
        notes = list(part.recurse().getElementsByClass(music21.note.Note))
        written_midis = sorted([n.pitch.midi for n in notes])
        assert written_midis == sorted(EXPECTED_WRITTEN_MIDI), (
            f"Written MIDIs {written_midis} != expected {sorted(EXPECTED_WRITTEN_MIDI)}"
        )


# ---------------------------------------------------------------------------
# Stage: tab_assignments
# ---------------------------------------------------------------------------

class TestTabAssignments:
    def test_tab_length(self, tab_assignments, quantized_notes):
        assert len(tab_assignments) == len(quantized_notes)

    def test_tab_all_fret_zero(self, tab_assignments):
        for assignment in tab_assignments:
            if assignment is not None:
                _, fret = assignment
                assert fret == 0, (
                    f"Open string expected fret 0, got {fret}"
                )

    def test_tab_exact_string_assignments(self, tab_assignments):
        # Compare in onset order (low string played first = ascending MIDI order)
        non_none = [a for a in tab_assignments if a is not None]
        assert non_none == EXPECTED_TAB, (
            f"Tab assignments {non_none} != expected {EXPECTED_TAB}"
        )
```

- [ ] **Step 2: Run the tests**

```bash
conda run -n auto-sheet-music pytest tests/integration/test_guitar_acoustic_line_in.py -v
```

Expected: all 18 tests pass. If `test_note_count` or `test_note_pitches` fails, run:

```bash
conda run -n auto-sheet-music python -c "
import librosa
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures
from improv_scribe.analysis.instrument_profiles import Instrument

sample_path = SAMPLE_ROOT / 'guitar' / '6_string_acoustic_line_in.mp3'
y, sr = librosa.load(str(sample_path), sr=44100, mono=True)

from improv_scribe.config import AppConfig
from improv_scribe.analysis.instrument_profiles import get_profile
from improv_scribe.analysis.pitch import PitchEstimator
from improv_scribe.analysis.onset import OnsetDetector
from improv_scribe.analysis.note_tracker import NoteTracker

config = AppConfig()
profile = get_profile(Instrument.GUITAR)
pitch_result = PitchEstimator(config, backend='pyin').estimate(y, profile)
onsets = OnsetDetector(config).detect(y)
note_events = NoteTracker(config, profile).process(pitch_result, onsets)
print(f'Onsets: {len(onsets)}')
print(f'NoteEvents: {len(note_events)}')
for e in note_events:
    print(f'  midi={e.midi_note}, freq={e.frequency_hz:.1f} Hz, onset={e.onset_s:.2f}s')
"
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_guitar_acoustic_line_in.py
git commit -m "test: add integration tests for guitar acoustic line-in sample"
```

---

## Task 3: Guitar — acoustic mic tests

**Files:**
- Create: `tests/integration/test_guitar_acoustic_mic.py`

- [ ] **Step 1: Create the test file**

Identical to `test_guitar_acoustic_line_in.py` with two constants changed:

```python
"""
tests/integration/test_guitar_acoustic_mic.py

End-to-end pipeline regression tests for:
    samples/guitar/6_string_acoustic_mic.mp3

The sample plays each open string of a calibrated acoustic guitar (mic)
from low to high: E2 A2 D3 G3 B3 E4. One note per string, six total.
"""

from __future__ import annotations

import numpy as np
import music21.clef
import music21.note

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "6_string_acoustic_mic.mp3"
INSTRUMENT = Instrument.GUITAR
NOTE_COUNT = 6
EXPECTED_DURATION_S = 13.4

EXPECTED_MIDI = [40, 45, 50, 55, 59, 64]
EXPECTED_WRITTEN_MIDI = [52, 57, 62, 67, 71, 76]
EXPECTED_TAB = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
EXPECTED_CLEF_SIGN = "G"

(
    audio,
    pitch_result,
    onsets,
    note_events,
    tempo_result,
    quantized_notes,
    score,
    tab_assignments,
) = make_pipeline_fixtures(SAMPLE_PATH, INSTRUMENT)


class TestAudio:
    def test_audio_shape(self, audio):
        y, _ = audio
        assert y.ndim == 1
        assert len(y) > 0
        assert y.dtype == np.float32

    def test_audio_sample_rate(self, audio):
        _, sr = audio
        assert sr == 44100

    def test_audio_duration(self, audio):
        y, sr = audio
        duration_s = len(y) / sr
        assert abs(duration_s - EXPECTED_DURATION_S) <= 2.0, (
            f"Expected ~{EXPECTED_DURATION_S}s, got {duration_s:.2f}s"
        )


class TestPitchResult:
    def test_pitch_voiced_frames_nonempty(self, pitch_result):
        assert len(pitch_result.voiced_frames) > 0

    def test_pitch_frequency_range(self, pitch_result):
        profile = get_profile(INSTRUMENT)
        for frame in pitch_result.voiced_frames:
            assert profile.freq_min_hz <= frame.freq_hz <= profile.freq_max_hz, (
                f"Frame freq {frame.freq_hz:.1f} Hz outside [{profile.freq_min_hz}, {profile.freq_max_hz}]"
            )


class TestOnsets:
    def test_onset_count(self, onsets):
        assert NOTE_COUNT <= len(onsets) <= NOTE_COUNT + 2, (
            f"Expected {NOTE_COUNT}–{NOTE_COUNT + 2} onsets, got {len(onsets)}"
        )

    def test_onsets_sorted(self, onsets):
        times = [o.time_s for o in onsets]
        assert times == sorted(times)


class TestNoteEvents:
    def test_note_count(self, note_events):
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} NoteEvents, got {len(note_events)}: "
            f"{[e.midi_note for e in note_events]}"
        )

    def test_note_pitches(self, note_events):
        for event, expected in zip(note_events, EXPECTED_MIDI):
            assert abs(event.midi_note - expected) <= 0.5, (
                f"Expected MIDI {expected}, got {event.midi_note} "
                f"({event.frequency_hz:.1f} Hz)"
            )

    def test_notes_in_instrument_range(self, note_events):
        profile = get_profile(INSTRUMENT)
        for event in note_events:
            assert profile.midi_min <= event.midi_note <= profile.midi_max, (
                f"MIDI {event.midi_note} outside instrument range "
                f"[{profile.midi_min}, {profile.midi_max}]"
            )


class TestTempoResult:
    def test_tempo_positive(self, tempo_result):
        assert 40.0 <= tempo_result.bpm <= 250.0, (
            f"BPM {tempo_result.bpm} outside [40, 250]"
        )


class TestQuantizedNotes:
    def test_quantized_note_count(self, quantized_notes):
        non_rests = [n for n in quantized_notes if not n.is_rest]
        assert len(non_rests) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} non-rest QuantizedNotes, "
            f"got {len(non_rests)} (total with rests: {len(quantized_notes)})"
        )

    def test_quantized_pitches_unchanged(self, quantized_notes, note_events):
        quantized_midis = [n.midi_note for n in quantized_notes if not n.is_rest]
        event_midis = [e.midi_note for e in note_events]
        assert quantized_midis == event_midis


class TestScore:
    def test_score_clef(self, score):
        part = score.parts[0]
        clefs = list(part.recurse().getElementsByClass(music21.clef.Clef))
        assert len(clefs) >= 1, "No clef found in score part"
        clef_obj = clefs[0]
        assert clef_obj.sign == EXPECTED_CLEF_SIGN, (
            f"Expected clef sign '{EXPECTED_CLEF_SIGN}', got '{clef_obj.sign}'"
        )
        assert clef_obj.octaveChange == -1, (
            f"Expected octaveChange -1 (8vb), got {clef_obj.octaveChange}"
        )

    def test_score_written_pitches(self, score):
        part = score.parts[0]
        notes = list(part.recurse().getElementsByClass(music21.note.Note))
        written_midis = sorted([n.pitch.midi for n in notes])
        assert written_midis == sorted(EXPECTED_WRITTEN_MIDI), (
            f"Written MIDIs {written_midis} != expected {sorted(EXPECTED_WRITTEN_MIDI)}"
        )


class TestTabAssignments:
    def test_tab_length(self, tab_assignments, quantized_notes):
        assert len(tab_assignments) == len(quantized_notes)

    def test_tab_all_fret_zero(self, tab_assignments):
        for assignment in tab_assignments:
            if assignment is not None:
                _, fret = assignment
                assert fret == 0, (
                    f"Open string expected fret 0, got {fret}"
                )

    def test_tab_exact_string_assignments(self, tab_assignments):
        non_none = [a for a in tab_assignments if a is not None]
        assert non_none == EXPECTED_TAB, (
            f"Tab assignments {non_none} != expected {EXPECTED_TAB}"
        )
```

- [ ] **Step 2: Run the tests**

```bash
conda run -n auto-sheet-music pytest tests/integration/test_guitar_acoustic_mic.py -v
```

Expected: all 18 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_guitar_acoustic_mic.py
git commit -m "test: add integration tests for guitar acoustic mic sample"
```

---

## Task 4: Guitar — electric line-in tests

**Files:**
- Create: `tests/integration/test_guitar_electric_line_in.py`

- [ ] **Step 1: Create the test file**

Identical to `test_guitar_acoustic_mic.py` with two constants changed (`SAMPLE_PATH` and the module docstring):

```python
"""
tests/integration/test_guitar_electric_line_in.py

End-to-end pipeline regression tests for:
    samples/guitar/6_string_electric_line_in.mp3

The sample plays each open string of a calibrated electric guitar (line-in)
from low to high: E2 A2 D3 G3 B3 E4. One note per string, six total.
"""

from __future__ import annotations

import numpy as np
import music21.clef
import music21.note

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "6_string_electric_line_in.mp3"
INSTRUMENT = Instrument.GUITAR
NOTE_COUNT = 6
EXPECTED_DURATION_S = 13.4

EXPECTED_MIDI = [40, 45, 50, 55, 59, 64]
EXPECTED_WRITTEN_MIDI = [52, 57, 62, 67, 71, 76]
EXPECTED_TAB = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
EXPECTED_CLEF_SIGN = "G"

(
    audio,
    pitch_result,
    onsets,
    note_events,
    tempo_result,
    quantized_notes,
    score,
    tab_assignments,
) = make_pipeline_fixtures(SAMPLE_PATH, INSTRUMENT)


class TestAudio:
    def test_audio_shape(self, audio):
        y, _ = audio
        assert y.ndim == 1
        assert len(y) > 0
        assert y.dtype == np.float32

    def test_audio_sample_rate(self, audio):
        _, sr = audio
        assert sr == 44100

    def test_audio_duration(self, audio):
        y, sr = audio
        duration_s = len(y) / sr
        assert abs(duration_s - EXPECTED_DURATION_S) <= 2.0, (
            f"Expected ~{EXPECTED_DURATION_S}s, got {duration_s:.2f}s"
        )


class TestPitchResult:
    def test_pitch_voiced_frames_nonempty(self, pitch_result):
        assert len(pitch_result.voiced_frames) > 0

    def test_pitch_frequency_range(self, pitch_result):
        profile = get_profile(INSTRUMENT)
        for frame in pitch_result.voiced_frames:
            assert profile.freq_min_hz <= frame.freq_hz <= profile.freq_max_hz, (
                f"Frame freq {frame.freq_hz:.1f} Hz outside [{profile.freq_min_hz}, {profile.freq_max_hz}]"
            )


class TestOnsets:
    def test_onset_count(self, onsets):
        assert NOTE_COUNT <= len(onsets) <= NOTE_COUNT + 2, (
            f"Expected {NOTE_COUNT}–{NOTE_COUNT + 2} onsets, got {len(onsets)}"
        )

    def test_onsets_sorted(self, onsets):
        times = [o.time_s for o in onsets]
        assert times == sorted(times)


class TestNoteEvents:
    def test_note_count(self, note_events):
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} NoteEvents, got {len(note_events)}: "
            f"{[e.midi_note for e in note_events]}"
        )

    def test_note_pitches(self, note_events):
        for event, expected in zip(note_events, EXPECTED_MIDI):
            assert abs(event.midi_note - expected) <= 0.5, (
                f"Expected MIDI {expected}, got {event.midi_note} "
                f"({event.frequency_hz:.1f} Hz)"
            )

    def test_notes_in_instrument_range(self, note_events):
        profile = get_profile(INSTRUMENT)
        for event in note_events:
            assert profile.midi_min <= event.midi_note <= profile.midi_max, (
                f"MIDI {event.midi_note} outside instrument range "
                f"[{profile.midi_min}, {profile.midi_max}]"
            )


class TestTempoResult:
    def test_tempo_positive(self, tempo_result):
        assert 40.0 <= tempo_result.bpm <= 250.0, (
            f"BPM {tempo_result.bpm} outside [40, 250]"
        )


class TestQuantizedNotes:
    def test_quantized_note_count(self, quantized_notes):
        non_rests = [n for n in quantized_notes if not n.is_rest]
        assert len(non_rests) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} non-rest QuantizedNotes, "
            f"got {len(non_rests)} (total with rests: {len(quantized_notes)})"
        )

    def test_quantized_pitches_unchanged(self, quantized_notes, note_events):
        quantized_midis = [n.midi_note for n in quantized_notes if not n.is_rest]
        event_midis = [e.midi_note for e in note_events]
        assert quantized_midis == event_midis


class TestScore:
    def test_score_clef(self, score):
        part = score.parts[0]
        clefs = list(part.recurse().getElementsByClass(music21.clef.Clef))
        assert len(clefs) >= 1, "No clef found in score part"
        clef_obj = clefs[0]
        assert clef_obj.sign == EXPECTED_CLEF_SIGN, (
            f"Expected clef sign '{EXPECTED_CLEF_SIGN}', got '{clef_obj.sign}'"
        )
        assert clef_obj.octaveChange == -1, (
            f"Expected octaveChange -1 (8vb), got {clef_obj.octaveChange}"
        )

    def test_score_written_pitches(self, score):
        part = score.parts[0]
        notes = list(part.recurse().getElementsByClass(music21.note.Note))
        written_midis = sorted([n.pitch.midi for n in notes])
        assert written_midis == sorted(EXPECTED_WRITTEN_MIDI), (
            f"Written MIDIs {written_midis} != expected {sorted(EXPECTED_WRITTEN_MIDI)}"
        )


class TestTabAssignments:
    def test_tab_length(self, tab_assignments, quantized_notes):
        assert len(tab_assignments) == len(quantized_notes)

    def test_tab_all_fret_zero(self, tab_assignments):
        for assignment in tab_assignments:
            if assignment is not None:
                _, fret = assignment
                assert fret == 0, (
                    f"Open string expected fret 0, got {fret}"
                )

    def test_tab_exact_string_assignments(self, tab_assignments):
        non_none = [a for a in tab_assignments if a is not None]
        assert non_none == EXPECTED_TAB, (
            f"Tab assignments {non_none} != expected {EXPECTED_TAB}"
        )
```

- [ ] **Step 2: Run the tests**

```bash
conda run -n auto-sheet-music pytest tests/integration/test_guitar_electric_line_in.py -v
```

Expected: all 18 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_guitar_electric_line_in.py
git commit -m "test: add integration tests for guitar electric line-in sample"
```

---

## Task 5: Bass — line-in tests

**Files:**
- Create: `tests/integration/test_bass_line_in.py`

- [ ] **Step 1: Create the test file**

```python
"""
tests/integration/test_bass_line_in.py

End-to-end pipeline regression tests for:
    samples/bass/4_string_bass_line_in.mp3

The sample plays each open string of a calibrated bass guitar (line-in)
from low to high: E1 A1 D2 G2. One note per string, four total.
"""

from __future__ import annotations

import numpy as np
import music21.clef
import music21.note

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

SAMPLE_PATH = SAMPLE_ROOT / "bass" / "4_string_bass_line_in.mp3"
INSTRUMENT = Instrument.BASS
NOTE_COUNT = 4
EXPECTED_DURATION_S = 12.3

# Concert (sounding) MIDI, low string → high string: E1 A1 D2 G2
EXPECTED_MIDI = [28, 33, 38, 43]

# Written MIDI: bass transpose_semitones=-12 → written = midi_note + 12
# Produces: E2 A2 D3 G3
EXPECTED_WRITTEN_MIDI = [40, 45, 50, 55]

# Tab: every open string → (string_idx, fret=0), 0-based from lowest string
EXPECTED_TAB = [(0, 0), (1, 0), (2, 0), (3, 0)]

# Clef: "bass8vb" → sign='F', octaveChange=-1
EXPECTED_CLEF_SIGN = "F"

# ---------------------------------------------------------------------------
# Fixture chain — runs the full pipeline once for this module
# ---------------------------------------------------------------------------

(
    audio,
    pitch_result,
    onsets,
    note_events,
    tempo_result,
    quantized_notes,
    score,
    tab_assignments,
) = make_pipeline_fixtures(SAMPLE_PATH, INSTRUMENT)

# ---------------------------------------------------------------------------
# Stage: audio
# ---------------------------------------------------------------------------

class TestAudio:
    def test_audio_shape(self, audio):
        y, _ = audio
        assert y.ndim == 1
        assert len(y) > 0
        assert y.dtype == np.float32

    def test_audio_sample_rate(self, audio):
        _, sr = audio
        assert sr == 44100

    def test_audio_duration(self, audio):
        y, sr = audio
        duration_s = len(y) / sr
        assert abs(duration_s - EXPECTED_DURATION_S) <= 2.0, (
            f"Expected ~{EXPECTED_DURATION_S}s, got {duration_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Stage: pitch_result
# ---------------------------------------------------------------------------

class TestPitchResult:
    def test_pitch_voiced_frames_nonempty(self, pitch_result):
        assert len(pitch_result.voiced_frames) > 0

    def test_pitch_frequency_range(self, pitch_result):
        profile = get_profile(INSTRUMENT)
        for frame in pitch_result.voiced_frames:
            assert profile.freq_min_hz <= frame.freq_hz <= profile.freq_max_hz, (
                f"Frame freq {frame.freq_hz:.1f} Hz outside [{profile.freq_min_hz}, {profile.freq_max_hz}]"
            )


# ---------------------------------------------------------------------------
# Stage: onsets
# ---------------------------------------------------------------------------

class TestOnsets:
    def test_onset_count(self, onsets):
        assert NOTE_COUNT <= len(onsets) <= NOTE_COUNT + 2, (
            f"Expected {NOTE_COUNT}–{NOTE_COUNT + 2} onsets, got {len(onsets)}"
        )

    def test_onsets_sorted(self, onsets):
        times = [o.time_s for o in onsets]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# Stage: note_events
# ---------------------------------------------------------------------------

class TestNoteEvents:
    def test_note_count(self, note_events):
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} NoteEvents, got {len(note_events)}: "
            f"{[e.midi_note for e in note_events]}"
        )

    def test_note_pitches(self, note_events):
        # midi_note is already rounded to int; ±0.5 is effectively exact match
        # for calibrated open-string recordings.
        for event, expected in zip(note_events, EXPECTED_MIDI):
            assert abs(event.midi_note - expected) <= 0.5, (
                f"Expected MIDI {expected}, got {event.midi_note} "
                f"({event.frequency_hz:.1f} Hz)"
            )

    def test_notes_in_instrument_range(self, note_events):
        profile = get_profile(INSTRUMENT)
        for event in note_events:
            assert profile.midi_min <= event.midi_note <= profile.midi_max, (
                f"MIDI {event.midi_note} outside instrument range "
                f"[{profile.midi_min}, {profile.midi_max}]"
            )


# ---------------------------------------------------------------------------
# Stage: tempo_result
# ---------------------------------------------------------------------------

class TestTempoResult:
    def test_tempo_positive(self, tempo_result):
        assert 40.0 <= tempo_result.bpm <= 250.0, (
            f"BPM {tempo_result.bpm} outside [40, 250]"
        )


# ---------------------------------------------------------------------------
# Stage: quantized_notes
# ---------------------------------------------------------------------------

class TestQuantizedNotes:
    def test_quantized_note_count(self, quantized_notes):
        non_rests = [n for n in quantized_notes if not n.is_rest]
        assert len(non_rests) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} non-rest QuantizedNotes, "
            f"got {len(non_rests)} (total with rests: {len(quantized_notes)})"
        )

    def test_quantized_pitches_unchanged(self, quantized_notes, note_events):
        quantized_midis = [n.midi_note for n in quantized_notes if not n.is_rest]
        event_midis = [e.midi_note for e in note_events]
        assert quantized_midis == event_midis


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_clef(self, score):
        part = score.parts[0]
        clefs = list(part.recurse().getElementsByClass(music21.clef.Clef))
        assert len(clefs) >= 1, "No clef found in score part"
        clef_obj = clefs[0]
        assert clef_obj.sign == EXPECTED_CLEF_SIGN, (
            f"Expected clef sign '{EXPECTED_CLEF_SIGN}', got '{clef_obj.sign}'"
        )
        assert clef_obj.octaveChange == -1, (
            f"Expected octaveChange -1 (8vb), got {clef_obj.octaveChange}"
        )

    def test_score_written_pitches(self, score):
        part = score.parts[0]
        notes = list(part.recurse().getElementsByClass(music21.note.Note))
        written_midis = sorted([n.pitch.midi for n in notes])
        assert written_midis == sorted(EXPECTED_WRITTEN_MIDI), (
            f"Written MIDIs {written_midis} != expected {sorted(EXPECTED_WRITTEN_MIDI)}"
        )


# ---------------------------------------------------------------------------
# Stage: tab_assignments
# ---------------------------------------------------------------------------

class TestTabAssignments:
    def test_tab_length(self, tab_assignments, quantized_notes):
        assert len(tab_assignments) == len(quantized_notes)

    def test_tab_all_fret_zero(self, tab_assignments):
        for assignment in tab_assignments:
            if assignment is not None:
                _, fret = assignment
                assert fret == 0, (
                    f"Open string expected fret 0, got {fret}"
                )

    def test_tab_exact_string_assignments(self, tab_assignments):
        non_none = [a for a in tab_assignments if a is not None]
        assert non_none == EXPECTED_TAB, (
            f"Tab assignments {non_none} != expected {EXPECTED_TAB}"
        )
```

- [ ] **Step 2: Run the tests**

```bash
conda run -n auto-sheet-music pytest tests/integration/test_bass_line_in.py -v
```

Expected: all 18 tests pass. Bass E1 (41.2 Hz) is the lowest note — if `test_note_count`
fails on the bass file specifically, it is likely a pYIN detection issue on the E1 string.
Inspect with the debug script from Task 2 Step 2 (substituting the bass path and `Instrument.BASS`).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_bass_line_in.py
git commit -m "test: add integration tests for bass line-in sample"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run the full integration suite**

```bash
conda run -n auto-sheet-music pytest tests/integration/ -v
```

Expected output: **72 passed** across 4 test modules. Each module runs its pipeline once
(4 pipeline runs total); the `-v` flag will show each test name and stage clearly.

- [ ] **Step 2: Run integration suite alongside existing unit tests**

```bash
conda run -n auto-sheet-music pytest -v
```

Expected: all existing unit tests still pass alongside the 72 new integration tests.
