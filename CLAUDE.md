# CLAUDE.md — improv_scribe

## Project Overview

`improv_scribe` is a macOS desktop application that captures live audio from a microphone
or USB audio interface and transcribes it into sheet music (PDF) and MIDI.

**Target instruments (MVP):** Standard guitar (E2–D6, MIDI 40–86), 4-string bass (E1–D4, MIDI 28–62)
**Transcription scope:** Monophonic + chords on guitar (via basic-pitch); bass is monophonic (`max_polyphony=1`)
**Platform:** macOS 13+ (Ventura or later)
**Python:** 3.13 (pinned in `environment.yaml`)

---

## Architecture

```
Audio Input (sounddevice)  [capture/audio_input.py]
  - Device management, streaming, ring buffer
  - Noise gate (RMS threshold)  [capture/noise_gate.py]
       │
       ▼
  Analysis Pipeline  [analysis/]
  - Onset detection  (librosa.onset.onset_detect)  [onset.py]
  - Pitch estimation [pitch.py] — pluggable backends:
      basic_pitch (default): polyphonic, emits assembled note events
      pyin / crepe (torchcrepe): monophonic, per-frame f0 + confidence
  - NoteTracker [note_tracker.py]: chord clustering, cluster-aware
    amplitude filtering, ring-over suppression, onset gating
       │
       ▼
  NoteEvent Stream  [{midi_notes tuple, onset_s, offset_s, confidences}]
       │
       ▼
  Rhythm Quantizer  [quantization/]
  - Tempo detection (librosa.beat.beat_track)  [tempo.py]
  - Grid snapping to nearest note value (whole → 32nd)  [grid.py]
  - Raw timing passthrough for MIDI export
       │
       ▼
  Notation  [notation/]
  - music21 Score construction  [score_builder.py]
  - Fret/string assignment via DP  [tab_builder.py]
       │
       ▼
  Exporters  [export/]
  - PDF: music21 → MusicXML → tab-staff injection [tab_xml.py]
         → MuseScore CLI  [pdf_exporter.py]
  - MIDI via music21  [midi_exporter.py]
       │
       ▼
  PyQt6 GUI  [gui/]
  - Live waveform + spectrogram
  - Transport bar (Record / Stop / Export)  [transport.py]
```

---

## Module Map

| Path | Responsibility |
|---|---|
| `src/improv_scribe/capture/audio_input.py` | CoreAudio device management, streaming, ring buffer |
| `src/improv_scribe/capture/noise_gate.py` | Energy-threshold noise gate |
| `src/improv_scribe/analysis/onset.py` | Librosa onset detection wrapper |
| `src/improv_scribe/analysis/pitch.py` | Pitch backends: basic_pitch (default), pyin, crepe |
| `src/improv_scribe/analysis/note_tracker.py` | NoteEvent assembly: chord clustering + filtering (bp path), onset+frame fusion (pyin/crepe path) |
| `src/improv_scribe/analysis/instrument_profiles.py` | Instrument profiles (range, clef, max_polyphony) |
| `src/improv_scribe/quantization/tempo.py` | BPM detection via librosa |
| `src/improv_scribe/quantization/grid.py` | Snap continuous timing to rhythmic grid |
| `src/improv_scribe/notation/score_builder.py` | Build music21 Score from QuantizedNotes |
| `src/improv_scribe/notation/tab_builder.py` | Chord-aware (string, fret) assignment via DP |
| `src/improv_scribe/export/pdf_exporter.py` | MusicXML → PDF via MuseScore CLI |
| `src/improv_scribe/export/tab_xml.py` | Inject linked TAB staff into MusicXML |
| `src/improv_scribe/export/midi_exporter.py` | music21 → MIDI |
| `src/improv_scribe/gui/main_window.py` | PyQt6 main window, layout |
| `src/improv_scribe/gui/waveform_widget.py` | Real-time waveform display (pyqtgraph) |
| `src/improv_scribe/gui/spectrogram_widget.py` | Live CQT spectrogram |
| `src/improv_scribe/gui/transport.py` | Transport bar (Record / Stop / Export) |
| `src/improv_scribe/config.py` | Central config / constants (env prefix `ATS_`) |
| `src/improv_scribe/cli.py` | Headless batch transcription |

---

## Key Design Decisions

### Pitch Detection: basic-pitch (default), with pyin/crepe fallbacks
Spotify's basic-pitch (ONNX) is the default backend (`ATS_PITCH_BACKEND=basic_pitch`)
because it handles polyphony — it emits assembled note events directly, enabling
chord transcription. `predict()` is called with the instrument profile's frequency
bounds to suppress out-of-range hallucinations at the source.

The model's raw recall is excellent; transcription quality is determined by the
**post-filtering in `note_tracker.py`**, which is cluster-context-aware (singleton
vs. chord-member amplitude floors, ring-over suppression of still-ringing strings,
onset gating of weak clusters, per-instrument `max_polyphony`). All thresholds in
`config.py` were calibrated against the sample corpus — see
`docs/precision_audit_basic_pitch.md` before changing any of them; it documents the
measured failure modes and the precision/recall sweeps behind each value.

Fallback backends: `pyin` (librosa, CPU-only, no model weights) and `crepe`
(torchcrepe port, MPS-accelerated) produce frame-level f0; the NoteTracker fuses
them with librosa onsets and applies octave-error correction. These remain
monophonic-only.

### Rhythm Quantization
Two modes are supported:
1. **Grid mode** — BPM detected via librosa.beat.beat_track, note onsets snapped to
   nearest grid division. Minimum note value: 32nd note. Produces standard notation.
2. **Raw mode** — Continuous timing preserved; exported directly to MIDI with no snapping.
   Useful for import into a DAW.

Grid snapping uses a dynamic programming approach that minimizes total quantization error
across the note sequence, rather than greedy nearest-neighbor, to avoid cascading drift.

### Audio Buffer Strategy
A thread-safe ring buffer accumulates frames from the sounddevice InputStream
callback (`capture/audio_input.py`); analysis runs batch-style on the captured
segment, not in the callback.

### Instrument Profiles
Two profiles for MVP (`analysis/instrument_profiles.py`):
- `guitar` — E2 (MIDI 40) to D6 (MIDI 86, 22nd fret on high E), `treble8vb` clef,
  `max_polyphony=6`
- `bass`   — E1 (MIDI 28) to D4 (MIDI 62), `bass8vb` clef, `max_polyphony=1`
  (monophonic: ringing bass strings otherwise form false dyads)

Range gates discard model outputs outside the instrument's physically possible
range. Scores are written at concert pitch; the 8vb clef alone carries the octave
offset (no `<transpose>` element — combining the two breaks MuseScore TAB frets).

### Tablature
`tab_builder.assign_frets()` picks (string, fret) per chord via DP minimizing hand
stretch + position shifts; transitions to/from all-open shapes are free, with a
small low-fret preference (`_POSITION_EPS`) breaking the resulting ties. The TAB
staff is injected into the MusicXML post-hoc (`export/tab_xml.py`) because
music21's tablature support is incomplete. Invariant: `tuning[string] + fret`
must equal the annotated note's MIDI on both staves.

---

## Environment Setup

This project uses **Anaconda** for environment management. Do not use `venv` or `pip` directly.

```bash
# 1. Install system dependencies
brew install musescore          # PDF rendering backend
brew install portaudio          # sounddevice C backend

# 2. Create the conda environment from the spec file
conda env create -f environment.yaml

# 3. Activate the environment
conda activate auto-sheet-music

# 4. Verify MuseScore is on PATH
mscore --version

# 5. Optional: install basic-pitch polyphonic backend (Phase 1+)
bash scripts/install_basic_pitch.sh
```

The environment name is **`auto-sheet-music`** (defined in `environment.yaml`).
Python version: **3.13** (as specified in `environment.yaml`).

Note: `pytest` and `ruff` are installed in the env but are not pinned in
`environment.yaml`.

To run any command in the environment without activating it interactively:
```bash
conda run -n auto-sheet-music <command>
```

---

## Running the App

```bash
conda activate auto-sheet-music

# Launch GUI
python -m improv_scribe

# CLI batch transcription (headless, useful for testing)
python -m improv_scribe.cli --input recording.wav --instrument guitar \
    --backend basic_pitch --mode auto \
    --output-pdf out/score.pdf --output-midi out/score.mid
```

Gotcha: the CLI's `--backend` default is `pyin`, while the config default
(`ATS_PITCH_BACKEND`, used by GUI and tests) is `basic_pitch`. Pass
`--backend basic_pitch` explicitly for chord support.

---

## Running Tests

```bash
conda run -n auto-sheet-music pytest                       # all tests
conda run -n auto-sheet-music pytest tests/analysis/       # specific module
conda run -n auto-sheet-music pytest -k "test_pitch"       # filter by name
conda run -n auto-sheet-music pytest --cov=improv_scribe  # with coverage
```

### Required integration tests after analysis pipeline changes

Any change to `src/improv_scribe/analysis/`, `src/improv_scribe/capture/`,
`src/improv_scribe/quantization/`, or `src/improv_scribe/config.py` **must** be
followed by a passing run of the full sample-based integration suite before the
work is considered complete:

```bash
conda run -n auto-sheet-music pytest tests/integration/ -v
```

These tests run the full pipeline end-to-end against real recordings and assert
exact pitch, fret, and tab correctness. They are the primary regression gate for
analysis accuracy. Core mono samples:

| Test file | Sample | Verifies |
|---|---|---|
| `tests/integration/test_guitar_electric_line_in.py` | `samples/guitar/6_string_electric_line_in.mp3` | Clean signal baseline — must pass first |
| `tests/integration/test_guitar_acoustic_mic.py` | `samples/guitar/6_string_acoustic_mic.mp3` | Mic noise, room acoustics, octave-error robustness |
| `tests/integration/test_guitar_acoustic_line_in.py` | `samples/guitar/6_string_acoustic_line_in.mp3` | Acoustic line-in (no mic noise) |
| `tests/integration/test_bass_line_in.py` | `samples/bass/4_string_bass_line_in.mp3` | Bass range, low-frequency accuracy |

Additional basic-pitch-only suites: `test_guitar_open_{A,C,D,E,G}_chord.py`
(strummed open chords), `test_guitar_dyad_{third,fifth,octave}.py` (interval
dyads), and `test_pdf_render_smoke.py` (full notation+TAB PDF render).

Start with the electric guitar line-in test (cleanest signal). If it fails, fix
that before checking the others — a failure there indicates a fundamental pipeline
bug rather than a noise/edge-case issue.

Expected tuples in the chord/dyad tests are musically-verified ground truth
re-derived 2026-06-10 (see `docs/precision_audit_basic_pitch.md`). Some encode
known limitations (octave-harmonic doubles on mic'd acoustic, ring-over members
on acoustic line-in) — these are documented in each test's comments. Do not
"recalibrate" expectations to whatever the pipeline currently emits; verify
musically first.

## Linting

Use **Ruff** for linting (included in the conda environment):

```bash
conda run -n auto-sheet-music ruff check src/ tests/
conda run -n auto-sheet-music ruff check --fix src/ tests/  # auto-fix safe issues
```

---

## Debugging Tips

### Audio device issues
```python
import sounddevice as sd
sd.query_devices()
sd.check_input_settings(device=N, samplerate=44100)
```

### Pitch debug output
Set `ATS_DEBUG_PITCH=1` to accumulate per-frame pitch/confidence rows (pyin/crepe
backends; basic-pitch emits no frame data) — written to `/tmp/ats_pitch_debug.csv`.

### MuseScore PDF export failures
- PDFExporter checks PATH (`mscore`/`musescore`) first, then falls back to
  `ATS_MUSESCORE_PATH` (default: `/Applications/MuseScore 4.app/Contents/MacOS/mscore`).
- Test manually: `mscore --force --export-to /tmp/test.pdf /tmp/test.musicxml`
- On failure, the intermediate MusicXML is kept at `/tmp/ats_last_export.musicxml`
  for post-mortem inspection.

### Tuning the basic-pitch filtering
All thresholds are env-overridable (`ATS_POLYPHONIC_*`, `ATS_RING_*`,
`ATS_ONSET_GATE_*`, `ATS_ONSET_GROUPING_WINDOW_MS`, ...) — see `config.py`.
Consult `docs/precision_audit_basic_pitch.md` before changing defaults.

---

## Planned Extensions (Post-MVP)

- [x] Chord detection (basic-pitch polyphonic model + cluster-aware filtering)
- [x] Tab notation output (guitar/bass tablature via MusicXML injection — see `notation/tab_builder.py`, `export/tab_xml.py`)
- [ ] LilyPond export for higher typesetting quality
- [ ] Spectral-presence check for octave-double / ring-over false positives (see audit report limitations)
- [ ] Real-time streaming transcription with sliding window buffer
- [ ] Key signature detection
- [ ] Dynamics (velocity) estimation from RMS envelope

---

## Dependencies Reference

| Package | Version | Purpose |
|---|---|---|
| sounddevice | >=0.4.6 | CoreAudio capture |
| numpy | >=1.26 | Array operations |
| scipy | — | Signal processing |
| librosa | >=0.11 | Onset detection, BPM, CQT, pyin |
| basic-pitch | via `scripts/install_basic_pitch.sh` | Default polyphonic pitch backend (ONNX) |
| onnxruntime | >=1.17 | basic-pitch inference |
| torch / torchcrepe | latest | Optional CREPE backend (MPS-accelerated) |
| soundfile | >=0.12 | WAV I/O for the basic-pitch backend |
| music21 | >=9.1 | Score representation, MusicXML, MIDI |
| pretty-midi / mido | >=0.2.9 / >=1.3 | MIDI plumbing |
| PyQt6 | >=6.6 | Desktop GUI |
| pyqtgraph | — | Real-time waveform/spectrogram |
| pytest / ruff | in env, unpinned | Testing / linting |

basic-pitch is installed by `scripts/install_basic_pitch.sh` *after*
`conda env create` (works around its TensorFlow base-dependency).


## Code Style

### Typing

Python 3.13 is required. Use modern built-in generic types throughout — do not
import or use deprecated aliases from `typing`:

- `list[str]` not `List[str]`
- `dict[str, Any]` not `Dict[str, Any]`
- `str | None` not `Optional[str]`
- `str | int` not `Union[str, int]`

`Literal` and `Any` are still imported from `typing` as they have no built-in
equivalent. `Callable` should be imported from `collections.abc` (Ruff UP035).

### Docstrings

Use NumPy-style docstrings for all public functions and classes. Follow the
wording and structural conventions used throughout the `improv-scribe` repository:

```python
def my_function(x: int, y: str | None = None) -> dict[str, Any]:
    """One-line summary of what the function does.

    Optional extended description if the behaviour needs more explanation.

    Parameters
    ----------
    x : int
        Description of x.
    y : str | None, optional
        Description of y, by default None.

    Returns
    -------
    dict[str, Any]
        Description of the return value.
    """
```

Class docstrings document `__init__` parameters under a `Parameters` section on
the class itself (not on `__init__`). Methods follow the same function pattern.