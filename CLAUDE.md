# CLAUDE.md — audio_to_sheet

## Project Overview

`audio_to_sheet` is a macOS desktop application that captures live audio from a microphone
or USB audio interface and transcribes it into sheet music (PDF) and MIDI.

**Target instruments (MVP):** Standard guitar (E2–E6), bass guitar (B0–G4)
**Transcription scope:** Monophonic (single-note) first; chord support is a planned extension
**Platform:** macOS 13+ (Ventura or later)
**Python:** 3.11+

---

## Architecture

```
Audio Input (sounddevice)
       │
       ▼
  Preprocessing  [capture/preprocessor.py]
  - DC offset removal
  - Noise gate (RMS threshold)
  - Hann windowing
       │
       ▼
  Analysis Pipeline  [analysis/]
  - Onset detection  (librosa.onset.onset_detect)
  - Pitch estimation (CREPE — monophonic, per-frame confidence)
       │
       ▼
  Note Event Stream  [{pitch_midi, onset_s, offset_s, confidence}]
       │
       ▼
  Rhythm Quantizer  [quantization/]
  - Tempo detection (librosa.beat.beat_track)
  - Grid snapping to nearest note value (whole → 32nd)
  - Tie/beam grouping
  - Raw timing passthrough for MIDI export
       │
       ▼
  Score Builder  [notation/]
  - music21 Stream construction
  - Instrument-aware clef/transposition
  - Time signature inference
       │
       ▼
  Exporters  [export/]
  - PDF  via music21 → MusicXML → MuseScore CLI
  - MIDI via music21
       │
       ▼
  PyQt6 GUI  [gui/]
  - Device selector
  - Live waveform + spectrogram
  - Transcription log (scrolling note events)
  - Record / Stop / Export controls
```

---

## Module Map

| Path | Responsibility |
|---|---|
| `src/audio_to_sheet/capture/device.py` | List / select CoreAudio devices via sounddevice |
| `src/audio_to_sheet/capture/stream.py` | Continuous audio capture, ring buffer |
| `src/audio_to_sheet/capture/preprocessor.py` | DC removal, noise gate, windowing |
| `src/audio_to_sheet/analysis/onset.py` | Librosa onset detection wrapper |
| `src/audio_to_sheet/analysis/pitch.py` | CREPE pitch estimation, confidence filtering |
| `src/audio_to_sheet/analysis/pipeline.py` | Combines onset + pitch into NoteEvent stream |
| `src/audio_to_sheet/quantization/tempo.py` | BPM detection via librosa |
| `src/audio_to_sheet/quantization/grid.py` | Snap continuous timing to rhythmic grid |
| `src/audio_to_sheet/notation/score.py` | Build music21 Score from NoteEvents |
| `src/audio_to_sheet/notation/instruments.py` | Instrument profiles (clef, range, transposition) |
| `src/audio_to_sheet/export/pdf.py` | MusicXML → PDF via MuseScore |
| `src/audio_to_sheet/export/midi.py` | music21 → MIDI |
| `src/audio_to_sheet/gui/main_window.py` | PyQt6 main window, layout |
| `src/audio_to_sheet/gui/waveform_widget.py` | Real-time waveform display (pyqtgraph) |
| `src/audio_to_sheet/gui/spectrogram_widget.py` | Live CQT spectrogram |
| `src/audio_to_sheet/gui/transcription_log.py` | Scrolling note event log |
| `src/audio_to_sheet/config.py` | Central config / constants |
| `src/audio_to_sheet/models.py` | Shared dataclasses (NoteEvent, SessionConfig) |

---

## Key Design Decisions

### Pitch Detection: CREPE
CREPE (Convolutional REpresentation for Pitch Estimation) is chosen over pYIN because:
- Lower error rate on guitar/bass in empirical benchmarks
- Per-frame confidence scores allow us to gate unreliable frames
- ONNX export available for future mobile/edge deployment
- Processes frames at 10ms hop rate (model-side), sufficient for 16th notes at 120 BPM

Trade-off: CREPE adds ~200ms latency per inference chunk on CPU. For MVP this is acceptable
since we are doing post-hoc transcription of captured segments, not true real-time.

### Rhythm Quantization
Two modes are supported:
1. **Grid mode** — BPM detected via librosa.beat.beat_track, note onsets snapped to
   nearest grid division. Minimum note value: 32nd note. Produces standard notation.
2. **Raw mode** — Continuous timing preserved; exported directly to MIDI with no snapping.
   Useful for import into a DAW.

Grid snapping uses a dynamic programming approach that minimizes total quantization error
across the note sequence, rather than greedy nearest-neighbor, to avoid cascading drift.

### Audio Buffer Strategy
A thread-safe ring buffer (collections.deque with maxlen) accumulates frames from the
sounddevice InputStream callback. A separate analysis thread drains the buffer in
non-overlapping chunks sized to CREPE's expected input (1024 samples at 16 kHz = 64ms).
Resampling to 16 kHz happens in the analysis thread, not the callback, to keep the
callback latency minimal.

### Instrument Profiles
Two profiles for MVP:
- `guitar` — E2 (MIDI 40) to E6 (MIDI 88), treble clef, concert pitch
- `bass`   — B0 (MIDI 23) to G4 (MIDI 67), bass clef, concert pitch

Range gates are applied during pitch validation to discard spurious CREPE outputs outside
the instrument's physically possible range.

---

## Environment Setup

```bash
# 1. Install system dependencies
brew install musescore          # PDF rendering backend
brew install portaudio          # sounddevice C backend

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -e ".[dev]"

# 4. Verify MuseScore is on PATH
mscore --version
```

---

## Running the App

```bash
source .venv/bin/activate

# Launch GUI
python -m audio_to_sheet

# CLI capture (headless, useful for testing)
python -m audio_to_sheet.cli --device 0 --instrument guitar --duration 30 --output out/
```

---

## Running Tests

```bash
pytest                          # all tests
pytest tests/analysis/          # specific module
pytest -k "test_pitch"          # filter by name
pytest --cov=audio_to_sheet     # with coverage
```

---

## Debugging Tips

### Audio device issues
```python
import sounddevice as sd
sd.query_devices()
sd.check_input_settings(device=N, samplerate=44100)
```

### CREPE pitch output
Set `AUDIO_TO_SHEET_DEBUG=1` env var to write per-frame pitch/confidence CSVs to `/tmp/ats_debug/`.

### MuseScore PDF export failures
- Ensure `mscore` or `mscore3` is on PATH: `which mscore`
- Test manually: `mscore -o /tmp/test.pdf /tmp/test.musicxml`
- MuseScore 4 uses `mscore4` on some installs; set `MUSESCORE_BINARY` env var to override.

### Quantization drift
Enable `--debug-quantization` CLI flag to dump the raw vs. quantized onset timeline as CSV.

---

## Planned Extensions (Post-MVP)

- [ ] Chord detection (chroma + template matching, or basic-pitch polyphonic model)
- [ ] Tab notation output (guitar tablature via music21)
- [ ] LilyPond export for higher typesetting quality
- [ ] On-device CREPE via CoreML (Apple Silicon acceleration)
- [ ] Real-time streaming transcription with sliding window buffer
- [ ] Key signature detection
- [ ] Dynamics (velocity) estimation from RMS envelope

---

## Dependencies Reference

| Package | Version | Purpose |
|---|---|---|
| sounddevice | >=0.4.6 | CoreAudio capture |
| numpy | >=1.26 | Array operations |
| scipy | >=1.12 | Signal processing |
| librosa | >=0.10 | Onset detection, BPM, CQT |
| crepe | >=0.0.13 | Pitch estimation |
| music21 | >=9.1 | Score representation, MusicXML, MIDI |
| PyQt6 | >=6.6 | Desktop GUI |
| pyqtgraph | >=0.13 | Real-time waveform/spectrogram |
| pytest | >=8.0 | Testing |
| pytest-cov | >=5.0 | Coverage |
| pytest-qt | >=4.4 | PyQt6 widget testing |


## Code Style

### Typing

Python 3.11+ is required. Use modern built-in generic types throughout — do not
import or use deprecated aliases from `typing`:

- `list[str]` not `List[str]`
- `dict[str, Any]` not `Dict[str, Any]`
- `str | None` not `Optional[str]`
- `str | int` not `Union[str, int]`

`Literal`, `Any`, and `Callable` are still imported from `typing` as they have
no built-in equivalent.

### Docstrings

Use NumPy-style docstrings for all public functions and classes. Follow the
wording and structural conventions used throughout the `graph-ai` repository:

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