# Sample-Based Integration Tests — Design Spec

**Date:** 2026-05-05
**Branch:** tab-generation
**Status:** Approved

---

## Goal

Build end-to-end regression tests that run the full `improv_scribe` pipeline on real
audio samples and assert correctness at each pipeline stage. Tests catch regressions in
analysis (pitch, onset) and generation (quantization, score, tab) introduced by future
code changes.

---

## Samples

| File | Instrument | Notes (concert pitch, low → high) | Duration |
|---|---|---|---|
| `samples/guitar/6_string_acoustic_line_in.mp3` | Guitar | E2 A2 D3 G3 B3 E4 | ~12.3 s |
| `samples/guitar/6_string_acoustic_mic.mp3` | Guitar | E2 A2 D3 G3 B3 E4 | ~13.4 s |
| `samples/guitar/6_string_electric_line_in.mp3` | Guitar | E2 A2 D3 G3 B3 E4 | ~13.4 s |
| `samples/bass/4_string_bass_line_in.mp3` | Bass | E1 A1 D2 G2 | ~12.3 s |

Each sample plays one open string at a time in low-to-high order. Instruments were
calibrated before recording, so pitch accuracy is expected to be within ±0.5 semitone.

---

## Approach

**Stage-chained module-scoped fixtures (Approach A).** Each test file defines a chain of
`scope="module"` pytest fixtures that build the pipeline incrementally. Each fixture
calls only the previous stage's fixture as input; the pipeline runs once per test
module. A fixture failure at stage N propagates as `ERROR` (not `FAIL`) to all downstream
tests, immediately identifying the broken stage.

---

## File Organisation

```
tests/integration/
    __init__.py
    conftest.py                        # SAMPLE_ROOT constant; pipeline fixture factory
    test_guitar_acoustic_line_in.py
    test_guitar_acoustic_mic.py
    test_guitar_electric_line_in.py
    test_bass_line_in.py
```

`conftest.py` provides a `make_pipeline_fixtures()` factory that each test module calls
with its sample path and instrument, returning the full fixture chain. This avoids
copy-pasting eight fixtures into each of four files.

---

## Fixture Chain

```
audio           → librosa.load(SAMPLE_PATH, sr=44100, mono=True)
                       ↓
pitch_result    → PitchEstimator(config, backend="pyin").estimate(audio, profile)
onsets          →  OnsetDetector(config).detect(audio)          ← sibling, depends only on audio
                       ↓  (both pitch_result and onsets feed note_events)
note_events     → NoteTracker(config, profile).process(pitch_result, onsets)
                       ↓
tempo_result    → TempoEstimator(config).estimate(note_events)
                       ↓
quantized_notes → RhythmQuantizer(tempo_result).quantize(note_events)
                       ↓
score           → ScoreBuilder(profile, tempo_result).build(quantized_notes)
tab_assignments → ScoreBuilder(profile, tempo_result).compute_tab_assignments(quantized_notes)
                                                      ← sibling to score, same inputs
```

`pitch_result` and `onsets` are siblings (both depend only on `audio`).
`score` and `tab_assignments` are siblings (both depend only on `quantized_notes`).

---

## Ground-Truth Constants

### Guitar (shared across all three guitar test files)

```python
NOTE_COUNT            = 6
INSTRUMENT            = Instrument.GUITAR

# Concert (sounding) MIDI notes, low string → high string
EXPECTED_MIDI         = [40, 45, 50, 55, 59, 64]   # E2 A2 D3 G3 B3 E4

# Written MIDI: guitar transpose_semitones=-12 → written_midi = midi_note + 12
EXPECTED_WRITTEN_MIDI = [52, 57, 62, 67, 71, 76]   # E3 A3 D4 G4 B4 E5

# Tab: every open string → fret 0, string_idx 0-based from lowest string
EXPECTED_TAB          = [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)]

EXPECTED_CLEF         = "treble8vb"
```

### Bass

```python
NOTE_COUNT            = 4
INSTRUMENT            = Instrument.BASS

# Concert (sounding) MIDI notes
EXPECTED_MIDI         = [28, 33, 38, 43]            # E1 A1 D2 G2

# Written MIDI: bass transpose_semitones=-12 → written_midi = midi_note + 12
EXPECTED_WRITTEN_MIDI = [40, 45, 50, 55]            # E2 A2 D3 G3

EXPECTED_TAB          = [(0,0),(1,0),(2,0),(3,0)]

EXPECTED_CLEF         = "bass8vb"
```

---

## Per-Stage Assertions

### Stage: audio

| Test | Assertion |
|---|---|
| `test_audio_shape` | 1-D array, `dtype=float32`, non-zero length |
| `test_audio_sample_rate` | `sr == 44100` |
| `test_audio_duration` | duration within ±2 s of expected file duration |

### Stage: pitch_result

| Test | Assertion |
|---|---|
| `test_pitch_voiced_frames_nonempty` | `len(voiced_frames) > 0` |
| `test_pitch_frequency_range` | all voiced frame `freq_hz` within `[profile.freq_min_hz, profile.freq_max_hz]` |

### Stage: onsets

| Test | Assertion |
|---|---|
| `test_onset_count` | `NOTE_COUNT ≤ len(onsets) ≤ NOTE_COUNT + 2` |
| `test_onsets_sorted` | onset times strictly increasing |

### Stage: note_events

| Test | Assertion |
|---|---|
| `test_note_count` | `len(note_events) == NOTE_COUNT` (exact) |
| `test_note_pitches` | each `midi_note` within ±0.5 semitone of sorted `EXPECTED_MIDI` |
| `test_notes_in_instrument_range` | all `midi_notes` within `[profile.midi_min, profile.midi_max]` |

### Stage: tempo_result

| Test | Assertion |
|---|---|
| `test_tempo_positive` | `40 ≤ bpm ≤ 250` |

### Stage: quantized_notes

| Test | Assertion |
|---|---|
| `test_quantized_note_count` | non-rest count == `NOTE_COUNT` |
| `test_quantized_pitches_unchanged` | non-rest `midi_notes` match `note_events` (quantizer must not alter pitch) |

### Stage: score

| Test | Assertion |
|---|---|
| `test_score_clef` | first clef in the part matches `EXPECTED_CLEF` |
| `test_score_written_pitches` | non-rest note pitches (as MIDI) match `EXPECTED_WRITTEN_MIDI` |

### Stage: tab_assignments

| Test | Assertion |
|---|---|
| `test_tab_length` | `len(tab_assignments) == len(quantized_notes)` |
| `test_tab_all_fret_zero` | every non-`None` assignment has `fret == 0` |
| `test_tab_exact_string_assignments` | non-`None` assignments in onset order match `EXPECTED_TAB` |

---

## Tolerance Notes

- **Pitch tolerance:** ±0.5 semitone at the `note_events` stage. `NoteEvent.midi_note`
  is already rounded to the nearest integer by `hz_to_midi()`, so comparing
  `abs(detected_midi - expected_midi) <= 0.5` where both sides are integers is
  effectively an exact-equality test. This is intentional: calibrated open-string
  recordings should land exactly on the correct semitone after rounding.
- **Onset count:** allowed up to `NOTE_COUNT + 2` to tolerate occasional double-fire on
  attack transients. The stricter `note_events` count test is the real "all notes
  detected" gate.
- **Tab assignments:** exact — no tolerance. Open strings always map to fret 0 on a
  known string; any other assignment is a regression. Assignments are compared in
  onset order (low string played first = ascending pitch order), not re-sorted.

---

## What Is Not Tested

- PDF / MIDI file output (requires MuseScore CLI; integration with external tools is out
  of scope for this regression suite).
- Chord detection (monophonic only).
- Real-time streaming performance.
