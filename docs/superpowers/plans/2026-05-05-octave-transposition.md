# Octave Transposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure guitar and bass scores are written one octave above sounding pitch (8vb convention), using the correct 8vb clef, without contaminating MIDI export with the written-pitch offset.

**Architecture:** The pipeline keeps sounding pitch throughout capture → analysis → quantization. The `+12` offset is applied exclusively inside `ScoreBuilder.build()` (for PDF/notation). `build_raw()` stays at sounding pitch and is used for MIDI. The 8vb clef visually indicates the convention to performers without affecting MuseScore's display logic.

**Tech Stack:** music21, pytest, Python 3.13

---

## File Map

| File | Change |
|---|---|
| `src/improv_scribe/analysis/instrument_profiles.py` | Guitar: `clef="treble8vb"`. Bass: `clef="bass8vb"`, `transpose_semitones=-12`. |
| `src/improv_scribe/notation/score_builder.py` | Replace `clefFromString(profile.clef)` with `_resolve_clef(profile.clef)` lookup. Fix `build_raw()` MetronomeMark rounding. |
| `src/improv_scribe/cli.py` | Build a sounding-pitch MIDI score via `build_raw(quantized_notes)` instead of reusing the written-pitch `score`. |
| `src/improv_scribe/gui/main_window.py` | Same: store `_last_midi_score` (sounding pitch) separately from `_last_score` (written pitch). |
| `tests/notation/test_score_builder.py` | Fix tempo test; add written-pitch and 8vb-clef assertions. |

---

### Task 1: Update instrument profiles

**Files:**
- Modify: `src/improv_scribe/analysis/instrument_profiles.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/notation/test_score_builder.py`:

```python
class TestOctaveTransposition:
    def test_guitar_note_written_up_octave(self, guitar_profile):
        """Sounding C4 (MIDI 60) must appear as C5 (MIDI 72) in the score."""
        import music21.note
        notes = [_make_note(60, 1.0)]
        score = ScoreBuilder(guitar_profile, _make_tempo()).build(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert score_notes[0].pitch.midi == 72

    def test_bass_note_written_up_octave(self, bass_profile):
        """Sounding E1 (MIDI 28) must appear as E2 (MIDI 40) in the score."""
        import music21.note
        notes = [_make_note(28, 1.0)]
        score = ScoreBuilder(bass_profile, _make_tempo()).build(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert score_notes[0].pitch.midi == 40

    def test_guitar_uses_treble8vb_clef(self, guitar_profile):
        import music21.clef
        score = ScoreBuilder(guitar_profile, _make_tempo()).build([_make_note(60, 1.0)])
        clefs = list(score.flatten().getElementsByClass(music21.clef.Clef))
        assert any(isinstance(c, music21.clef.Treble8vbClef) for c in clefs)

    def test_bass_uses_8vb_clef(self, bass_profile):
        import music21.clef
        score = ScoreBuilder(bass_profile, _make_tempo()).build([_make_note(28, 1.0)])
        clefs = list(score.flatten().getElementsByClass(music21.clef.Clef))
        # Bass8vbClef or BassClef with octaveChange=-1
        assert any(
            (isinstance(c, music21.clef.BassClef) and getattr(c, "octaveChange", 0) == -1)
            or type(c).__name__ == "Bass8vbClef"
            for c in clefs
        )

    def test_build_raw_stays_at_sounding_pitch(self, guitar_profile):
        """build_raw() must NOT apply the octave offset — it feeds MIDI export."""
        import music21.note
        notes = [_make_note(60, 1.0)]
        score = ScoreBuilder(guitar_profile, _make_tempo()).build_raw(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert score_notes[0].pitch.midi == 60

    def test_tempo_mark_is_rounded(self, guitar_profile):
        """Fractional BPM from librosa must be rounded to nearest integer."""
        notes = [_make_note(60, 1.0)]
        builder = ScoreBuilder(guitar_profile, _make_tempo(bpm=142.7))
        score = builder.build(notes)
        marks = list(score.flatten().getElementsByClass(music21.tempo.MetronomeMark))
        assert marks[0].number == 143
```

- [ ] **Step 2: Run to confirm all six new tests fail**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py::TestOctaveTransposition -v
```

Expected: 6 FAILED (bass transpose=0, plain clefs, tempo not rounded).

- [ ] **Step 3: Update `instrument_profiles.py`**

Replace the `PROFILES` dict:

```python
PROFILES: dict[Instrument, InstrumentProfile] = {
    Instrument.GUITAR: InstrumentProfile(
        name="Guitar (standard)",
        instrument=Instrument.GUITAR,
        freq_min_hz=73.42,    # D2 — 2 semitones below E2 for CREPE headroom
        freq_max_hz=1174.66,  # D6
        midi_min=40,          # E2
        midi_max=98,          # D6
        clef="treble8vb",     # 8vb treble: written an octave above sounding
        transpose_semitones=-12,
    ),
    Instrument.BASS: InstrumentProfile(
        name="Bass Guitar (standard scale)",
        instrument=Instrument.BASS,
        freq_min_hz=38.89,    # D1 — 2 semitones below E1 for CREPE headroom
        freq_max_hz=293.66,   # D4
        midi_min=28,          # E1 (sounding)
        midi_max=62,          # D4 (sounding)
        clef="bass8vb",       # 8vb bass: written an octave above sounding
        transpose_semitones=-12,
        noise_gate_rms_override=0.015,
    ),
}
```

- [ ] **Step 4: Run the new tests** — all should still fail (clef resolution not done yet)

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py::TestOctaveTransposition -v
```

Expected: `test_bass_note_written_up_octave` now PASSES (transpose=-12 takes effect). Others still FAIL.

- [ ] **Step 5: Commit the profile change**

```bash
git add src/improv_scribe/analysis/instrument_profiles.py tests/notation/test_score_builder.py
git commit -m "feat: set bass transpose_semitones=-12 and 8vb clefs for both instruments"
```

---

### Task 2: Resolve 8vb clef strings in ScoreBuilder

**Files:**
- Modify: `src/improv_scribe/notation/score_builder.py`

- [ ] **Step 1: Add `_resolve_clef` and fix `build_raw` rounding**

In `score_builder.py`, add a module-level mapping and a helper just before the `ScoreBuilder` class definition:

```python
import music21.clef

# Map profile clef strings to music21 clef constructors.
# BassClef with octaveChange=-1 is the portable substitute for Bass8vbClef
# across music21 versions that may not define the subclass explicitly.
def _resolve_clef(clef_str: str) -> music21.clef.Clef:
    if clef_str == "treble8vb":
        return music21.clef.Treble8vbClef()
    if clef_str in ("bass8vb",):
        try:
            return music21.clef.Bass8vbClef()
        except AttributeError:
            c = music21.clef.BassClef()
            c.octaveChange = -1
            return c
    return music21.clef.clefFromString(clef_str)
```

Replace the clef insertion line inside `build()`:

```python
        # Clef
        clef_obj = _resolve_clef(self._profile.clef)
        part.append(clef_obj)
```

Also fix `build_raw()` — it currently passes a raw float BPM to MetronomeMark:

```python
    def build_raw(self, notes: list[QuantizedNote]) -> music21.stream.Score:
        score = music21.stream.Score()
        part = music21.stream.Part()
        part.insert(0, music21.tempo.MetronomeMark(number=round(self._tempo_result.bpm)))
        ...
```

- [ ] **Step 2: Run all new tests**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py::TestOctaveTransposition -v
```

Expected: all 6 PASS.

- [ ] **Step 3: Run full test suite to check regressions**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py -v
```

Expected: all tests pass, including the pre-existing `test_bass_profile_uses_bass_clef` (bass8vb clef has `sign=="F"`, satisfying that assertion).

- [ ] **Step 4: Commit**

```bash
git add src/improv_scribe/notation/score_builder.py
git commit -m "feat: resolve treble8vb/bass8vb clef strings; round BPM in build_raw"
```

---

### Task 3: Fix MIDI export to use sounding pitch

**Context:** `ScoreBuilder.build()` now emits written pitch (+12). `MIDIExporter.quantized_from_score()` writes whatever pitch is in the score directly to MIDI. This causes MIDI notes to be 12 semitones too high. The fix: use `build_raw(quantized_notes)` to produce a sounding-pitch score for MIDI export.

**Files:**
- Modify: `src/improv_scribe/cli.py`
- Modify: `src/improv_scribe/gui/main_window.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/notation/test_score_builder.py` inside `TestOctaveTransposition`:

```python
    def test_build_raw_quantized_preserves_timing(self, guitar_profile):
        """build_raw with quantized notes must preserve beat offsets (for MIDI)."""
        import music21.note
        notes = [_make_note(60, 1.0, 0.0), _make_note(64, 1.0, 1.0)]
        score = ScoreBuilder(guitar_profile, _make_tempo()).build_raw(notes)
        score_notes = list(score.flatten().getElementsByClass(music21.note.Note))
        assert len(score_notes) == 2
        assert score_notes[0].pitch.midi == 60
        assert score_notes[1].pitch.midi == 64
```

- [ ] **Step 2: Run to confirm it passes** (build_raw already works — this is a guard test)

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py::TestOctaveTransposition::test_build_raw_quantized_preserves_timing -v
```

Expected: PASS.

- [ ] **Step 3: Update `cli.py`**

In the `run()` function, after building `score` and `tab_assignments`, build a separate score for MIDI at sounding pitch:

```python
    if args.mode == "auto":
        print("Quantizing rhythm …")
        quantizer = RhythmQuantizer(tempo_result)
        quantized_notes = quantizer.quantize(events)
        score_builder = ScoreBuilder(profile, tempo_result, title=input_path.stem)
        score = score_builder.build(quantized_notes)          # written pitch → PDF
        midi_score = score_builder.build_raw(quantized_notes) # sounding pitch → MIDI
        tab_assignments = score_builder.compute_tab_assignments(quantized_notes)
        print(f"  {len(quantized_notes)} quantized elements (notes + rests)")
```

Then in the MIDI export block, replace `score` with `midi_score`:

```python
    if args.output_midi:
        print(f"Exporting MIDI → {args.output_midi} …")
        midi_exporter = MIDIExporter(config)
        if args.mode == "auto":
            out = midi_exporter.quantized_from_score(midi_score, Path(args.output_midi))
        else:
            out = midi_exporter.raw_from_events(events, tempo_result, Path(args.output_midi))
        print(f"  MIDI written: {out}")
```

- [ ] **Step 4: Update `gui/main_window.py`**

Add `_last_midi_score` to `__init__`:

```python
        self._last_score = None
        self._last_events = None
        self._last_quantized_notes = None
        self._last_tab_assignments = None
        self._last_profile: object = None
        self._last_midi_score = None   # sounding-pitch score for MIDI export
```

In `_run_pipeline()`, after building `score`:

```python
            if rhythm_mode == "auto":
                quantizer = RhythmQuantizer(tempo_result)
                quantized_notes = quantizer.quantize(events)
                score_builder = ScoreBuilder(profile, tempo_result)
                score = score_builder.build(quantized_notes)            # written pitch → PDF
                midi_score = score_builder.build_raw(quantized_notes)  # sounding pitch → MIDI
                self._last_quantized_notes = quantized_notes
                self._last_tab_assignments = score_builder.compute_tab_assignments(quantized_notes)
                self._last_profile = profile
                self._last_midi_score = midi_score
```

In `_on_processing_done()`, store it:

```python
    def _on_processing_done(self, score: object, events: object) -> None:
        self._last_score = score
        self._last_events = events
        ...
```

(No change needed here — `_last_midi_score` is already set in `_run_pipeline()` before the signal fires.)

In `_on_export_midi()`, use `_last_midi_score`:

```python
    def _on_export_midi(self) -> None:
        if self._last_events is None:
            return
        ...
        try:
            exporter = MIDIExporter(self._config)
            from improv_scribe.quantization.tempo import TempoEstimator
            tempo_estimator = TempoEstimator(self._config)
            tempo_result = tempo_estimator.estimate(self._last_events)

            if self._last_midi_score is not None and self._rhythm_mode == "auto":
                out = exporter.quantized_from_score(self._last_midi_score, Path(path))
            else:
                out = exporter.raw_from_events(self._last_events, tempo_result, Path(path))
            ...
```

- [ ] **Step 5: Run the full test suite**

```bash
conda run -n auto-sheet-music pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/improv_scribe/cli.py src/improv_scribe/gui/main_window.py
git commit -m "fix: use sounding-pitch score for MIDI export; written-pitch score only for PDF"
```

---

## Self-Review

**Spec coverage:**
- Guitar written up octave → Task 1 (profile) + Task 2 (clef) ✓
- Bass written up octave → Task 1 (transpose_semitones=-12) ✓
- High E4 on top space of treble clef → E4 sounding + 12 = E5, which is the top space of Treble8vbClef ✓
- MIDI unaffected → Task 3 ✓
- TAB unaffected → tab_builder uses `qn.midi_note` (sounding pitch), unchanged ✓

**Placeholder scan:** None.

**Type consistency:** `_resolve_clef` returns `music21.clef.Clef`; used as `clef_obj` passed to `part.append()` — matches usage pattern throughout.
