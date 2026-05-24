# Phase 4 — Quantizer Overlap Fix + CLI Backend Choice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `RhythmQuantizer.quantize()` to snap onsets and offsets to a single fine grid (1/12 beat default) using a "largest catalog duration that fits" rule, preventing the MusicXML overlap that blocks PDF rendering on chord samples. Add `basic_pitch` to the CLI's `--backend` choices.

**Architecture:** Two contained edits. (1) `src/improv_scribe/quantization/grid.py` — `RhythmQuantizer` gains a single `grid_beats` attribute computed at init; `quantize()` rewritten around snap-both-endpoints + largest-fitting-duration. (2) `src/improv_scribe/cli.py` — argparse `--backend` choices list gains `"basic_pitch"`. Five new unit tests in `tests/quantization/test_grid.py` assert the tiling invariant on adversarial inputs. One new PDF smoke test in `tests/integration/test_pdf_render_smoke.py` asserts the G chord sample renders to a non-trivial PDF.

**Tech Stack:** Python 3.13, pytest, music21, MuseScore CLI

**Spec reference:** [docs/superpowers/specs/2026-05-23-phase-4-quantizer-and-cli-design.md](../specs/2026-05-23-phase-4-quantizer-and-cli-design.md)

**Phase gate (definition of done):**
```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
```
Expected: basic_pitch **318 passed, 4 skipped** (was 312/4 — +6 new tests); crepe **289 passed, 33 skipped** (was 284/32 — +5 unit tests, +1 chord-only PDF smoke skip).

---

## File Map

| File | Change |
|---|---|
| `src/improv_scribe/quantization/grid.py` | `RhythmQuantizer.__init__` computes `self._grid_beats`. `quantize()` rewritten: snap onset+offset to grid, pick largest fitting NoteDuration, recompute offset from chosen catalog duration, emit rests from snapped-onset-to-onset gap. Helpers `_largest_fitting_duration` and `_snap_to_grid` retained/refactored. |
| `src/improv_scribe/cli.py` | argparse `--backend` choices: `["pyin", "crepe"]` → `["pyin", "crepe", "basic_pitch"]`. Help text updated. |
| `tests/quantization/test_grid.py` | Append `TestQuantizerTiling` class with 5 tests. |
| `tests/integration/test_pdf_render_smoke.py` *(new)* | One test: G chord renders to >5KB PDF. |
| `docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md` | Append §17 (Phase 4 outcome) after implementation. |

---

## Task 1: Confirm baseline + read existing code

**Files:** none (read-only).

- [ ] **Step 1: Verify clean state**

```bash
cd /Users/davehollander/Documents/Personal/Projects/audio_to_sheet
git status
git log --oneline -3
```

Expected: clean working tree, HEAD at `8276ccd` (Phase 4 spec commit) or later.

- [ ] **Step 2: Confirm editable install + baseline tests**

```bash
conda run -n auto-sheet-music pip show improv_scribe | grep "Editable"
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
```

Expected:
- `Editable project location: /Users/davehollander/Documents/Personal/Projects/audio_to_sheet`
- basic_pitch: 312 passed, 4 skipped
- crepe: 284 passed, 32 skipped

If any baseline fails, STOP and fix before proceeding.

- [ ] **Step 3: Read the files you'll be editing**

```bash
cat src/improv_scribe/quantization/grid.py
cat src/improv_scribe/cli.py | head -180
cat tests/quantization/test_grid.py | head -60
```

You should understand:
- `RhythmQuantizer.__init__` builds `self._candidates` from the duration catalog.
- The current `quantize()` snaps onset, snaps duration independently, computes `prev_end_beat` from `snapped_onset + dur_beats` (which can advance differently than the next event's `snapped_onset` — this is the bug).
- The `_min_dur_beats()` helper returns `min(frac * 4 for frac in candidates)` — this is what the new `_grid_beats` attribute replaces.

---

## Task 2: Failing unit tests for the tiling invariant

**Files:**
- Modify: `tests/quantization/test_grid.py` (append `TestQuantizerTiling` at end of file).

- [ ] **Step 1: Append the test class**

Add to the end of `tests/quantization/test_grid.py`:

```python
class TestQuantizerTiling:
    """Phase 4 — RhythmQuantizer.quantize() must produce non-overlapping output.

    The tiling invariant: for any consecutive pair of QuantizedNote entries
    in the output list, prev.onset_beat + prev.duration_beats <= next.onset_beat.
    """

    def _quantize(self, events: list[NoteEvent], bpm: float = 120.0) -> list:
        from improv_scribe.quantization.grid import RhythmQuantizer
        return RhythmQuantizer(_make_tempo(bpm)).quantize(events)

    def _assert_tiling(self, quantized: list) -> None:
        """Assert no entry overlaps the next."""
        for i in range(len(quantized) - 1):
            prev = quantized[i]
            nxt = quantized[i + 1]
            end = prev.onset_beat + prev.duration_beats
            assert end <= nxt.onset_beat + 1e-9, (
                f"Overlap at index {i}: prev ends at {end}, next starts at {nxt.onset_beat}"
            )

    def test_consecutive_notes_tile_exactly(self):
        # Two abutting quarter notes at 120 BPM: 0.0-0.5s, 0.5-1.0s
        # = beats 0.0-1.0, 1.0-2.0
        events = [
            _make_event(0.0, 0.5),
            _make_event(0.5, 1.0),
        ]
        q = self._quantize(events)
        # No rest expected (they abut)
        non_rests = [n for n in q if not n.is_rest]
        assert len(non_rests) == 2
        self._assert_tiling(q)
        # Tile invariant strict here: 0.0 + 1.0 == 1.0
        assert non_rests[0].onset_beat + non_rests[0].duration_beats == pytest.approx(non_rests[1].onset_beat)

    def test_rest_fills_gap_exactly(self):
        # Two quarter notes with a 1-beat gap at 120 BPM:
        # 0.0-0.5s, then gap, then 1.0-1.5s = beats 0-1, gap 1-2, note 2-3
        events = [
            _make_event(0.0, 0.5),
            _make_event(1.0, 1.5),
        ]
        q = self._quantize(events)
        # Expect: note, rest, note
        assert len(q) == 3
        assert q[0].is_rest is False
        assert q[1].is_rest is True
        assert q[2].is_rest is False
        self._assert_tiling(q)
        # The rest should exactly fill the gap
        assert q[1].duration_beats == pytest.approx(1.0)

    def test_overlap_regression_dyad_sample_scenario(self):
        # Reproduce the failing dyad-sample shape: 40 BPM tempo, onsets
        # at irregular fractional beat positions that triggered the
        # original §14.3 overlap bug.
        # At 40 BPM: 1 beat = 1.5s. Onsets at 0.290s, 2.079s, 3.821s,
        # 5.541s match the octave-dyads sample (basic-pitch output).
        events = [
            _make_event(0.290, 1.265, midi=40),
            _make_event(2.079, 3.229, midi=41),
            _make_event(3.821, 4.797, midi=55),
            _make_event(5.541, 6.970, midi=45),
        ]
        q = self._quantize(events, bpm=40.0)
        # Crucial: no overlap. This is the regression guard.
        self._assert_tiling(q)
        # All durations are exact catalog values
        for entry in q:
            ql = entry.quarter_length
            assert ql > 0, f"zero or negative quarter_length at {entry}"

    def test_triplet_quarter_duration_preserved(self):
        # An event whose duration is exactly 2/3 beat at 120 BPM
        # (1 beat = 0.5s; 2/3 beat = 0.333s). Triplet-quarter is 2/3 beat.
        events = [_make_event(0.0, 0.333)]
        q = self._quantize(events)
        non_rests = [n for n in q if not n.is_rest]
        assert len(non_rests) == 1
        assert non_rests[0].duration_type == NoteDuration.TRIPLET_QUARTER
        # Triplet-quarter = 2/3 beat = 2/3 quarter-length
        assert non_rests[0].duration_beats == pytest.approx(2.0 / 3.0)

    def test_phase_1_2_3_aligned_inputs_unchanged(self):
        # Simulate clean-grid inputs as produced by CREPE/basic-pitch on the
        # mono open-string samples at 120 BPM: onsets at beats 1,2,3,4,5,6
        # with 1-beat durations.
        events = [
            _make_event(0.5, 1.0),   # beat 1, dur 1
            _make_event(1.0, 1.5),   # beat 2, dur 1
            _make_event(1.5, 2.0),   # beat 3, dur 1
            _make_event(2.0, 2.5),   # beat 4, dur 1
            _make_event(2.5, 3.0),   # beat 5, dur 1
            _make_event(3.0, 3.5),   # beat 6, dur 1
        ]
        q = self._quantize(events)
        non_rests = [n for n in q if not n.is_rest]
        assert len(non_rests) == 6
        # All snap to integer beat positions
        for i, entry in enumerate(non_rests):
            assert entry.onset_beat == pytest.approx(float(i + 1))
            assert entry.duration_type == NoteDuration.QUARTER
            assert entry.duration_beats == pytest.approx(1.0)
        self._assert_tiling(q)
```

- [ ] **Step 2: Run the tests; confirm they fail**

```bash
conda run -n auto-sheet-music pytest tests/quantization/test_grid.py::TestQuantizerTiling -v 2>&1 | tail -15
```

Expected: most/all 5 tests FAIL — the current `quantize()` algorithm doesn't guarantee tiling, doesn't always pick triplet-quarter for 0.667-beat durations, etc.

Specifically:
- `test_overlap_regression_dyad_sample_scenario` should fail with overlap (the original bug)
- `test_consecutive_notes_tile_exactly` may pass or fail depending on the current "rest threshold" behaviour
- The other tests likely fail in various ways

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/quantization/test_grid.py
git commit -m "test(quantization): add TestQuantizerTiling regression tests (Phase 4)

Five tests asserting the tiling invariant
(prev.onset + prev.dur <= next.onset) on:
- Abutting notes
- Notes with a rest-filled gap
- The original §14.3 overlap scenario (40 BPM, irregular onsets)
- A triplet-quarter duration
- Clean-grid Phase 1/2/3 input shape (no drift)

Tests fail against the current quantizer; Task 3 implements the fix.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Implement the quantizer rewrite

**Files:**
- Modify: `src/improv_scribe/quantization/grid.py:141-243` (RhythmQuantizer class body).

- [ ] **Step 1: Replace `RhythmQuantizer.__init__`, `quantize()`, and helpers**

Locate the existing class (around lines 126–243) and replace its body. Keep the class docstring at lines 126–139. Replace `__init__` and everything below it with:

```python
    def __init__(
        self,
        tempo_result: TempoResult,
        time_signature: tuple[int, int] = (4, 4),
        include_triplets: bool = True,
        smallest_duration: NoteDuration = NoteDuration.SIXTEENTH,
    ) -> None:
        self._bpm = tempo_result.bpm
        self._time_sig = time_signature
        self._beat_duration_s = 60.0 / self._bpm  # seconds per quarter note

        # Build the set of candidate durations (sorted ascending by duration)
        min_frac = _DURATION_FRACTIONS[smallest_duration]
        self._candidates: list[tuple[NoteDuration, float]] = sorted(
            (
                (dur, frac)
                for dur, frac in _SORTED_DURATIONS
                if frac >= min_frac - 1e-9 and (include_triplets or "triplet" not in dur.value)
            ),
            key=lambda dur_frac: dur_frac[1],
        )

        # Common grid: smallest candidate duration in beats. Snapping both
        # onsets and offsets to this grid guarantees the tiling invariant.
        # 1/12 beat ≈ 0.0833 with triplets; 1/16 = 0.0625 without.
        self._grid_beats: float = min(frac for _dur, frac in self._candidates) * 4.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, events: list[NoteEvent]) -> list[QuantizedNote]:
        """Quantize raw NoteEvents into grid-aligned QuantizedNotes.

        Phase 4 algorithm: snap both onset and offset to ``self._grid_beats``.
        Pick the largest catalog NoteDuration that fits within the snapped
        duration; the chosen catalog duration may be ≤ the raw snapped
        duration (note shortens slightly to fit) but never longer (no
        overlap). Rests fill the gap between the previous note's chosen
        end and the next note's snapped onset using the same rule.

        Tiling invariant (asserted by tests): for consecutive entries,
        ``prev.onset_beat + prev.duration_beats <= next.onset_beat``.

        Parameters
        ----------
        events : list[NoteEvent]
            Sorted by onset_s.

        Returns
        -------
        list[QuantizedNote]
            Notes and rests in score order.
        """
        if not events:
            return []

        quantized: list[QuantizedNote] = []
        prev_end_beat = 0.0

        for event in events:
            snapped_onset = self._snap_to_grid(self._s_to_beat(event.onset_s))
            snapped_offset = self._snap_to_grid(self._s_to_beat(event.offset_s))
            # Ensure at least one grid cell of duration
            if snapped_offset < snapped_onset + self._grid_beats:
                snapped_offset = snapped_onset + self._grid_beats

            snapped_dur_beats = snapped_offset - snapped_onset
            dur_type, dur_beats = self._largest_fitting_duration(snapped_dur_beats)
            # Note's chosen end is `snapped_onset + dur_beats`. This may be
            # ≤ snapped_offset (we shrink to a catalog value); the leftover
            # is absorbed into the rest after this note.

            note_end = snapped_onset + dur_beats

            # Insert rest if there's a gap from the previous note's end
            # to this note's snapped onset.
            gap = snapped_onset - prev_end_beat
            if gap >= self._grid_beats - 1e-9:
                rest = self._make_rest(prev_end_beat, gap)
                if rest is not None:
                    quantized.append(rest)

            quantized.append(QuantizedNote(
                midi_notes=event.midi_notes,
                frequencies_hz=event.frequencies_hz,
                confidences=event.confidences,
                cents_deviations=event.cents_deviations,
                onset_beat=snapped_onset,
                duration_beats=dur_beats,
                duration_type=dur_type,
                quarter_length=to_quarter_length(dur_type),
            ))

            prev_end_beat = note_end

        return quantized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _s_to_beat(self, time_s: float) -> float:
        """Convert seconds to beat position (quarter note = 1 beat)."""
        return time_s / self._beat_duration_s

    def _snap_to_grid(self, beat: float) -> float:
        """Snap a beat position to the nearest ``self._grid_beats`` multiple."""
        return round(beat / self._grid_beats) * self._grid_beats

    def _largest_fitting_duration(
        self, dur_beats: float
    ) -> tuple[NoteDuration, float]:
        """Return the largest catalog (NoteDuration, beat_value) that fits.

        "Fits" means ``catalog_beat_value <= dur_beats + 1e-9``. If no catalog
        value fits (i.e. ``dur_beats < smallest catalog``), returns the
        smallest catalog value — this is the only case where the chosen
        duration exceeds the requested duration (needed to avoid emitting
        a zero-duration note, which music21 rejects).
        """
        # _candidates is sorted ascending by fraction.
        # Walk descending to find the largest that fits.
        smallest = self._candidates[0]
        for dur, frac in reversed(self._candidates):
            beat_value = frac * 4.0
            if beat_value <= dur_beats + 1e-9:
                return dur, beat_value
        # Nothing fits — return smallest (must expand to avoid zero-duration)
        return smallest[0], smallest[1] * 4.0

    def _make_rest(self, start_beat: float, gap_beats: float) -> QuantizedNote | None:
        """Create a rest QuantizedNote for a gap between notes.

        Returns ``None`` if the gap is smaller than ``self._grid_beats``.
        The rest's chosen duration is the largest catalog value that fits
        the gap; any leftover (gap minus chosen duration) is absorbed as
        quantization noise — the next note's snapped onset is unchanged.
        """
        if gap_beats < self._grid_beats - 1e-9:
            return None
        dur_type, dur_beats = self._largest_fitting_duration(gap_beats)
        return QuantizedNote(
            midi_notes=(),
            frequencies_hz=(),
            confidences=(),
            cents_deviations=(),
            onset_beat=start_beat,
            duration_beats=dur_beats,
            duration_type=dur_type,
            quarter_length=to_quarter_length(dur_type),
            is_rest=True,
        )

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def time_signature(self) -> tuple[int, int]:
        return self._time_sig
```

Note: this replaces the old `_min_dur_beats()` (use `self._grid_beats` directly) and `_snap_duration()` (replaced by `_largest_fitting_duration()`).

- [ ] **Step 2: Run the TestQuantizerTiling tests**

```bash
conda run -n auto-sheet-music pytest tests/quantization/test_grid.py::TestQuantizerTiling -v 2>&1 | tail -15
```

Expected: all 5 tests PASS.

If any fail, the most likely causes:
- `test_consecutive_notes_tile_exactly` failing on "len(non_rests) == 2" — `_make_rest` may be emitting a spurious rest. Check the `gap >= self._grid_beats - 1e-9` condition; if gap is exactly 0 (notes truly abut), no rest should be inserted.
- `test_triplet_quarter_duration_preserved` failing on duration_type — `_largest_fitting_duration` may be picking eighth (0.5 beat) instead of triplet-quarter (0.667 beat) for a 0.667-beat input. Verify `_candidates` is sorted ascending by fraction and the descending walk picks the right value.

- [ ] **Step 3: Run the existing test_grid.py tests**

```bash
conda run -n auto-sheet-music pytest tests/quantization/test_grid.py -v 2>&1 | tail -10
```

Expected: all existing tests PASS (the rewrite preserves behaviour on clean-grid inputs). If a pre-existing test fails, the rewrite changed observable behaviour outside the Phase 4 spec — diagnose and either update the test (with a clear rationale) or fix the rewrite.

- [ ] **Step 4: Run the full integration gauntlet**

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected:
- basic_pitch: same passing count as before (the chord tests' ground truth in spec §15.5 was captured under the OLD quantizer behaviour, but those tests assert on `note_events`, not `quantized_notes`, so the rewrite shouldn't affect them).
- crepe: same passing count as before.

If integration tests fail with `EXPECTED_MIDI` mismatches, the chord-aware ground truth from §13.5/§15.5 was captured under the old quantizer and may need a small update (one grid-cell shift). Report the diff and pause for review.

- [ ] **Step 5: Commit**

```bash
git add src/improv_scribe/quantization/grid.py
git commit -m "fix(quantization): common-grid snapping prevents MusicXML overlap (Phase 4)

RhythmQuantizer.__init__ now stores self._grid_beats (smallest catalog
duration in beats, default 1/12). quantize() snaps both onset and
offset to that grid, then picks the largest catalog NoteDuration that
fits the snapped duration. Rests fill the gap between prev note's
chosen end and next note's snapped onset using the same rule.

Result: chosen durations never extend past the original snapped
offset, so MusicXML <duration> and <type> stay consistent and
MuseScore stops rejecting the output with exit 40.

Tiling invariant (asserted by 5 new TestQuantizerTiling tests):
  prev.onset_beat + prev.duration_beats <= next.onset_beat

Existing test_grid.py tests and integration tests continue to pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: PDF render smoke test

**Files:**
- Create: `tests/integration/test_pdf_render_smoke.py`.

- [ ] **Step 1: Create the test file**

```python
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
    """End-to-end: sample → quantizer → ScoreBuilder → PDFExporter."""
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
    tab_assignments = builder.compute_tab_assignments(quantized)

    pdf_path = tmp_path / "open_G_chord.pdf"
    out = PDFExporter(config).export(
        score,
        pdf_path,
        tab_notes=quantized,
        tab_assignments=tab_assignments,
        tab_profile=profile,
    )
    assert out.exists(), f"PDF was not written to {out}"
    assert out.stat().st_size > 5000, (
        f"PDF size {out.stat().st_size} bytes is suspiciously small "
        "(MuseScore may have produced an empty or broken file)"
    )
```

- [ ] **Step 2: Run the smoke test**

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration/test_pdf_render_smoke.py -v 2>&1 | tail -8
```

Expected: PASS. The G chord already rendered successfully in Phase 3 §16, so this should work even without the Phase 4 fix; after the fix, the same render still works.

- [ ] **Step 3: Run under CREPE to confirm skip**

```bash
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration/test_pdf_render_smoke.py -v 2>&1 | tail -5
```

Expected: SKIPPED with "PDF render test requires basic_pitch backend" message.

- [ ] **Step 4: Verify a previously-failing chord sample now renders**

This is a *manual verification step* outside pytest. The original spec §14.3 bug blocked PDF rendering on E, A, D, C chord samples. Try one:

```bash
cd /Users/davehollander/Documents/Personal/Projects/audio_to_sheet
conda run -n auto-sheet-music python -c "
from pathlib import Path
import librosa
from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.note_tracker import NoteTracker
from improv_scribe.analysis.onset import OnsetDetector
from improv_scribe.analysis.pitch import PitchEstimator
from improv_scribe.config import AppConfig
from improv_scribe.export.pdf_exporter import PDFExporter
from improv_scribe.notation.score_builder import ScoreBuilder
from improv_scribe.quantization.grid import RhythmQuantizer
from improv_scribe.quantization.tempo import TempoEstimator

config = AppConfig()
profile = get_profile(Instrument.GUITAR)
sample = Path('samples/guitar/chords/6_string_electric_open_E_chord.mp3')
y, _ = librosa.load(str(sample), sr=44100, mono=True)
result = PitchEstimator(config, backend='basic_pitch').estimate(y, profile)
onsets = OnsetDetector(config).detect(y)
events = NoteTracker(config, profile).process(result, onsets, audio=y)
tempo = TempoEstimator(config).estimate(events)
quantized = RhythmQuantizer(tempo).quantize(events)
builder = ScoreBuilder(profile, tempo, title='Phase 4 E chord proof')
score = builder.build(quantized)
assignments = builder.compute_tab_assignments(quantized)
out = PDFExporter(config).export(
    score, Path('/tmp/phase4_E_chord.pdf'),
    tab_notes=quantized, tab_assignments=assignments, tab_profile=profile,
)
print(f'PDF written: {out} ({out.stat().st_size} bytes)')
" 2>&1 | tail -3
```

Expected: PDF written successfully (e.g. `PDF written: /tmp/phase4_E_chord.pdf (~30000 bytes)`). If it fails with the same MuseScore exit-40 error, the quantizer rewrite didn't fully fix the §14.3 bug — STOP and diagnose.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_pdf_render_smoke.py
git commit -m "test(integration): PDF render smoke test on open G chord (Phase 4)

Regression guard for spec §14.3 quantizer overlap bug: runs the
full pipeline (basic_pitch → quantizer → ScoreBuilder → PDFExporter)
on the open G chord sample, asserts the PDF file exists and is
> 5KB (catches the case where MuseScore exits silently with an
empty or broken file).

CREPE/pyin skip with explanatory message.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: CLI --backend basic_pitch

**Files:**
- Modify: `src/improv_scribe/cli.py:163-165`.

- [ ] **Step 1: Update the argparse choices and help text**

Locate (around line 163):

```python
    p.add_argument(
        "--backend", choices=["pyin", "crepe"], default="pyin",
        help="Pitch detection backend (default: pyin)"
    )
```

Replace with:

```python
    p.add_argument(
        "--backend", choices=["pyin", "crepe", "basic_pitch"], default="pyin",
        help=(
            "Pitch detection backend (default: pyin). "
            "'basic_pitch' requires `bash scripts/install_basic_pitch.sh` after env creation."
        ),
    )
```

- [ ] **Step 2: Verify --help displays the new choice**

```bash
conda run -n auto-sheet-music python -m improv_scribe.cli --help 2>&1 | grep -A2 backend
```

Expected output includes:

```
  --backend {pyin,crepe,basic_pitch}
                        Pitch detection backend (default: pyin). 'basic_pitch' requires
                        `bash scripts/install_basic_pitch.sh` after env creation.
```

- [ ] **Step 3: Run the full test suite to confirm no regression**

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
```

Expected:
- basic_pitch: 318 passed, 4 skipped (was 312/4 — 5 new unit + 1 PDF smoke = +6).
- crepe: 289 passed, 33 skipped (was 284/32 — 5 new unit pass + 1 PDF smoke skip).

If counts differ, report the actual numbers.

- [ ] **Step 4: Commit**

```bash
git add src/improv_scribe/cli.py
git commit -m "feat(cli): add basic_pitch to --backend choices (Phase 4)

argparse choices list now includes 'basic_pitch'. Help text notes
the install requirement (scripts/install_basic_pitch.sh).

Users can now select basic_pitch from --help-visible choices instead
of having to discover the ATS_PITCH_BACKEND env var.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Phase 4 outcome in spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md` (append §17 after §16).

- [ ] **Step 1: Append §17 to the polyphonic detection spec**

Edit `docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md`. Find the end of §16 (the Phase 3 outcome) and append:

```markdown
---

## 17. Phase 4 Outcome (landed 2026-05-23)

Phase 4 — quantizer overlap fix + CLI backend choice — completed in N
task commits on `chord-detection`. The phase delivered:

- `RhythmQuantizer.quantize()` rewritten around a single fine grid
  (1/12 beat default). Both onsets and offsets snap to the grid; the
  chosen catalog `NoteDuration` is the largest value that fits the
  snapped duration. Rests fill the inter-note gap using the same rule.
- Tiling invariant guaranteed (asserted by 5 new `TestQuantizerTiling`
  tests): for consecutive QuantizedNote entries,
  ``prev.onset_beat + prev.duration_beats <= next.onset_beat``.
- PDF rendering now works on the previously-failing chord samples
  (E, A, D, C in addition to G). The §14.3 MuseScore exit-40 issue
  is resolved.
- CLI `--backend` choices list includes `basic_pitch`. Help text
  notes the installer-script requirement.

### 17.1 Phase gate results

| Backend | Result |
|---|---|
| `ATS_PITCH_BACKEND=basic_pitch` | NN passed, MM skipped |
| `ATS_PITCH_BACKEND=crepe` | NN passed, MM skipped |

(Fill in from Task 5 Step 3.)

### 17.2 §16.3 priority list status

- ✓ #1 Quantizer overlap bug — fixed in this phase.
- ✓ #2 CLI `--backend` missing `basic_pitch` — fixed in this phase.
- Open: #3 Chord-recall improvement, #4 Decay-tail singleton filter,
  #5 Chord-name detection — deferred to a future phase, scoped only
  when there's evidence of user demand.

### 17.3 Spec §14.3 status

The pre-existing quantizer issue documented in §14.3 is now resolved.
PDF rendering on chord samples works end-to-end without manual
intervention.
```

Fill in N (commit count from `git log --oneline 8276ccd..HEAD | wc -l`) and the pass/skip counts.

- [ ] **Step 2: Commit the spec update**

```bash
git add docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md
git commit -m "docs(spec): record Phase 4 outcome (quantizer + CLI fixes)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 3: Final summary check**

```bash
git log --oneline -10
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
```

Expected: clean log showing the Phase 4 commits, both backends at the §4.3 phase-gate counts.

- [ ] **Step 4: Declare Phase 4 done**

Phase 4 ships when:
1. All 5 unit tests in `TestQuantizerTiling` pass.
2. PDF smoke test passes on the G chord.
3. Manual E chord PDF render succeeds (Task 4 Step 4).
4. CLI `--help` lists `basic_pitch`.
5. Both backends meet the §4.3 phase-gate test counts.
6. Spec §17 records the outcome.

---

## What's next (out of scope for this plan)

The §16.3 priority list still has three open items (#3 chord recall, #4 decay-tail filter, #5 chord-name detection). None are blocking; brainstorm them individually when user demand surfaces.
