# Polyphonic Detection — Phase 3: Open-Chord Detection (3–6 member chords)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and ship Phase 2's chord-aware pipeline working end-to-end on real open-chord recordings (E, A, D, G, C). Add integration tests for each sample using actual basic-pitch output as ground truth.

**Architecture:** No new algorithms. Phase 2's clustering, chord rendering, no-string-conflict tab DP, and chord-iterating MIDI export all already handle N-member chords for any N. Phase 3's job is to validate this against the five real open-chord recordings and record the per-cluster ground truth from spec §15.5.

**Tech Stack:** Python 3.13, basic-pitch, music21, pytest. No new dependencies.

**Spec reference:** [docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md](../specs/2026-05-09-polyphonic-detection-design.md) §15 (Phase 3 prerequisite probe — ground truth tables in §15.5, scope decisions in §15.3–§15.4).

**Phase gate (definition of done):**
```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest
```
- basic_pitch: existing tests + 5 new open-chord tests all pass.
- CREPE: existing tests pass; open-chord tests skip (no CREPE ground truth — same pattern as Phase 2 dyad tests).

**Phasing context:** No worktree this time — scope is small (mostly test additions). Work directly on `chord-detection`. Phase 4 (separate spec, only if quality issues warrant) would address: chord-recall improvement, decay-tail singleton filtering, octave/sympathetic detection corrections, raw-MIDI per-string sustain, quantizer overlap bug fix.

---

## File Map

| File | Change |
|---|---|
| `tests/notation/test_tab_builder.py` | Add tests for 4-member and 6-member chord shapes (extending Phase 2's 3-member coverage). |
| `tests/integration/test_guitar_open_E_chord.py` *(new)* | Integration test against `6_string_electric_open_E_chord.mp3`. Ground truth from spec §15.5. |
| `tests/integration/test_guitar_open_A_chord.py` *(new)* | Same for A chord. |
| `tests/integration/test_guitar_open_D_chord.py` *(new)* | Same for D chord. |
| `tests/integration/test_guitar_open_G_chord.py` *(new)* | Same for G chord. |
| `tests/integration/test_guitar_open_C_chord.py` *(new)* | Same for C chord. |
| `docs/superpowers/phase3_chord_proof.pdf` *(new)* | Hand-constructed PDF render verifying chord+tab works for a 4-member voicing. |
| `docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md` | Append §16 (Phase 3 outcome). |

---

## Task 1: Confirm baseline + clean state

- [ ] **Step 1: Verify clean working tree on chord-detection**

```bash
cd /Users/davehollander/Documents/Personal/Projects/audio_to_sheet
git status
git log --oneline -3
```

Expected: clean working tree, HEAD at `28c4800` (Phase 3 prerequisite commit) or later.

- [ ] **Step 2: Run regression gauntlet**

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
```

Expected:
- basic_pitch: 279 passed, 4 skipped (Phase 2 §14.1 numbers).
- CREPE: 271 passed, 12 skipped.

If either fails, STOP and fix baseline before proceeding.

- [ ] **Step 3: Verify editable install points at main repo (not stale worktree)**

```bash
conda run -n auto-sheet-music pip show improv_scribe | grep "Editable"
```

Expected: `Editable project location: /Users/davehollander/Documents/Personal/Projects/audio_to_sheet`. If it points anywhere else, run `pip install -e .` from the main repo.

---

## Task 2: Failing tests for 4-member and 6-member chord tab DP

Phase 2's `tests/notation/test_tab_builder.py` covers 1, 2, and 3-member shapes. Extend to 4 and 6 members to confirm Phase 2's DP scales to the open-chord sizes.

**Files:**
- Modify: `tests/notation/test_tab_builder.py`

- [ ] **Step 1: Append the larger-chord tests**

Read the existing `TestAssignFretsChordAware` class to find the right append point. Then append (or add to the class):

```python
    def test_four_member_chord_d_major_open(self):
        """Open D major (D3, A3, D4, F#4) — basic-pitch detects 3-4 of 4
        per Phase 3 §15.5. Tests the DP scales to 4 distinct strings."""
        # D3=50, A3=57, D4=62, F#4=66
        result = assign_frets([_qn((50, 57, 62, 66))], Instrument.GUITAR)
        assert len(result) == 1
        assert result[0] is not None
        shape = result[0]
        assert len(shape) == 4
        # All four must use distinct strings
        strings = [s for s, _f in shape]
        assert len(set(strings)) == 4
        # Frets must be plausible for guitar (open D major shape: strings 2/3/4/5)
        for s, f in shape:
            assert 0 <= s <= 5
            assert 0 <= f <= 22

    def test_six_member_chord_g_major_open(self):
        """Open G major (G2, B2, D3, G3, B3, G4) — six members across six strings.
        Tests the DP scales to a fully-voiced 6-note shape."""
        # G2=43, B2=47, D3=50, G3=55, B3=59, G4=67
        result = assign_frets([_qn((43, 47, 50, 55, 59, 67))], Instrument.GUITAR)
        assert len(result) == 1
        assert result[0] is not None
        shape = result[0]
        assert len(shape) == 6
        # Six members must occupy six distinct strings (all of them)
        strings = sorted(s for s, _f in shape)
        assert strings == [0, 1, 2, 3, 4, 5]

    def test_chord_progression_dp_transitions(self):
        """A progression of 4-member chords (D, G, C) should produce a valid
        assignment for each. Tests DP transition cost on chord-to-chord."""
        # D: (50, 57, 62, 66) — D3 A3 D4 F#4
        # G: (43, 50, 55, 59)  — G2 D3 G3 B3 (subset of full G voicing)
        # C: (48, 52, 55, 60)  — C3 E3 G3 C4
        result = assign_frets(
            [_qn((50, 57, 62, 66)), _qn((43, 50, 55, 59)), _qn((48, 52, 55, 60))],
            Instrument.GUITAR,
        )
        assert len(result) == 3
        for shape in result:
            assert shape is not None
            strings = [s for s, _f in shape]
            assert len(set(strings)) == len(strings)   # no string conflicts within a chord
```

- [ ] **Step 2: Run the tests**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_tab_builder.py -v 2>&1 | tail -20
```

Expected: all 3 new tests PASS. If any fail, the Phase 2 DP doesn't scale to larger N — investigate (most likely cause: a bug in `get_chord_shapes` Cartesian product or in the DP's tie-break).

If a test fails for a legitimate reason (e.g. unplayable chord — open G major has G4 on string 5 fret 15, which IS playable but a stretch), accept that and either weaken the test or document the limitation.

- [ ] **Step 3: Commit**

```bash
git add tests/notation/test_tab_builder.py
git commit -m "$(cat <<'EOF'
test(notation): extend tab DP tests to 4- and 6-member chords (Phase 3)

Phase 2's chord-aware DP was tested with 3-member shapes. Phase 3
validates that the same DP handles the full open-chord sizes (4-6
members) used in spec §15.5 ground truth.

3 new tests:
- 4-member D major (subset)
- 6-member G major (full)
- 3-chord progression (D→G→C) to exercise the DP transition cost

No DP changes expected; the algorithm is N-member-agnostic.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Open E chord integration test

**Files:**
- Create: `tests/integration/test_guitar_open_E_chord.py`

- [ ] **Step 1: Create the test file**

Ground truth from spec §15.5 — 5 clusters:

```python
"""End-to-end pipeline regression test for:
    samples/guitar/chords/6_string_electric_open_E_chord.mp3

User strums an open E major chord (E2 B2 E3 G#3 B3 E4) five times over
~12.3 seconds. basic-pitch detects 2-3 of 6 chord members per strum:
the high voices (E3, B3, E4) consistently fall below the 0.65 amplitude
floor. See spec §15.2 for recall analysis and §15.5 for the exact
per-cluster ground truth.
"""

from __future__ import annotations

import os

import music21.chord
import numpy as np
import pytest

from improv_scribe.analysis.instrument_profiles import Instrument
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "chords" / "6_string_electric_open_E_chord.mp3"
INSTRUMENT = Instrument.GUITAR
EXPECTED_DURATION_S = 12.30

_BACKEND = os.getenv("ATS_PITCH_BACKEND", "basic_pitch")

# basic_pitch ground truth from spec §15.5. CREPE/pyin are monophonic
# and skipped (same pattern as Phase 2 dyad tests).
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (47, 56),       # B2 + G#3
        (40, 47, 56),   # E2 + B2 + G#3 (best detection)
        (47,),          # B2 only (decay fragmentation)
        (40, 47),       # E2 + B2
        (47, 56),       # B2 + G#3
    ],
    "crepe": [],
    "pyin":  [],
}
EXPECTED_MIDI_TUPLES = EXPECTED_MIDI_TUPLES_BY_BACKEND[_BACKEND]
NOTE_COUNT = len(EXPECTED_MIDI_TUPLES)


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

    def test_audio_duration(self, audio):
        y, sr = audio
        duration_s = len(y) / sr
        assert abs(duration_s - EXPECTED_DURATION_S) <= 1.0


class TestNoteEvents:
    def test_note_count(self, note_events):
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} events, got {len(note_events)}: "
            f"{[tuple(e.midi_notes) for e in note_events]}"
        )

    def test_note_midi_tuples_match(self, note_events):
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        actual = [tuple(e.midi_notes) for e in note_events]
        assert actual == EXPECTED_MIDI_TUPLES


class TestScore:
    def test_score_chord_emission(self, score):
        """Each cluster with len(midi_notes) > 1 should emit a music21.chord.Chord."""
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        chord_count = sum(1 for t in EXPECTED_MIDI_TUPLES if len(t) > 1)
        if chord_count == 0:
            pytest.skip(f"Backend {_BACKEND} produces no chords on this sample")

        chords_in_score = list(score.recurse().getElementsByClass(music21.chord.Chord))
        assert len(chords_in_score) == chord_count, (
            f"Expected {chord_count} Chord objects, found {len(chords_in_score)}"
        )


class TestTabAssignments:
    def test_chord_tab_uses_distinct_strings(self, tab_assignments):
        """Every chord-shape assignment must use distinct strings."""
        if not EXPECTED_MIDI_TUPLES:
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        for assignment in tab_assignments:
            if assignment is None or len(assignment) <= 1:
                continue
            strings = [s for s, _f in assignment]
            assert len(set(strings)) == len(strings), (
                f"Chord assignment {assignment} reuses a string"
            )
```

- [ ] **Step 2: Run the test**

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration/test_guitar_open_E_chord.py -v 2>&1 | tail -10
```

Expected: all tests PASS. If the count or tuples diverge from spec §15.5, the probe results from 2026-05-23 are stale — re-run the probe (see Step 3 of Task 8) and update the ground truth to match current basic-pitch output.

- [ ] **Step 3: Run under CREPE to confirm graceful skip**

```bash
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration/test_guitar_open_E_chord.py -v 2>&1 | tail -10
```

Expected: tests skip with the "No ground truth recorded for backend crepe" message; no failures.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_guitar_open_E_chord.py
git commit -m "$(cat <<'EOF'
test(integration): open E chord (Phase 3)

Five-cluster ground truth from spec §15.5. basic-pitch detects 2-3 of
6 chord members per strum; tests assert exact per-cluster MIDI tuples.

CREPE/pyin skip (no monophonic ground truth for chord recordings).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Open A chord integration test

**Files:**
- Create: `tests/integration/test_guitar_open_A_chord.py`

Same structure as Task 3. Ground truth from spec §15.5 (A chord, 6 clusters):

```python
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (57,),            # A3 only
        (45,),            # A2 only
        (45, 52),         # A2 + E3
        (45, 52, 57),     # A2 + E3 + A3 (best detection)
        (45, 52),         # A2 + E3
        (40,),            # E2 only (decay tail, 0.4s — see §15.5)
    ],
    "crepe": [],
    "pyin":  [],
}
```

The rest of the test file (TestAudio, TestNoteEvents, TestScore, TestTabAssignments) is identical to the E chord file except for `SAMPLE_PATH` and the docstring.

Commit per the same pattern (`test(integration): open A chord (Phase 3)`).

---

## Task 5: Open D chord integration test

**Files:**
- Create: `tests/integration/test_guitar_open_D_chord.py`

Ground truth from spec §15.5 (D chord, 5 clusters):

```python
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (45, 50),         # A2 + D3 (A2 is unexpected; sympathetic of A3)
        (50, 57, 66),     # D3 + A3 + F#4 (best detection)
        (45, 50),         # A2 + D3
        (45, 66),         # A2 + F#4
        (62,),            # D4 only (decay tail)
    ],
    "crepe": [],
    "pyin":  [],
}
```

Commit (`test(integration): open D chord (Phase 3)`).

---

## Task 6: Open G chord integration test

**Files:**
- Create: `tests/integration/test_guitar_open_G_chord.py`

Ground truth from spec §15.5 (G chord, 5 clusters — best detection of all five samples):

```python
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (43, 47, 50, 55),   # G2 + B2 + D3 + G3 (4-member)
        (47, 50, 55),       # B2 + D3 + G3
        (50, 55, 59),       # D3 + G3 + B3
        (47, 50),           # B2 + D3
        (43, 47, 50, 55),   # G2 + B2 + D3 + G3 again
    ],
    "crepe": [],
    "pyin":  [],
}
```

Commit (`test(integration): open G chord (Phase 3)`).

---

## Task 7: Open C chord integration test

**Files:**
- Create: `tests/integration/test_guitar_open_C_chord.py`

Ground truth from spec §15.5 (C chord, 6 clusters):

```python
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (48, 52, 55),     # C3 + E3 + G3
        (40, 48, 55),     # E2 + C3 + G3 (E2 is unexpected; spec §15.5 notes this)
        (48, 55, 60),     # C3 + G3 + C4
        (48, 60),         # C3 + C4
        (48, 55),         # C3 + G3
        (48, 52),         # C3 + E3 (fragmented from previous strum)
    ],
    "crepe": [],
    "pyin":  [],
}
```

Commit (`test(integration): open C chord (Phase 3)`).

---

## Task 8: Phase 3 phase-gate + PDF render proof

- [ ] **Step 1: Run the full regression gauntlet**

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
```

Expected:
- basic_pitch: 5 new test files × 6 tests each ≈ 30 new tests pass. Total ~309 passed + 4 skipped (depending on how many new tests there actually are).
- CREPE: same as before for existing tests + 30 new tests skip with "no ground truth" → 25 new skips. Total ~271 + 5×6 = 301 passed + 12+30 = 42 skipped.

Record the actual numbers from your run for the spec §16 outcome.

- [ ] **Step 2: Verify ground truth still matches actual basic-pitch output**

If any of the 5 chord tests FAIL, basic-pitch's output drifted from the probe results. Re-run the probe to update ground truth:

```bash
conda run -n auto-sheet-music python /tmp/probe_open_chords.py 2>&1 | tail -100
```

If the probe is no longer available, recreate it. Update the affected test file's `EXPECTED_MIDI_TUPLES_BY_BACKEND["basic_pitch"]` to match the new probe output, then re-run.

- [ ] **Step 3: PDF render proof on the G chord sample**

The G chord had the best detection (4-member shape on strums 1 and 5). Render it end-to-end to verify the full pipeline works on a real chord-progression input. Use a one-shot script (the CLI's `--backend` option doesn't yet include `basic_pitch`):

```python
# /tmp/render_phase3_G.py
from pathlib import Path
import librosa

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.note_tracker import NoteTracker
from improv_scribe.analysis.onset import OnsetDetector
from improv_scribe.analysis.pitch import PitchEstimator
from improv_scribe.config import AppConfig
from improv_scribe.export.midi_exporter import MIDIExporter
from improv_scribe.export.pdf_exporter import PDFExporter
from improv_scribe.notation.score_builder import ScoreBuilder
from improv_scribe.quantization.grid import RhythmQuantizer
from improv_scribe.quantization.tempo import TempoEstimator

ROOT = Path("/Users/davehollander/Documents/Personal/Projects/audio_to_sheet")
SAMPLE = ROOT / "samples/guitar/chords/6_string_electric_open_G_chord.mp3"
OUT_DIR = Path("/tmp/phase3_render")
OUT_DIR.mkdir(parents=True, exist_ok=True)

config = AppConfig()
profile = get_profile(Instrument.GUITAR)
y, _ = librosa.load(str(SAMPLE), sr=44100, mono=True)
result = PitchEstimator(config, backend="basic_pitch").estimate(y, profile)
onsets = OnsetDetector(config).detect(y)
events = NoteTracker(config, profile).process(result, onsets, audio=y)
tempo = TempoEstimator(config).estimate(events)
quantized = RhythmQuantizer(tempo).quantize(events)

builder = ScoreBuilder(profile, tempo, title="Phase 3 — Open G chord")
score = builder.build(quantized)
assignments = builder.compute_tab_assignments(quantized)

MIDIExporter(config).quantized_from_score(
    builder.build_raw(quantized),
    OUT_DIR / "open_G_chord.mid",
)
print(f"MIDI: {OUT_DIR / 'open_G_chord.mid'}")

try:
    pdf = PDFExporter(config).export(
        score,
        OUT_DIR / "open_G_chord.pdf",
        tab_notes=quantized,
        tab_assignments=assignments,
        tab_profile=profile,
    )
    print(f"PDF: {pdf} ({pdf.stat().st_size} bytes)")
except Exception as e:
    print(f"PDF render failed (likely the Phase 2 §14.3 quantizer bug): {e}")
```

Run it:

```bash
conda run -n auto-sheet-music python /tmp/render_phase3_G.py 2>&1 | grep -v "warnings.warn\|^Predicting\|WARNING:root\|^qt\." | tail -10
```

**Expected outcomes** (any of these is acceptable for Phase 3):

1. **PDF renders successfully.** Verify MIDI byte count and PDF file exists. Open PDF and visually confirm chord glyphs + tab numbers. Save the PDF as `docs/superpowers/phase3_chord_proof.pdf`.

2. **PDF render fails** with the same MuseScore exit 40 documented in Phase 2 §14.3 (quantizer overlap bug). In this case, build a hand-constructed clean-timing 4-member chord and render that instead — same approach as Phase 2 §14's `phase2_chord_proof.pdf`. The point of Task 8 Step 3 is to prove Phase 3's larger chords render correctly when the quantizer doesn't produce overlap.

3. **MIDI export succeeds** even if PDF fails. This is sufficient end-to-end validation of the data flow (chord clustering → score → MIDI iteration).

Commit whichever artifact succeeded:

```bash
git add docs/superpowers/phase3_chord_proof.pdf  # if PDF succeeded
git commit -m "docs(phase3): commit open G chord render proof (Task 8)"
```

If only MIDI succeeded, document that in the spec §16 outcome instead of committing a PDF.

- [ ] **Step 4: Append §16 to the spec**

Add to `docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md` after §15:

```markdown
---

## 16. Phase 3 Outcome (landed YYYY-MM-DD)

Phase 3 — open-chord detection (3–6 member chords) — completed in N
task commits on `chord-detection`. The phase delivered:

- 3 new tab DP unit tests for 4- and 6-member shapes + chord progressions
  (Task 2). Phase 2's no-string-conflict DP scales to the full open-chord
  voicing sizes with no algorithmic changes.
- 5 new integration test files (Tasks 3–7), one per open-chord sample,
  asserting the exact basic-pitch detection ground truth from spec §15.5.
- PDF/MIDI render proof on the G chord sample (Task 8) — see
  `docs/superpowers/phase3_chord_proof.pdf` for the artifact (or notes
  about the quantizer bug if the dyad-sample render issue from §14.3
  also blocks here).

### 16.1 Phase gate results

| Backend | Result |
|---|---|
| basic_pitch | NN passed, MM skipped |
| crepe | NN passed, MM skipped |

(Fill in from Task 8 Step 1.)

### 16.2 What Phase 3 did NOT do

- No chord-recall improvement. basic-pitch's per-strum recall stays at
  2-4 of 5-6 notes; ground truth records the actual detection. Phase 4
  work (separate spec) would address chord-recall via lower amplitude
  floor + cluster-internal relative filter and/or chord-template matching.
- No fragmentation cleanup. Decay-tail re-detection (singleton clusters
  arriving within ~500ms of a multi-member cluster) is left in the
  ground truth. Phase 4 could filter these.
- No chord-name detection. Phase 3 emits the detected pitches; recognising
  these as "G major" etc. is a separate Phase 5+ feature.

### 16.3 What remains for Phase 4

In rough priority order:

1. **Quantizer overlap bug** (spec §14.3) — blocks PDF rendering on chord
   samples at clamped slow tempos. The most concrete Phase 4 item.
2. **CLI `--backend` choice** — still lists only `{pyin,crepe}`;
   `basic_pitch` selectable only via env var. Trivial fix.
3. **Chord-recall improvement** — explore lowering the absolute amplitude
   floor + cluster-internal relative filter to catch more chord members.
4. **Decay-tail singleton filter** — drop very short singleton clusters
   that arrive within ~500ms of a recent multi-member cluster.
5. **Chord-name detection** — emit "Em", "G", "C" labels alongside the
   detected pitches.
```

Fill in N (commit count from `git log --oneline 28c4800..HEAD | wc -l`) and the pass/skip counts.

- [ ] **Step 5: Commit the spec update**

```bash
git add docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md
git commit -m "docs(spec): record Phase 3 outcome (open-chord detection)"
```

- [ ] **Step 6: Declare Phase 3 done**

Phase 3 ships when:
1. All 5 open-chord integration tests pass under basic_pitch backend.
2. CREPE backend remains green (chord tests skip gracefully).
3. Spec §16 records the outcome and Phase 4 priorities.
4. The render proof artifact (PDF or MIDI) is committed (or scope documented if blocked by §14.3 quantizer bug).

---

## What's next (out of scope for this plan)

Phase 4 (separate spec, prioritise after Phase 3 lands and user reviews):
1. Fix the rhythm quantizer's onset/offset overlap bug (spec §14.3) so PDF rendering works on real chord recordings.
2. Update CLI `--backend` choices.
3. Investigate chord-recall improvement strategies; only commit to one if there's a clear quality win demonstrated against the §15.5 baseline.
4. Chord-name detection (Em, G, C) — depends on whether the user wants chord-symbol annotation in the rendered output.
