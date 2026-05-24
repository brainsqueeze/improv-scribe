# Phase 4 — Quantizer Overlap Fix + CLI Backend Choice

**Status:** Draft (pending user review)
**Date:** 2026-05-23
**Owner:** Dave Hollander
**Target branch:** `chord-detection`
**Predecessor specs:** [2026-05-09-polyphonic-detection-design.md](2026-05-09-polyphonic-detection-design.md) (§16.3 lists this as Phase 4 priorities #1 and #2)

---

## 1. Goal

Resolve the two highest-priority items in the polyphonic-detection §16.3
follow-up list:

1. **Quantizer overlap bug** — `RhythmQuantizer.quantize()` produces
   MusicXML with inconsistent `<duration>` and `<type>` element pairings
   on certain real-audio inputs (slow tempo + irregular onset spacing
   when triplet durations are in the catalog). MuseScore silently
   rejects these files (exit 40) and the PDF render fails. The bug
   blocks end-to-end PDF rendering on the E, A, D, C dyad samples and
   four of the five open-chord samples (only the G chord happens to
   dodge it).
2. **CLI `--backend` missing `basic_pitch`** — the argparse `choices`
   list in [src/improv_scribe/cli.py](../../../src/improv_scribe/cli.py)
   currently has `["pyin", "crepe"]`. Users can only select the
   basic-pitch backend via the `ATS_PITCH_BACKEND=basic_pitch` env var,
   which is not discoverable from `--help`.

### Non-goals

- Chord-recall improvement (Phase 5+ candidate). The "missing high
  voices" pattern documented in §15.2 is left as-is; this phase does
  not change `_BasicPitchBackend.estimate()` or `_cluster_basic_pitch_notes()`.
- Decay-tail singleton filtering (Phase 5+ candidate). The spurious
  short clusters documented in §15.5 are left in the ground truth.
- Chord-name detection (Phase 5+ candidate). No chord-symbol labels.
- Rhythm-mode work beyond fixing the tiling invariant. The "raw" mode
  is untouched.

---

## 2. Quantizer common-grid snapping

### 2.1 Root cause recap

The current `RhythmQuantizer` ([quantization/grid.py](../../../src/improv_scribe/quantization/grid.py))
snaps three quantities independently against the standard `NoteDuration`
catalog:

- `_snap_to_grid(onset_beat)` — onset position
- `_snap_duration(raw_dur_beats)` — note duration
- `_snap_duration(gap_beats)` — rest duration filling the gap to the next onset

Because the catalog mixes triplet durations (1/12 whole-note) and regular
durations (1/16 whole-note), and the snapping decisions are independent,
the chosen rest duration can extend past the next note's snapped onset.
Example from Phase 2 §14.3 / 2026-05-23 diagnostic:

```
Note (triplet-quarter, dur=0.667 beats) ends at beat 3.167
Rest with raw gap 0.583 snaps to 0.667 (closest catalog match) → ends at beat 3.834
Next note onset snaps to 3.750
Overlap of 0.084 beats. MusicXML <duration> 7560 says eighth note,
<type> says quarter — MuseScore exit 40, no useful error.
```

### 2.2 New algorithm

`RhythmQuantizer.quantize()` is rewritten around a single fine grid
(`grid_beats`) chosen at construction:

```python
grid_beats = min(frac * 4 for _name, frac in self._candidates)
```

For the default catalog (sixteenth + triplets enabled) this is
**1/12 beat ≈ 0.0833**. With `include_triplets=False` it's
**1/16 beat = 0.0625** (sixteenth note). Either way the grid divides
every candidate duration evenly.

The per-event procedure becomes:

1. Convert `event.onset_s`, `event.offset_s` → beats.
2. **Snap both onset and offset to `grid_beats`** by `round(beat / grid_beats) * grid_beats`. Clamp `snapped_offset ≥ snapped_onset + grid_beats` so every event has a minimum 1-grid-cell duration.
3. Compute `dur_beats = snapped_offset - snapped_onset`. By construction this is a positive multiple of `grid_beats`.
4. **Pick the largest `NoteDuration` that fits** (catalog value ≤ `dur_beats`). This is the key rule that prevents overlap: by preferring "fits-within" over "closest", the chosen duration never extends past the original snapped offset. The chosen `closest_dur_beats` is therefore ≤ `dur_beats`. Update `snapped_offset = snapped_onset + closest_dur_beats` (may retreat by up to nearly one catalog gap, ~0.083 beats at the 1/12 grid). If `dur_beats` is smaller than the smallest catalog value (`sixteenth = 0.25 beats` or `triplet-eighth = 0.333 beats`), pick the smallest catalog value and accept that the note's snapped offset advances by the difference — this is the only case where the note expands; needed because zero-duration notes are invalid in music21.
5. Emit the note's `QuantizedNote` with `onset_beat=snapped_onset`, `duration_beats=closest_dur_beats`, `quarter_length=to_quarter_length(closest_dur_type)`.

Between consecutive notes:

6. **Rest gap** = `next_note.snapped_onset - prev_note.snapped_offset`. By construction this is a non-negative multiple of `grid_beats` (modulo the rare clamp from step 4's smallest-catalog case, which can produce a tiny negative gap; treat negative gap as zero and skip the rest).
7. If `gap >= grid_beats`, emit a rest. Pick the largest catalog value that fits (same rule as step 4 for notes). If the chosen rest duration is less than the actual gap, the leftover is absorbed as quantization noise — the next note's `snapped_onset` is unchanged, but there's a small unfilled region in the measure timeline (music21 fills this with implicit rests or absorbs it into the surrounding measure structure).
8. If `gap < grid_beats`, emit no rest. This matches the existing `_make_rest` behaviour for sub-minimum gaps.

### 2.3 Tiling invariant

After quantization, for any consecutive pair of `QuantizedNote` entries
in the output list:

```
prev.onset_beat + prev.duration_beats <= next.onset_beat
```

(non-overlapping, not strictly tiling). The unit tests assert this
non-overlap invariant. Strict equality holds when the gap is a clean
multiple of the smallest catalog duration; otherwise a small unfilled
region remains in the measure timeline that music21 absorbs into
implicit rests during MusicXML serialisation.

### 2.4 Boundary behaviour

- **Already-aligned input** (existing Phase 1/2 mono integration tests
  at 120 BPM): onsets at 1.0, 2.0, ... beats land on grid points
  exactly. Durations also align. The snap is a no-op and the existing
  `EXPECTED_MIDI` / `EXPECTED_TAB` ground truth is preserved.
- **Triplet-feel content**: 1/12-beat grid resolves triplet quarters
  (4/12 = 1/3 beat = 0.333), triplet eighths (2/12 = 1/6), and all
  regular durations (1/4 = 3/12 = 0.25, etc.) without conflict.
- **Very short events** (< `grid_beats`): clamped to one grid cell.
  This may extend a transient detection but prevents zero-duration
  notes which music21 rejects.

### 2.5 Files affected

- Modify: `src/improv_scribe/quantization/grid.py` — `RhythmQuantizer.__init__`, `quantize()`, internal helpers (`_snap_to_grid`, `_snap_duration`, `_make_rest`).
- Modify: `tests/quantization/test_grid.py` — add `TestQuantizerTiling` class (5 tests, see §4.1).

The integration tests for existing samples (Phase 1/2/3) should pass
unchanged because their inputs already snap cleanly. If any fail, the
ground truth needs a one-cell adjustment.

---

## 3. CLI `--backend` choice update

### 3.1 Change

In [src/improv_scribe/cli.py](../../../src/improv_scribe/cli.py), the
existing argparse line:

```python
parser.add_argument(
    "--backend",
    choices=["pyin", "crepe"],
    default=None,
    help="Pitch detection backend (default: pyin)",
)
```

becomes:

```python
parser.add_argument(
    "--backend",
    choices=["pyin", "crepe", "basic_pitch"],
    default=None,
    help=(
        "Pitch detection backend. 'basic_pitch' requires running "
        "`bash scripts/install_basic_pitch.sh` after env creation."
    ),
)
```

### 3.2 Default-resolution wiring

The CLI's `--backend` argument currently flows into the `AppConfig`
constructor or via `PitchEstimator(backend=...)`. Verify the routing
correctly handles `"basic_pitch"`; no additional plumbing should be
needed since `_BACKENDS` dict in `pitch.py` already maps the string.

### 3.3 Files affected

- Modify: `src/improv_scribe/cli.py` — 5-line change (choices list + help text).

No new test files — the help-text update is the regression gate (visible
in `--help` output).

---

## 4. Testing

### 4.1 Unit tests — `tests/quantization/test_grid.py::TestQuantizerTiling`

Five tests, all using fabricated `NoteEvent` inputs (no real audio):

1. **`test_consecutive_notes_tile_exactly`** — two abutting notes; assert no rest is inserted and `q[0].onset_beat + q[0].duration_beats == q[1].onset_beat`.
2. **`test_rest_fills_gap_exactly`** — two notes with a gap; assert one rest inserted, sum of rest + note durations equals the gap-spanning distance.
3. **`test_overlap_regression_dyad_sample_scenario`** — feed `NoteEvent`s with raw timings that reproduced the failing dyad case (40 BPM tempo, onsets near triplet-quarter boundaries). Assert tiling invariant holds on all consecutive pairs.
4. **`test_triplet_quarter_duration_preserved`** — feed an event with `offset_s - onset_s` corresponding to exactly 2/3 of a beat at the test tempo. Assert quantized `duration_type == NoteDuration.TRIPLET_QUARTER` and `duration_beats == pytest.approx(2/3)`.
5. **`test_phase_1_2_3_aligned_inputs_unchanged`** — feed simulated events at 120 BPM with `onset_s` values at `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]` (= beats 1, 2, 3, 4, 5, 6) and `offset_s - onset_s = 0.5` (= 1 beat each). Assert all snap to integer beat positions with `NoteDuration.QUARTER` (no shift, no drift — matches Phase 1/2/3 ground truth for clean-grid inputs).

### 4.2 Integration smoke test — `tests/integration/test_pdf_render_smoke.py`

New file. One test:

```python
def test_open_g_chord_renders_to_pdf(tmp_path):
    """Full pipeline: G chord sample → ScoreBuilder → PDFExporter.

    Asserts PDF render succeeds AND the output file is non-trivial.
    Regression guard for spec §14.3 quantizer overlap bug.
    """
    # Load sample, run pipeline, call PDFExporter.export() with tab injection
    # Assert pdf_path.exists() and pdf_path.stat().st_size > 5000
```

Uses the G chord sample (already-passing pre-fix; the fix should not
regress it). After the quantizer fix, this test should also pass on the
E/A/D/C samples — but adding all five would 5× the runtime. **One sample
is sufficient as a smoke gate;** the unit tests carry the per-input
correctness load.

Skipped under `ATS_PITCH_BACKEND=crepe` (no chord-detection ground
truth on CREPE — matches existing chord-test skip pattern).

### 4.3 Phase 4 phase gate

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
```

Expected:
- basic_pitch: **318 passed, 4 skipped** (was 312/4 — +6 new tests: 5 unit + 1 PDF smoke).
- crepe: **289 passed, 33 skipped** (was 284/32 — +5 unit, +1 skip for the chord-backend-only PDF smoke).

If existing ground-truth tests fail because the quantizer now produces
slightly different beat positions (e.g. shifted by 1 grid cell), update
the affected `EXPECTED_*` fixture and document the diff. This is
acceptable provided the new values still make musical sense.

### 4.4 Phase 4 outcome

After implementation, append §17 to the polyphonic spec with:
- Commit list
- Final pytest counts (both backends)
- Confirmation that PDF render now works on at least the G chord (smoke) and ideally all five open-chord samples (manual verification)
- A note in §16.3 Phase 4 priorities marking #1 and #2 as ✓ done

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Existing integration ground truth shifts by 1 grid cell** — Phase 1/2/3 `EXPECTED_*` fixtures may not match the new quantizer output exactly | Medium | Test 5 above guards against this: simulated clean-grid inputs at 120 BPM produce unchanged output. If real-audio integration tests fail, treat as ground-truth update (one commit). |
| **Note end-shift of ±½ grid cell** (§2.2 step 4) audibly clips short notes | Low | Grid is 1/12 beat ≈ 50 ms at 120 BPM. ½ grid = 25 ms shift. Inaudible at typical tempos. |
| **PDF smoke test slow** (~10s per render) | Low | Single render per backend run. Acceptable cost for the strongest regression gate. |
| **CLI choice change breaks downstream scripts** | Very low | Adding to `choices` doesn't remove existing values. Users still call `--backend crepe` etc. |
| **Quantizer rewrite introduces new edge cases not caught by §4.1 tests** | Medium | The 5 unit tests cover: abutting notes, gapped notes, the original failing scenario, triplets, and Phase 1/2/3 aligned inputs. Anything else can be added on a per-bug basis. |

---

## 6. Dependency additions

None. Phase 4 is pure code changes in existing modules.

---

## 7. Migration completion checklist

After Phase 4 implementation:

- [ ] All 5 unit tests in `TestQuantizerTiling` pass.
- [ ] PDF smoke test passes (open G chord renders to >5KB PDF).
- [ ] CLI `--help` lists `basic_pitch` as a `--backend` choice.
- [ ] Both backends pass full pytest gauntlet at the expected counts (§4.3).
- [ ] Spec §17 records the Phase 4 outcome with commit SHAs and final counts.
- [ ] Manual verification: at least one of E/A/D/C chord samples renders to PDF (proves the original §14.3 blocker is resolved).
