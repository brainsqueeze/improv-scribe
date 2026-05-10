# Polyphonic Detection — Phase 0: Data Model Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `NoteEvent` and `QuantizedNote` from a single-pitch `midi_note: int` model to a chord-capable `midi_notes: tuple[int, ...]` model, with back-compat properties so all existing call-sites keep working unchanged. No algorithmic change visible to the integration tests.

**Architecture:** A pure data-model refactor. Both dataclasses gain tuple-typed fields for `midi_notes`, `frequencies_hz`, `confidences`, `cents_deviations`. Read-only `midi_note`, `frequency_hz`, `confidence`, `cents_deviation` properties wrap `[0]` (or mean for confidence) so existing code paths see no change. The `_merge_consecutive_same_pitch()` helper is rewritten to compare full tuples and average element-wise — this **is** an algorithmic change at the helper level, but on monophonic singletons it produces identical results to today.

**Tech Stack:** Python 3.13 dataclasses, pytest

**Spec reference:** [docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md](../specs/2026-05-09-polyphonic-detection-design.md) §3.1, §3.3.

**Phase gate (definition of done):**
```bash
ATS_PITCH_BACKEND=pyin   conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=crepe  conda run -n auto-sheet-music pytest tests/integration -v
```
Both must pass with no test changes other than what this plan introduces.

**Phasing context:** This plan covers **Phase 0 only** of the four-phase polyphonic spec. Phase 1 (basic-pitch backend) requires an empirical prototype of basic-pitch's actual return shape before its tasks can be written without placeholders. Phase 2 requires Phase 1's empirical calibration. Phase 3 requires user-provided real chord recordings. Plans for Phases 1–3 will be drafted after Phase 0 lands.

---

## File Map

| File | Change |
|---|---|
| `src/improv_scribe/analysis/note_tracker.py` | `NoteEvent` gains tuple fields + back-compat properties; `_merge_consecutive_same_pitch` rewritten for tuple semantics; `NoteTracker.process()` constructs singleton tuples. |
| `src/improv_scribe/quantization/grid.py` | `QuantizedNote` gains tuple fields + back-compat properties; `RhythmQuantizer.quantize()` and `_make_rest()` construct singleton tuples (rests use empty tuple). |
| `tests/analysis/test_note_event.py` *(new)* | Unit tests for the new `NoteEvent` shape, back-compat properties, and merge semantics. |
| `tests/quantization/test_quantized_note.py` *(new)* | Unit tests for the new `QuantizedNote` shape and back-compat properties. |
| `tests/integration/*.py` | **Unchanged.** Existing tests access `event.midi_note` via the back-compat property. The phase gate proves no regression. |

Files that read `.midi_note` and rely on the back-compat shim (no change needed in Phase 0):
- `src/improv_scribe/notation/score_builder.py:138`
- `src/improv_scribe/notation/tab_builder.py:77`
- `src/improv_scribe/export/midi_exporter.py:120-121`
- `src/improv_scribe/export/tab_xml.py` (if any)
- `src/improv_scribe/gui/main_window.py` (any pitch display)
- `src/improv_scribe/gui/transcription_log.py` (any pitch display)

These will be migrated to read `midi_notes` directly in Phase 2 when the back-compat shim is removed.

---

## Task 1: Confirm baseline integration tests are green

**Files:** none (read-only verification step)

- [ ] **Step 1: Run the regression gauntlet on `chord-detection` branch as-is**

```bash
ATS_PITCH_BACKEND=pyin   conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=crepe  conda run -n auto-sheet-music pytest tests/integration -v
```

Expected: both PASS. If either fails, **STOP** — fix the baseline before starting the migration. A pre-existing failure will mask Phase 0 regressions.

---

## Task 2: Add unit tests for the new `NoteEvent` shape

**Files:**
- Create: `tests/analysis/test_note_event.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for NoteEvent's chord-capable shape.

These tests lock down the data-model contract that the rest of the
polyphonic detection pipeline relies on. Phase 0 keeps back-compat
properties so existing single-pitch consumers see no behaviour change.
"""

from __future__ import annotations

import pytest

from improv_scribe.analysis.note_tracker import NoteEvent


def _make(midi_notes: tuple[int, ...] = (60,), **overrides) -> NoteEvent:
    """Helper: build a NoteEvent with sensible defaults for testing."""
    n = len(midi_notes)
    defaults = {
        "onset_s": 0.0,
        "offset_s": 1.0,
        "midi_notes": midi_notes,
        "frequencies_hz": tuple(440.0 * 2 ** ((m - 69) / 12) for m in midi_notes),
        "confidences": (0.9,) * n,
        "cents_deviations": (0.0,) * n,
    }
    defaults.update(overrides)
    return NoteEvent(**defaults)


class TestNoteEventShape:
    def test_singleton_construction(self):
        event = _make(midi_notes=(60,))
        assert event.midi_notes == (60,)
        assert len(event.frequencies_hz) == 1
        assert len(event.confidences) == 1
        assert len(event.cents_deviations) == 1

    def test_chord_construction(self):
        event = _make(midi_notes=(60, 64, 67))   # C major triad
        assert event.midi_notes == (60, 64, 67)
        assert len(event.frequencies_hz) == 3
        assert len(event.confidences) == 3
        assert len(event.cents_deviations) == 3

    def test_is_chord_property(self):
        assert _make(midi_notes=(60,)).is_chord is False
        assert _make(midi_notes=(60, 64)).is_chord is True
        assert _make(midi_notes=(60, 64, 67)).is_chord is True

    def test_duration_s(self):
        event = _make(onset_s=0.5, offset_s=2.5)
        assert event.duration_s == pytest.approx(2.0)

    def test_duration_s_clamps_to_zero(self):
        event = _make(onset_s=2.0, offset_s=1.0)
        assert event.duration_s == 0.0


class TestNoteEventBackCompatProperties:
    """Phase 0 keeps these properties for callers that haven't migrated yet.
    Phase 2 removes them; the migration completion check is a grep for `.midi_note`
    and `.frequency_hz` returning zero hits in `src/`."""

    def test_midi_note_returns_first_element(self):
        assert _make(midi_notes=(60,)).midi_note == 60
        assert _make(midi_notes=(60, 64, 67)).midi_note == 60   # lowest

    def test_frequency_hz_returns_first_element(self):
        event = _make(midi_notes=(69,))   # A4 = 440 Hz
        assert event.frequency_hz == pytest.approx(440.0)

    def test_confidence_returns_mean(self):
        event = _make(midi_notes=(60, 64), confidences=(0.8, 0.6))
        assert event.confidence == pytest.approx(0.7)

    def test_cents_deviation_returns_first_element(self):
        event = _make(midi_notes=(60, 64), cents_deviations=(5.0, -3.0))
        assert event.cents_deviation == pytest.approx(5.0)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_event.py -v
```

Expected: FAIL with `TypeError` on `NoteEvent(...)` or attribute errors — the new fields don't exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/analysis/test_note_event.py
git commit -m "test(analysis): add NoteEvent chord-shape contract tests (Phase 0)"
```

---

## Task 3: Refactor `NoteEvent` to tuple-based fields

**Files:**
- Modify: `src/improv_scribe/analysis/note_tracker.py:37-64`

- [ ] **Step 1: Replace the `NoteEvent` dataclass**

Replace the existing definition (lines 37–64) with:

```python
@dataclass
class NoteEvent:
    """A detected note or chord event with timing and pitch.

    All times are in seconds relative to the start of the recorded session.
    A monophonic detection emits singleton tuples (length 1). Chord
    detections emit tuples of length 2+, with `midi_notes` sorted ascending
    so chord identity is canonical.

    Phase 0 of the polyphonic migration introduces the tuple fields and
    keeps single-element back-compat properties for callers that have not
    yet been migrated. The properties will be removed in Phase 2.

    Parameters
    ----------
    onset_s : float
        Note start time.
    offset_s : float
        Note end time (next onset or last voiced frame).
    midi_notes : tuple[int, ...]
        MIDI note numbers (0–127), sorted ascending. Empty tuple for rests
        (rests are represented in QuantizedNote, not NoteEvent — NoteEvent
        always has at least one pitch).
    frequencies_hz : tuple[float, ...]
        Median f0 across active frames, parallel to midi_notes.
    confidences : tuple[float, ...]
        Mean voiced-probability across active frames, parallel to midi_notes.
    cents_deviations : tuple[float, ...]
        Deviation from equal temperament (-50 to +50), parallel to midi_notes.
    """
    onset_s: float
    offset_s: float
    midi_notes: tuple[int, ...]
    frequencies_hz: tuple[float, ...]
    confidences: tuple[float, ...]
    cents_deviations: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.offset_s - self.onset_s)

    @property
    def is_chord(self) -> bool:
        return len(self.midi_notes) > 1

    # ------------------------------------------------------------------
    # Back-compat shims — removed in Phase 2
    # ------------------------------------------------------------------

    @property
    def midi_note(self) -> int:
        """Lowest MIDI note. Back-compat shim — prefer `midi_notes[0]`."""
        return self.midi_notes[0]

    @property
    def frequency_hz(self) -> float:
        """First-pitch frequency. Back-compat shim — prefer `frequencies_hz[0]`."""
        return self.frequencies_hz[0]

    @property
    def confidence(self) -> float:
        """Mean confidence across chord members. Back-compat shim."""
        return sum(self.confidences) / len(self.confidences)

    @property
    def cents_deviation(self) -> float:
        """First-pitch cents deviation. Back-compat shim."""
        return self.cents_deviations[0]

    def __repr__(self) -> str:
        if self.is_chord:
            return (
                f"NoteEvent(midi={list(self.midi_notes)}, "
                f"onset={self.onset_s:.3f}s, "
                f"dur={self.duration_s:.3f}s, "
                f"conf={self.confidence:.2f})"
            )
        return (
            f"NoteEvent(midi={self.midi_notes[0]}, "
            f"onset={self.onset_s:.3f}s, "
            f"dur={self.duration_s:.3f}s, "
            f"conf={self.confidence:.2f})"
        )
```

- [ ] **Step 2: Run unit tests to verify they pass**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_event.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 3: Run integration tests to confirm no regression**

```bash
ATS_PITCH_BACKEND=pyin   conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=crepe  conda run -n auto-sheet-music pytest tests/integration -v
```

Expected: both PASS — but at this point `NoteTracker.process()` still constructs the *old* `NoteEvent(midi_note=...)` so the integration tests will FAIL with `TypeError: NoteEvent.__init__() got an unexpected keyword argument 'midi_note'`. **This is expected and addressed in Task 4.**

If they fail with that error, proceed to Task 4. If they fail with a different error (e.g. one of the back-compat properties missing), fix it before continuing.

---

## Task 4: Update `NoteTracker.process()` to construct singleton tuples

**Files:**
- Modify: `src/improv_scribe/analysis/note_tracker.py:389-396` (the `events.append(NoteEvent(...))` call inside the loop)

- [ ] **Step 1: Replace the `events.append(NoteEvent(...))` block**

Locate the current code (around line 389):

```python
events.append(NoteEvent(
    onset_s=t_start + chunk_offset_s,
    offset_s=t_end + chunk_offset_s,
    midi_note=midi_note,
    frequency_hz=median_freq,
    confidence=mean_conf,
    cents_deviation=cents_dev,
))
```

Replace with:

```python
events.append(NoteEvent(
    onset_s=t_start + chunk_offset_s,
    offset_s=t_end + chunk_offset_s,
    midi_notes=(midi_note,),
    frequencies_hz=(median_freq,),
    confidences=(mean_conf,),
    cents_deviations=(cents_dev,),
))
```

- [ ] **Step 2: Run integration tests**

```bash
ATS_PITCH_BACKEND=pyin   conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=crepe  conda run -n auto-sheet-music pytest tests/integration -v
```

Expected: both PASS — back-compat properties handle the integration tests' `event.midi_note` reads.

If they fail, the most likely cause is the merge helper still using the old field names. Proceed to Task 5.

- [ ] **Step 3: Commit**

```bash
git add src/improv_scribe/analysis/note_tracker.py
git commit -m "refactor(analysis): NoteEvent uses tuple-based pitch fields (Phase 0)

Add midi_notes/frequencies_hz/confidences/cents_deviations as tuples.
Keep singular back-compat properties for callers that haven't migrated.
NoteTracker.process() emits singleton tuples for monophonic events.

Behaviour visible to existing tests is unchanged."
```

---

## Task 5: Add unit tests for `_merge_consecutive_same_pitch` chord semantics

**Files:**
- Create or extend: `tests/analysis/test_note_event.py`

- [ ] **Step 1: Add merge tests to the existing test file**

Append to `tests/analysis/test_note_event.py`:

```python
from improv_scribe.analysis.note_tracker import _merge_consecutive_same_pitch


class TestMergeConsecutiveSamePitch:
    """The merge helper collapses back-to-back same-pitch events caused by
    spurious re-onsets on sustained notes.

    Phase 0 changes: comparison is now over full midi_notes tuples (chord
    identity), parallel-tuple arithmetic for averaged fields, and a separate
    gap threshold for chord events (200 ms) vs mono events (600 ms).
    """

    def test_merges_consecutive_singletons_close_in_time(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60,),
                   frequencies_hz=(261.6,), confidences=(0.9,))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60,),
                   frequencies_hz=(262.0,), confidences=(0.85,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 1
        assert merged[0].onset_s == 0.0
        assert merged[0].offset_s == 1.0
        assert merged[0].midi_notes == (60,)
        # Frequencies averaged element-wise
        assert merged[0].frequencies_hz[0] == pytest.approx((261.6 + 262.0) / 2)
        assert merged[0].confidences[0] == pytest.approx((0.9 + 0.85) / 2)

    def test_does_not_merge_singletons_with_large_gap(self):
        # Mono gap > 600 ms must NOT merge
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60,))
        e2 = _make(onset_s=1.5, offset_s=2.0, midi_notes=(60,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_does_not_merge_different_singletons(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60,))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(64,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_merges_consecutive_chords_with_identical_pitches(self):
        # Same chord, < 200 ms gap -> merge
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64, 67),
                   frequencies_hz=(261.6, 329.6, 392.0),
                   confidences=(0.9, 0.85, 0.8),
                   cents_deviations=(0.0, 0.0, 0.0))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60, 64, 67),
                   frequencies_hz=(262.0, 330.0, 392.5),
                   confidences=(0.85, 0.8, 0.75),
                   cents_deviations=(0.0, 0.0, 0.0))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 1
        assert merged[0].midi_notes == (60, 64, 67)
        assert merged[0].frequencies_hz[0] == pytest.approx((261.6 + 262.0) / 2)
        assert merged[0].frequencies_hz[1] == pytest.approx((329.6 + 330.0) / 2)
        assert merged[0].frequencies_hz[2] == pytest.approx((392.0 + 392.5) / 2)

    def test_does_not_merge_chords_with_different_pitches(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60, 67))   # one note differs
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_does_not_merge_chord_to_singleton(self):
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64))
        e2 = _make(onset_s=0.6, offset_s=1.0, midi_notes=(60,))
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_chord_gap_threshold_is_tighter_than_mono(self):
        # Chord gap of 300 ms (eighth notes at 100 BPM) must NOT merge
        e1 = _make(onset_s=0.0, offset_s=0.5, midi_notes=(60, 64))
        e2 = _make(onset_s=0.8, offset_s=1.3, midi_notes=(60, 64))   # gap = 300 ms
        merged = _merge_consecutive_same_pitch([e1, e2])
        assert len(merged) == 2

    def test_empty_list_returns_empty(self):
        assert _merge_consecutive_same_pitch([]) == []

    def test_single_event_returns_single(self):
        e = _make(midi_notes=(60,))
        merged = _merge_consecutive_same_pitch([e])
        assert merged == [e]
```

- [ ] **Step 2: Run new tests to confirm most fail**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_event.py::TestMergeConsecutiveSamePitch -v
```

Expected: 6 of 9 tests FAIL.
- `test_merges_consecutive_singletons_close_in_time` — FAIL: tuple addition raises `TypeError` on `frequency_hz` averaging.
- `test_merges_consecutive_chords_with_identical_pitches` — FAIL: ditto.
- `test_does_not_merge_chord_to_singleton` — FAIL: comparison happens to pass on first element via `midi_note` shim, falsely merging.
- `test_does_not_merge_chords_with_different_pitches` — FAIL: ditto.
- `test_chord_gap_threshold_is_tighter_than_mono` — FAIL: chord uses mono 600 ms threshold.
- `test_does_not_merge_different_singletons` — should PASS (different `midi_note`).
- `test_does_not_merge_singletons_with_large_gap` — should PASS.
- The two trivial tests (empty, single) — should PASS.

If the failure pattern differs significantly, stop and investigate before continuing.

---

## Task 6: Rewrite `_merge_consecutive_same_pitch` for chord semantics

**Files:**
- Modify: `src/improv_scribe/analysis/note_tracker.py:253-285`

- [ ] **Step 1: Replace the merge helper**

Replace the existing `_MERGE_GAP_S` constant and `_merge_consecutive_same_pitch` function with:

```python
# Maximum silence between two same-pitch single-note events to treat as one note.
# Spurious re-onsets caused by harmonic evolution appear within 600 ms;
# intentional repeated notes at >=80 BPM have gaps >=375 ms but are accompanied
# by a fresh attack, so we use a conservative 600 ms ceiling.
_MERGE_GAP_S: float = 0.600

# Tighter threshold for chord events. Eighth-note strums at 100 BPM are 300 ms
# apart and must NOT merge into one held chord.
_MERGE_GAP_CHORD_S: float = 0.200


def _avg_tuples(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    """Element-wise average of two parallel tuples.

    Pre-condition: len(a) == len(b). Used by the merge helper after the
    caller verifies tuple equality of midi_notes (which guarantees parallel
    structure).
    """
    return tuple((x + y) / 2.0 for x, y in zip(a, b, strict=True))


def _merge_consecutive_same_pitch(events: list[NoteEvent]) -> list[NoteEvent]:
    """Merge back-to-back NoteEvents whose midi_notes are identical.

    Handles phantom re-onsets caused by onset_detect firing on harmonic
    evolution of a sustained note (e.g. the B3 string triggering twice).

    The merge gap threshold differs by chord size:
    - Singleton (mono) events: 600 ms — covers harmonic-evolution
      false re-onsets on a decaying single note.
    - Chord events: 200 ms — tighter, because eighth-note repeated chords
      at 100 BPM are 300 ms apart and must NOT merge.
    """
    if not events:
        return events

    merged: list[NoteEvent] = [events[0]]
    for current in events[1:]:
        prev = merged[-1]
        gap = current.onset_s - prev.offset_s
        same_pitches = current.midi_notes == prev.midi_notes
        threshold = _MERGE_GAP_CHORD_S if prev.is_chord else _MERGE_GAP_S
        if same_pitches and gap <= threshold:
            merged[-1] = NoteEvent(
                onset_s=prev.onset_s,
                offset_s=current.offset_s,
                midi_notes=prev.midi_notes,
                frequencies_hz=_avg_tuples(prev.frequencies_hz, current.frequencies_hz),
                confidences=_avg_tuples(prev.confidences, current.confidences),
                cents_deviations=_avg_tuples(prev.cents_deviations, current.cents_deviations),
            )
        else:
            merged.append(current)
    return merged
```

- [ ] **Step 2: Run merge unit tests to verify all pass**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_event.py::TestMergeConsecutiveSamePitch -v
```

Expected: all 9 tests PASS.

- [ ] **Step 3: Run integration tests to confirm no regression**

```bash
ATS_PITCH_BACKEND=pyin   conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=crepe  conda run -n auto-sheet-music pytest tests/integration -v
```

Expected: both PASS. The merge logic on monophonic singletons produces identical results to the pre-refactor code (same comparison via tuple equality, same averaging via `_avg_tuples` on length-1 tuples).

- [ ] **Step 4: Commit**

```bash
git add src/improv_scribe/analysis/note_tracker.py tests/analysis/test_note_event.py
git commit -m "refactor(analysis): chord-aware _merge_consecutive_same_pitch (Phase 0)

Compare full midi_notes tuples for chord identity. Average parallel-tuple
fields element-wise via _avg_tuples. Use a tighter 200 ms gap threshold
for chord events (eighth-note repeated chords at 100 BPM must not merge).

Mono singleton behaviour is preserved: tuple equality reduces to int
equality, and length-1 tuple averaging reduces to scalar averaging."
```

---

## Task 7: Add unit tests for the new `QuantizedNote` shape

**Files:**
- Create: `tests/quantization/test_quantized_note.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for QuantizedNote's chord-capable shape (Phase 0)."""

from __future__ import annotations

import pytest

from improv_scribe.quantization.grid import NoteDuration, QuantizedNote


def _make_qn(
    midi_notes: tuple[int, ...] = (60,),
    is_rest: bool = False,
    **overrides,
) -> QuantizedNote:
    """Helper: build a QuantizedNote with sensible defaults for testing."""
    n = len(midi_notes) if not is_rest else 0
    defaults = {
        "midi_notes": () if is_rest else midi_notes,
        "frequencies_hz": () if is_rest else (440.0,) * n,
        "confidences": () if is_rest else (0.9,) * n,
        "cents_deviations": () if is_rest else (0.0,) * n,
        "onset_beat": 0.0,
        "duration_beats": 1.0,
        "duration_type": NoteDuration.QUARTER,
        "quarter_length": 1.0,
        "is_rest": is_rest,
    }
    defaults.update(overrides)
    return QuantizedNote(**defaults)


class TestQuantizedNoteShape:
    def test_singleton_construction(self):
        qn = _make_qn(midi_notes=(60,))
        assert qn.midi_notes == (60,)
        assert qn.is_rest is False

    def test_chord_construction(self):
        qn = _make_qn(midi_notes=(60, 64, 67))
        assert qn.midi_notes == (60, 64, 67)
        assert len(qn.frequencies_hz) == 3
        assert len(qn.confidences) == 3

    def test_rest_has_empty_tuples(self):
        qn = _make_qn(is_rest=True)
        assert qn.is_rest is True
        assert qn.midi_notes == ()
        assert qn.frequencies_hz == ()
        assert qn.confidences == ()
        assert qn.cents_deviations == ()


class TestQuantizedNoteBackCompatProperties:
    """Removed in Phase 2."""

    def test_midi_note_returns_first_element(self):
        assert _make_qn(midi_notes=(60,)).midi_note == 60
        assert _make_qn(midi_notes=(60, 64, 67)).midi_note == 60

    def test_midi_note_returns_zero_for_rest(self):
        # Existing rest convention is midi_note=0.
        assert _make_qn(is_rest=True).midi_note == 0

    def test_frequency_hz_returns_first_element(self):
        assert _make_qn(midi_notes=(60,), frequencies_hz=(261.6,)).frequency_hz == pytest.approx(261.6)

    def test_frequency_hz_returns_zero_for_rest(self):
        assert _make_qn(is_rest=True).frequency_hz == 0.0

    def test_confidence_returns_mean(self):
        qn = _make_qn(midi_notes=(60, 64), confidences=(0.8, 0.6))
        assert qn.confidence == pytest.approx(0.7)

    def test_confidence_returns_one_for_rest(self):
        # Existing rest convention is confidence=1.0.
        assert _make_qn(is_rest=True).confidence == 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n auto-sheet-music pytest tests/quantization/test_quantized_note.py -v
```

Expected: FAIL with `TypeError` on `QuantizedNote(...)` — the new fields don't exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/quantization/test_quantized_note.py
git commit -m "test(quantization): add QuantizedNote chord-shape contract tests (Phase 0)"
```

---

## Task 8: Refactor `QuantizedNote` to tuple-based fields

**Files:**
- Modify: `src/improv_scribe/quantization/grid.py:76-90`

- [ ] **Step 1: Replace the `QuantizedNote` dataclass**

Replace the existing definition (lines 76–90) with:

```python
@dataclass
class QuantizedNote:
    """A NoteEvent after rhythm quantization, chord-capable.

    Phase 0 of the polyphonic migration introduces tuple-typed pitch fields
    matching NoteEvent. Rests carry empty tuples for all pitch fields.
    Back-compat properties wrap the first element so existing consumers
    (notation, tab, MIDI export) continue to work without modification.
    The properties will be removed in Phase 2.

    Parameters
    ----------
    midi_notes : tuple[int, ...]
        MIDI notes, sorted ascending. Empty tuple for rests.
    frequencies_hz : tuple[float, ...]
        Parallel to midi_notes. Empty for rests.
    confidences : tuple[float, ...]
        Parallel to midi_notes. Empty for rests.
    cents_deviations : tuple[float, ...]
        Parallel to midi_notes. Empty for rests.
    onset_beat : float
        Beat position (quarter note = 1).
    duration_beats : float
        Duration in beats.
    duration_type : NoteDuration
        Standard music notation duration name.
    quarter_length : float
        music21 quarterLength.
    is_rest : bool
        True for inserted rest entries.
    """
    midi_notes: tuple[int, ...]
    frequencies_hz: tuple[float, ...]
    confidences: tuple[float, ...]
    cents_deviations: tuple[float, ...]

    onset_beat: float
    duration_beats: float
    duration_type: NoteDuration
    quarter_length: float

    is_rest: bool = False

    # ------------------------------------------------------------------
    # Back-compat shims — removed in Phase 2
    # ------------------------------------------------------------------

    @property
    def midi_note(self) -> int:
        """Lowest MIDI note, or 0 for a rest. Back-compat shim."""
        return self.midi_notes[0] if self.midi_notes else 0

    @property
    def frequency_hz(self) -> float:
        """First-pitch frequency, or 0.0 for a rest. Back-compat shim."""
        return self.frequencies_hz[0] if self.frequencies_hz else 0.0

    @property
    def confidence(self) -> float:
        """Mean confidence across chord members, or 1.0 for a rest."""
        if not self.confidences:
            return 1.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def cents_deviation(self) -> float:
        """First-pitch cents deviation, or 0.0 for a rest. Back-compat shim."""
        return self.cents_deviations[0] if self.cents_deviations else 0.0
```

- [ ] **Step 2: Run unit tests to verify they pass**

```bash
conda run -n auto-sheet-music pytest tests/quantization/test_quantized_note.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 3: Run integration tests — they will fail (expected)**

```bash
ATS_PITCH_BACKEND=pyin conda run -n auto-sheet-music pytest tests/integration -v
```

Expected: FAIL — `RhythmQuantizer.quantize()` and `_make_rest()` still construct the old single-pitch `QuantizedNote`. Addressed in Task 9.

---

## Task 9: Update `RhythmQuantizer.quantize()` and `_make_rest()`

**Files:**
- Modify: `src/improv_scribe/quantization/grid.py:178-188` (the `quantize()` append) and `src/improv_scribe/quantization/grid.py:221-236` (`_make_rest`)

- [ ] **Step 1: Update the `quantize()` `quantized.append(...)` call**

Locate (around line 178):

```python
quantized.append(QuantizedNote(
    midi_note=event.midi_note,
    frequency_hz=event.frequency_hz,
    confidence=event.confidence,
    cents_deviation=event.cents_deviation,
    onset_beat=snapped_onset,
    duration_beats=dur_beats,
    duration_type=dur_type,
    quarter_length=to_quarter_length(dur_type),
))
```

Replace with:

```python
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
```

- [ ] **Step 2: Update `_make_rest()`**

Replace the body (around line 221):

```python
def _make_rest(self, start_beat: float, gap_beats: float) -> QuantizedNote | None:
    """Create a rest QuantizedNote for a gap between notes."""
    dur_type, dur_frac = self._snap_duration(gap_beats / 4.0 * 4.0)
    if _DURATION_FRACTIONS[dur_type] < self._min_dur_beats() / 4.0 - 1e-9:
        return None
    return QuantizedNote(
        midi_note=0,
        frequency_hz=0.0,
        confidence=1.0,
        cents_deviation=0.0,
        onset_beat=start_beat,
        duration_beats=_DURATION_FRACTIONS[dur_type] * 4.0,
        duration_type=dur_type,
        quarter_length=to_quarter_length(dur_type),
        is_rest=True,
    )
```

With:

```python
def _make_rest(self, start_beat: float, gap_beats: float) -> QuantizedNote | None:
    """Create a rest QuantizedNote for a gap between notes."""
    dur_type, dur_frac = self._snap_duration(gap_beats / 4.0 * 4.0)
    if _DURATION_FRACTIONS[dur_type] < self._min_dur_beats() / 4.0 - 1e-9:
        return None
    return QuantizedNote(
        midi_notes=(),
        frequencies_hz=(),
        confidences=(),
        cents_deviations=(),
        onset_beat=start_beat,
        duration_beats=_DURATION_FRACTIONS[dur_type] * 4.0,
        duration_type=dur_type,
        quarter_length=to_quarter_length(dur_type),
        is_rest=True,
    )
```

- [ ] **Step 3: Run integration tests under both backends**

```bash
ATS_PITCH_BACKEND=pyin   conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=crepe  conda run -n auto-sheet-music pytest tests/integration -v
```

Expected: both PASS. All consumers of `qn.midi_note`, `qn.frequency_hz`, etc. resolve through the back-compat properties.

If they fail, the most likely cause is a `score_builder` or `tab_builder` access that takes a different code path on `is_rest`. Read the failure carefully and patch the consumer rather than removing the back-compat property.

- [ ] **Step 4: Commit**

```bash
git add src/improv_scribe/quantization/grid.py tests/quantization/test_quantized_note.py
git commit -m "refactor(quantization): QuantizedNote uses tuple-based pitch fields (Phase 0)

Mirror the NoteEvent migration: tuple fields with single-element back-compat
properties. Rests carry empty tuples; the back-compat properties return the
existing rest sentinels (midi_note=0, frequency_hz=0.0, confidence=1.0).

RhythmQuantizer.quantize() and _make_rest() construct the new shape.
All four monophonic integration tests pass under both pyin and crepe backends."
```

---

## Task 10: Final phase-gate verification

**Files:** none (read-only verification)

- [ ] **Step 1: Run the full integration gauntlet under both backends**

```bash
ATS_PITCH_BACKEND=pyin   conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=crepe  conda run -n auto-sheet-music pytest tests/integration -v
```

Expected: both runs PASS with the same number of tests as the baseline run from Task 1.

- [ ] **Step 2: Run unit tests added by this phase**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_event.py tests/quantization/test_quantized_note.py -v
```

Expected: all PASS.

- [ ] **Step 3: Run Ruff to verify no lint regressions**

```bash
conda run -n auto-sheet-music ruff check src/ tests/
```

Expected: clean exit (or only pre-existing warnings — diff against the baseline before starting if unsure).

- [ ] **Step 4: Sanity-check the back-compat shim coverage**

```bash
grep -rn '\.midi_note\b' src/ | grep -v 'def midi_note'
```

Expected output: this lists every call-site that still relies on the back-compat property. These are the targets for Phase 2 migration. The list should include at least:
- `src/improv_scribe/notation/score_builder.py:138`
- `src/improv_scribe/notation/tab_builder.py:77`
- `src/improv_scribe/export/midi_exporter.py:120-121`

Record this list at the bottom of [docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md](../specs/2026-05-09-polyphonic-detection-design.md) under a new "## Phase 0 Outcome" heading so Phase 2 has the migration target list ready.

- [ ] **Step 5: Confirm Phase 0 is complete**

Phase 0 ships when:
1. The two-backend integration gauntlet passes.
2. New unit tests pass.
3. Ruff is clean.
4. The Phase 2 migration target list is recorded in the spec.

At this point the `chord-detection` branch is in a state where Phase 1 (basic-pitch backend) can begin. **Stop here.** Do not start Phase 1 in the same execution session — the Phase 1 plan has not been written yet because it requires an empirical prototype of basic-pitch's API before its tasks can be specified without placeholders.

---

## What's next (out of scope for this plan)

After Phase 0 ships, the next planning step is:

1. **Prototype basic-pitch's API** — a 30-minute throwaway script that:
   - Imports `basic_pitch.inference.predict`
   - Calls it on `samples/guitar/6_string_electric_line_in.mp3`
   - Prints the actual return shape (we expect `(model_output, midi_data, note_events)` with `note_events` as a list of 5-tuples, but the installed version may differ)
   - Measures cold-start latency on the local machine
   - Confirms whether `predict()` accepts a numpy array or only a path

2. **Write the Phase 1 plan** based on the prototype's findings — likely a new file `docs/superpowers/plans/YYYY-MM-DD-polyphonic-phase-1-basic-pitch-backend.md`.

3. **Phase 2 and 3 plans** are written when their respective phase gates approach — they require user-provided real recordings (real dyads, real chord progression) that can be recorded between phases.
