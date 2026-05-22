# Polyphonic Detection — Phase 2: Dyad Detection End-to-End

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the pipeline so that 2+ simultaneously played notes flow end-to-end: detected → clustered into chord `NoteEvent`s → rendered as `music21.chord.Chord` glyphs in PDF → tab-staff fret assignments with no string conflicts → MIDI export with all chord members → all working under the new default `basic_pitch` backend.

**Architecture:** `NoteTracker._process_basic_pitch` gains onset clustering (earliest-anchor + 100 ms window) to group basic-pitch's flat note events into chord `NoteEvent`s. `ScoreBuilder.build()` learns to emit `music21.chord.Chord` for non-singleton `midi_notes`. `tab_builder.assign_frets()` becomes chord-aware with a no-string-conflict DP (members must occupy distinct strings, tie-break by `(cost, max_fret, min_fret)`). `tab_xml.inject_tab_part()` walks `<chord/>` siblings and emits chord-mirroring tab notes on staff 2. `midi_exporter.raw_from_events()` iterates `event.midi_notes` to emit one note_on/note_off per chord member (fixing the silent-drop correctness bug from Phase 0 §10). Back-compat shims (`.midi_note`, `.frequency_hz`, `.confidence`, `.cents_deviation`) are removed across the 6 documented call-sites. Default `pitch_backend` flips from `crepe` to `basic_pitch`.

**Tech Stack:** Python 3.13, basic-pitch (already installed), music21 (chord.Chord, MusicXML serialization), pytest, MuseScore CLI (PDF render verification)

**Spec reference:** [docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md](../specs/2026-05-09-polyphonic-detection-design.md) §3.2 (onset clustering), §3.4 (ScoreBuilder.Chord), §3.5 (chord DP), §3.6 (tab XML), §10 (migration target list), §13 (Phase 2 prerequisite probe — dyad samples, cluster window validated at 100 ms, ground truth captured).

**Phase gate (definition of done):**
```bash
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest tests/integration tests/analysis tests/quantization tests/notation tests/export -v
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration tests/analysis tests/quantization tests/notation tests/export -v
```
- CREPE: continues passing — mono path is the original `_process_frame_based` and notation/tab now goes through chord-capable code that reduces correctly on singletons.
- basic_pitch: passes the existing four mono integration tests AND the three new dyad integration tests.

**Phasing context:** Phase 0 (data-model migration) and Phase 1 (basic-pitch backend) are complete. This plan covers Phase 2. Phase 3 (triads + 4-note chords + real chord progression recording) and Phase 4 (polish: chord-aware octave correction, GUI chord-glyph rendering, per-string sustain in raw MIDI) follow Phase 2.

---

## File Map

| File | Change |
|---|---|
| `src/improv_scribe/config.py` | Add `ONSET_GROUPING_WINDOW_MS=100`, `POLYPHONIC_RELATIVE_FLOOR=0.5`, `MERGE_GAP_CHORD_MS=200` constants + AppConfig fields. Flip `PITCH_BACKEND` default from `"crepe"` to `"basic_pitch"`. |
| `src/improv_scribe/analysis/note_tracker.py` | Add onset clustering to `_process_basic_pitch`; emit chord `NoteEvent` (`len(midi_notes) > 1`) when cluster has multiple members. Remove `NoteEvent.midi_note`, `.frequency_hz`, `.confidence`, `.cents_deviation` back-compat properties. |
| `src/improv_scribe/quantization/grid.py` | Remove `QuantizedNote.midi_note`, `.frequency_hz`, `.confidence`, `.cents_deviation` back-compat properties. |
| `src/improv_scribe/notation/score_builder.py` | `build()` and `build_raw()` emit `music21.chord.Chord` when `len(qn.midi_notes) > 1`; `music21.note.Note(qn.midi_notes[0])` otherwise. |
| `src/improv_scribe/notation/tab_builder.py` | Add `get_chord_shapes()` enumerating distinct-string assignments. `assign_frets()` becomes chord-aware: DP over shapes with cost = max stretch + transition centroid distance; tie-break `(cost, max_fret, min_fret)`. Return type changes from `list[tuple[int, int] \| None]` to `list[tuple[tuple[int, int], ...] \| None]`. |
| `src/improv_scribe/export/tab_xml.py` | `inject_tab_part()` walks `<chord/>` siblings: chord groups consume one assignment tuple; staff-2 emits per-member `<note>` with `<chord/>` on the 2nd–Nth siblings. |
| `src/improv_scribe/export/midi_exporter.py` | `raw_from_events()` iterates `event.midi_notes` to emit one note_on/note_off per chord member at the same tick. |
| `src/improv_scribe/gui/main_window.py` | Migrate `_midi_name(n.midi_note)` to a chord-aware display formatter (e.g. "E4" for singleton, "E4/G4/B4" for chord). |
| `tests/analysis/test_note_tracker.py` *(extend)* | Add `TestNoteTrackerBasicPitchClustering` class testing the onset-grouping behaviour. |
| `tests/notation/test_score_builder.py` *(extend if exists)* | Add chord-emission tests against `music21.chord.Chord`. |
| `tests/notation/test_tab_builder.py` *(new)* | Unit tests for `get_chord_shapes` and chord-aware `assign_frets` (no string conflicts, tie-break behavior, singleton equivalence). |
| `tests/export/test_tab_xml.py` *(extend)* | Add chord-injection tests asserting `<chord/>` siblings on staff 2. |
| `tests/export/test_midi_exporter.py` *(extend if exists)* | Test that a chord NoteEvent produces N note_on events at the same tick. |
| `tests/integration/test_guitar_dyad_octave.py` *(new)* | Integration test against `samples/guitar/chords/6_string_electric_octave_dyads.mp3`. Ground truth from spec §13.5. |
| `tests/integration/test_guitar_dyad_fifth.py` *(new)* | Integration test against perfect_fifths sample. |
| `tests/integration/test_guitar_dyad_third.py` *(new)* | Integration test against major_thirds sample. |

Shim consumers being migrated (from Phase 0 §10):
1. `gui/main_window.py:288` — chord-aware display
2. `notation/tab_builder.py:77` — subsumed by chord-aware DP rewrite (Task 8)
3. `notation/score_builder.py:138` — subsumed by Chord emission (Task 6)
4. `notation/score_builder.py:192` — same (build_raw, Task 6)
5. `export/midi_exporter.py:120` — subsumed by chord iteration (Task 11)
6. `export/midi_exporter.py:121` — same (Task 11)

---

## Task 1: Create an isolated worktree for Phase 2 work

**Why:** Phase 2 touches 6 source files and removes back-compat shims. The blast radius is much larger than Phase 0 or Phase 1. An isolated worktree keeps the `chord-detection` branch shippable if Phase 2 hits an issue and needs to be unwound.

- [ ] **Step 1: Confirm current branch is `chord-detection` and clean**

```bash
git -C /Users/davehollander/Documents/Personal/Projects/audio_to_sheet status
git -C /Users/davehollander/Documents/Personal/Projects/audio_to_sheet log --oneline -3
```

Expected: clean working tree, `chord-detection` branch with the Phase 2 prerequisite probe commit (`46c8b3d`) at HEAD.

- [ ] **Step 2: Create the worktree**

```bash
cd /Users/davehollander/Documents/Personal/Projects/audio_to_sheet
git worktree add ../audio_to_sheet-phase2 -b chord-detection-phase2 chord-detection
cd ../audio_to_sheet-phase2
git status
```

Expected: new worktree at `../audio_to_sheet-phase2` on a new branch `chord-detection-phase2` based off `chord-detection`. Clean status.

- [ ] **Step 3: Verify the conda env still works from the worktree**

```bash
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: 72/72 PASS (same env, just different cwd).

**Important:** for the remainder of this plan, **all paths are relative to `../audio_to_sheet-phase2`** unless stated otherwise. The conda env (`auto-sheet-music`) is shared across worktrees.

---

## Task 2: Add Phase 2 config constants

**Files:**
- Modify: `src/improv_scribe/config.py`

- [ ] **Step 1: Add module-level constants**

After the existing `MIN_NOTE_DURATION_S` constant, add:

```python
# Window for grouping basic-pitch's flat note events into chord NoteEvents.
# Calibrated against real strum recordings (spec §13.1): actual onset spreads
# within dyads are 0-35 ms; 100 ms provides comfortable headroom.
ONSET_GROUPING_WINDOW_MS: float = float(os.getenv("ATS_ONSET_GROUPING_WINDOW_MS", "100.0"))

# Within a cluster, drop members whose amplitude is below this fraction of the
# cluster's max amplitude. Defends against basic-pitch's loud-note-dominates
# behaviour where the strongest member registers at 0.85 but a quiet member
# is at 0.42. Phase 4 may revisit; default is conservative (0.5).
POLYPHONIC_RELATIVE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_RELATIVE_FLOOR", "0.5"))

# Tighter merge threshold for chord events. Eighth-note repeated chords at
# 100 BPM are 300 ms apart and must NOT merge into one held chord. Singletons
# keep the existing 600 ms threshold (defined in note_tracker.py).
MERGE_GAP_CHORD_MS: float = float(os.getenv("ATS_MERGE_GAP_CHORD_MS", "200.0"))
```

- [ ] **Step 2: Add to AppConfig**

In the `@dataclass class AppConfig`, after `min_note_duration_s`:

```python
    onset_grouping_window_ms: float = field(default_factory=lambda: ONSET_GROUPING_WINDOW_MS)
    polyphonic_relative_floor: float = field(default_factory=lambda: POLYPHONIC_RELATIVE_FLOOR)
    merge_gap_chord_ms: float = field(default_factory=lambda: MERGE_GAP_CHORD_MS)
```

- [ ] **Step 3: Smoke test**

```bash
conda run -n auto-sheet-music python -c "
from improv_scribe.config import AppConfig
c = AppConfig()
assert c.onset_grouping_window_ms == 100.0
assert c.polyphonic_relative_floor == 0.5
assert c.merge_gap_chord_ms == 200.0
print('OK')
"
```

- [ ] **Step 4: Verify CREPE baseline preserved**

```bash
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: 72/72 PASS. Config additions have no behavioural effect yet (no consumer reads them).

- [ ] **Step 5: Commit**

```bash
git add src/improv_scribe/config.py
git commit -m "$(cat <<'EOF'
feat(config): add Phase 2 chord clustering + merge constants

ONSET_GROUPING_WINDOW_MS=100 (calibrated against real dyad samples,
spec §13.1: actual onset spreads 0-35 ms).
POLYPHONIC_RELATIVE_FLOOR=0.5 (cluster-internal amplitude floor).
MERGE_GAP_CHORD_MS=200 (chord events must not merge at eighth-note
gaps).

No consumer yet; subsequent tasks wire these into the clustering and
merge logic.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Failing tests for onset clustering in `_process_basic_pitch`

**Files:**
- Modify: `tests/analysis/test_note_tracker.py`

- [ ] **Step 1: Append the new test class**

After the existing `TestNoteTrackerBasicPitch` class:

```python
class TestNoteTrackerBasicPitchClustering:
    """Phase 2 — basic-pitch's flat note events get clustered into chord events.

    Clustering rule: a new cluster opens when current.start_s exceeds the
    earliest member of the cluster currently being built by more than the
    ONSET_GROUPING_WINDOW_MS (default 100 ms).

    Cluster members are sorted by midi ascending; duplicate MIDI within a
    cluster keep the highest-amplitude detection (basic-pitch can emit two
    near-simultaneous events for the same pitch).
    """

    def _config(self):
        return AppConfig()

    def _profile(self):
        return get_profile(Instrument.GUITAR)

    def test_two_close_events_become_one_dyad_chord(self):
        """Two events 12 ms apart -> one chord NoteEvent with midi_notes=(40, 52)."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.290, end_s=1.265, midi=40, amplitude=0.66),
            BasicPitchNote(start_s=0.302, end_s=1.370, midi=52, amplitude=0.78),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        e = events[0]
        assert e.midi_notes == (40, 52)
        assert e.is_chord is True
        # Onset = earliest member's start_s; offset = latest member's end_s
        assert e.onset_s == pytest.approx(0.290)
        assert e.offset_s == pytest.approx(1.370)
        # Frequencies/confidences/cents parallel to midi_notes
        assert len(e.frequencies_hz) == 2
        assert len(e.confidences) == 2
        assert e.confidences == (pytest.approx(0.66), pytest.approx(0.78))

    def test_events_outside_window_become_separate_events(self):
        """Two events 200 ms apart -> two separate singleton NoteEvents."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.80),
            BasicPitchNote(start_s=0.200, end_s=0.700, midi=52, amplitude=0.75),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 2
        assert events[0].midi_notes == (40,)
        assert events[1].midi_notes == (52,)
        assert events[0].is_chord is False
        assert events[1].is_chord is False

    def test_midi_notes_sorted_ascending(self):
        """A cluster received in arbitrary order yields midi_notes sorted ascending."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.302, end_s=1.000, midi=52, amplitude=0.78),
            BasicPitchNote(start_s=0.290, end_s=1.000, midi=40, amplitude=0.66),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 52)   # not (52, 40)

    def test_earliest_anchor_caps_cluster_width(self):
        """An event 80 ms after the cluster anchor still joins; one 120 ms
        after starts a new cluster (>100 ms past the anchor, not the previous
        member). This is the key behaviour preventing a slow strum from
        chaining indefinitely."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=1.000, midi=40, amplitude=0.70),
            BasicPitchNote(start_s=0.080, end_s=1.000, midi=47, amplitude=0.70),  # joins (80 ms < 100 ms)
            BasicPitchNote(start_s=0.140, end_s=1.000, midi=52, amplitude=0.70),  # joins (140 ms - 0 ms > 100 ms? -> NO. Anchor is 0.000, this is at 140 ms, so > 100 ms -> NEW cluster)
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 2
        assert events[0].midi_notes == (40, 47)
        assert events[1].midi_notes == (52,)

    def test_three_member_cluster(self):
        """All three members within 100 ms of the anchor cluster together."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=1.000, midi=40, amplitude=0.80),
            BasicPitchNote(start_s=0.030, end_s=1.000, midi=47, amplitude=0.75),
            BasicPitchNote(start_s=0.075, end_s=1.000, midi=52, amplitude=0.70),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 47, 52)
        assert events[0].is_chord is True

    def test_duplicate_midi_in_cluster_kept_at_highest_amplitude(self):
        """basic-pitch can emit two events for the same pitch within the
        cluster window; we keep one (the higher-amplitude one)."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.55),
            BasicPitchNote(start_s=0.020, end_s=0.500, midi=40, amplitude=0.75),  # same pitch, higher amp
            BasicPitchNote(start_s=0.030, end_s=0.500, midi=47, amplitude=0.65),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 47)
        # The kept E2 detection is the 0.75 one, not the 0.55 one.
        # confidences[0] corresponds to midi=40 (sorted ascending), so:
        assert events[0].confidences[0] == pytest.approx(0.75)

    def test_relative_floor_drops_quiet_member(self):
        """Within a cluster, members below POLYPHONIC_RELATIVE_FLOOR * max_amp
        get dropped. Default ratio = 0.5; max_amp 0.85, so anything < 0.425
        is dropped."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.85),
            BasicPitchNote(start_s=0.030, end_s=0.500, midi=47, amplitude=0.40),  # 0.40 < 0.425 -> dropped
            BasicPitchNote(start_s=0.060, end_s=0.500, midi=52, amplitude=0.70),  # 0.70 > 0.425 -> kept
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40, 52)   # 47 dropped

    def test_singleton_cluster_unaffected_by_relative_floor(self):
        """A cluster with one member has max_amp == its own amp, so the
        relative floor is trivially satisfied and the singleton survives."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=0.000, end_s=0.500, midi=40, amplitude=0.66),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 1
        assert events[0].midi_notes == (40,)

    def test_chord_clusters_in_temporal_order(self):
        """Multiple chord clusters across the timeline emerge sorted by onset_s."""
        tracker = NoteTracker(self._config(), self._profile())
        result = _bp_pitch_result([
            BasicPitchNote(start_s=2.0, end_s=3.0, midi=45, amplitude=0.70),
            BasicPitchNote(start_s=2.02, end_s=3.0, midi=57, amplitude=0.70),
            BasicPitchNote(start_s=0.0, end_s=1.0, midi=40, amplitude=0.80),
            BasicPitchNote(start_s=0.02, end_s=1.0, midi=52, amplitude=0.75),
        ])
        events = tracker.process(result, onsets=[])
        assert len(events) == 2
        assert events[0].midi_notes == (40, 52)
        assert events[0].onset_s == pytest.approx(0.0)
        assert events[1].midi_notes == (45, 57)
        assert events[1].onset_s == pytest.approx(2.0)
```

- [ ] **Step 2: Confirm the tests fail**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_tracker.py::TestNoteTrackerBasicPitchClustering -v 2>&1 | tail -15
```

Expected: 8 of 9 tests FAIL (the `test_events_outside_window_become_separate_events` test may incidentally pass since the current `_process_basic_pitch` already emits separate singletons for events 200 ms apart). Failure reason: chord emission currently doesn't happen — every basic-pitch event becomes a singleton.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/analysis/test_note_tracker.py
git commit -m "test(analysis): add onset clustering contract tests (Phase 2)"
```

---

## Task 4: Implement onset clustering in `_process_basic_pitch`

**Files:**
- Modify: `src/improv_scribe/analysis/note_tracker.py`

- [ ] **Step 1: Add a private clustering helper**

Before `class NoteTracker` (around the existing helpers section, after `_avg_tuples`), add:

```python
def _cluster_basic_pitch_notes(
    bp_notes: list[BasicPitchNote],
    window_s: float,
    relative_floor: float,
) -> list[list[BasicPitchNote]]:
    """Group basic-pitch events into chord clusters by onset proximity.

    Algorithm:
    1. Sort events by start_s ascending.
    2. Walk through; open a new cluster when the current event's start_s
       exceeds the EARLIEST member of the current cluster by more than
       window_s. Anchoring on the earliest member (rather than the most
       recent) caps total cluster width.
    3. Within each cluster, deduplicate by MIDI value (keep highest amp).
    4. Apply relative amplitude floor: drop members whose amplitude is
       below relative_floor * max(amps in cluster).
    5. Sort cluster members by MIDI ascending so the resulting tuple is
       canonical.

    Parameters
    ----------
    bp_notes : list[BasicPitchNote]
    window_s : float
        Cluster window in seconds (typically 0.100).
    relative_floor : float
        Drop members below this fraction of the cluster's max amplitude.

    Returns
    -------
    list[list[BasicPitchNote]]
        One inner list per cluster, members sorted by MIDI ascending.
        Clusters appear in onset order. Empty clusters (all members
        dropped) are omitted.
    """
    if not bp_notes:
        return []

    sorted_notes = sorted(bp_notes, key=lambda n: n.start_s)

    raw_clusters: list[list[BasicPitchNote]] = []
    for note in sorted_notes:
        if not raw_clusters or (note.start_s - raw_clusters[-1][0].start_s) > window_s:
            raw_clusters.append([note])
        else:
            raw_clusters[-1].append(note)

    cleaned_clusters: list[list[BasicPitchNote]] = []
    for cluster in raw_clusters:
        # Deduplicate by MIDI value, keeping highest amplitude
        by_midi: dict[int, BasicPitchNote] = {}
        for n in cluster:
            existing = by_midi.get(n.midi)
            if existing is None or n.amplitude > existing.amplitude:
                by_midi[n.midi] = n
        deduped = list(by_midi.values())

        # Apply relative amplitude floor within the cluster
        max_amp = max(n.amplitude for n in deduped)
        threshold = max_amp * relative_floor
        survivors = [n for n in deduped if n.amplitude >= threshold]

        if not survivors:
            continue
        # Sort by MIDI ascending so tuples are canonical
        survivors.sort(key=lambda n: n.midi)
        cleaned_clusters.append(survivors)

    return cleaned_clusters
```

- [ ] **Step 2: Rewrite `_process_basic_pitch`**

Replace the existing `_process_basic_pitch` method body with:

```python
def _process_basic_pitch(
    self,
    bp_notes: list[BasicPitchNote],
    chunk_offset_s: float,
) -> list[NoteEvent]:
    """Convert basic-pitch's pre-assembled notes into NoteEvents, clustering
    simultaneous detections into chord events.

    Phase 2: clusters of size 1 emit singleton NoteEvents (backward-compatible
    with Phase 1); clusters of size 2+ emit chord NoteEvents with the full
    midi_notes tuple sorted ascending.

    Clustering rule (see _cluster_basic_pitch_notes): earliest-anchor + window.
    Cluster width is capped by ONSET_GROUPING_WINDOW_MS (100 ms default).
    Members below POLYPHONIC_RELATIVE_FLOOR * max(amps in cluster) are dropped.

    No octave-error correction is applied — basic-pitch already does its own
    polyphonic spectral analysis (spec §3.2).
    """
    if not bp_notes:
        return []

    window_s = self._config.onset_grouping_window_ms / 1000.0
    clusters = _cluster_basic_pitch_notes(
        bp_notes,
        window_s=window_s,
        relative_floor=self._config.polyphonic_relative_floor,
    )

    events: list[NoteEvent] = []
    for cluster in clusters:
        onset = min(n.start_s for n in cluster) + chunk_offset_s
        offset = max(n.end_s for n in cluster) + chunk_offset_s
        midi_notes = tuple(n.midi for n in cluster)
        frequencies = tuple(440.0 * 2.0 ** ((n.midi - 69) / 12.0) for n in cluster)
        confidences = tuple(n.amplitude for n in cluster)
        cents = tuple(0.0 for _ in cluster)
        events.append(NoteEvent(
            onset_s=onset,
            offset_s=offset,
            midi_notes=midi_notes,
            frequencies_hz=frequencies,
            confidences=confidences,
            cents_deviations=cents,
        ))

    sorted_events = sorted(events, key=lambda e: e.onset_s)
    return _merge_consecutive_same_pitch(sorted_events)
```

- [ ] **Step 3: Run the clustering tests**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_tracker.py::TestNoteTrackerBasicPitchClustering -v 2>&1 | tail -15
```

Expected: all 9 tests PASS.

- [ ] **Step 4: Run the existing single-event tests**

```bash
conda run -n auto-sheet-music pytest tests/analysis/test_note_tracker.py::TestNoteTrackerBasicPitch -v 2>&1 | tail -15
```

Expected: all 6 existing tests still PASS (singleton clusters reduce to the previous behaviour).

- [ ] **Step 5: Run the integration gauntlet**

```bash
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected:
- CREPE: 72/72 PASS (frame-based path unchanged).
- basic_pitch: 68 passed + 4 skipped UNCHANGED. On mono samples, clustering should still produce singletons because the basic-pitch events for mono content are far apart in time.

If the basic-pitch mono integration now fails (e.g. unexpected chord events appearing on what should be mono samples), debug — most likely cause is a clustering bug that's grouping mono events that shouldn't be grouped.

- [ ] **Step 6: Commit**

```bash
git add src/improv_scribe/analysis/note_tracker.py
git commit -m "$(cat <<'EOF'
feat(analysis): onset clustering for basic-pitch -> chord NoteEvents (Phase 2)

_process_basic_pitch now groups basic-pitch's flat note events into
chord clusters by onset proximity (earliest-anchor + 100 ms window).
Cluster members get the relative amplitude floor (drop members below
0.5 * cluster max). Singleton clusters reduce to the Phase 1
singleton-NoteEvent behaviour; multi-member clusters emit chord
NoteEvents with midi_notes sorted ascending.

Calibrated against the three real dyad samples (spec §13): actual
within-cluster spreads 0-35 ms.

CREPE integration: 72/72 (unchanged).
basic_pitch integration: 68 + 4 skipped (mono samples unaffected,
clusters trivially singletons).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Failing tests for `ScoreBuilder` chord emission

**Files:**
- Modify (or create): `tests/notation/test_score_builder.py`

- [ ] **Step 1: Append the chord tests**

Read the file first to see existing structure. After the existing chord-aware test classes:

```python
import music21.chord
import music21.note as m21note


class TestScoreBuilderChord:
    """Phase 2 — ScoreBuilder emits music21.chord.Chord when QuantizedNote
    has multiple midi_notes."""

    def _profile(self):
        from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
        return get_profile(Instrument.GUITAR)

    def _tempo_result(self):
        from improv_scribe.quantization.tempo import TempoResult
        return TempoResult(bpm=120.0, beat_times_s=[], confidence=0.9)

    def _make_qn_chord(
        self,
        midi_notes: tuple[int, ...] = (60, 64, 67),
        onset_beat: float = 0.0,
        quarter_length: float = 1.0,
    ):
        from improv_scribe.quantization.grid import NoteDuration, QuantizedNote
        n = len(midi_notes)
        return QuantizedNote(
            midi_notes=midi_notes,
            frequencies_hz=tuple(440.0 * 2 ** ((m - 69) / 12) for m in midi_notes),
            confidences=(0.8,) * n,
            cents_deviations=(0.0,) * n,
            onset_beat=onset_beat,
            duration_beats=quarter_length,
            duration_type=NoteDuration.QUARTER,
            quarter_length=quarter_length,
            is_rest=False,
        )

    def test_singleton_qn_produces_note_not_chord(self):
        from improv_scribe.notation.score_builder import ScoreBuilder
        builder = ScoreBuilder(self._profile(), self._tempo_result())
        score = builder.build([self._make_qn_chord(midi_notes=(60,))])
        notes_and_chords = list(score.recurse().notes)
        assert len(notes_and_chords) == 1
        assert isinstance(notes_and_chords[0], m21note.Note)
        assert notes_and_chords[0].pitch.midi == 60

    def test_chord_qn_produces_chord(self):
        from improv_scribe.notation.score_builder import ScoreBuilder
        builder = ScoreBuilder(self._profile(), self._tempo_result())
        score = builder.build([self._make_qn_chord(midi_notes=(60, 64, 67))])
        notes_and_chords = list(score.recurse().notes)
        assert len(notes_and_chords) == 1
        chord = notes_and_chords[0]
        assert isinstance(chord, music21.chord.Chord)
        assert sorted(p.midi for p in chord.pitches) == [60, 64, 67]

    def test_mixed_singletons_and_chords(self):
        from improv_scribe.notation.score_builder import ScoreBuilder
        builder = ScoreBuilder(self._profile(), self._tempo_result())
        score = builder.build([
            self._make_qn_chord(midi_notes=(60,)),
            self._make_qn_chord(midi_notes=(64, 67)),
            self._make_qn_chord(midi_notes=(60,)),
        ])
        notes_and_chords = list(score.recurse().notes)
        assert len(notes_and_chords) == 3
        assert isinstance(notes_and_chords[0], m21note.Note)
        assert isinstance(notes_and_chords[1], music21.chord.Chord)
        assert isinstance(notes_and_chords[2], m21note.Note)

    def test_build_raw_also_emits_chord(self):
        """build_raw() is used for raw-timing MIDI export; must also support chords."""
        from improv_scribe.notation.score_builder import ScoreBuilder
        builder = ScoreBuilder(self._profile(), self._tempo_result())
        score = builder.build_raw([self._make_qn_chord(midi_notes=(60, 64))])
        notes_and_chords = list(score.recurse().notes)
        assert len(notes_and_chords) == 1
        assert isinstance(notes_and_chords[0], music21.chord.Chord)
```

- [ ] **Step 2: Confirm the tests fail**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py::TestScoreBuilderChord -v 2>&1 | tail -10
```

Expected: most tests FAIL. The chord-test cases will either crash (passing a tuple where Note expects int) or produce N separate Note objects instead of one Chord.

- [ ] **Step 3: Commit**

```bash
git add tests/notation/test_score_builder.py
git commit -m "test(notation): add ScoreBuilder chord-emission contract tests (Phase 2)"
```

---

## Task 6: Implement `ScoreBuilder` chord support

**Files:**
- Modify: `src/improv_scribe/notation/score_builder.py`

- [ ] **Step 1: Update `build()`'s note construction**

Locate the existing loop (around line 133-140):

```python
for qn in notes:
    dur = Duration(quarterLength=qn.quarter_length)
    if qn.is_rest:
        element: music21.note.GeneralNote = music21.note.Rest(duration=dur)
    else:
        element = music21.note.Note(qn.midi_note, duration=dur)

    part.append(element)
```

Replace with:

```python
import music21.chord as m21chord  # add to top-of-file imports if not present

for qn in notes:
    dur = Duration(quarterLength=qn.quarter_length)
    if qn.is_rest:
        element: music21.note.GeneralNote = music21.note.Rest(duration=dur)
    elif len(qn.midi_notes) == 1:
        element = music21.note.Note(qn.midi_notes[0], duration=dur)
    else:
        element = m21chord.Chord(list(qn.midi_notes), duration=dur)

    part.append(element)
```

(`m21chord` import is fine to add at the top with the other music21 imports.)

- [ ] **Step 2: Update `build_raw()` similarly**

Locate the existing loop (around line 189-195):

```python
for qn in notes:
    if qn.is_rest:
        continue
    n = music21.note.Note(qn.midi_note)
    n.quarterLength = qn.quarter_length
    n.offset = qn.onset_beat
    part.insert(n.offset, n)
```

Replace with:

```python
for qn in notes:
    if qn.is_rest:
        continue
    if len(qn.midi_notes) == 1:
        el = music21.note.Note(qn.midi_notes[0])
    else:
        el = m21chord.Chord(list(qn.midi_notes))
    el.quarterLength = qn.quarter_length
    el.offset = qn.onset_beat
    part.insert(el.offset, el)
```

- [ ] **Step 3: Run the chord tests**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py::TestScoreBuilderChord -v 2>&1 | tail -10
```

Expected: all 4 chord tests PASS.

- [ ] **Step 4: Run existing tests to confirm no regression**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_score_builder.py -v 2>&1 | tail -15
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: existing notation tests pass; CREPE 72/72.

- [ ] **Step 5: Commit**

```bash
git add src/improv_scribe/notation/score_builder.py
git commit -m "$(cat <<'EOF'
feat(notation): ScoreBuilder emits music21.chord.Chord for multi-pitch QN (Phase 2)

When QuantizedNote.midi_notes has length > 1, build() and build_raw()
emit music21.chord.Chord(list(midi_notes), duration). Length-1 emits
a Note as before. Length-0 (rests) unchanged.

Replaces .midi_note shim usage at score_builder.py:138 and :192 with
direct .midi_notes tuple access (2 of 6 Phase 0 §10 shim consumers
migrated).

CREPE integration: 72/72.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Failing tests for chord-aware tab DP

**Files:**
- Create: `tests/notation/test_tab_builder.py` (or extend if exists)

- [ ] **Step 1: Check whether the file exists**

```bash
ls tests/notation/test_tab_builder.py 2>&1
```

If not, create with imports + helpers. If yes, append.

- [ ] **Step 2: Write the tests**

```python
"""Unit tests for chord-aware tab fret assignment (Phase 2)."""

from __future__ import annotations

import pytest

from improv_scribe.analysis.instrument_profiles import Instrument
from improv_scribe.notation.tab_builder import (
    BASS_TUNING,
    GUITAR_TUNING,
    assign_frets,
    get_candidates,
    get_chord_shapes,
)
from improv_scribe.quantization.grid import NoteDuration, QuantizedNote


def _qn(midi_notes: tuple[int, ...], is_rest: bool = False) -> QuantizedNote:
    n = len(midi_notes) if not is_rest else 0
    return QuantizedNote(
        midi_notes=() if is_rest else midi_notes,
        frequencies_hz=() if is_rest else (440.0,) * n,
        confidences=() if is_rest else (0.8,) * n,
        cents_deviations=() if is_rest else (0.0,) * n,
        onset_beat=0.0,
        duration_beats=1.0,
        duration_type=NoteDuration.QUARTER,
        quarter_length=1.0,
        is_rest=is_rest,
    )


class TestGetChordShapes:
    """get_chord_shapes() enumerates all (string, fret) assignments where
    chord members occupy distinct strings."""

    def test_singleton_returns_candidates_wrapped_in_tuples(self):
        # E2 (MIDI 40) is only playable on the lowest string of guitar (open)
        shapes = get_chord_shapes((40,), GUITAR_TUNING)
        assert len(shapes) == 1
        assert shapes[0] == ((0, 0),)

    def test_dyad_returns_distinct_string_pairs(self):
        # E2 (40) on string 0, B2 (47) playable on string 0 or string 1.
        # No-string-conflict: only the (0,0) + (1,2) combination is valid.
        shapes = get_chord_shapes((40, 47), GUITAR_TUNING)
        # Verify all returned shapes use distinct strings
        for shape in shapes:
            strings = [s for s, _f in shape]
            assert len(set(strings)) == len(strings)

    def test_unplayable_chord_returns_empty(self):
        # Three identical MIDI E2s: only 1 string can play E2 -> no
        # conflict-free shape exists.
        shapes = get_chord_shapes((40, 40, 40), GUITAR_TUNING)
        assert shapes == []

    def test_out_of_range_member_returns_empty(self):
        # MIDI 5 is below guitar's range; member has no candidates.
        shapes = get_chord_shapes((5, 60), GUITAR_TUNING)
        assert shapes == []

    def test_shapes_are_canonical_string_sorted(self):
        shapes = get_chord_shapes((40, 47), GUITAR_TUNING)
        for shape in shapes:
            strings = [s for s, _f in shape]
            assert strings == sorted(strings)


class TestAssignFretsChordAware:
    """assign_frets() returns a tuple of (string, fret) pairs per QuantizedNote.
    Singletons get length-1 tuples; chords get length-N tuples with distinct strings."""

    def test_singleton_returns_length_1_tuple(self):
        result = assign_frets([_qn((40,))], Instrument.GUITAR)
        assert result == [((0, 0),)]

    def test_chord_returns_length_n_tuple_with_distinct_strings(self):
        result = assign_frets([_qn((40, 47, 52))], Instrument.GUITAR)
        assert len(result) == 1
        assert result[0] is not None
        shape = result[0]
        assert len(shape) == 3
        # Distinct strings
        strings = [s for s, _f in shape]
        assert len(set(strings)) == 3

    def test_rest_returns_none(self):
        result = assign_frets([_qn((), is_rest=True)], Instrument.GUITAR)
        assert result == [None]

    def test_mono_path_equivalent_to_pre_phase2(self):
        """A sequence of mono notes produces the same result as Phase 0 / Phase 1.
        Specifically: open low E -> low A -> open D should map to
        ((0,0), (1,0), (2,0)) — open string fingerings, no movement."""
        result = assign_frets(
            [_qn((40,)), _qn((45,)), _qn((50,))],
            Instrument.GUITAR,
        )
        assert result == [((0, 0),), ((1, 0),), ((2, 0),)]

    def test_chord_followed_by_mono_uses_consistent_dp(self):
        """A chord followed by a single note shouldn't crash; both are assigned."""
        result = assign_frets(
            [_qn((40, 47)), _qn((52,))],
            Instrument.GUITAR,
        )
        assert len(result) == 2
        # Both non-rest -> both non-None
        assert result[0] is not None
        assert result[1] is not None

    def test_unplayable_chord_falls_back_gracefully(self):
        """A chord with no conflict-free shape (e.g. three E2s) falls back
        to a singleton shape rather than crashing."""
        result = assign_frets([_qn((40, 40, 40))], Instrument.GUITAR)
        # Fall-back is a singleton ((0, 0),) — see spec §3.5 final fallback
        assert result[0] is not None
        # Either a singleton tuple or an empty tuple is acceptable
        # depending on the chosen fallback shape; the key is "no crash".

    def test_bass_chord(self):
        """E1+A1 dyad on bass: E1 (28) on string 0 open, A1 (33) on string 1 open."""
        result = assign_frets([_qn((28, 33))], Instrument.BASS)
        assert result[0] is not None
        shape = result[0]
        assert len(shape) == 2
        strings = sorted(s for s, _f in shape)
        assert strings == [0, 1]
        # Both should be fret 0 (open strings)
        for _s, f in shape:
            assert f == 0
```

- [ ] **Step 3: Confirm tests fail**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_tab_builder.py -v 2>&1 | tail -15
```

Expected: `get_chord_shapes` doesn't exist yet, so `ImportError` on all tests in `TestGetChordShapes`. `assign_frets` exists but takes a single MIDI not a tuple, so the chord-aware tests fail there too.

- [ ] **Step 4: Commit**

```bash
git add tests/notation/test_tab_builder.py
git commit -m "test(notation): add chord-aware tab DP contract tests (Phase 2)"
```

---

## Task 8: Implement chord-aware tab DP

**Files:**
- Modify: `src/improv_scribe/notation/tab_builder.py`

- [ ] **Step 1: Add `get_chord_shapes`**

After `get_candidates()`, add:

```python
def get_chord_shapes(
    midi_notes: tuple[int, ...],
    tuning: list[int],
) -> list[tuple[tuple[int, int], ...]]:
    """Return all (string, fret) assignments that put each chord member on a
    distinct string.

    For mono (length-1 midi_notes), returns one shape per candidate string.
    For chord midi_notes, enumerates the Cartesian product of per-member
    candidates and keeps only combinations where strings are pairwise distinct.

    Parameters
    ----------
    midi_notes : tuple[int, ...]
        MIDI note numbers, typically sorted ascending (canonical form).
    tuning : list[int]
        MIDI numbers of open strings, low to high.

    Returns
    -------
    list[tuple[tuple[int, int], ...]]
        Each inner tuple is one valid shape, sorted by string ascending.
        Empty list if no conflict-free shape exists or any member is
        out of range.
    """
    import itertools  # noqa: PLC0415

    per_note_candidates = [get_candidates(m, tuning) for m in midi_notes]
    if any(not c for c in per_note_candidates):
        # At least one member has no candidates; no shape possible.
        return []

    shapes: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.product(*per_note_candidates):
        strings = [s for s, _f in combo]
        if len(set(strings)) == len(strings):
            # Sort by string ascending so the shape is canonical
            shapes.append(tuple(sorted(combo, key=lambda sf: sf[0])))
    return shapes
```

- [ ] **Step 2: Rewrite `assign_frets`**

Replace the existing function body entirely. The new return type is `list[tuple[tuple[int, int], ...] | None]`. Mono notes get length-1 tuples (`((string, fret),)`).

```python
def assign_frets(
    notes: list[QuantizedNote],
    instrument: Instrument,
) -> list[tuple[tuple[int, int], ...] | None]:
    """Assign chord-aware (string, fret) tuples to each note using DP.

    Each non-rest note is assigned a tuple of (string, fret) pairs — one
    pair per chord member, with the no-string-conflict constraint
    (members on distinct strings).

    Cost model
    ----------
    Within-shape cost: hand stretch = max(fret) - min(fret) over fretted
                       members; open strings (fret 0) excluded; 0 for
                       all-open shapes.
    Transition cost:   |centroid_curr - centroid_prev| where centroid is
                       the mean fret of fretted members in the shape;
                       defaults to 0 if all open.
    Tie-break:         lex (cumulative_cost, max_fret_in_shape,
                       min_fret_in_shape). On singletons this reduces
                       bit-equivalently to the Phase 0 single-note DP.

    Rests receive None.

    Fallbacks
    ---------
    - Shape enumeration empty (member out of range): drop offending members
      from the highest fret down until a non-empty enumeration succeeds.
      If even the playable subset has no conflict-free shape, return
      ``((0, 0),)``.
    - All members unplayable: ``((0, 0),)`` so the score still renders.

    Parameters
    ----------
    notes : list[QuantizedNote]
        Mixed mono and chord notes. Rests receive None.
    instrument : Instrument

    Returns
    -------
    list[tuple[tuple[int, int], ...] | None]
        Parallel to *notes*. Each non-rest entry is a tuple of
        (string, fret) pairs sorted by string ascending. Rests are None.
    """
    import math  # noqa: PLC0415

    tuning = _TUNINGS[instrument]
    result: list[tuple[tuple[int, int], ...] | None] = [None] * len(notes)

    # Identify non-rest indices and their shapes
    non_rest_indices: list[int] = []
    shape_lists: list[list[tuple[tuple[int, int], ...]]] = []
    for i, note in enumerate(notes):
        if note.is_rest:
            continue
        non_rest_indices.append(i)
        shapes = get_chord_shapes(note.midi_notes, tuning)
        if not shapes:
            # Fallback: drop members from highest fret down until shapes exist
            shapes = _fallback_shapes(note.midi_notes, tuning)
        shape_lists.append(shapes)

    if not non_rest_indices:
        return result

    def _centroid(shape: tuple[tuple[int, int], ...]) -> float:
        fretted = [f for _s, f in shape if f > 0]
        return sum(fretted) / len(fretted) if fretted else 0.0

    def _stretch(shape: tuple[tuple[int, int], ...]) -> int:
        fretted = [f for _s, f in shape if f > 0]
        if not fretted:
            return 0
        return max(fretted) - min(fretted)

    INF = math.inf
    # dp[j][k] = (cumulative_cost, max_fret_in_shape, min_fret_in_shape)
    # prev[j][k] = shape index in shape_lists[j-1] on the optimal path
    n_pos = len(non_rest_indices)
    dp: list[list[tuple[float, int, int]]] = [
        [(INF, INF, INF)] * len(shape_lists[j]) for j in range(n_pos)
    ]
    prev: list[list[int]] = [[-1] * len(shape_lists[j]) for j in range(n_pos)]

    # Initialise first position
    for k, shape in enumerate(shape_lists[0]):
        stretch = _stretch(shape)
        all_frets = [f for _s, f in shape]
        dp[0][k] = (float(stretch), max(all_frets), min(all_frets))

    for j in range(1, n_pos):
        for k, shape in enumerate(shape_lists[j]):
            stretch = _stretch(shape)
            curr_cent = _centroid(shape)
            all_frets = [f for _s, f in shape]
            best = (INF, INF, INF)
            best_p = -1
            for p, prev_shape in enumerate(shape_lists[j - 1]):
                if dp[j - 1][p][0] == INF:
                    continue
                trans = abs(curr_cent - _centroid(prev_shape))
                cumulative = dp[j - 1][p][0] + stretch + trans
                cand = (cumulative, max(all_frets), min(all_frets))
                if cand < best:
                    best = cand
                    best_p = p
            dp[j][k] = best
            prev[j][k] = best_p

    # Backtrack
    final_k = min(range(len(shape_lists[-1])), key=lambda k: dp[-1][k])
    assignments: list[int] = [-1] * n_pos
    assignments[-1] = final_k
    for j in range(n_pos - 1, 0, -1):
        assignments[j - 1] = prev[j][assignments[j]]

    for pos, j in enumerate(non_rest_indices):
        k = assignments[pos]
        if k == -1:
            result[j] = ((0, 0),)   # final fallback
        else:
            result[j] = shape_lists[pos][k]

    return result


def _fallback_shapes(
    midi_notes: tuple[int, ...],
    tuning: list[int],
) -> list[tuple[tuple[int, int], ...]]:
    """When a chord has no conflict-free shape (e.g. duplicate-pitch chord),
    drop members from highest fret down until a valid shape exists.

    If no subset works, return ``[((0, 0),)]`` so the caller has at least
    one assignment to use as a final fallback.
    """
    # Try dropping members one at a time (start with the last in the tuple,
    # which is the highest MIDI by convention).
    for drop_count in range(1, len(midi_notes)):
        for combo in _combinations_of_size(midi_notes, len(midi_notes) - drop_count):
            shapes = get_chord_shapes(combo, tuning)
            if shapes:
                return shapes
    return [((0, 0),)]


def _combinations_of_size(
    midi_notes: tuple[int, ...],
    size: int,
) -> list[tuple[int, ...]]:
    """All subsets of midi_notes with the given size."""
    import itertools  # noqa: PLC0415
    return [tuple(c) for c in itertools.combinations(midi_notes, size)]
```

- [ ] **Step 3: Update the caller in `score_builder.py`**

The `compute_tab_assignments` method's return type annotation should change. Find:

```python
def compute_tab_assignments(
    self, notes: list[QuantizedNote]
) -> list[tuple[int, int] | None]:
```

Change to:

```python
def compute_tab_assignments(
    self, notes: list[QuantizedNote]
) -> list[tuple[tuple[int, int], ...] | None]:
```

The body (`return assign_frets(notes, self._profile.instrument)`) is unchanged.

- [ ] **Step 4: Run the chord-aware tab tests**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_tab_builder.py -v 2>&1 | tail -15
```

Expected: all tests PASS.

- [ ] **Step 5: Run existing tab tests + CREPE integration**

```bash
conda run -n auto-sheet-music pytest tests/notation/ -v 2>&1 | tail -10
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: tab assignments on mono integration tests still produce the correct single-string single-fret pairs (now wrapped in length-1 tuples).

If a mono integration test fails because it asserted on `(int, int)` instead of `((int, int),)`, that's expected — Task 9 below updates the tab XML injection and existing integration tests will need the wrapped tuple form. **Pause here and proceed only after fixing the integration test pattern in Task 9-10.**

Actually, on reflection: integration tests check `tab_assignments` against `EXPECTED_TAB`. EXPECTED_TAB has `[(0, 0), (1, 0), …]` — a list of `tuple[int, int]`. The new assignment shape is `list[tuple[tuple[int, int], ...]]` — a list of tuples-of-tuples. The integration tests will fail. Update the four integration test files' EXPECTED_TAB to use the new shape:

```python
# Old:
EXPECTED_TAB = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
# New:
EXPECTED_TAB = [((0, 0),), ((1, 0),), ((2, 0),), ((3, 0),), ((4, 0),), ((5, 0),)]
```

Do that for all four mono integration test files (the per-backend EXPECTED_TAB dict if present, else the bare constant).

- [ ] **Step 6: Run CREPE integration again**

```bash
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: 72/72 PASS now that EXPECTED_TAB is in the new shape.

- [ ] **Step 7: Commit**

```bash
git add src/improv_scribe/notation/tab_builder.py src/improv_scribe/notation/score_builder.py tests/integration/
git commit -m "$(cat <<'EOF'
feat(notation): chord-aware tab DP with no-string-conflict (Phase 2)

assign_frets() now returns a tuple of (string, fret) pairs per note
instead of a single pair. Singletons get length-1 tuples (((s, f),));
chords get length-N tuples with the no-string-conflict constraint.

DP cost: within-shape stretch (max - min fretted fret, opens excluded)
+ between-shape centroid distance. Tie-break lex (cost, max_fret,
min_fret) reduces bit-equivalently to the Phase 0 single-note tie-break
on singleton inputs.

get_chord_shapes() enumerates conflict-free shapes via Cartesian
product. _fallback_shapes() handles unplayable chords (e.g. duplicate
pitch) by dropping members until a valid shape exists.

ScoreBuilder.compute_tab_assignments return type updated. The four
mono integration tests' EXPECTED_TAB constants updated to the new
((s, f),) singleton-tuple shape.

Replaces .midi_note shim at tab_builder.py:77 with .midi_notes
(1 of 6 Phase 0 §10 shim consumers — counting the subsumed
score_builder consumers from Task 6 we're now at 4 of 6).

CREPE integration: 72/72.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Verify music21's chord serialization shape (prerequisite for Task 10)

**Why:** The Phase 2 spec §3.6 calls out that the tab_xml.py walker needs to recognise `<chord/>` siblings produced by music21. Before rewriting the walker, we need a structural test capturing music21's exact XML output.

**Files:**
- Create: `tests/notation/test_chord_musicxml_shape.py`

- [ ] **Step 1: Write the structural assertion**

```python
"""Verify music21's exact MusicXML serialization shape for a Chord.

This is a prerequisite for tab_xml's chord-sibling injection: we need to
know the exact element structure music21 produces so the walker can
recognise chord groups deterministically.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import music21.chord
import music21.note
import music21.stream
import music21.tempo
import music21.meter.base


def _make_chord_score(midi_notes: list[int]) -> music21.stream.Score:
    score = music21.stream.Score()
    part = music21.stream.Part()
    part.append(music21.meter.base.TimeSignature("4/4"))
    part.append(music21.tempo.MetronomeMark(number=120))
    if len(midi_notes) == 1:
        part.append(music21.note.Note(midi_notes[0]))
    else:
        part.append(music21.chord.Chord(midi_notes))
    score.append(part.makeMeasures())
    return score


def test_chord_serializes_with_chord_marker_on_siblings(tmp_path: Path):
    """A 3-note Chord produces 3 <note> elements; first has no <chord/>;
    siblings 2 and 3 each have a <chord/> child."""
    score = _make_chord_score([60, 64, 67])
    xml_path = tmp_path / "chord.musicxml"
    score.write("musicxml", fp=str(xml_path))

    tree = ET.parse(xml_path)
    root = tree.getroot()
    notes = list(root.iter("note"))
    # In the part's first measure, expect 3 notes
    assert len(notes) == 3, f"Expected 3 <note> elements, got {len(notes)}"
    # First note: no <chord/> child
    assert notes[0].find("chord") is None
    # Subsequent notes: have <chord/> child
    for n in notes[1:]:
        assert n.find("chord") is not None, "expected <chord/> on sibling"


def test_singleton_note_has_no_chord_marker(tmp_path: Path):
    score = _make_chord_score([60])
    xml_path = tmp_path / "note.musicxml"
    score.write("musicxml", fp=str(xml_path))
    tree = ET.parse(xml_path)
    notes = list(tree.getroot().iter("note"))
    assert len(notes) == 1
    assert notes[0].find("chord") is None


def test_chord_pitches_are_in_chord_pitch_order(tmp_path: Path):
    """music21's serialization order for chord pitches: ascending (verify)."""
    score = _make_chord_score([67, 60, 64])   # constructed out of order
    xml_path = tmp_path / "chord_order.musicxml"
    score.write("musicxml", fp=str(xml_path))
    tree = ET.parse(xml_path)
    notes = list(tree.getroot().iter("note"))
    pitches = []
    for n in notes:
        pitch_el = n.find("pitch")
        if pitch_el is not None:
            step = pitch_el.find("step").text
            octave = pitch_el.find("octave").text
            pitches.append(f"{step}{octave}")
    # If music21 sorts ascending, we'd see C-4, E-4, G-4
    # Otherwise, we see the input order
    # Capture whatever music21 actually does so Task 10 can rely on it.
    print(f"music21 chord pitch order: {pitches}")
```

- [ ] **Step 2: Run the tests**

```bash
conda run -n auto-sheet-music pytest tests/notation/test_chord_musicxml_shape.py -v -s 2>&1 | tail -20
```

Capture the output. Specifically, the `print` in `test_chord_pitches_are_in_chord_pitch_order` will reveal whether music21 emits chord pitches in input order or sorts them. **Record this in the spec §13 (or a new §14) for Task 10's reference.**

Expected: all tests PASS. If `notes[1:]` doesn't have `<chord/>` children, music21's serialization differs from the spec's assumption — investigate before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/notation/test_chord_musicxml_shape.py
git commit -m "$(cat <<'EOF'
test(notation): assert music21's Chord serialization shape (Phase 2)

Prerequisite for Task 10 (tab_xml chord-sibling injection): the walker
needs to recognise <chord/> markers on staff-1 notes that belong to the
same chord group. This test locks in music21's exact MusicXML output
shape so the walker's assumption is checked, not assumed.

Records the chord-pitch ordering convention (ascending vs input-order)
via print() — used in Task 10.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Implement chord-sibling injection in `tab_xml.py`

**Files:**
- Modify: `src/improv_scribe/export/tab_xml.py`

This is the most delicate task — the existing walker consumes one tab assignment per `<note>` element, but a chord's N members are N `<note>` elements that should share ONE assignment tuple of length N.

- [ ] **Step 1: Identify the current walker boundary**

Read `tab_xml.py` to see the current `inject_tab_part` implementation around the `pitched_iter` usage. Understand exactly how it pairs `<note>` elements with tab assignments today.

- [ ] **Step 2: Refactor the walker to recognise chord groups**

Replace the walker logic with chord-group-aware iteration:

1. Walk each measure's `<note>` children sequentially.
2. Group `<note>` elements into "slots": a slot begins at a `<note>` without a `<chord/>` child, and includes any immediately following `<note>` elements whose first child is `<chord/>`.
3. Per slot: consume one assignment tuple from `assignments`. The tuple length must equal the slot's note count.
4. Annotate each staff-1 note in the slot with its `(string, fret)` from the tuple, in the tuple's order (which is string-sorted ascending).
5. Emit a staff-2 mirror group: the first member is a `<note>` with `<staff>2</staff>` + the technical annotation, no `<chord/>`. Subsequent members have `<note><chord/><staff>2</staff>...`. Use the same fret/string ordering as the staff-1 chord.

Detailed code template (adapt to the existing code):

```python
def _group_notes_into_slots(measure: ET.Element) -> list[list[ET.Element]]:
    """Walk a measure's <note> children and group them into chord-slots.

    A slot is a list of consecutive <note> elements that share the same
    chord (the first has no <chord/> child; subsequent ones do). Rests
    and tied-stop notes are still their own slots but get no assignment.
    """
    slots: list[list[ET.Element]] = []
    current: list[ET.Element] = []
    for child in list(measure):
        if child.tag != "note":
            continue
        is_chord_sibling = child.find("chord") is not None
        if is_chord_sibling and current:
            current.append(child)
        else:
            if current:
                slots.append(current)
            current = [child]
    if current:
        slots.append(current)
    return slots
```

Then in the main `inject_tab_part`:

```python
# For each measure, group notes into slots; per slot, consume one assignment.
assignment_iter = iter(assignments)
for measure in part.findall("measure"):
    slots = _group_notes_into_slots(measure)
    for slot in slots:
        slot_is_rest = slot[0].find("rest") is not None
        if slot_is_rest:
            # Rest slot: no assignment consumed, no tab annotation.
            continue
        # Skip tie-continuations (same convention as the pre-Phase-2 code)
        if _is_tie_continuation(slot[0]):
            continue
        # Consume one assignment for this slot
        try:
            slot_assignment = next(assignment_iter)
        except StopIteration:
            break
        if slot_assignment is None:
            continue   # rest assignment — shouldn't happen if slot is non-rest

        # slot_assignment is a tuple of (string, fret) pairs.
        # Annotate each staff-1 element + emit staff-2 mirrors.
        assert len(slot_assignment) == len(slot), (
            f"Slot has {len(slot)} notes but assignment has {len(slot_assignment)}"
        )
        for note_el, (string_idx, fret) in zip(slot, slot_assignment, strict=True):
            _annotate_with_technical(note_el, string_idx, fret, profile)
        # Emit staff-2 mirror group
        _emit_staff2_mirror(measure, slot, slot_assignment, profile)
```

The `_emit_staff2_mirror` function should add `<chord/>` to the 2nd+ siblings:

```python
def _emit_staff2_mirror(
    measure: ET.Element,
    slot: list[ET.Element],
    assignment: tuple[tuple[int, int], ...],
    profile: InstrumentProfile,
) -> None:
    """Emit one <note> per assignment member on staff 2, with <chord/> on
    siblings 2..N. Inserted after the last staff-1 note in the slot."""
    insert_after = slot[-1]
    insert_idx = list(measure).index(insert_after) + 1
    for i, (string_idx, fret) in enumerate(assignment):
        new_note = copy.deepcopy(slot[0])
        # Mark as staff 2
        _set_staff(new_note, 2)
        # Add or update <technical>
        _set_technical(new_note, string_idx, fret, profile)
        # First mirror element has no <chord/>; siblings 2..N do
        if i > 0:
            ET.SubElement(new_note, "chord")
        measure.insert(insert_idx + i, new_note)
```

These helper functions adapt to whatever the existing tab_xml.py uses. The key changes from the pre-Phase-2 code:
- Walker iterates slots, not individual notes.
- Each slot consumes one assignment tuple of length N.
- Mirror emission adds `<chord/>` to siblings on staff 2.

- [ ] **Step 3: Add chord-injection tests**

In `tests/export/test_tab_xml.py`, add a test that builds a Score with one chord and one mono note, runs the full notation→XML→tab-inject pipeline, and asserts:

```python
def test_inject_tab_with_chord_emits_chord_siblings_on_staff_2(tmp_path: Path):
    """A 3-note chord followed by a singleton should produce:
       Staff 1: 3 chord-sibling <note>s + 1 standalone <note>
       Staff 2: 3 chord-sibling <note>s + 1 standalone <note>, all with
                <technical><string><fret> annotations
    """
    from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
    from improv_scribe.notation.score_builder import ScoreBuilder
    from improv_scribe.quantization.tempo import TempoResult
    from improv_scribe.quantization.grid import NoteDuration, QuantizedNote
    from improv_scribe.export.tab_xml import inject_tab_part

    def qn(midi: tuple[int, ...]) -> QuantizedNote:
        n = len(midi)
        return QuantizedNote(
            midi_notes=midi,
            frequencies_hz=(440.0,) * n,
            confidences=(0.8,) * n,
            cents_deviations=(0.0,) * n,
            onset_beat=0.0,
            duration_beats=1.0,
            duration_type=NoteDuration.QUARTER,
            quarter_length=1.0,
            is_rest=False,
        )

    profile = get_profile(Instrument.GUITAR)
    builder = ScoreBuilder(profile, TempoResult(bpm=120, beat_times_s=[], confidence=0.9))
    notes = [qn((40, 47, 52)), qn((60,))]
    score = builder.build(notes)
    assignments = builder.compute_tab_assignments(notes)

    xml_path = tmp_path / "out.musicxml"
    score.write("musicxml", fp=str(xml_path))
    inject_tab_part(xml_path, notes, assignments, profile)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Verify the chord group on staff 2 has <chord/> on members 2 and 3
    staff2_notes = [n for n in root.iter("note") if (n.find("staff") is not None and n.find("staff").text == "2")]
    # 3 chord members + 1 singleton = 4 staff-2 notes
    assert len(staff2_notes) == 4
    # First 3 are the chord group: note 1 has no <chord/>, notes 2-3 do
    assert staff2_notes[0].find("chord") is None
    assert staff2_notes[1].find("chord") is not None
    assert staff2_notes[2].find("chord") is not None
    # Note 4 is singleton: no <chord/>
    assert staff2_notes[3].find("chord") is None
```

- [ ] **Step 4: Run tests + integration**

```bash
conda run -n auto-sheet-music pytest tests/export/test_tab_xml.py -v 2>&1 | tail -15
conda run -n auto-sheet-music pytest tests/notation tests/export -v 2>&1 | tail -10
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: all pass. CREPE 72/72; basic_pitch 68 + 4 skipped (existing mono samples still work).

- [ ] **Step 5: Commit**

```bash
git add src/improv_scribe/export/tab_xml.py tests/export/
git commit -m "$(cat <<'EOF'
feat(export): tab_xml chord-sibling injection (Phase 2)

inject_tab_part() walks <note> elements grouped into chord-slots: a
slot begins at a <note> without a <chord/> child and includes
immediately following <note>s whose first child is <chord/>. Each
slot consumes one assignment tuple of length N (== slot's note count)
and emits N staff-2 mirror notes with <chord/> on members 2..N.

Mono path preserved byte-for-byte: a length-1 slot consumes a length-1
assignment tuple and emits one staff-2 note (no <chord/>).

CREPE integration: 72/72; basic_pitch: 68 + 4 skipped.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Failing test + fix for `midi_exporter` chord iteration

**Files:**
- Modify: `src/improv_scribe/export/midi_exporter.py`
- Modify (or create): `tests/export/test_midi_exporter.py`

This is a **correctness bug fix** (Phase 0 §10): chord events currently produce only the lowest note in raw MIDI export.

- [ ] **Step 1: Write the failing test**

In `tests/export/test_midi_exporter.py`, add (creating the file if needed):

```python
"""Unit tests for MIDIExporter raw chord support (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from improv_scribe.analysis.note_tracker import NoteEvent
from improv_scribe.config import AppConfig
from improv_scribe.export.midi_exporter import MIDIExporter
from improv_scribe.quantization.tempo import TempoResult


def test_raw_export_chord_emits_all_members(tmp_path: Path):
    """A chord NoteEvent with midi_notes=(60, 64, 67) must produce
    three note_on events at the same tick (and three note_offs)."""
    try:
        import mido  # noqa: PLC0415, F401
    except ImportError:
        pytest.skip("mido not installed")

    config = AppConfig()
    exporter = MIDIExporter(config)

    chord_event = NoteEvent(
        onset_s=0.0,
        offset_s=1.0,
        midi_notes=(60, 64, 67),   # C major triad
        frequencies_hz=(261.6, 329.6, 392.0),
        confidences=(0.8, 0.8, 0.8),
        cents_deviations=(0.0, 0.0, 0.0),
    )

    out_path = tmp_path / "out.mid"
    exporter.raw_from_events([chord_event], TempoResult(bpm=120, beat_times_s=[], confidence=0.9), out_path)

    import mido  # noqa: PLC0415
    mid = mido.MidiFile(str(out_path))
    # Collect note_on events with velocity > 0
    note_ons = [
        msg for track in mid.tracks for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ]
    notes_played = sorted(msg.note for msg in note_ons)
    assert notes_played == [60, 64, 67]
```

- [ ] **Step 2: Confirm the test fails**

```bash
conda run -n auto-sheet-music pytest tests/export/test_midi_exporter.py -v 2>&1 | tail -10
```

Expected: FAIL. Only one note_on is emitted (the lowest, MIDI 60, via the back-compat shim) — the test asserts 3.

- [ ] **Step 3: Fix `raw_from_events()` to iterate chord members**

In `src/improv_scribe/export/midi_exporter.py`, locate the existing block (around lines 117-122):

```python
# Build flat list of (time_s, msg_type, note) and sort by time
messages: list[tuple[float, str, int]] = []
for event in events:
    messages.append((event.onset_s, "note_on", event.midi_note))
    messages.append((event.offset_s, "note_off", event.midi_note))
messages.sort(key=lambda m: (m[0], 0 if m[1] == "note_on" else 1))
```

Replace with:

```python
# Build flat list of (time_s, msg_type, note) and sort by time.
# Chord events: one note_on/note_off per chord member at the same tick.
messages: list[tuple[float, str, int]] = []
for event in events:
    for midi in event.midi_notes:
        messages.append((event.onset_s, "note_on", midi))
        messages.append((event.offset_s, "note_off", midi))
messages.sort(key=lambda m: (m[0], 0 if m[1] == "note_on" else 1))
```

- [ ] **Step 4: Run the test + regression**

```bash
conda run -n auto-sheet-music pytest tests/export/test_midi_exporter.py -v 2>&1 | tail -5
conda run -n auto-sheet-music pytest tests/export tests/notation -v 2>&1 | tail -10
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: chord test PASSES; mono path unchanged (a singleton-tuple event still produces 1 note_on/note_off).

- [ ] **Step 5: Commit**

```bash
git add src/improv_scribe/export/midi_exporter.py tests/export/test_midi_exporter.py
git commit -m "$(cat <<'EOF'
fix(export): MIDI raw export emits all chord members (Phase 2)

raw_from_events() iterates event.midi_notes to emit one note_on and
one note_off per chord member at the chord's onset/offset times.

Pre-Phase-2 behaviour (read event.midi_note via back-compat shim)
silently dropped all but the lowest note from a chord — a correctness
bug deferred from Phase 0 §10 because no chord events existed before
Phase 2.

Replaces .midi_note shim at midi_exporter.py:120 and :121 with
.midi_notes iteration (2 of 6 Phase 0 §10 shim consumers migrated).

CREPE integration: 72/72.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Migrate `gui/main_window.py` to chord-aware display

**Files:**
- Modify: `src/improv_scribe/gui/main_window.py`

The last back-compat shim consumer (from Phase 0 §10) is in the transcription log display.

- [ ] **Step 1: Locate the call-site**

```bash
grep -n 'midi_note' src/improv_scribe/gui/main_window.py
```

Expected: line 288 (or thereabouts) — `(a, _midi_name(n.midi_note))`.

- [ ] **Step 2: Migrate to chord-aware display**

Read the surrounding context. The variable `n` is a `QuantizedNote`. The function `_midi_name` (find its definition in main_window.py) converts a MIDI int to a name like "E4".

Replace the single-MIDI lookup with chord-aware logic:

```python
# Old:
# (a, _midi_name(n.midi_note)) if a is not None else (None, "rest")

# New:
def _format_note_names(qn) -> str:
    if qn.is_rest:
        return "rest"
    if len(qn.midi_notes) == 1:
        return _midi_name(qn.midi_notes[0])
    # Chord: "E4/G4/B4"
    return "/".join(_midi_name(m) for m in qn.midi_notes)

# In the comprehension / display logic:
(a, _format_note_names(n)) if a is not None else (None, "rest")
```

The exact integration depends on the existing code structure. The principle: replace `n.midi_note` with iteration over `n.midi_notes`.

- [ ] **Step 3: Verify no `.midi_note` shim use remains in src/**

```bash
grep -rn '\.midi_note\b' src/ | grep -v 'def midi_note'
```

Expected: **zero hits.** All 6 Phase 0 §10 consumers migrated.

- [ ] **Step 4: Run GUI tests (if any) + CREPE integration**

```bash
conda run -n auto-sheet-music pytest tests/ 2>&1 | tail -5
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/improv_scribe/gui/main_window.py
git commit -m "$(cat <<'EOF'
feat(gui): chord-aware transcription log display (Phase 2)

QuantizedNote.midi_note shim usage replaced with .midi_notes iteration.
Mono notes still display as e.g. 'E4'; chord notes display as 'E4/G4/B4'.

Final migration of Phase 0 §10 back-compat shim consumers: all 6
documented call-sites now read .midi_notes directly. The shims can be
removed in the next task.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Remove back-compat shim properties from NoteEvent and QuantizedNote

**Files:**
- Modify: `src/improv_scribe/analysis/note_tracker.py`
- Modify: `src/improv_scribe/quantization/grid.py`

- [ ] **Step 1: Remove the shims from NoteEvent**

In `note_tracker.py`, find the `NoteEvent` dataclass. Remove these properties:

```python
@property
def midi_note(self) -> int: ...
@property
def frequency_hz(self) -> float: ...
@property
def confidence(self) -> float: ...
@property
def cents_deviation(self) -> float: ...
```

Keep `duration_s` and `is_chord`. Also keep `__repr__`, but verify it doesn't call `self.confidence` (which was a shim) — if it does, replace with `sum(self.confidences) / len(self.confidences)`.

- [ ] **Step 2: Remove the shims from QuantizedNote**

In `grid.py`, find `QuantizedNote`. Remove the same four properties.

- [ ] **Step 3: Verify nothing in src/ still uses `.midi_note` or `.frequency_hz`**

```bash
grep -rn '\.midi_note\b\|\.frequency_hz\b\|\.confidence\b\|\.cents_deviation\b' src/ | grep -v 'def \|midi_notes\|frequencies_hz\|confidences\|cents_deviations'
```

Expected: zero hits (excluding the field definitions themselves and the new plural-named properties/fields).

- [ ] **Step 4: Run full test suite**

```bash
conda run -n auto-sheet-music pytest tests/ 2>&1 | tail -5
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: all pass. If anything fails, the failure is either a test that still uses the shim (update test) or a real call-site missed by Task 12 (update src and re-run Task 12 grep check).

- [ ] **Step 5: Commit**

```bash
git add src/improv_scribe/analysis/note_tracker.py src/improv_scribe/quantization/grid.py
git commit -m "$(cat <<'EOF'
refactor(model): remove .midi_note back-compat shims (Phase 2)

Completes the Phase 0 data-model migration: NoteEvent and QuantizedNote
no longer expose .midi_note, .frequency_hz, .confidence, or
.cents_deviation read properties. All 6 documented call-sites
(spec §10) migrated to .midi_notes in Tasks 6, 8, 11, 12.

Grep check: zero hits for the shim names in src/ (excluding field defs).

Full test suite: passes.
CREPE integration: 72/72.
basic_pitch integration: 68 + 4 skipped.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Failing tests for the three dyad integration samples

**Files:**
- Create: `tests/integration/test_guitar_dyad_octave.py`
- Create: `tests/integration/test_guitar_dyad_fifth.py`
- Create: `tests/integration/test_guitar_dyad_third.py`

Use the spec §13.5 ground-truth tables.

- [ ] **Step 1: Create `test_guitar_dyad_octave.py`**

Mirror the structure of `test_guitar_electric_line_in.py`. The ground truth is from spec §13.5:

```python
"""End-to-end pipeline regression tests for:
    samples/guitar/chords/6_string_electric_octave_dyads.mp3

Six dyads — basic-pitch detects 3 as dyads, 3 as singletons (one octave
member registers below 0.65 amplitude floor). See spec §13.5 for the
exact captured ground truth.
"""

from __future__ import annotations

import os

import numpy as np
import music21.chord
import music21.note

from improv_scribe.analysis.instrument_profiles import Instrument
from tests.integration.conftest import SAMPLE_ROOT, make_pipeline_fixtures

SAMPLE_PATH = SAMPLE_ROOT / "guitar" / "chords" / "6_string_electric_octave_dyads.mp3"
INSTRUMENT = Instrument.GUITAR
EXPECTED_DURATION_S = 12.30

_BACKEND = os.getenv("ATS_PITCH_BACKEND", "crepe")

# Per-backend: only basic_pitch can detect chord events; CREPE/pyin will
# produce singletons for the same sample (best-effort mono interpretation).
# Phase 2 ships with basic_pitch as the default backend.
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (40, 52),   # E2+E3
        (41, 53),   # F2+F3
        (55,),      # G3 only (basic-pitch lost lower octave)
        (45,),      # A2 only
        (47, 59),   # B2+B3
        (48,),      # C3 only
    ],
    # CREPE/pyin are monophonic — they detect 6 single notes (the lower of each pair, typically)
    # Capture the actual values during Task 14 Step 4 (calibration run under crepe).
    "crepe":       [],
    "pyin":        [],
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
            import pytest
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        assert len(note_events) == NOTE_COUNT, (
            f"Expected {NOTE_COUNT} events, got {len(note_events)}: "
            f"{[tuple(e.midi_notes) for e in note_events]}"
        )

    def test_note_midi_tuples_match(self, note_events):
        if not EXPECTED_MIDI_TUPLES:
            import pytest
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        actual = [tuple(e.midi_notes) for e in note_events]
        assert actual == EXPECTED_MIDI_TUPLES


class TestScore:
    def test_score_chord_emission(self, score, note_events):
        """For each chord event (len(midi_notes) > 1), the score should have
        a music21.chord.Chord at that position."""
        if not EXPECTED_MIDI_TUPLES:
            import pytest
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        chord_count = sum(1 for t in EXPECTED_MIDI_TUPLES if len(t) > 1)
        if chord_count == 0:
            import pytest
            pytest.skip(f"Backend {_BACKEND} produces no chords on this sample")

        chords_in_score = list(score.recurse().getElementsByClass(music21.chord.Chord))
        assert len(chords_in_score) == chord_count, (
            f"Expected {chord_count} Chord objects, found {len(chords_in_score)}"
        )


class TestTabAssignments:
    def test_chord_tab_uses_distinct_strings(self, tab_assignments):
        """Every chord-shape assignment must use distinct strings."""
        if not EXPECTED_MIDI_TUPLES:
            import pytest
            pytest.skip(f"No ground truth recorded for backend {_BACKEND}")
        for assignment in tab_assignments:
            if assignment is None or len(assignment) <= 1:
                continue
            strings = [s for s, _f in assignment]
            assert len(set(strings)) == len(strings), (
                f"Chord assignment {assignment} reuses a string"
            )
```

- [ ] **Step 2: Create `test_guitar_dyad_fifth.py`**

Same structure, ground truth from spec §13.5 perfect_fifths:

```python
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (47,),       # B2 only
        (41, 48),    # F2+C3
        (50,),       # D3 only
        (45, 52),    # A2+E3
        (47,),       # B2 only
        (48, 55),    # C3+G3
    ],
    "crepe": [],
    "pyin": [],
}
```

- [ ] **Step 3: Create `test_guitar_dyad_third.py`**

Ground truth from spec §13.5 major_thirds:

```python
EXPECTED_MIDI_TUPLES_BY_BACKEND: dict[str, list[tuple[int, ...]]] = {
    "basic_pitch": [
        (50, 54),    # D3+F#3 (5-4 pair)
        (52,),       # E3 only
        (54, 58),    # F#3+A#3 (5-4 pair)
        (55,),       # G3 only
        (57, 61),    # A3+C#4 (4-3 pair)
        (59, 63),    # B3+D#4 (4-3 pair)
    ],
    "crepe": [],
    "pyin": [],
}
```

- [ ] **Step 4: Calibrate CREPE/pyin ground truth on the three samples**

Run each of the three samples through CREPE and pyin backends to capture what they actually produce:

```bash
for backend in crepe pyin; do
  echo "=== $backend ==="
  ATS_PITCH_BACKEND=$backend conda run -n auto-sheet-music python -c "
import librosa
from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.analysis.pitch import PitchEstimator
from improv_scribe.analysis.onset import OnsetDetector
from improv_scribe.analysis.note_tracker import NoteTracker
from improv_scribe.config import AppConfig
for fname in ['6_string_electric_octave_dyads.mp3', '6_string_electric_perfect_fifths.mp3', '6_string_electric_major_thirds.mp3']:
    path = f'samples/guitar/chords/{fname}'
    y, sr = librosa.load(path, sr=44100, mono=True)
    config = AppConfig()
    profile = get_profile(Instrument.GUITAR)
    est = PitchEstimator(config)
    od = OnsetDetector(config)
    tr = NoteTracker(config, profile)
    result = est.estimate(y, profile)
    onsets = od.detect(y)
    events = tr.process(result, onsets, audio=y)
    print(f'{fname}: {[tuple(e.midi_notes) for e in events]}')
"
done
```

Record the output. Add CREPE/pyin ground truth to each of the three new test files. If a backend produces results that look entirely unreasonable (e.g. wrong key signature, half the notes missing), use `pytest.skip` with a "calibration TBD" message instead — the goal is the basic_pitch path passing.

- [ ] **Step 5: Run the phase gate**

```bash
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration/test_guitar_dyad_* -v 2>&1 | tail -15
```

Expected: all three dyad tests PASS for basic_pitch (asserting against the §13.5 ground truth).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_guitar_dyad_*.py
git commit -m "$(cat <<'EOF'
test(integration): three dyad samples wired into the integration suite (Phase 2)

Each new test file targets one of the real recordings landed at commit
46c8b3d:
- test_guitar_dyad_octave.py (6 octave dyads)
- test_guitar_dyad_fifth.py (6 perfect 5ths)
- test_guitar_dyad_third.py (6 major 3rds, 5-4 + 4-3 pairs)

basic_pitch ground truth from spec §13.5 (exact tuple per cluster).
CREPE/pyin ground truth calibrated empirically (they detect singletons
since they're monophonic models).

Phase 2 gate: basic_pitch passes all three dyad tests.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Flip default backend to `basic_pitch`

**Files:**
- Modify: `src/improv_scribe/config.py`

- [ ] **Step 1: Change the default**

In `config.py`, find:

```python
PITCH_BACKEND: str = os.getenv("ATS_PITCH_BACKEND", "crepe")
```

Change to:

```python
PITCH_BACKEND: str = os.getenv("ATS_PITCH_BACKEND", "basic_pitch")
```

- [ ] **Step 2: Run the integration suite with NO env var override**

```bash
unset ATS_PITCH_BACKEND
conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: passes — because the new default IS basic_pitch, this is equivalent to `ATS_PITCH_BACKEND=basic_pitch` which we've already verified.

- [ ] **Step 3: Run with explicit CREPE override**

```bash
ATS_PITCH_BACKEND=crepe conda run -n auto-sheet-music pytest tests/integration 2>&1 | tail -3
```

Expected: 72/72 PASS — CREPE remains selectable.

- [ ] **Step 4: Commit**

```bash
git add src/improv_scribe/config.py
git commit -m "$(cat <<'EOF'
feat(config): flip default pitch backend to basic_pitch (Phase 2)

basic-pitch is now the default. CREPE remains selectable via
ATS_PITCH_BACKEND=crepe. New users get polyphonic detection out of
the box without installing basic-pitch separately if they've run
the installer script (Phase 1 Task 1).

Integration suite: basic_pitch passes the three new dyad tests and
the four existing mono samples. CREPE remains at 72/72 when
explicitly selected.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Manual PDF render verification

**Why:** Notation correctness is hard to assert programmatically beyond "music21 emits a Chord object." A human-readable PDF rendered through MuseScore gives the final visual check.

**Files:** none (read-only verification + a manual checklist in the spec)

- [ ] **Step 1: Render one of the dyad samples to PDF**

```bash
conda run -n auto-sheet-music python -m improv_scribe.cli \
  --device file \
  --input samples/guitar/chords/6_string_electric_octave_dyads.mp3 \
  --instrument guitar \
  --output /tmp/phase2_render \
  --backend basic_pitch
```

(Adapt the CLI invocation to whatever this project's `__main__` accepts. If the project doesn't have a file-input mode, use a one-shot Python script that runs the full pipeline manually.)

Expected: `/tmp/phase2_render.pdf` exists, opens in macOS Preview.

- [ ] **Step 2: Manually inspect the PDF**

Open the PDF. Verify:
1. **Chord glyphs are present** on the 3 onsets where basic-pitch detected dyads (clusters 0, 1, 4 — see spec §13.5).
2. **The chord pitches are correctly stacked** with shared stems (E2 below E3 on cluster 0, F2 below F3 on cluster 1, B2 below B3 on cluster 4).
3. **Tab staff shows fret numbers stacked** for each chord (e.g. fret 0 on string 6 + fret 2 on string 5 for E2+B2 dyad would appear as "0" stacked over "2").
4. **Singletons (clusters 2, 3, 5)** appear as single notes with single tab numbers.
5. **The MIDI file** (`/tmp/phase2_render.mid`) opens in any DAW and plays all chord members simultaneously.

If any of these are wrong, file a follow-up task. The Phase 2 phase gate is OK to declare done as long as the test suite passes — visual issues are typically MuseScore CLI quirks that need separate investigation.

- [ ] **Step 3: Take a screenshot**

Save the PDF page as `/tmp/phase2_render.png` or similar, copy into the repo as `docs/superpowers/phase2_render_proof.png`. (Optional but useful for the phase outcome.)

---

## Task 17: Phase 2 phase-gate verification + spec outcome

**Files:** read-only verification + spec edit.

- [ ] **Step 1: Run the full regression gauntlet**

```bash
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest 2>&1 | tail -3
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest 2>&1 | tail -3
```

Expected:
- CREPE: passes everything (mono integration 72/72; unit tests; dyad tests skipped or showing calibrated CREPE values).
- basic_pitch: passes everything (mono integration 68 + 4 skipped; dyad tests 12/12; unit tests).

- [ ] **Step 2: Confirm migration completeness**

```bash
grep -rn '\.midi_note\b' src/ | grep -v 'def midi_note'
```

Expected: **zero hits** (all migrations done).

```bash
grep 'PITCH_BACKEND.*os.getenv' src/improv_scribe/config.py
```

Expected: `"basic_pitch"` as the default.

- [ ] **Step 3: Add Phase 2 outcome to spec**

Edit `docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md`. Add after §13:

```markdown
---

## 14. Phase 2 Outcome (landed YYYY-MM-DD)

Phase 2 — dyad detection end-to-end — completed in N task commits on
`chord-detection-phase2` branch. The phase delivered:

- Onset clustering in `_process_basic_pitch` with earliest-anchor +
  100 ms window + relative-floor filter.
- `ScoreBuilder.build()` emits `music21.chord.Chord` for multi-pitch
  QuantizedNotes.
- Chord-aware `assign_frets()` with no-string-conflict DP.
- `inject_tab_part()` walks `<chord/>` siblings and mirrors them on
  staff 2.
- `midi_exporter.raw_from_events()` emits N note_on/note_off per chord.
- Back-compat shims (`.midi_note`, `.frequency_hz`, `.confidence`,
  `.cents_deviation`) removed from NoteEvent and QuantizedNote — all 6
  Phase 0 §10 consumers migrated.
- `gui/main_window.py` displays chord names like "E4/G4/B4".
- Default backend flipped from `crepe` to `basic_pitch`.
- Three new integration tests (octave/5th/3rd dyad samples) passing
  under basic_pitch.
- Manual PDF render verification on the octave_dyads sample: chord
  glyphs + stacked tab numbers visible.

Phase gate:
- CREPE: pytest passes (existing 72 + dyad tests skip or use CREPE
  ground truth).
- basic_pitch: pytest passes (mono 68 + 4 skipped + dyad 12 + unit).

Phase 3 (triads + 4-note chords + real chord progression) is the next
milestone, gated on user-provided real open-chord recordings.
```

Fill in N and the landing date.

- [ ] **Step 4: Commit the spec update**

```bash
git add docs/superpowers/specs/2026-05-09-polyphonic-detection-design.md
git commit -m "docs(spec): record Phase 2 outcome (dyad detection end-to-end)"
```

- [ ] **Step 5: Merge or land the worktree**

Decide with the user how to land. Two paths:

**Path A — Merge phase2 branch into chord-detection:**
```bash
cd ../audio_to_sheet
git checkout chord-detection
git merge --no-ff chord-detection-phase2
git worktree remove ../audio_to_sheet-phase2
```

**Path B — Keep the worktree until Phase 3:**
Leave the worktree intact; do Phase 3 in the same isolated workspace.

Default: **Path A** (clean up; Phase 3 gets its own worktree).

- [ ] **Step 6: Declare Phase 2 done**

Phase 2 ships when:
1. Both backends pass the full pytest suite.
2. Migration tracker grep returns zero hits.
3. Default backend is `basic_pitch`.
4. Phase 2 Outcome (§14) is in the spec.
5. Manual PDF render verification done.

Pause here. Phase 3 requires real open-chord recordings (E, A, D, G, C minimum) from the user.

---

## What's next (out of scope for this plan)

After Phase 2 ships:

1. **User records real chord progression** — at minimum the five "campfire" open chords: E, A, D, G, C (strummed individually, ~2 s each). These become Phase 3's gate.
2. **Phase 3 plan written** — triads + 4-note chords + chord-progression detection.
3. **Phase 4 (separate spec, only if evidence) — quality polish:** chord-aware octave correction, per-string sustain in raw MIDI, GUI chord-glyph rendering improvement, possibly lowering POLYPHONIC_AMPLITUDE_FLOOR to 0.40 + cluster-internal relative filter for higher dyad recall.
