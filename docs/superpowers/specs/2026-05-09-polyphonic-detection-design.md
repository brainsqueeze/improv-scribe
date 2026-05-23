# Polyphonic Note Detection — Design Spec

**Status:** Draft (pending user review)
**Date:** 2026-05-09
**Owner:** Dave Hollander
**Target branch:** `chord-detection`

---

## 1. Goal

Extend the `improv_scribe` pipeline to detect, quantize, notate, and tab-render
**two or more simultaneously played notes** on guitar and bass, with zero
regression to the existing single-note pipeline.

The detection target is general polyphony — any combination of pitches
playable on the instrument, whether or not they form an established chord
shape. Chord-shape templates are explicitly **not** the basis of detection.

### Non-goals

- Real-time / streaming polyphonic transcription (the existing chunk-based
  batch pipeline is preserved).
- Voice separation across staves (single-staff chord notation only).
- Strum-direction or articulation inference.
- Rhythm voice splitting (e.g. melody + accompaniment on a single staff).
- Per-string sustain in raw MIDI export — flagged as a Phase 4 polish,
  not part of MVP.

---

## 2. Scope phasing

The work is staged so each phase is shippable on its own:

| Phase | Deliverable | Stop condition |
|---|---|---|
| **0** | `NoteEvent` / `QuantizedNote` data-model migration to `midi_notes: tuple[int, ...]`. Includes the chord-aware refactor of `_merge_consecutive_same_pitch` (tuple equality, element-wise averaging) — this is **not** a no-op. | All four existing integration tests pass under `crepe` and `pyin` backends. Mono semantics unchanged because monophonic events are singleton tuples and the new merge logic collapses identically. |
| **1** | `basic-pitch` registered as a third pitch backend. Mono-only validation: each `BasicPitchNote` becomes a singleton `NoteEvent`. **No octave correction applied to basic-pitch outputs.** Per-backend `EXPECTED_MIDI` ground truth in integration tests, calibrated empirically. | All four existing monophonic samples pass under `ATS_PITCH_BACKEND=basic_pitch` with backend-specific ground truth and tolerances. Phase 1 prerequisite (basic-pitch API prototype) complete. |
| **2** | Dyad detection end-to-end (analysis → quantization → notation → tab → **MIDI**). Default backend flips to `basic_pitch`. **`midi_exporter.py` updated to iterate chord members for note_on/note_off** — exporting only the lowest note of a chord is a correctness bug, not a Phase 4 polish. Cluster-window calibration against real strum recordings. | Synthetic dyad fixtures pass exact `midi_notes` assertions; user-provided real-dyad recordings pass too; tab assignments use distinct strings; PDF renders chord glyphs; MIDI plays *all* chord members. |
| **3** | Triads and 4+ chords. User-provided real chord progression is the phase gate (not optional). | Synthetic triad/4-note fixtures pass; user-provided real open-chord progression (E, A, D, G, C minimum) produces correct PDF + MIDI + tab. |
| **4** *(separate spec, not in this MVP)* | Per-string sustain in raw MIDI export, chord-aware octave correction if needed, GUI chord-glyph rendering. | — |

This spec covers **Phases 0–3.** Phase 4 will be scoped from real-world feedback
after Phase 3 lands.

---

## 3. Architectural changes

### 3.1 Data model — single source-of-truth change

`NoteEvent` ([analysis/note_tracker.py](../../../src/improv_scribe/analysis/note_tracker.py))
becomes the canonical chord-capable event:

```python
@dataclass
class NoteEvent:
    onset_s: float
    offset_s: float
    midi_notes: tuple[int, ...]            # was: midi_note: int — sorted ascending
    frequencies_hz: tuple[float, ...]      # was: frequency_hz: float
    confidences: tuple[float, ...]         # per-note confidence
    cents_deviations: tuple[float, ...]    # per-note tuning deviation

    @property
    def is_chord(self) -> bool:
        return len(self.midi_notes) > 1

    @property
    def midi_note(self) -> int:
        """Back-compat shim: returns lowest pitch. Removed at end of Phase 2."""
        return self.midi_notes[0]

    @property
    def confidence(self) -> float:
        """Back-compat shim: mean confidence across chord members."""
        return sum(self.confidences) / len(self.confidences)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.offset_s - self.onset_s)
```

`QuantizedNote` ([quantization/grid.py](../../../src/improv_scribe/quantization/grid.py))
mirrors the same change. Rests stay singleton with `midi_notes=()`.

**Invariants:**

- `midi_notes` is sorted ascending (lowest first). Matches MusicXML chord
  serialisation and makes equality comparison stable.
- `len(midi_notes) == len(frequencies_hz) == len(confidences) == len(cents_deviations)`.
- A monophonic detection emits `(midi,)` — singleton tuple, never a bare int.
- `midi_notes=()` only in rests.
- Back-compat `midi_note` and `confidence` properties exist transiently. After
  Phase 2 the migration completion check is:
  `grep -rn '\.midi_note\b' src/ | grep -v 'def midi_note'` returns no hits.

### 3.2 Analysis — `basic-pitch` as a third backend

#### New backend in `analysis/pitch.py`

```python
@dataclass
class BasicPitchNote:
    """A polyphonic note event from basic-pitch's predict() output."""
    start_s: float
    end_s: float
    midi: int
    amplitude: float    # basic-pitch's mean frame activation, in [0, 1]

@dataclass
class PitchResult:
    frames: list[PitchFrame]                   # populated by pyin/crepe; empty for basic-pitch
    bp_notes: list[BasicPitchNote] | None      # populated by basic-pitch; None for pyin/crepe
    sample_rate: int
    hop_length: int
```

**basic-pitch's actual API** (confirmed by the Phase 1 prerequisite probe —
see §11). `basic_pitch.inference.predict()` returns a 3-tuple
`(model_output_dict, midi_data, note_events)` where `note_events` is
`list[tuple[float, float, int, float, list[int] | None]]` —
`(start_s, end_s, midi, amplitude, pitch_bends)` positionally. We unpack
positionally and ignore `pitch_bends` (we already encode microtonal deviation
in `cents_deviations`, but we set those to 0.0 for basic-pitch outputs since
they emit integer MIDI directly). The model is loaded lazily on first
`predict()` call (~0.32 s measured cold-call on Apple Silicon — ONNX runtime
only, no TF/TFLite). We keep a module-level cache of the loaded model so
subsequent chunks re-use it.

**`predict()` requires a file path; numpy arrays are not accepted.**
`_BasicPitchBackend.estimate()` therefore writes the input audio buffer to a
temporary WAV file (via `soundfile.write`), calls `predict(str(tmp_path))`,
then deletes the temp file in a `finally` block (exception-safe). The wrapper
unpacks `note_events`, filters by `InstrumentProfile.midi_min/midi_max`,
`POLYPHONIC_AMPLITUDE_FLOOR` (final calibrated value **0.65** — see §6 and
§12), and `MIN_NOTE_DURATION_S` (50 ms), and returns a `PitchResult` with
`bp_notes` populated and `frames=[]`.

`PITCH_BACKENDS = {"pyin", "crepe", "basic_pitch"}`. Default flips at the end
of Phase 2.

#### Onset clustering in `NoteTracker`

basic-pitch returns a flat list of note events; multi-string plucks have
onsets staggered by tens of milliseconds and are not pre-grouped into chords.
The existing librosa-onset-driven assembly is bypassed when the pitch backend
is `basic_pitch`.

```python
class NoteTracker:
    def process(
        self,
        pitch_result: PitchResult,
        onsets: list[Onset],            # used only when bp_notes is None
        chunk_offset_s: float = 0.0,
        audio: np.ndarray | None = None,
    ) -> list[NoteEvent]:
        if pitch_result.bp_notes is not None:
            return self._process_basic_pitch(pitch_result.bp_notes, chunk_offset_s)
        return self._process_frame_based(pitch_result, onsets, chunk_offset_s, audio)
```

`_process_basic_pitch()` algorithm:

1. Sort `bp_notes` by `start_s`.
2. Walk the list, opening a new cluster when the current note's `start_s`
   exceeds **the earliest member of the cluster currently being built**
   by more than `ONSET_GROUPING_WINDOW_S` (default **100 ms**, env-overridable
   as `ATS_ONSET_GROUPING_WINDOW_MS`). Comparing against the earliest member
   (rather than the most recent member) caps total cluster width — a
   slow strum across 4 strings spanning 80 ms with 20 ms inter-onset gaps
   does not run away into one giant cluster.

   **Default rationale.** A vigorous guitar strum across all six strings
   typically spans 80–120 ms; an arpeggiated chord intentionally spans
   longer. 100 ms is the conservative midpoint that captures most strums
   without merging deliberate arpeggios. Phase 2 includes a calibration
   step against real strum recordings (see §4) to confirm or adjust this
   default before the basic-pitch backend becomes the project default.
3. Within each cluster:
   - Drop members whose `amplitude < max(amps) * POLYPHONIC_RELATIVE_FLOOR`
     (default **0.5**, env-overridable). This is **relative**, not absolute,
     because chord members vary widely in amplitude (the bass note is often
     loudest by 6–10 dB).
   - Drop members whose MIDI is outside `InstrumentProfile.midi_min/midi_max`.
   - Deduplicate by MIDI value (keep the one with the highest amplitude).
4. Sort surviving members by MIDI ascending. Emit one `NoteEvent` with:
   - `onset_s` = earliest member's `start_s + chunk_offset_s`
   - `offset_s` = latest member's `end_s + chunk_offset_s`
   - `midi_notes` = sorted tuple of surviving MIDI values
   - `frequencies_hz` = MIDI-derived 440-tuned frequency for each (no
     `cents_deviation` — basic-pitch outputs integer MIDI, not f0)
   - `confidences` = each member's `amplitude`
   - `cents_deviations` = `(0.0,) * len(midi_notes)`
5. Apply the **chord-aware** merge step. `_merge_consecutive_same_pitch()`
   extends to:
   - Merge only when `prev.midi_notes == current.midi_notes` (full tuple
     equality, sorted, so chord identity is unambiguous).
   - Average element-wise per index when merging (e.g. `frequencies_hz[i] =
     (prev.frequencies_hz[i] + current.frequencies_hz[i]) / 2.0`). This
     avoids the tuple-arithmetic bug — pre-Phase-0 code averages floats; the
     refactored version averages parallel tuples by index.
   - **Use a different gap threshold for chord events vs mono.** Mono keeps
     `_MERGE_GAP_S = 600 ms` (handles harmonic-evolution false re-onsets on
     decaying single notes). Chord events use `_MERGE_GAP_CHORD_S = 200 ms`
     because legitimate repeated-chord rhythms (eighth notes at 100 BPM ≈
     300 ms apart) must not collapse. A regression test asserts two
     identical chords at 300 ms gap are preserved as two events.

**Octave correction is OFF for the basic-pitch backend in Phases 1–3.**
basic-pitch already does its own polyphonic spectral analysis with stronger
priors than `_correct_octave_error`'s post-hoc spectral fallback (which was
designed to rescue CREPE on noisy mic'd guitar). Force-running the spectral
fallback on basic-pitch's already-cleaned outputs would risk pulling
legitimate high notes an octave low when body-resonance energy is present
in the lower octave. The midi_min+24 ceiling on the existing fallback covers
most of the playable guitar range, so the risk surface is large.

If empirical Phase 2/3 results show octave-doubling errors specifically on
basic-pitch outputs, Phase 4 introduces a basic-pitch-specific filter
(separate from `_correct_octave_error`) that drops a chord member which is
exactly an octave above another member when its amplitude is below
**`OCTAVE_DOUBLING_AMPLITUDE_RATIO` (0.4 of the lower note's amplitude)**.
This is not built in MVP; we wait for evidence.

#### Onset detection (`analysis/onset.py`)

Unchanged. It is bypassed in the `basic_pitch` path but still required for
the `crepe` and `pyin` backends and for `TempoEstimator` (which still derives
the synthetic onset envelope from `NoteEvent.onset_s` values, not from
librosa onsets directly — re-read [tempo.py](../../../src/improv_scribe/quantization/tempo.py)
to confirm: yes, it builds the envelope from event onsets, so it works
unchanged for chord events that have one onset per chord).

### 3.3 Quantization — minimal change

[quantization/grid.py](../../../src/improv_scribe/quantization/grid.py)
operates on `(onset_s, offset_s)` and is pitch-agnostic. The only change is
the `QuantizedNote` field rename / type change. The quantizer body is
unchanged because it never inspects pitch.

**Tempo accuracy under chord-dense input:** chord events have a single onset
each, so onset density per beat may decrease relative to a phrase of single
notes at the same tempo. This is not a bug per se, but it widens the
beat-tracker's confidence band. Polyphonic integration tests use a wider BPM
tolerance (±15%) than monophonic tests (±5%).

### 3.4 Notation — `music21.chord.Chord`

The single change in [notation/score_builder.py](../../../src/improv_scribe/notation/score_builder.py):

```python
for qn in notes:
    dur = Duration(quarterLength=qn.quarter_length)
    if qn.is_rest:
        element = music21.note.Rest(duration=dur)
    elif len(qn.midi_notes) == 1:
        element = music21.note.Note(qn.midi_notes[0], duration=dur)
    else:
        element = music21.chord.Chord(list(qn.midi_notes), duration=dur)
    part.append(element)
```

Three observations validated against music21:

1. The clef-octave handling preserved at score_builder.py:14-21 propagates
   uniformly to all chord members — music21 applies `<clef-octave-change>` to
   every pitch in a `Chord`, no extra work.
2. `makeBeams(inPlace=True)` treats a `Chord` as a single rhythmic unit and
   beams chord chains identically to note chains.
3. `Chord` instances serialise to MusicXML as
   `<note>...</note><note><chord/>...</note>...` — first member without a
   `<chord/>` element, subsequent members with one. We rely on this in §3.6.

### 3.5 Tab fret assignment — chord-shape DP

[notation/tab_builder.py](../../../src/improv_scribe/notation/tab_builder.py)
extends to assign a tuple of `(string, fret)` pairs per event.

Result type:

```python
# Was:
def assign_frets(notes, instrument) -> list[tuple[int, int] | None]: ...

# Becomes:
def assign_frets(notes, instrument) -> list[tuple[tuple[int, int], ...] | None]: ...
# Each non-rest entry: tuple of (string, fret) pairs sorted by string ascending.
# Mono notes are length-1 tuples, e.g. (((0, 0),)). Rests: None.
```

#### Chord-shape enumeration

```python
def get_chord_shapes(
    midi_notes: tuple[int, ...], tuning: list[int]
) -> list[tuple[tuple[int, int], ...]]:
    """All assignments of distinct strings to chord members."""
    per_note_candidates = [get_candidates(m, tuning) for m in midi_notes]
    if any(not c for c in per_note_candidates):
        return []  # at least one member out of range — handled by fallback
    shapes = []
    for combo in itertools.product(*per_note_candidates):
        strings = [s for s, _ in combo]
        if len(set(strings)) == len(strings):
            shapes.append(tuple(sorted(combo)))   # canonical order
    return shapes
```

#### DP cost function

- **Within-shape cost (hand stretch):** for shapes with ≥1 fretted note,
  `max(fret) − min(fret)` over fretted notes only (open strings excluded —
  a free finger). 0 for all-open shapes.
- **Transition cost between shapes (i → i+1):** absolute difference of
  *fretted-fret centroids* (mean fret excluding 0s; defaults to 0 if all
  open). Singleton notes reduce to the existing case naturally because the
  shape has one element and "centroid" equals "fret."
- **Total cost** for an assignment over the phrase: sum of within-shape
  costs + sum of transition costs.
- **Tie-breaking.** The DP value at each (event, shape) is the lexicographic
  triple `(cumulative_cost, max_fret_in_shape, min_fret_in_shape)`. Ties on
  cumulative cost prefer the shape with the lower top fret (fewer hand-shifts
  for upcoming phrases); ties on max-fret prefer the lower bottom fret.
  For singleton shapes this collapses to `(cost, fret, fret)`, which is
  bit-equivalent to the existing single-note DP's `(cost, fret)` tie-break
  rule — verified by the Tier 1 mono-DP regression test.

#### Combinatorial bound

Worst case: a 6-note chord on guitar where each pitch has 2 candidate
strings → 64 shapes per chord. A 6-chord phrase: 64 × 64 × 5 transitions =
20,480 ops. Trivial; no beam search needed.

#### Out-of-range fallback and conflict resolution

Two distinct failure modes for `get_chord_shapes()`:

**Out-of-range members:** if at least one chord member has zero candidate
strings (above the 22-fret limit or below the lowest open string):
1. Drop the offending members, log a warning.
2. Re-run shape enumeration on the playable subset.

**No conflict-free shape:** if all members have candidates but no
combination produces pairwise-distinct strings (e.g. an unplayable
duplicate-pitch chord like {E2, E2, E2} where only 2 strings can sound E2):
1. Detected upstream by the basic-pitch boundary's MIDI dedup step (§3.2
   step 3 collapses identical MIDI values, keeping highest amplitude). So
   this case shouldn't arise from the normal flow.
2. As a defence-in-depth, if shape enumeration still returns empty, fall
   back to dropping members from the highest fret downward until a
   conflict-free shape exists. Log each drop.

**Final fallback** (both subsets empty): emit `((0, 0),)` so the score still
renders rather than failing. The dropped pitches still appear on the
**notation staff** (music21 doesn't care about playability); only the
**tab staff** loses them. This is the correct behaviour — a guitarist
reading the score sees what was played, but the tab can only show what is
playable on this instrument.

### 3.6 MusicXML tab injection — chord siblings on staff 2

[export/tab_xml.py](../../../src/improv_scribe/export/tab_xml.py) currently
walks `<note>` siblings and consumes one tab assignment per non-rest, non-tie-stop
element. Music21 emits each member of a `chord.Chord` as its own `<note>` element
with a `<chord/>` child on the 2nd–Nth members. **The existing iterator would
therefore consume N tab assignments per chord rather than one tuple** — a
silent off-by-N bug if not addressed.

The change is therefore deeper than a "small extension":

#### Prerequisite — verify music21's chord serialisation shape

Before implementation, write a one-shot test that builds a 3-pitch
`music21.chord.Chord`, serialises it through `score.write('musicxml')`,
and asserts the exact element structure produced (presence/absence of
`<chord/>` children, location of `<duration>`, location of `<staff>`,
behaviour of `<tie>` siblings within a chord). The implementation is
written against the structure observed, not assumed.

#### `inject_tab_part()` algorithm

1. **Group `<note>` siblings into "slots."** A slot is a chord group:
   the first `<note>` in the group has no `<chord/>` child; each subsequent
   sibling whose first child is `<chord/>` belongs to the same slot until
   the next chord-group boundary. For non-chord notes, a slot is one
   element. The number of slots in a measure equals the number of
   assignment tuples to consume from the assignments list.
2. **For each slot, fetch one assignment tuple** of length `len(slot)`.
3. **Annotate staff-1 elements** with `<staff>1</staff>` + per-member
   `<technical><string><fret>` matching the assignment tuple in
   string-ascending order.
4. **Emit staff-2 mirror elements**, one per slot member, preserving the
   `<chord/>` topology:
   - Slot member 1: `<note><staff>2</staff>` + `<technical>` for
     `(s1, f1)` — no `<chord/>`.
   - Slot members 2…N: `<note><chord/><staff>2</staff>` + `<technical>`
     for `(si, fi)`.
5. `<tie type="stop">` handling: if any slot member has a tie-stop, the
   slot is treated as a tie continuation (assignment is consumed but no new
   `<technical>` annotation on staff 1, mirror still emitted on staff 2 to
   keep visual continuity). The exact rule is finalised in the Phase 2
   prerequisite test above.

Mono path is preserved byte-for-byte because a length-1 slot produces the
same single `<note>` we emit today, and a length-1 assignment tuple
contains exactly one `(string, fret)` pair.

A new Tier 1 unit test (`tests/export/test_tab_xml_chord.py`) asserts both
the byte-equivalence of the mono path and the exact structure of the
chord path.

---

## 4. Testing strategy

### 4.0 Two-tier sample strategy — synthetic *and* real

Synthetic samples are valuable for fast iteration (tuning the cluster
window, the relative confidence floor, the merge gap) because they have
deterministic ground truth and zero recording overhead. **They do not
validate real-world chord performance** — summed isolated-string segments
lack string coupling, sympathetic resonance, and shared body reverb.
basic-pitch could pass synthetics perfectly and fail on real chords (or
vice versa), so synthetics alone cannot be the phase gate.

The strategy is therefore hybrid:

| Use | Source | Phase gate? |
|---|---|---|
| Tuning cluster window, confidence floors, merge thresholds; fast unit-style regression | Synthetic (mixed open-string segments) | No — never blocks on its own |
| Phase 2 sign-off | User-provided real dyad recordings (4–6 takes covering octave, fifth, third, third-on-different-strings) | Yes |
| Phase 3 sign-off | User-provided real open-chord progression (E, A, D, G, C minimum) | Yes |

The user records the real samples between Phases 1 and 2 (and between 2 and
3). Implementation work in Phase 2/3 can begin against synthetics, but the
phase cannot be marked done until the real recordings pass.

### 4.1 Synthetic sample generation

A new `tests/integration/_helpers/polyphonic_synth.py` builds reproducible
polyphonic samples from existing isolated open-string recordings.

```python
def synthesize_chord(
    sample_path: Path,
    note_indices: list[int],     # which onset(s) from the source to mix
    instrument: Instrument,
    cache_dir: Path,
) -> tuple[np.ndarray, int]:
    """Mix segments of `sample_path` to produce a synthetic dyad/chord.
    Each note_indices[i] selects one detected onset from the source recording;
    those segments are time-aligned to t=0 and summed (with peak-normalisation
    to avoid clipping). Cached on a content hash of source file + indices."""
```

Source recordings:

- Guitar: `samples/guitar/6_string_electric_line_in.mp3` — six clean
  isolated open-string hits (E2 A2 D3 G3 B3 E4)
- Bass: `samples/bass/4_string_bass_line_in.mp3` — four open-string hits
  (E1 A1 D2 G2)

Generated fixtures (cached under `tests/integration/_synth_cache/`,
regenerated on source-file SHA256 change):

| Fixture | MIDI set | Purpose |
|---|---|---|
| `dyad_octave_e2_e3` | {40, 52} | Octave dyad — basic-pitch's known weakness on string instruments |
| `dyad_fifth_e2_b2` | {40, 47} | Perfect 5th (open E + B); strong shared harmonics |
| `dyad_third_g3_b3` | {55, 59} | Major 3rd, mid-register |
| `dyad_unison_e2_e2_dup` | {40} | Two copies of E2 mixed — must collapse to a single-note event after dedup |
| `triad_e_major_open` | {40, 47, 52} | E major lower three notes |
| `bass_dyad_e1_a1` | {28, 33} | Low-frequency dyad on bass |

### 4.2 Test layering

#### Tier 1 — Unit tests (new files)

- `tests/analysis/test_basic_pitch_backend.py` — backend-level: CREPE vs
  basic-pitch on the same monophonic sample produce the same MIDI set.
- `tests/analysis/test_onset_clustering.py` — synthetic basic-pitch event
  lists with controlled jitter; verify clustering boundary (60 ms ± epsilon).
- `tests/notation/test_chord_tab_dp.py` — chord {40, 47, 52} (E2+B2+E3) on
  guitar tuning yields `((0,0), (1,2), (2,2))`. No two members share a
  string. Mono inputs produce identical assignments to the existing single-
  note DP (binary-equal regression check).
- `tests/export/test_tab_xml_chord.py` — given a 2-pitch QuantizedNote,
  produced MusicXML has `<chord/>` markers in the correct positions on both
  staves.

#### Tier 2 — Existing monophonic integration tests (preserved, per-backend)

All four existing integration tests stay green and are parametrised over
`pitch_backend ∈ {crepe, basic_pitch}`. **Each backend gets its own ground
truth.** CREPE values stay identical to today (no regression). basic-pitch
ground truth is calibrated empirically in Phase 1 — basic-pitch may differ
from CREPE by a semitone on edge cases like E2 (82 Hz, near its lower
reliable range) or under acoustic-mic noise. We accept the difference and
record per-backend `EXPECTED_MIDI` rather than pretending equivalence.

The `EXPECTED_TAB` fixtures stay identical (the tab DP is deterministic
given the MIDI input). If basic-pitch differs from CREPE on MIDI, the tab
will differ too — that's expected and reflected in per-backend tab
expectations where they diverge.

A single mono sample may need a per-backend block at the top of its test
file:

```python
EXPECTED_MIDI_CREPE = [40, 45, 50, 55, 59, 64]
EXPECTED_MIDI_BASIC_PITCH = [40, 45, 50, 55, 59, 64]   # calibrated in Phase 1
```

#### Tier 3 — New polyphonic integration tests

- `test_dyad_octave_synth.py`, `test_dyad_fifth_synth.py`,
  `test_triad_open_e_synth.py`, `test_bass_dyad_synth.py`, etc.
- Same fixture-chain pattern as `make_pipeline_fixtures`, with a sibling
  `make_polyphonic_pipeline_fixtures()` factory.
- Assertions:
  - `len(note_events)` matches expected onset count (one event per chord)
  - Each event's `midi_notes` exactly equals the expected sorted tuple
  - Tab assignments use distinct strings within each chord
  - Score has `music21.chord.Chord` objects in the right positions
  - Tab MusicXML contains correct `<chord/>` markers on staff 2

### 4.3 Regression gate per phase

```bash
ATS_PITCH_BACKEND=crepe       conda run -n auto-sheet-music pytest tests/integration -v
ATS_PITCH_BACKEND=basic_pitch conda run -n auto-sheet-music pytest tests/integration -v
```

Both must pass before crossing a phase boundary.

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| basic-pitch octave doubling on guitar (model emits both fundamental and octave for one plucked note) | Medium | Phase 2 collects empirical data on synthetic dyads. If octave doubling appears, Phase 4 introduces a chord-aware octave filter. Do **not** extend `_correct_octave_error` — its assumptions are monophonic. |
| Onset cluster window mis-tuning (too tight = chord splits into rapid arpeggio; too loose = legitimate fast passages collapse to chord) | Medium | `ATS_ONSET_GROUPING_WINDOW_MS` is env-tunable; default 60 ms. Add a debug CSV (mirror of `ATS_DEBUG_PITCH`) logging cluster decisions: each cluster's members, onsets, and the "why we cut" boundary. |
| Tempo accuracy degrades with chord-dense input | Low–medium | ±15% tolerance band on polyphonic BPM assertions. If real-world degradation is severe, fall back to `_ioi_median` more aggressively when chord events dominate. |
| Per-note absolute confidence floor causes "missing notes" in chords (a quiet member at amp 0.39 next to amp 0.95 is dropped) | High | `POLYPHONIC_RELATIVE_FLOOR` (default 0.5 of cluster max), not absolute. Tunable via `ATS_POLYPHONIC_RELATIVE_FLOOR`. |
| Existing CREPE path bit-rot during refactor | Low | Phase 0 is a pure data-model refactor; CREPE/pYIN tests are the primary regression gate. Phases 1–3 keep CREPE as a selectable backend; the same integration tests run against both. |
| Tab XML chord injection breaks existing single-note tab path | Medium | Tier 1 unit tests assert mono case is byte-identical to today's output. The mono path is taken whenever the assignment tuple has length 1. |
| `midi_note` back-compat property obscures incomplete migrations | Low | `grep -rn '\.midi_note\b' src/ \| grep -v 'def midi_note'` must return zero hits at end of Phase 2; that's the migration completion check. The property is then deleted. |
| Synthetic samples mask real-world issues | High | Phase 3 explicitly requires the user to provide real recordings as a "realism" sanity check before declaring chord support shipped. Synthetic tests are necessary but not sufficient. |
| basic-pitch dependency size and MPS support | Low | basic-pitch ships an ONNX model (~30 MB). It runs on CPU comfortably; no MPS expectation. Added to `envionment.yaml`. |
| basic-pitch produces spurious very short notes on attack transients | Medium | Filter at backend boundary: drop `BasicPitchNote` with `(end_s - start_s) < MIN_NOTE_DURATION_S` (default 50 ms). |
| **`midi_exporter.py` silently drops chord members via the `midi_note` back-compat shim** | High (correctness bug if not addressed) | Promoted from Phase 4 polish to Phase 2 deliverable. Exporter must iterate `midi_notes` and emit one note_on/note_off per chord member. The Phase 0 grep check expanded to cover `event.midi_note` in `midi_exporter.py` and `gui/main_window.py`. |
| **Tempo IOI fallback under chord-dense input** — chord events drop event count by ≤4× per chord, making `_ioi_median` (≤ 4 events) more likely to fire with 0.3 confidence | Medium | The ±15% BPM tolerance on polyphonic tests covers the IOI-fallback case. If real recordings show degradation, revisit by weighting onsets by chord size in the synthetic envelope. |
| **basic-pitch first-call latency** — measured 0.32 s on Apple Silicon (ONNX runtime, no TF) | Negligible (downgraded after probe) | Module-level model cache keeps subsequent chunks at warm-call latency. Original 1–3 s concern was based on TF/TFLite assumptions that don't apply to the ONNX-only install path. |

---

## 6. Configuration additions

New entries in [config.py](../../../src/improv_scribe/config.py):

```python
# Polyphonic detection (Phase 1+)
# Phase 1 (LANDED): POLYPHONIC_AMPLITUDE_FLOOR and MIN_NOTE_DURATION_S
ONSET_GROUPING_WINDOW_MS: float = float(os.getenv("ATS_ONSET_GROUPING_WINDOW_MS", "100.0"))   # Phase 2
POLYPHONIC_AMPLITUDE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_AMPLITUDE_FLOOR", "0.65")) # calibrated Phase 1
POLYPHONIC_RELATIVE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_RELATIVE_FLOOR", "0.5"))    # Phase 2
MIN_NOTE_DURATION_S: float = float(os.getenv("ATS_MIN_NOTE_DURATION_S", "0.050"))             # Phase 1
MERGE_GAP_CHORD_MS: float = float(os.getenv("ATS_MERGE_GAP_CHORD_MS", "200.0"))               # Phase 2
DEBUG_CLUSTERING: bool = os.getenv("ATS_DEBUG_CLUSTERING", "0") == "1"                         # Phase 2
```

These propagate into `AppConfig` with the same field-default pattern used
today. `POLYPHONIC_AMPLITUDE_FLOOR` was initially proposed at 0.10 then
revised to 0.40 then to 0.65 — see §11.6 and §12 for the empirical
calibration trail.

---

## 7. Dependency additions

**Phase 1 actually-shipped install path** (supersedes the simpler bullet
that originally appeared here — see §11.1 and §11.2 for why):

`pyproject.toml` gains a `basic-pitch` optional-dependency extra with the
six transitives that resolve normally on Python 3.13:

- `onnxruntime>=1.17`
- `pretty-midi>=0.2.9`
- `mir-eval>=0.6`
- `mido>=1.3`
- `importlib_resources>=5.0`
- `soundfile>=0.12` (needed for the temp-WAV path described in §3.2)

`envionment.yaml` mirrors these in its `pip:` section.

`basic-pitch` itself is **NOT** in the extra because pip's resolver tries
to pull `tensorflow-macos<2.15.1` for Python >3.11, and no such wheel exists.
The installer script `scripts/install_basic_pitch.sh` runs the install in
two passes: first `pip install -e ".[basic-pitch]"` for the transitives,
then `pip install --no-deps 'basic-pitch>=0.4'` for the package itself.

`resampy` is already in the env as a `torchcrepe` transitive — do not
remove it during any cleanup pass (it would break CREPE). No system-level
additions (no Homebrew formula).

---

## 8. Migration completion checklist

Before declaring the polyphonic MVP shipped:

- [ ] All Phase 0–3 steps complete and committed.
- [ ] `grep -rn '\.midi_note\b' src/ | grep -v 'def midi_note'` returns no hits.
- [ ] `grep -rn '\.frequency_hz\b' src/ | grep -v 'def frequency_hz'` returns no hits.
- [ ] Both `ATS_PITCH_BACKEND=crepe` and `ATS_PITCH_BACKEND=basic_pitch`
      pass `pytest tests/integration -v`.
- [ ] Default `pitch_backend` is `basic_pitch` in `config.py`.
- [ ] User-provided real dyad recordings (Phase 2 gate) render correctly.
- [ ] User-provided real open-chord progression (Phase 3 gate) renders correctly.
- [ ] `midi_exporter.py` exports all chord members (verified by listening to
      a multi-note MIDI file and by an automated test parsing the MIDI back
      and asserting all `note_on` events present).
- [ ] Risk-register items marked High have a corresponding mitigation
      that has actually fired in tests (i.e. we have evidence the
      mitigation works, not just that it exists).

---

## 9. What we explicitly do not reinvent

For the audit trail, this is the dependency-coverage matrix that justified
the design:

| Capability | Provided by | Status |
|---|---|---|
| Polyphonic pitch detection | `basic-pitch` (Spotify) | Reuse |
| Per-note event assembly with onset/offset/pitch/amplitude | `basic_pitch.inference.predict()` | Reuse |
| Onset detection in polyphonic path | basic-pitch (internal) | Reuse |
| Notation chord rendering | `music21.chord.Chord` | Reuse |
| MusicXML `<chord/>` markers on staff 1 | music21 serialiser | Reuse |
| MIDI simultaneous-note encoding | mido / music21 | Reuse |
| Tempo / beat tracking | librosa | Reuse, unchanged |
| Onset clustering of basic-pitch flat events into chord events | (none — bespoke, ~15 LOC) | New |
| Chord-aware tab DP with no-string-conflict | (none — extension to existing DP, ~50 LOC) | New |
| Tab MusicXML chord injection on staff 2 | (none — extension to existing tab_xml.py, ~20 LOC) | New |
| `NoteEvent` chord migration | (data model evolution) | New |

---

## 10. Phase 0 Outcome (landed 2026-05-12)

Phase 0 — data-model migration — completed in 9 commits on `chord-detection`
branch (155f8d4 → 3b45f16). Net result:

- `NoteEvent.midi_notes: tuple[int, ...]` and parallel tuple fields land
  with read-only back-compat properties (`midi_note`, `frequency_hz`,
  `confidence`, `cents_deviation`).
- `QuantizedNote` mirrors the same shape; rests use empty tuples.
- `_merge_consecutive_same_pitch` is chord-aware: tuple equality + parallel
  averaging via `_avg_tuples` + 200 ms threshold for chord events.
- Integration tests preserved at baseline: CREPE 72/72, pyin 69/72 (same 3
  pre-existing failures unrelated to Phase 0).
- 30 new unit tests cover the chord-capable contract (19 NoteEvent + 11
  QuantizedNote).
- Ruff: pre-existing errors unchanged, no new ones introduced by Phase 0.

### Phase 2 migration target list — `.midi_note` back-compat shim consumers

Phase 2 removes the `midi_note` / `frequency_hz` / `confidence` / `cents_deviation`
properties on `NoteEvent` and `QuantizedNote`. These six call-sites currently
rely on the shim and must be migrated to read the tuple fields directly:

| File | Line | Current code (via shim) | Migration target |
|---|---:|---|---|
| `src/improv_scribe/gui/main_window.py` | 288 | `_midi_name(n.midi_note)` | iterate `n.midi_notes` for chord display |
| `src/improv_scribe/notation/tab_builder.py` | 77 | `get_candidates(note.midi_note, tuning)` | Phase 2 chord-shape DP rewrite uses `note.midi_notes` |
| `src/improv_scribe/notation/score_builder.py` | 138 | `Note(qn.midi_note, …)` | Phase 2: `Chord(list(qn.midi_notes), …)` when `len > 1` |
| `src/improv_scribe/notation/score_builder.py` | 192 | `Note(qn.midi_note)` | same — chord-aware in `build_raw()` |
| `src/improv_scribe/export/midi_exporter.py` | 120 | `("note_on", event.midi_note)` | iterate `event.midi_notes` for chord note_on |
| `src/improv_scribe/export/midi_exporter.py` | 121 | `("note_off", event.midi_note)` | iterate `event.midi_notes` for chord note_off |

The midi_exporter changes (lines 120-121) are a **correctness bug** today
under the back-compat shim — a chord event silently drops all but the
lowest note in MIDI output. Phase 2 fixes this; Phase 0 deliberately
defers it (no chord events exist until basic-pitch ships in Phase 1, so
the bug is unreachable today).

---

## 11. Phase 1 Prerequisite — basic-pitch API Prototype (run 2026-05-12)

Recorded findings from running `scripts/basic_pitch_probe.py` against
`samples/guitar/6_string_electric_line_in.mp3`. These findings update §3.2
and §7 before Phase 1 begins.

### 11.1 Install path on Python 3.13

basic-pitch 0.4.0's `pyproject.toml` declares `tensorflow-macos<2.15.1,>=2.4.1`
as a base dependency gated on `platform_system == "Darwin" and python_version > "3.11"`.
**No wheel for `tensorflow-macos` exists for Python 3.13.** Standard
`pip install basic-pitch` and `pip install 'basic-pitch[onnx]'` both fail with
`No matching distribution found for tensorflow-macos`.

The `[onnx]` extra does NOT remove the TF base dep — it adds onnxruntime on
top, which is unreachable when the base dep resolution fails.

**Working install on Python 3.13 + macOS arm64:**

```bash
pip install --no-deps basic-pitch onnxruntime coremltools pretty-midi mir-eval resampy
pip install mido importlib_resources
```

The ONNX model (`nmp.onnx`) ships with the basic-pitch wheel at
`<site-packages>/basic_pitch/saved_models/icassp_2022/nmp.onnx` — no separate
model download required.

### 11.2 envionment.yaml additions

Phase 1's environment update must enumerate the transitive deps explicitly
(because we're using `--no-deps` for basic-pitch). The minimal set:

- `basic-pitch>=0.4` (installed with `--no-deps`)
- `onnxruntime` (the runtime; coremltools and tflite-runtime are NOT needed)
- `pretty-midi>=0.2.9`
- `mir-eval>=0.6`
- `mido` (used by pretty-midi)
- `importlib_resources` (used by pretty-midi's fluidsynth shim)

`resampy` is **already in the env** as a torchcrepe transitive dep — do not
remove it during any cleanup pass (lesson learned during the prototype:
removing it broke the CREPE backend's import). `coremltools` is NOT required
for the ONNX backend despite the misleading import-time warning
("Coremltools is not installed"). The warning is harmless when ONNX is
present.

### 11.3 Return shape (confirmed)

```python
result = predict(audio_path)
# result is tuple[dict, pretty_midi.PrettyMIDI, list[tuple]]

model_output, midi_data, note_events = result
# model_output: dict with keys {'note', 'onset', 'contour'} — raw model logits
# midi_data:    pretty_midi.PrettyMIDI object — already-assembled MIDI
# note_events:  list[tuple[float, float, int, float, list[int]]]
#               (start_s, end_s, midi, amplitude, pitch_bend)
```

The `pitch_bend` field is a list of int values (mostly 1s, occasional 0/2)
representing 1/3-semitone offsets per frame. We will ignore it in Phase 1 —
basic-pitch emits integer MIDI directly, and our `cents_deviations` field is
zeroed for basic-pitch outputs anyway.

### 11.4 numpy in-memory input — NOT SUPPORTED

The spec §3.2 originally assumed `predict()` accepts a numpy array directly.
**It does not.** `predict(numpy_array)` calls `str(audio_path)` internally
and treats the result as a file path, producing `FileNotFoundError` with the
array's repr as the "missing path."

**Workaround for Phase 1:** the `_BasicPitchBackend.estimate()` wrapper writes
the input audio to a temporary WAV file (via `soundfile.write`), calls
`predict(temp_path)`, then deletes the temp file. The wrapper signature
`estimate(audio: np.ndarray, …)` stays unchanged; only the internal plumbing
shifts. Phase 4 polish: explore whether basic-pitch has a private "audio
array" entry point that bypasses the path layer (we can check
`basic_pitch.inference._run_inference` or similar).

### 11.5 Cold-call latency (confirmed: very fast)

Measured **0.32 s** for the cold-call (first import + first predict). The
spec's concern about 1–3 s TF/TFLite cold-start does not apply because ONNX
is much lighter. The model-load cache (module-level singleton) is still
worth keeping for warm-call performance, but it's no longer a critical perf
concern.

### 11.6 Mono-content over-detection — critical Phase 1 calibration finding

Running basic-pitch on `samples/guitar/6_string_electric_line_in.mp3` (6
isolated open-string plucks: E2 A2 D3 G3 B3 E4) produced **18 note events**,
not 6. Categorisation:

- **Six correct fundamentals** are present (MIDI 40, 45, 50, 55, 59, 64),
  with amplitudes 0.39–0.84.
- **Duplicate detections** of E2 (MIDI 40) appear as 5 separate events —
  basic-pitch fragments the sustained low note into multiple events.
- **Spurious higher-octave detections** include MIDI 71 (B4), 76 (E5), 97
  (E7), 62 (D4), 52 (E3) — these are harmonics/octave errors with
  amplitudes 0.31–0.42.

**Phase 1 implications:**

1. The default `POLYPHONIC_AMPLITUDE_FLOOR = 0.10` is far too low — even the
   spurious detections register at 0.31+. Recommend **default raised to
   `0.40`** for the mono validation phase, and re-tuned empirically once we
   have polyphonic samples.
2. The merge helper (`_merge_consecutive_same_pitch`) will collapse the
   5 duplicate E2 detections into one after Phase 1 wires basic-pitch into
   the existing pipeline, but only if the inter-detection gap is below
   `_MERGE_GAP_S = 600 ms`. The probe data shows E2 detections at 0.4s,
   3.2s, 4.9s, 9.2s, 11.6s — gaps of 1.5–4.3 s, far above the threshold.
   These will NOT be merged. Phase 1's mono validation test must therefore
   either (a) accept many same-pitch events as the test outcome, (b) add a
   "same-pitch deduplication" pass tuned for mono recordings, or (c) accept
   that basic-pitch is just worse than CREPE on isolated mono content and
   record per-backend ground truth in tests.
3. The per-backend `EXPECTED_MIDI` strategy from §4.2 is **necessary, not
   optional.** basic-pitch will produce more events than CREPE on the
   existing mono samples; the integration tests have to accept the new count.

### 11.7 Spec sections to update before Phase 1 implementation

- §3.2 "basic-pitch's actual API" paragraph: replace "predict() takes either
  a path or a numpy array" with "predict() takes a path only; the wrapper
  writes a temp WAV file from numpy input."
- §6 Configuration additions: `POLYPHONIC_AMPLITUDE_FLOOR` default raised
  from 0.10 to 0.40 (calibration may revise this further).
- §7 Dependency additions: list all 7 transitive deps explicitly; install
  via `--no-deps` for basic-pitch.
- §5 Risk register: downgrade "basic-pitch first-call latency" from Low to
  Negligible; add a new risk "basic-pitch fragments sustained mono notes
  into multiple events" with Medium likelihood and the §11.6 mitigation
  options.
- §2 Phase table — Phase 1 prerequisite is **DONE**; the install path,
  return shape, and calibration finding are all locked in.

---

## 12. Phase 1 Outcome (landed 2026-05-12)

Phase 1 — basic-pitch as third backend, mono-only validation — completed
across 10 task commits + 1 follow-up on `chord-detection` branch.

- `_BasicPitchBackend` registered; selectable via `ATS_PITCH_BACKEND=basic_pitch`.
- Default backend stays `crepe`; basic-pitch is opt-in until Phase 2.
- All four mono integration samples produce correct MIDI under both backends.
- **Final calibrated `POLYPHONIC_AMPLITUDE_FLOOR = 0.65`** (raised from the
  initial 0.50 estimate after empirical run against all four samples).
- Octave-error correction is NOT applied to basic-pitch outputs (deliberate).
- Phase 2 work (onset clustering, chord events, score/tab chord support) is
  ready to begin once user-provided real dyad recordings land.

### Integration test outcome

| Backend | Result |
|---|---|
| `ATS_PITCH_BACKEND=crepe` | **72/72 passed** (byte-identical to pre-Phase-1 baseline) |
| `ATS_PITCH_BACKEND=basic_pitch` | **68 passed + 4 skipped, 0 failures** (the 4 skips are `test_pitch_frequency_range` tests that assert on per-frame f0 data, which basic-pitch doesn't produce — gracefully skipped with `pytest.skip()` rather than weakening the CREPE assertion) |

### Per-backend ground truth divergence

| Sample | CREPE | basic_pitch | Difference |
|---|---|---|---|
| guitar electric line-in | 6 events [40,45,50,55,59,64] | 6 events [40,45,50,55,59,64] | None |
| guitar acoustic line-in | 6 events [40,45,50,55,59,64] | 6 events [40,45,50,55,59,64] | None |
| guitar acoustic mic | 6 events [40,45,50,55,59,64] | 6 events [40,45,50,55,59,64] | None |
| bass line-in | 4 events [28,33,38,43] | **5 events** [28,33,38,43,28] | One sympathetic E1 detection at 6.749s (amp 0.74), can't be filtered without losing legitimate notes |

The bass divergence is recorded as a per-backend `NOTE_COUNT_BY_BACKEND`
and `EXPECTED_TAB_BY_BACKEND` in the bass integration test. Mono-content
sympathetic-resonance fragmentation is a known characteristic of
basic-pitch and would be cleaner to address with a chord-aware
post-processor that we have not built yet (Phase 4 or later, only if
sympathetic-resonance becomes a real-world quality issue on chord
content).

### Phase 0 follow-up surfaced

Phase 1's Task 10 full unit-suite verification surfaced that two
pre-existing test files (`tests/quantization/test_grid.py` and
`tests/quantization/test_tempo.py`) had been broken since Phase 0 landed
because they constructed `NoteEvent` with the old scalar kwargs. The
Phase 0 phase gate ran only `tests/integration`, so those failures
were masked. Phase 1 commit `bb9b267` migrated both files to the
tuple-kwarg shape.

**Lesson recorded:** future phase gates should run BOTH the integration
suite AND the unit-test suite. Phase 2's phase-gate command updated
accordingly.

---

## 13. Phase 2 Prerequisite — Real Dyad Recording Probe (run 2026-05-22)

User-provided real dyad recordings landed in `samples/guitar/chords/` —
three files on a USB-in electric guitar:

- `6_string_electric_octave_dyads.mp3` (six octave dyads)
- `6_string_electric_perfect_fifths.mp3` (six perfect 5ths)
- `6_string_electric_major_thirds.mp3` (six major 3rds, mixed string pairs:
  4 on the 5-4 pair, 2 on the 4-3 pair)

Ran `_BasicPitchBackend` on each (with the Phase 1 calibrated config:
`POLYPHONIC_AMPLITUDE_FLOOR=0.65`, no clustering yet) and inspected the
raw note_events output.

### 13.1 Cluster window validated at 100ms

Onset spreads within detected dyads ranged from **0 ms to 35 ms** across
all 18 dyads detected. The Phase 2 default `ONSET_GROUPING_WINDOW_MS=100`
has comfortable headroom over the largest observed spread. **No revision
needed.**

### 13.2 Dyad detection rate: 3–4 of 6 per file

basic-pitch at floor=0.65 detects:

| Sample | Dyads | Singletons (one member lost) | Hit rate |
|---|---|---|---|
| octave_dyads | 3 | 3 (lost lower octave on G/A/C) | 50% |
| perfect_fifths | 3 | 3 (lost the 5th on D5/D3/B2) | 50% |
| major_thirds | 4 | 2 (lost the 3rd on E3/G3) | 67% |

The lost members register below 0.65 amplitude — picking-dynamics variation
on a real performance. The 6-onset cluster structure is intact in every
file; what we're losing is pitch *coverage* within some clusters, not the
clusters themselves.

### 13.3 Phase 2 ground-truth decision: accept actual detection

User decision (recorded 2026-05-22): Phase 2 integration tests assert
**what basic-pitch actually detects** (mixed dyads + singletons), not the
ideal 6-of-6 dyads. Rationale: this is production-realistic. Phase 4 may
revisit the absolute floor and/or add cluster-internal relative
filtering, but only with evidence from real-world chord performances.

### 13.4 Implications for Phase 2 plan

- The cluster window default of 100 ms is empirically confirmed; no
  Phase 2 plan task needed to re-tune.
- The cluster algorithm itself (earliest-anchor + 100 ms window) is
  validated against real strums.
- Integration test ground truth for the three new samples records the
  actual basic-pitch output (mixed dyads/singletons) — Phase 2's Tier 3
  test file lists each cluster's expected `midi_notes` tuple explicitly.
- The `POLYPHONIC_RELATIVE_FLOOR` config field (defaulted 0.5 in §6) is
  still introduced in Phase 2 for forward-compat, but its effect on the
  current samples is small — most surviving dyad members are within 25%
  of their cluster's max amplitude, so a 50% relative floor doesn't
  drop them.
- Phase 4 / quality-improvement scope (not Phase 2): explore lowering
  absolute floor + per-cluster relative filtering for higher dyad-recall
  rate. Out of scope for this MVP.

### 13.5 Exact basic-pitch output (for Phase 2 ground truth)

The Phase 2 plan's integration tests will assert these exact MIDI tuples
per cluster, captured by `scripts/basic_pitch_probe.py` on the real
recordings:

**octave_dyads** (6 clusters):
1. `(40, 52)` at 0.290s — E2+E3
2. `(41, 53)` at 2.079s — F2+F3
3. `(55,)` at 3.821s — G3 only
4. `(45,)` at 5.541s — A2 only
5. `(47, 59)` at 7.307s — B2+B3
6. `(48,)` at 9.014s — C3 only

**perfect_fifths** (6 clusters):
1. `(47,)` at 0.453s — B2 only
2. `(41, 48)` at 1.998s — F2+C3
3. `(50,)` at 3.484s — D3 only
4. `(45, 52)` at 4.983s — A2+E3
5. `(47,)` at 6.355s — B2 only
6. `(48, 55)` at 7.771s — C3+G3

**major_thirds** (6 clusters):
1. `(50, 54)` at 0.302s — D3+F#3 (5-4 pair)
2. `(52,)` at 1.091s — E3 only
3. `(54, 58)` at 1.927s — F#3+A#3 (5-4 pair)
4. `(55,)` at 2.834s — G3 only
5. `(57, 61)` at 3.670s — A3+C#4 (4-3 pair)
6. `(59, 63)` at 4.530s — B3+D#4 (4-3 pair)

---

## 14. Phase 2 Outcome (landed 2026-05-23)

Phase 2 — dyad detection end-to-end — completed in 12 task commits on
`chord-detection-phase2` worktree branch. The phase delivered:

- Onset clustering in `_process_basic_pitch` with earliest-anchor +
  100 ms window + relative-floor filter (Task 4).
- `ScoreBuilder.build()` emits `music21.chord.Chord` for multi-pitch
  QuantizedNotes (Task 6).
- Chord-aware `assign_frets()` with no-string-conflict DP (Task 8).
- music21 chord serialization shape verified (Task 9 — input-order
  preservation, not ascending).
- `inject_tab_part()` walks `<chord/>` siblings and mirrors them on
  staff 2 with MIDI-ordered assignment alignment (Task 10).
- `midi_exporter.raw_from_events()` emits N note_on/note_off per chord
  (Task 11). Phase 0 §10 silent-drop bug fixed.
- `gui/main_window.py` displays chord names like "E4/G4/B4" (Task 12).
- Back-compat shims (`.midi_note`, `.frequency_hz`, `.confidence`,
  `.cents_deviation`) removed from NoteEvent and QuantizedNote — all 6
  Phase 0 §10 consumers migrated (Task 13).
- Three new dyad integration tests (octave/5th/3rd samples) keyed off
  spec §13.5 ground truth (Task 14).
- Default backend flipped from `crepe` to `basic_pitch` (Task 15).
- Manual PDF chord+tab render verification (Task 16) — see
  `docs/superpowers/phase2_chord_proof.pdf` for the artifact.

### 14.1 Phase gate results

| Backend | Result |
|---|---|
| `ATS_PITCH_BACKEND=crepe` | 271 passed, 12 skipped |
| `ATS_PITCH_BACKEND=basic_pitch` | 279 passed, 4 skipped |

### 14.2 Per-backend ground truth divergence on dyad samples

All three dyad samples produce identical results to spec §13.5 ground truth.

### 14.3 Discovered pre-existing quantizer issue (deferred to Phase 4)

Task 16's PDF render verification surfaced a pre-existing quantizer
bug not specific to chord support: `RhythmQuantizer.quantize()` snaps
onset and offset independently, which can produce rests and notes
with overlapping beat ranges. When the resulting MusicXML is rendered
via MuseScore, the inconsistent `<duration>` and `<type>` element
pairings cause MuseScore to exit with code 40 (silent rejection).

Reproduces with the three dyad samples at the clamped 40 BPM tempo.
Does NOT reproduce with the four existing monophonic integration
samples — those snap to clean integer beat positions.

The bug is in `quantization/grid.py` and pre-dates Phase 2. The chord
support itself is verified working end-to-end via
`docs/superpowers/phase2_chord_proof.pdf` (a hand-constructed score
with clean timing rendered through the full Phase 2 pipeline).

Phase 4 scope (separate spec): fix the quantizer to ensure rest
+ note duration consistency. Until then, the dyad integration tests
assert on the data model (NoteEvent.midi_notes tuples) and PDF
rendering is verified separately on hand-constructed inputs.

### 14.4 Lessons recorded for Phase 3

1. **PDF render is a useful regression gate.** The integration tests
   assert on music21 Score state but not on MuseScore output. Phase 3
   should consider adding at least one PDF render smoke test (hand-
   constructed Score, render to PDF, assert PDF file > 1 KB) so future
   quantizer bugs surface in CI.
2. **Worktree editable install requires re-pip from the worktree.**
   The conda env's site-packages can only point at one location;
   `pip install -e .` from the worktree path makes that worktree the
   source-of-truth. Document in CLAUDE.md if Phase 3 also uses a
   worktree.
3. **basic-pitch's mono detection differs from CREPE.** The bass
   sympathetic-resonance detection (spec §12, bass_line_in test)
   was extended in Phase 2: with clustering enabled, the sympathetic
   E1 merges with the G2 into a chord. This is expected, but
   Phase 3 should be aware that clustering can change basic-pitch's
   per-onset event counts on mono samples.

---

## 15. Phase 3 Prerequisite — Open-Chord Recording Probe (run 2026-05-23)

User-provided real open-chord recordings landed in
`samples/guitar/chords/` — five files on a USB-in electric guitar, each
~12.3 s of strummed open-chord content (5-6 strums per recording):

- `6_string_electric_open_E_chord.mp3` (E2 B2 E3 G#3 B3 E4 — 6 notes)
- `6_string_electric_open_A_chord.mp3` (A2 E3 A3 C#4 E4 — 5 notes)
- `6_string_electric_open_D_chord.mp3` (D3 A3 D4 F#4 — 4 notes)
- `6_string_electric_open_G_chord.mp3` (G2 B2 D3 G3 B3 G4 — 6 notes)
- `6_string_electric_open_C_chord.mp3` (C3 E3 G3 C4 E4 — 5 notes)

Ran the full Phase 2 pipeline (`_BasicPitchBackend` → clustering →
chord NoteEvents) against each.

### 15.1 Cluster count matches strum count

The user is strumming each chord 5–6 times. Per-recording cluster counts:

| Sample | Strums detected (clusters) | Comments |
|---|---|---|
| E chord | 5 | one strum fragments across two clusters at 4.774/4.925s |
| A chord | 6 | last "strum" at 11.489s (0.4s dur) is likely a decay tail |
| D chord | 5 | clean |
| G chord | 5 | clean |
| C chord | 6 | last two clusters at 8.643/8.794s are one strum fragmented |

The cluster algorithm is doing its job. The fragmentations happen when a
chord decays past the 100ms window with one note dropping below the
amplitude floor mid-decay and re-detecting.

### 15.2 Per-strum note recall drops sharply vs dyads

basic-pitch reliably detects 2–4 of the 4–6 chord members per strum.
High strings (G4, E4, C#4, D4) are consistently missed — they have
lower amplitude and shorter duration than the bass notes. Best per-strum
recall by chord:

| Chord | Expected | Best detected per-strum |
|---|---|---|
| E (6 notes) | E2 B2 E3 G#3 B3 E4 | (40, 47, 56) = E2 B2 G#3 (3/6) |
| A (5 notes) | A2 E3 A3 C#4 E4 | (45, 52, 57) = A2 E3 A3 (3/5) |
| D (4 notes) | D3 A3 D4 F#4 | (50, 57, 66) = D3 A3 F#4 (3/4) |
| G (6 notes) | G2 B2 D3 G3 B3 G4 | (43, 47, 50, 55) = G2 B2 D3 G3 (4/6) |
| C (5 notes) | C3 E3 G3 C4 E4 | (48, 52, 55) = C3 E3 G3 (3/5) |

basic-pitch's amplitude floor (0.65) drops the higher-voice members.
Lowering the floor catches them but re-introduces spurious mid-range
detections — Phase 2 calibration recorded 0.65 as the sweet spot for
mono samples, and changing it now would invalidate the Phase 1/Phase 2
gates.

### 15.3 Phase 3 ground-truth decision: accept actual detection

User decision (recorded 2026-05-23): same approach as Phase 2 §13.3 —
Phase 3 integration tests assert **what basic-pitch actually detects**
per strum, not idealised 5-6 of N chord-member recall. The ground truth
is recorded per-cluster, exactly. The chord-rendering pipeline (
`music21.chord.Chord`, no-string-conflict tab DP, MIDI iteration) is
the thing being validated; basic-pitch's recall on full chord voicings
is a quality issue for Phase 4 (separate spec, only if we want to
invest in chord-recall improvements via chord-template matching, lower
absolute floor + cluster-internal relative filter, or chord-aware
amplitude smoothing).

### 15.4 Implications for Phase 3 plan

- The clustering algorithm doesn't need changes. It correctly groups
  strum-events; the recall issue is upstream (basic-pitch's amplitude
  filter at the backend boundary), not in clustering.
- Phase 3's tab DP (chord-aware DP from Phase 2 Task 8) already handles
  3–4 member chords correctly — the no-string-conflict constraint
  applies regardless of chord size.
- Phase 3 integration tests follow Phase 2's pattern: one test file per
  open-chord sample (`test_guitar_open_E_chord.py` etc.) with per-cluster
  `EXPECTED_MIDI_TUPLES_BY_BACKEND["basic_pitch"]` set from the probe.
- Phase 3 should NOT attempt to "complete" partial chords by inferring
  the missing voicings — that's a chord-recognition feature out of
  Phase 3 scope.

### 15.5 Exact basic-pitch output for Phase 3 ground truth

**E chord** (5 clusters):
1. `(47, 56)` at 2.288s — B2 + G#3
2. `(40, 47, 56)` at 4.774s — E2 + B2 + G#3
3. `(47,)` at 4.925s — B2 only (decay re-detection)
4. `(40, 47)` at 7.156s — E2 + B2
5. `(47, 56)` at 9.409s — B2 + G#3

**A chord** (6 clusters):
1. `(57,)` at 1.939s — A3 only
2. `(45,)` at 4.054s — A2 only
3. `(45, 52)` at 6.041s — A2 + E3
4. `(45, 52, 57)` at 7.887s — A2 + E3 + A3
5. `(45, 52)` at 9.711s — A2 + E3
6. `(40,)` at 11.489s — spurious E2 detection at decay tail (0.4s duration)

**D chord** (5 clusters):
1. `(45, 50)` at 1.115s — A2 + D3 (A2 is below the open-D voicing; harmonic of A3)
2. `(50, 57, 66)` at 3.043s — D3 + A3 + F#4
3. `(45, 50)` at 5.041s — A2 + D3
4. `(45, 66)` at 8.550s — A2 + F#4 (no D3?)
5. `(62,)` at 9.850s — D4 only (decay tail)

**G chord** (5 clusters):
1. `(43, 47, 50, 55)` at 1.091s — G2 + B2 + D3 + G3 (best chord detection)
2. `(47, 50, 55)` at 2.881s — B2 + D3 + G3
3. `(50, 55, 59)` at 4.705s — D3 + G3 + B3
4. `(47, 50)` at 6.529s — B2 + D3
5. `(43, 47, 50, 55)` at 8.329s — G2 + B2 + D3 + G3 (matches #1)

**C chord** (6 clusters):
1. `(48, 52, 55)` at 2.033s — C3 + E3 + G3
2. `(40, 48, 55)` at 3.658s — E2 + C3 + G3 (E2 is unexpected; sympathetic from C major root)
3. `(48, 55, 60)` at 5.308s — C3 + G3 + C4
4. `(48, 60)` at 7.132s — C3 + C4
5. `(48, 55)` at 8.643s — C3 + G3
6. `(48, 52)` at 8.794s — C3 + E3 (fragmented from previous strum)

### 15.6 Observations for Phase 4 (not Phase 3)

- The "decay re-detection" pattern (E chord cluster 3, A chord cluster 6,
  D chord cluster 5, C chord cluster 6) suggests a Phase 4 enhancement:
  drop very short (<500 ms) singleton clusters when they fall within the
  decay tail of a recent multi-member cluster.
- Octave/sympathetic detections (A2 in D chord, E2 in C chord) are
  recurring patterns that a chord-template matcher could correct, but
  Phase 4 first needs evidence whether these confuse downstream
  notation/tab output enough to warrant the complexity.
- Low-string root-note misdetection: D chord cluster 4 missing the D3
  root, A chord clusters 1-2 detecting only the A note (no other
  members). The amplitude profile of basic-pitch on guitar strums
  appears strongly biased toward whichever string was struck loudest,
  not toward the harmonic structure.

---

## 16. Phase 3 Outcome (landed 2026-05-23)

Phase 3 — open-chord detection (3–6 member chords) — completed in 4
task commits on `chord-detection`. No worktree (small scope). The
phase delivered:

- 3 new tab DP unit tests for 4- and 6-member shapes + chord
  progressions (Task 2, commit `b8e1439`). Phase 2's no-string-conflict
  DP scales to the full open-chord voicing sizes with no algorithmic
  changes. All 38 tab tests now pass.
- 5 new integration test files (Tasks 3-7, batched in commit `3b3079f`),
  one per open-chord sample, asserting the exact basic-pitch detection
  ground truth from spec §15.5. **All 30 chord tests pass under
  basic_pitch on first run** — the §15.5 probe values match current
  detection exactly. CREPE backend skips them gracefully.
- PDF/MIDI render proof on the G chord sample (Task 8, commit `643740d`)
  — see `docs/superpowers/phase3_chord_proof.pdf` (38 KB). The §14.3
  quantizer bug did NOT fire on this sample, so the full real-audio
  pipeline (basic-pitch → clustering → score → chord-aware tab DP →
  MusicXML → MuseScore PDF) is verified working on actual recorded
  audio with 4-member chord shapes.

### 16.1 Phase gate results

| Backend | Result |
|---|---|
| `ATS_PITCH_BACKEND=basic_pitch` | **312 passed, 4 skipped** (was 279/4 pre-Phase-3) |
| `ATS_PITCH_BACKEND=crepe` | **284 passed, 32 skipped** (was 271/12 pre-Phase-3) |

New tests added: 3 tab DP + 30 chord integration = 33 new test functions.
Under CREPE, 20 of the 30 chord-test functions skip with "No ground
truth recorded for backend crepe" + 10 audio-shape tests pass.

### 16.2 What Phase 3 did NOT do

- **No chord-recall improvement.** basic-pitch's per-strum recall stays
  at 2–4 of 5–6 notes; ground truth records the actual detection.
  Phase 4 work (separate spec) would address chord-recall via lower
  amplitude floor + cluster-internal relative filter and/or
  chord-template matching.
- **No fragmentation cleanup.** Decay-tail re-detection (singleton
  clusters arriving within ~500ms of a multi-member cluster, see §15.5
  D/A/C chord tails) is left in the ground truth. Phase 4 could filter
  these.
- **No chord-name detection.** Phase 3 emits the detected pitches;
  recognising these as "G major" labels is a separate Phase 5+ feature.
- **No quantizer fix.** The §14.3 overlap bug affects PDF rendering on
  some samples; the G chord happened to dodge it. Phase 4 should fix
  it so all chord samples render reliably.

### 16.3 What remains for Phase 4

In rough priority order based on Phase 3 evidence:

1. **Quantizer overlap bug** (§14.3) — blocks PDF rendering on chord
   samples at clamped slow tempos. The most concrete Phase 4 item.
   Affects: E, A, D, C chord samples (G chord happens to dodge it).
2. **CLI `--backend` choice** — still lists only `{pyin,crepe}`;
   `basic_pitch` selectable only via env var. Trivial fix.
3. **Chord-recall improvement** — explore lowering the absolute
   amplitude floor + cluster-internal relative filter to catch the
   missing high voices (G4, E4, C#4, D4 in the §15.5 ground truth).
4. **Decay-tail singleton filter** — drop very short (<500 ms)
   singleton clusters that arrive within ~500ms of a recent
   multi-member cluster (would clean up §15.5 clusters 3 of E, 6 of A,
   5 of D, 6 of C).
5. **Chord-name detection** — emit "Em", "G", "C" labels alongside the
   detected pitches.

### 16.4 Phase 3 confirmed Phase 2's design

The most important Phase 3 finding: **Phase 2's algorithms required
zero changes to handle 4–6 member chords.** The N-member-agnostic
design choices (Cartesian-product shape enumeration, lexicographic DP
tie-break, chord-sibling MusicXML walker, MIDI iteration over
midi_notes) all scale without modification. This validates the design
decisions made during Phase 2 brainstorming and review.
