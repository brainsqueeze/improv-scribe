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

**basic-pitch's actual API.** `basic_pitch.inference.predict()` returns a
3-tuple `(model_output_dict, midi_data, note_events)` where `note_events` is
`list[tuple[float, float, int, float, list[int] | None]]` —
`(start_s, end_s, midi, amplitude, pitch_bends)` positionally. We unpack
positionally and ignore `pitch_bends` (we already encode microtonal deviation
in `cents_deviations`, but we set those to 0.0 for basic-pitch outputs since
they emit integer MIDI directly). The model is loaded lazily on first
`predict()` call (~1–3 s, mostly TF/TFLite import + ONNX weight load). We
keep a module-level cache of the loaded model so subsequent chunks re-use it.

**Phase 1 prerequisite — prototype the import + first call** before locking
the wrapper interface. The prototype confirms (a) the exact tuple shape the
installed version returns, (b) whether `predict()` accepts a numpy array
in-memory or only a file path (we expect the former), and (c) the actual
TF/TFLite cold-start cost. If the prototype reveals a different API shape,
this section gets a small revision before implementation continues.

`_BasicPitchBackend.estimate()` calls `predict()` on the in-memory audio
array, unpacks `note_events`, filters by `InstrumentProfile.midi_min/midi_max`
and an absolute amplitude floor (`POLYPHONIC_AMPLITUDE_FLOOR = 0.10`),
and returns a `PitchResult` with `bp_notes` populated.

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
| **basic-pitch first-call latency** — TF/TFLite import + ONNX weight load is 1–3 s on first `predict()` | Low | Cache the loaded model at module level; cold start is amortised over a session. Acceptable since the pipeline is batch (post-hoc), not real-time. |

---

## 6. Configuration additions

New entries in [config.py](../../../src/improv_scribe/config.py):

```python
# Polyphonic detection (Phase 1+)
ONSET_GROUPING_WINDOW_MS: float = float(os.getenv("ATS_ONSET_GROUPING_WINDOW_MS", "100.0"))
POLYPHONIC_AMPLITUDE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_AMPLITUDE_FLOOR", "0.10"))
POLYPHONIC_RELATIVE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_RELATIVE_FLOOR", "0.5"))
MIN_NOTE_DURATION_S: float = float(os.getenv("ATS_MIN_NOTE_DURATION_S", "0.050"))
MERGE_GAP_CHORD_MS: float = float(os.getenv("ATS_MERGE_GAP_CHORD_MS", "200.0"))
DEBUG_CLUSTERING: bool = os.getenv("ATS_DEBUG_CLUSTERING", "0") == "1"
```

These propagate into `AppConfig` with the same field-default pattern used
today.

---

## 7. Dependency additions

`envionment.yaml` gains:

- `basic-pitch>=0.4` (Spotify's polyphonic transcription model)

basic-pitch's own dependencies (pretty-midi, librosa, mir_eval, etc.) are
all already in our environment or are transitively pulled. No system-level
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
