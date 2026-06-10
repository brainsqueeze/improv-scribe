# Precision Audit: basic-pitch Tab Transcription Pipeline

**Date:** 2026-06-10
**Scope:** Localize where fret/string/tab precision degrades in the
basic-pitch path: model output → filtering → clustering → NoteEvents →
fret/string assignment → tab MusicXML.

---

## Method

Every pipeline boundary was instrumented and replayed against all 12 sample
recordings (5 open chords, 3 dyad sequences, 3 mono guitar, 1 bass).
Raw `basic_pitch.inference.predict()` events (with per-note amplitudes) were
dumped *before* any filtering and compared against musical ground truth, then
each filter's drops were attributed. Candidate fixes were scored with
per-(event, pitch) precision/recall over all samples. Suspicious
"false positives" were verified spectrally (FFT band energy at the
fundamental vs. semitone-offset control bands) to separate model
hallucinations from strings that physically sounded in the recording.

---

## Verdict: where precision is actually lost

| Stage | Verdict |
|---|---|
| 1. Pitch detection (basic-pitch model) | **Not the problem.** The raw model detects essentially 100 % of ground-truth chord members on every sample. |
| 2. Note identification (our filtering + clustering in `pitch.py` / `note_tracker.py`) | **This is where ~all of the loss happens.** The 0.65 absolute amplitude floor silently deletes 2–4 of 6 chord members per strum; secondary bugs in the range gate, cluster window, and fragment handling compound it. |
| 3. Fret/string assignment (`tab_builder.py` DP) | Correct *given its input*. With full chords restored, the DP is forced into the natural open-chord shapes. Garbage-in was the issue. |
| 4. Tab MusicXML (`tab_xml.py`) | Correct. String numbering (`n_strings - string_idx`), staff-tuning lines, MIDI-ordered chord zipping, and backup-cursor handling all check out. |

The integration test suite passes (117/117) **because the chord-test
"ground truths" were recorded from the broken pipeline itself** — e.g.
`test_guitar_open_E_chord.py` *expects* `(47, 56)` for a six-note E-major
strum. The tests regression-lock the degradation; they don't measure
accuracy.

---

## Findings (ranked by impact)

### F1 — `POLYPHONIC_AMPLITUDE_FLOOR = 0.65` destroys chord recall (critical)

Raw amplitudes on the open-E sample: the six chord members register at
0.42–0.76 *per strum*, but the floor keeps only those ≥ 0.65 (typically E2,
B2, G#3). High voices (E3/B3/E4: 0.42–0.60) are deleted **before
clustering**. Measured chord-member recall: **33 % (open E), 30 % (open A),
48 % (open C)**. Meanwhile genuine ghosts (harmonic re-detections of ringing
strings) sit at 0.29–0.43 — the distributions overlap, so *no* absolute
floor can separate them. Separation requires context (cluster membership,
attack alignment, ring history — see fixes).

### F2 — Guitar `midi_max = 98` is an octave-error bug (high)

`instrument_profiles.py` says `midi_max=98  # D6 (sounding)` — but D6 is
MIDI **86**; 98 is D7. `freq_max_hz=1174.66` (D6) and `MAX_FRET=22` (E4+22
= 86) both agree on 86. Consequence: hallucinated B6/A#6/C#7 events (amp
0.29–0.40, present in most samples) pass the range gate and are only
stopped today by the same over-aggressive floor that kills real chord
members. Fixing the floor *requires* fixing this gate.

### F3 — `predict()` is called with no frequency bounds (high)

`predict()` accepts `minimum_frequency` / `maximum_frequency`, which crop the
model's posteriorgram before note assembly. We pass neither, so the model
freely emits E7/F7/B6 hallucinations that downstream code must clean up.
Passing the instrument profile's bounds suppresses them at the source.

### F4 — 100 ms cluster window splits real strums (medium)

`ONSET_GROUPING_WINDOW_MS = 100` was calibrated on *strong-string* onset
spreads (0–35 ms). But basic-pitch emits weak strings late (activation
crosses threshold mid-attack): measured spreads run to ~280 ms (open G
strum 2: G2 at 2.764 s → G4 at 2.962 s). Members past the window become
singleton clusters and die at the singleton floor. 250 ms captures the
measured spreads while still separating eighth-note strums at 100 BPM.

### F5 — Attack fragmentation mis-handled (medium)

basic-pitch frequently splits one re-articulated note into
`[140 ms fragment][long event]` with ~0 gap. The fragment and the long event
land in different clusters; `MIN_NOTE_DURATION_S` deletes fragments that
carried the true onset time. Naive gap-merging is wrong too (it fuses
*re-articulations across strums* — B2 in open E merges into one 9.5 s
note). Correct rule: absorb **short** (<180 ms) fragments into the adjacent
same-pitch event (preferring forward/re-attack), never merge two long events.

### F6 — Ring-over ghosts and octave re-detections (medium)

With the floor lowered, weak re-detections of *still-ringing earlier notes*
(e.g. E2 re-emitted at amp 0.41 during the D3 pluck) join clusters as false
chord members. Spectral checks confirmed some are pure hallucinations (E2
"during B3" on acoustic line-in: band-energy ratio 0.5 vs. control — nothing
there). Discriminator that works: a non-max cluster member whose pitch
chain-links (gap ≤ 3 s) to an earlier detection with ≥ 1.67× its amplitude
is a ring-over, not a new note.

### F7 — Onset information is unused in the basic-pitch path (medium)

`NoteTracker.process()` receives librosa onsets but `_process_basic_pitch()`
ignores them. Attack alignment is exactly the evidence that separates weak
real strums from decay-phase ghost clusters. Caveat measured on real data:
librosa missed 2 of 6 open-A strums, so gating must be **conditional** —
only weak clusters (max amp < 0.65) require onset corroboration; strong
clusters stand on their own. Unconditional gating costs more recall than it
buys precision.

### F8 — Bass is monophonic but clustered polyphonically (low)

Long-ringing bass strings produce strong (amp 0.6+) octave/unison
re-detections that form false dyads — e.g. `(28, 38)` while D2 plays. The
MVP scope for bass is monophonic; keeping only the max-amplitude member per
bass cluster makes the bass sample score perfect 1.00/1.00. Added as a
per-profile `max_polyphony` (bass = 1, guitar = 6).

### F9 — Test "ground truth" is recorded pipeline output (process issue)

Chord/dyad integration tests assert the under-detected tuples and only
shape-level tab properties ("distinct strings"), never actual fret/string
correctness for chords. Two of the "expected" values even contain
**spectrally-confirmed real notes labelled as artifacts** (the low-E string
audibly sounds in the open-A/C strums — verified: band-energy ratio
1 400–10 800× over control; and A2 sounds in every open-D strum). Tests
should assert musically-verified truth.

### Verified non-issues

- **Tab XML** (`tab_xml.py`): MusicXML string numbering, `<staff-tuning>`
  line order, chord-slot grouping, tie handling, `<backup>` durations — all
  correct by inspection and PDF smoke test.
- **Fret DP** (`tab_builder.py`): with complete chords, candidate
  enumeration + stretch/transition cost lands on standard open-chord shapes
  (e.g. full open E is *forced* to 022100 by the distinct-strings
  constraint). Cost model is sane for MVP.
- **Quantizer**: pitch passthrough confirmed by existing tests.

---

## Fix plan (executed)

All thresholds were selected by sweep over the 12-sample corpus; each rule
was ablated to confirm net contribution.

1. **`instrument_profiles.py`** — guitar `midi_max` 98 → **86**; add
   `max_polyphony` field (guitar 6, bass 1).
2. **`pitch.py` (backend)** — pass `minimum_frequency`/`maximum_frequency`
   from the profile into `predict()`; replace the 0.65 floor with a
   permissive pre-floor (**0.25**); add forward-preferring attack-fragment
   merge (frag < 180 ms, gap ≤ 100 ms); keep range + min-duration gates.
3. **`note_tracker.py` (cluster stage)** — widen cluster window to
   **250 ms**; apply context-aware floors after dedupe:
   singleton clusters ≥ **0.65**; multi-member clusters ≥ max(**0.30**,
   0.5 × cluster max). Add ring-over suppression (non-max member, chain gap
   ≤ 3 s, amp < 0.6 × chain max → drop). Add conditional onset gating
   (clusters with max amp < 0.65 must align with a librosa onset,
   −200/+150 ms). Enforce `profile.max_polyphony`.
4. **`config.py`** — new tunables with env overrides for all of the above.
5. **Tests** — re-derive integration expectations from musically-verified
   truth; update unit tests for the new filter semantics.

### Measured outcome (12-sample corpus, corrected ground truth)

| Pipeline | Precision | Recall | F1 |
|---|---|---|---|
| Current production | 0.94 | 0.70 | 0.84 |
| **Fixed (this plan)** | **0.94** | **0.96** | **0.95** |

Chord-member recall on the worst samples: open E 0.33 → 0.83, open A
0.30 → 0.81, open C 0.48 → 1.00, octave dyads 0.75 → 1.00, thirds/fifths
→ 1.00. Mono guitar and bass remain/become exact.

### Known remaining limitations

- Acoustic mic/line samples emit 2 octave-double FPs (e.g. E3 with E2) —
  the 2nd harmonic of a mic'd acoustic is strong enough that amplitude alone
  cannot separate it from a played octave dyad without breaking the octave
  dyad sample. A future spectral-presence check (audio is already plumbed
  into `NoteTracker.process`) is the right tool.
- Decay-tail clusters occasionally survive as a final junk event when
  librosa reports an onset there (open E emits one `(50, 56)` tail event).
- Open A loses E4 in 5 of 6 strums (raw amp 0.34–0.42 against cluster max
  0.7+, below the relative floor); recoverable only with per-string source
  separation, out of MVP scope.
