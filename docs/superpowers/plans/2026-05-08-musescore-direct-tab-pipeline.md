# MuseScore-Direct Tab Pipeline — Roadmap

> **Status:** Roadmap / not yet scheduled. Captures intent and design so this can be picked up cleanly in a future session.

**Goal:** Add an *opt-in* alternate export path that does **not** inject `<technical><string>/<fret>` annotations from our DP fret-assigner. Instead, hand MuseScore a clean two-staff score (notation + tab) and let MuseScore choose string/fret itself. Run side-by-side with the existing DP-driven path so we can A/B compare tab quality on real samples.

**Why:** We currently compute fret assignments via `tab_builder.assign_frets()` (a phrase-aware DP that minimizes left-hand position shift) and inject them as MusicXML `<technical>` annotations. MuseScore has its own tab-rendering logic and sometimes ignores those annotations in subtle MusicXML configurations (see the bug fixed on 2026-05-08 — redundant `<transpose>` + `<clef-octave-change>` caused MuseScore to override explicit `<fret>` for the first non-rest note). Having a parallel path that trusts MuseScore lets us:
1. Verify our DP earns its keep for melodic/positional playing (where lowest-fret-anywhere is suboptimal).
2. Catch future MuseScore-vs-DP drift early.
3. Keep an escape hatch if MuseScore changes its tab logic in a future release.

**Non-goal:** Removing the DP. This is purely additive — the existing path stays the default.

---

## File Map

| File | Change |
|---|---|
| `src/improv_scribe/export/tab_xml.py` | Split `inject_tab_part` into two functions: existing one keeps DP injection; new `inject_tab_part_auto(mxl_path, profile)` adds only the staff-2 structure (clef, staff-details, staff-tuning, `<staff>` annotations) and **omits** `<technical>`. |
| `src/improv_scribe/export/pdf_exporter.py` | `PDFExporter.export()` accepts a new keyword `tab_strategy: Literal["dp", "musescore"] = "dp"`. When `"musescore"`, calls the auto variant and ignores `tab_assignments`. |
| `src/improv_scribe/cli.py` | Add `--tab-strategy {dp,musescore}` flag; thread through to exporter. |
| `tests/export/test_tab_xml.py` | New `TestInjectTabPartAuto` class: verifies two staves, tab clef, staff-tuning, **no `<technical>` elements**, all notes have `<staff>1</staff>` or `<staff>2</staff>`. |
| `tests/integration/test_*` | Add a parametric layer: each sample asserts BOTH strategies produce a non-empty PDF. Compare frame-of-reference: print rendered tab as ASCII for human inspection in test output. |
| `docs/comparisons/` (new dir) | Side-by-side rendered PDFs of each sample under both strategies — committed for visual diff review. |

---

## Design Decisions

1. **Where to branch.** Branch at the export layer, not the score layer. The `music21.Score` is the same; only the MusicXML post-processing differs. Keeps the analysis/quantization pipeline single-path.

2. **Selection mechanism.** A `tab_strategy` keyword on `PDFExporter.export()`, surfaced as a CLI flag. No env var, no config file — explicit per-call.

3. **What MuseScore needs to render tab on its own.** Just the staff-2 scaffolding:
   - `<staves>2</staves>`
   - `<clef number="2"><sign>TAB</sign></clef>`
   - `<staff-details number="2">` with `<staff-type>tab</staff-type>` and 6 (or 4) `<staff-tuning>` entries (low → high)
   - `<staff>1</staff>` on existing notes, plus a deep-copy of each note tagged `<staff>2</staff>` after a `<backup>` of measure duration.
   - **No `<technical>`** — the absence of explicit string/fret is what triggers MuseScore's auto-pick.

4. **Tied notes.** Same handling as the DP path — staff-2 copies are deep copies of the existing `<note>` elements including any `<tie>` elements.

5. **Octave handling.** Concert pitch + `treble8vb`/`bass8vb` clef, no `<transpose>` injection — same as the post-2026-05-08 baseline. MuseScore needs unambiguous pitch info to pick frets correctly.

6. **A/B compare in tests.** Don't try to assert MuseScore-side fret values from inside Python tests (we'd be re-implementing MuseScore's logic to know what to assert). Instead, render both PDFs, extract tab text via `pdftotext`, and emit a diff report. Failures: no PDF output, missing TAB clef, zero rendered fret numbers. Tab-quality differences are reviewed by a human.

---

## Open Questions

- **Does MuseScore re-derive string/fret per measure or per phrase?** Affects whether MuseScore's auto-pick can ever match a DP that minimizes shift across barlines. Worth a small experiment with a sample where the DP and lowest-fret would diverge (e.g. melody crossing strings 4–6).

- **Capo / alt-tuning support.** Out of scope for the first cut. If we add it later, both paths need consistent tuning input.

- **Bass: 5-string future support.** Plan assumes 4-string bass. If we add 5-string, both `BASS_TUNING` (in `tab_builder.py`) and `BASS_STAFF_TUNING` (in `tab_xml.py`) need to grow together — consider deriving one from the other to prevent skew.

---

## Suggested First Slice

Smallest useful prototype, ~half a day:

1. Add `inject_tab_part_auto()` next to `inject_tab_part()` in `tab_xml.py`. Refactor shared helpers (`_inject_two_staves_attributes`, the staff-2 deep-copy walk) into a private function both call.
2. Wire `tab_strategy="musescore"` through `PDFExporter.export()`.
3. Add a single integration smoke test on `samples/guitar/6_string_electric_line_in.mp3`: assert both strategies produce a PDF > 10 KB containing a TAB clef.
4. Render both PDFs by hand, eyeball the tab. Capture findings in this file.

If the eyeball check shows MuseScore picking equivalent frets on open-string samples, expand to mic/acoustic/bass samples and write the human-review test harness.
