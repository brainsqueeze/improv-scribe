"""
export/tab_xml.py — Injects linked tablature staves into an existing MusicXML file.

MuseScore requires a single part with two staves for proper notation+TAB rendering:
  - Staff 1: standard notation (existing)
  - Staff 2: tablature (injected)

MusicXML structure produced:
  <part id="P1">
    <measure number="1">
      <attributes>
        <staves>2</staves>
        <clef number="1">...</clef>        ← existing, gets number="1"
        <clef number="2"><sign>TAB</sign></clef>   ← injected
        <staff-details number="2">         ← injected
          <staff-type>tab</staff-type>
          <staff-lines>6</staff-lines>
          <staff-tuning line="1">...</staff-tuning>
          ...
        </staff-details>
      </attributes>
      <note>                        ← existing note, gains <staff>1</staff> + <technical>
        <staff>1</staff>
        <notations><technical><string>N</string><fret>M</fret></technical></notations>
      </note>
      <note>                        ← new staff-2 copy
        <staff>2</staff>
        <notations><technical><string>N</string><fret>M</fret></technical></notations>
      </note>
    </measure>
  </part>
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from improv_scribe.analysis.instrument_profiles import Instrument, InstrumentProfile
from improv_scribe.quantization.grid import QuantizedNote

# ---------------------------------------------------------------------------
# Tuning data for <staff-tuning> elements
# Strings ordered high → low, indexed by MusicXML line number (line 1 = high
# string in our numbering, which aligns with <string>1</string> = high string).
# Values are SOUNDING pitch.  inject_tab_part also writes <transpose chromatic=N>
# so MuseScore correctly converts written → sounding before computing fret positions.
# ---------------------------------------------------------------------------

# Guitar: string 1 (high E4 sounding) → string 6 (low E2 sounding)
GUITAR_STAFF_TUNING: list[tuple[str, int]] = [
    ("E", 4), ("B", 3), ("G", 3), ("D", 3), ("A", 2), ("E", 2),
]

# Bass: string 1 (high G2 sounding) → string 4 (low E1 sounding)
BASS_STAFF_TUNING: list[tuple[str, int]] = [
    ("G", 2), ("D", 2), ("A", 1), ("E", 1),
]

_STAFF_TUNING: dict[Instrument, list[tuple[str, int]]] = {
    Instrument.GUITAR: GUITAR_STAFF_TUNING,
    Instrument.BASS: BASS_STAFF_TUNING,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_tab_part(
    mxl_path: Path,
    notes: list[QuantizedNote],
    assignments: list[tuple[int, int] | None],
    profile: InstrumentProfile,
) -> None:
    """
    Modify *mxl_path* in-place to add a linked TAB staff to the existing notation staff.

    The existing P1 part is modified to contain two staves:
    staff 1 (notation) and staff 2 (TAB). Each non-rest note gains a ``<staff>1</staff>``
    element plus ``<technical><string>/<fret>`` annotations, and a staff-2 copy is
    inserted immediately after.

    Parameters
    ----------
    mxl_path : Path
        Path to an existing MusicXML file (will be overwritten).
    notes : list[QuantizedNote]
        The same note list used to build the score (in order).
    assignments : list[tuple[int, int] | None]
        Fret assignments from tab_builder.assign_frets(). Parallel to notes.
        Each entry is (string_idx, fret) or None for rests.
    profile : InstrumentProfile
        Used for string count and tuning info.
    """
    tree = ET.parse(mxl_path)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def tag(name: str) -> str:
        return f"{ns}{name}"

    p1 = root.find(f".//{tag('part')}")
    if p1 is None:
        raise ValueError("MusicXML file has no <part> element.")

    tuning_data = _STAFF_TUNING.get(profile.instrument, GUITAR_STAFF_TUNING)
    n_strings = len(tuning_data)

    # -----------------------------------------------------------------------
    # Build note_assignment_map: id(note_el) → (string_idx, fret)
    # Only non-rest XML notes are matched; music21-inserted rests are skipped.
    # Tied continuation notes (tie type="stop") do not consume a new assignment;
    # they reuse the assignment from the note they continue so that the TAB staff
    # shows the same fret for the duration of the tie.
    # -----------------------------------------------------------------------
    pitched_assignments = [a for a in assignments if a is not None]
    pitched_iter = iter(pitched_assignments)

    def _is_rest_element(note_el: ET.Element) -> bool:
        return note_el.find(tag("rest")) is not None

    def _is_tie_continuation(note_el: ET.Element) -> bool:
        return any(t.get("type") == "stop" for t in note_el.findall(tag("tie")))

    note_assignment_map: dict[int, tuple[int, int]] = {}
    current_assignment: tuple[int, int] | None = None
    for measure_el in p1.findall(tag("measure")):
        for note_el in measure_el.findall(tag("note")):
            if _is_rest_element(note_el):
                pass  # rests never appear mid-tie; don't reset current_assignment
            elif _is_tie_continuation(note_el):
                if current_assignment is not None:
                    note_assignment_map[id(note_el)] = current_assignment
            else:
                current_assignment = next(pitched_iter, None)
                if current_assignment is not None:
                    note_assignment_map[id(note_el)] = current_assignment

    # -----------------------------------------------------------------------
    # Process the first measure: inject <staves>, tab clef, staff-details
    # -----------------------------------------------------------------------
    first_measure = p1.find(tag("measure"))
    if first_measure is None:
        raise ValueError("MusicXML part has no measures.")

    attrs_el = first_measure.find(tag("attributes"))
    if attrs_el is None:
        attrs_el = ET.Element(tag("attributes"))
        first_measure.insert(0, attrs_el)

    _inject_two_staves_attributes(attrs_el, tuning_data, tag, profile.transpose_semitones)

    # -----------------------------------------------------------------------
    # For every measure: annotate staff-1 notes and insert staff-2 copies
    # -----------------------------------------------------------------------
    for measure_el in p1.findall(tag("measure")):
        _annotate_measure(measure_el, note_assignment_map, n_strings, tag)

    tree.write(str(mxl_path), encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _inject_two_staves_attributes(
    attrs_el: ET.Element,
    tuning_data: list[tuple[str, int]],
    tag: Callable[[str], str],
    transpose_semitones: int = 0,
) -> None:
    """Add ``<staves>2</staves>``, a numbered tab ``<clef>``, ``<staff-details>``,
    and (when *transpose_semitones* is non-zero) a ``<transpose>`` element.

    The ``<transpose>`` element is required when the score is in written pitch
    (e.g. guitar, written an octave above sounding).  Without it MuseScore treats
    note pitch as concert pitch and cannot correctly compute TAB fret positions
    against the sounding-pitch string tuning values.
    """
    n_strings = len(tuning_data)

    # <staves>2</staves> — must appear before <clef> per MusicXML schema
    staves_el = ET.Element(tag("staves"))
    staves_el.text = "2"
    attrs_el.insert(0, staves_el)

    # Number the existing clef as staff 1 (if any)
    existing_clef = attrs_el.find(tag("clef"))
    if existing_clef is not None and not existing_clef.get("number"):
        existing_clef.set("number", "1")

    # <clef number="2"><sign>TAB</sign></clef>
    tab_clef = ET.SubElement(attrs_el, tag("clef"))
    tab_clef.set("number", "2")
    sign_el = ET.SubElement(tab_clef, tag("sign"))
    sign_el.text = "TAB"

    # <staff-details number="2">
    staff_details = ET.SubElement(attrs_el, tag("staff-details"))
    staff_details.set("number", "2")

    staff_type = ET.SubElement(staff_details, tag("staff-type"))
    staff_type.text = "tab"

    staff_lines_el = ET.SubElement(staff_details, tag("staff-lines"))
    staff_lines_el.text = str(n_strings)

    for line_num, (step, octave) in enumerate(tuning_data, start=1):
        st = ET.SubElement(staff_details, tag("staff-tuning"))
        st.set("line", str(line_num))
        step_el = ET.SubElement(st, tag("tuning-step"))
        step_el.text = step
        octave_el = ET.SubElement(st, tag("tuning-octave"))
        octave_el.text = str(octave)

    if transpose_semitones != 0:
        transpose_el = ET.SubElement(attrs_el, tag("transpose"))
        chromatic_el = ET.SubElement(transpose_el, tag("chromatic"))
        chromatic_el.text = str(transpose_semitones)


def _annotate_measure(
    measure_el: ET.Element,
    note_assignment_map: dict[int, tuple[int, int]],
    n_strings: int,
    tag: Callable[[str], str],
) -> None:
    """
    For each note in *measure_el*:
    - Add ``<staff>1</staff>`` and fret/string technical annotation (non-rest notes only)
    - Append a staff-2 copy after a ``<backup>`` element that resets the cursor

    MusicXML cursor semantics: each ``<note>`` without ``<chord/>`` advances the
    cursor by its duration.  Staff-2 notes must be preceded by a ``<backup>`` that
    resets the cursor to the beginning of the measure; otherwise they appear at
    beat positions past the barline and are not rendered.
    """
    original_notes = list(measure_el.findall(tag("note")))
    staff2_notes: list[ET.Element] = []
    total_duration = 0

    for note_el in original_notes:
        is_rest = note_el.find(tag("rest")) is not None
        assignment = note_assignment_map.get(id(note_el))

        # Add <staff>1</staff> to the original note
        staff1_el = ET.SubElement(note_el, tag("staff"))
        staff1_el.text = "1"

        # Add technical annotation to staff-1 note (non-rest with assignment)
        if not is_rest and assignment is not None:
            string_idx, fret = assignment
            string_num = n_strings - string_idx
            _add_technical_annotation(note_el, string_num, fret, tag)

        # Accumulate measure duration from notes that advance the cursor.
        # Notes with <chord/> play simultaneously with the preceding note and
        # do not advance the cursor — skip them.
        if note_el.find(tag("chord")) is None:
            dur_el = note_el.find(tag("duration"))
            if dur_el is not None:
                try:
                    total_duration += int(dur_el.text or "0")
                except ValueError:
                    pass

        # Build staff-2 copy
        staff2_note = copy.deepcopy(note_el)
        staff_el = staff2_note.find(tag("staff"))
        if staff_el is not None:
            staff_el.text = "2"
        else:
            s = ET.SubElement(staff2_note, tag("staff"))
            s.text = "2"
        staff2_notes.append(staff2_note)

    if not staff2_notes or total_duration == 0:
        return

    # <backup> resets the MusicXML cursor to the start of the measure so
    # the staff-2 notes are placed at the correct beat positions.
    backup_el = ET.Element(tag("backup"))
    backup_dur = ET.SubElement(backup_el, tag("duration"))
    backup_dur.text = str(total_duration)
    measure_el.append(backup_el)

    for note in staff2_notes:
        measure_el.append(note)


def _add_technical_annotation(
    note_el: ET.Element,
    string_num: int,
    fret: int,
    tag: Callable[[str], str],
) -> None:
    """Inject ``<notations><technical><string>/<fret>`` into *note_el*."""
    notations = note_el.find(tag("notations"))
    if notations is None:
        notations = ET.SubElement(note_el, tag("notations"))
    technical = ET.SubElement(notations, tag("technical"))
    string_el = ET.SubElement(technical, tag("string"))
    string_el.text = str(string_num)
    fret_el = ET.SubElement(technical, tag("fret"))
    fret_el.text = str(fret)
